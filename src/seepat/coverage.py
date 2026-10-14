"""Opt-in Train-only timing pilot; never decodes media or exports training inputs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.data.sampling import canonical_source_group
from seepat.live_progress import ProgressCallback
from seepat.preprocessing.events import parse_segments
from seepat.training.dataset import sequence_frame_indices

VERSION = "context-coverage-pilot-v1"
CATEGORIES = {"real", "audio_modified", "visual_modified", "both_modified"}


@dataclass(frozen=True)
class CoveragePilotJob:
    name: str
    source_manifest: Path
    video_manifest: Path
    event_manifest: Path
    output_dir: Path
    expected_sha256: dict[str, str]
    max_videos: int = 128
    seed: int = 20261013

    def validate(self) -> None:
        if not self.name or type(self.max_videos) is not int or not 1 <= self.max_videos <= 128:
            raise ValueError("Coverage pilot needs a name and max_videos between 1 and 128")
        if type(self.seed) is not int:
            raise ValueError("Coverage pilot seed must be an integer")
        if set(self.expected_sha256) != {"source_manifest", "video_manifest", "event_manifest"}:
            raise ValueError("Coverage pilot requires all three expected input hashes")
        for key, value in self.expected_sha256.items():
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"Invalid SHA-256 for {key}")
            if getattr(self, key).resolve().is_relative_to(self.output_dir.resolve()):
                raise ValueError("Pilot output cannot contain an input file")


def select_pilot_sources(rows: list[dict[str, str]], count: int, seed: int) -> list[dict]:
    """Rank canonical groups without consulting category, annotation or eligibility."""
    groups = [canonical_source_group(r.get("source_group") or r.get("original"), r["file"])
              for r in rows]
    if len(set(groups)) != len(groups):
        raise ValueError("Pilot source manifest contains duplicate source groups")
    if len({r["file"] for r in rows}) != len(rows) or len(rows) < count:
        raise ValueError("Pilot source manifest has duplicate files or insufficient rows")
    ranked = sorted(zip(groups, rows), key=lambda pair: (stable_id(f"{seed}\0{pair[0]}", 64),
                                                        pair[0]))
    return [row for _, row in ranked[:count]]


def context_windows(frame_count: int, fps: float) -> list[dict[str, object]]:
    """At most eight 0.64 s windows at equal-bin centers; 16 frames/masks each."""
    if type(frame_count) is not int or frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Context planning requires positive finite frame metadata")
    span = min(frame_count, max(1, round(0.64 * fps)))
    windows = []
    seen = set()
    indices, mask = sequence_frame_indices(span, 16)
    for slot in range(8):
        start = min(frame_count - span, max(0, round((slot + 0.5) * frame_count / 8 - span / 2)))
        if start in seen:
            continue
        seen.add(start)
        windows.append({
            "slot": slot,
            "start_frame": start,
            "stop_frame_exclusive": start + span,
            "source_frames": [start + i for i in indices],
            "frame_mask": mask,
        })
    return windows


def frames_overlap(frames: set[int], fps: float, segments: list[tuple[float, float]]) -> bool:
    return any(min((f + 1) / fps, end) > max(f / fps, start)
               for f in frames for start, end in segments)


def _contract(job: CoveragePilotJob) -> dict[str, object]:
    job.validate()
    inputs = {}
    for key, expected in job.expected_sha256.items():
        path = getattr(job, key)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen coverage-pilot input changed: {key}")
        inputs[key] = {"path": path.as_posix(), "sha256": actual}
    return {
        "version": VERSION, "name": job.name, "inputs": inputs,
        "max_videos": job.max_videos, "seed": job.seed,
        "context_window_cap": 8, "window_seconds": 0.64, "sequence_length": 16,
        "code_sha256": file_sha256(Path(__file__)),
        "frame_helper_sha256": file_sha256(Path(__file__).parent / "training/dataset.py"),
    }


def coverage_outputs_are_current(job: CoveragePilotJob) -> bool:
    try:
        contract = _contract(job)
        saved = json.loads((job.output_dir / "summary.json").read_text(encoding="utf-8"))
        return (saved.get("status") == "complete" and saved.get("contract") == contract
                and json.loads((job.output_dir / "contract.json").read_text()) == contract
                and set(saved["output_sha256"]) == {"context_plan.csv", "video_coverage.csv"}
                and all(file_sha256(job.output_dir / name) == sha
                        for name, sha in saved["output_sha256"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _inputs(job: CoveragePilotJob) -> tuple[list[dict], dict[str, dict], dict[str, list]]:
    sources = read_csv_rows(job.source_manifest)
    videos = read_csv_rows(job.video_manifest)
    events = read_csv_rows(job.event_manifest)
    for rows, split_key in ((sources, "split"), (videos, "split"), (events, "dataset_split")):
        if any(r.get(split_key) != "train" or r.get("dataset_source", "AV-Deepfake1M++")
               != "AV-Deepfake1M++" for r in rows):
            raise ValueError("Coverage pilot accepts original AV++ Train tables only")
    if any(r.get("modify_type") not in CATEGORIES for r in sources):
        raise ValueError("Unsupported source category")
    by_file = {r["file"]: r for r in videos}
    if len(by_file) != len(videos) or {r["file"] for r in sources} != set(by_file):
        raise ValueError("Source/terminal video membership must match without replacements")
    if any(r["video_id"] != stable_id(r["file"]) for r in videos):
        raise ValueError("Video IDs disagree with source files")
    sources_by_file = {r["file"]: r for r in sources}
    grouped = defaultdict(list)
    event_ids = set()
    for row in events:
        source = sources_by_file.get(row["file"])
        if (source is None or row["event_id"] in event_ids
                or row["video_id"] != stable_id(row["file"])
                or canonical_source_group(row["source_group"], row["file"])
                != canonical_source_group(source.get("source_group") or source.get("original"),
                                          source["file"])):
            raise ValueError("Event identity/source-group provenance mismatch")
        event_ids.add(row["event_id"])
        grouped[row["video_id"]].append(row)
    return sources, by_file, grouped


def _audit_video(source: dict, video: dict, events: list[dict]) -> tuple[dict, list[dict]]:
    identifier = video["video_id"]
    row = {"video_id": identifier, "file": source["file"],
           "source_group": canonical_source_group(
               source.get("source_group") or source.get("original"), source["file"]),
           "category": source["modify_type"], "baseline_events": len(events),
           "context_windows": 0, "status": "unavailable", "reason": ""}
    plans = []
    if video.get("pipeline_status") != "complete" or video.get("eligibility_status") != "eligible":
        row["reason"] = "existing_video_incomplete_or_ineligible"
        return row, plans
    if not events:
        row["reason"] = "no_retained_bilabial_events"
        return row, plans
    try:
        fps = float(video["fps"])
        frame_count = int(video.get("video_frame_count") or source["video_frames"])
        windows = context_windows(frame_count, fps)
        base_frames = set()
        for event in events:
            frames = json.loads(event["mouth_crop_frame_indices_json"])
            if not frames or any(type(f) is not int or not 0 <= f < frame_count for f in frames):
                raise ValueError("Invalid baseline source-frame map")
            chosen, mask = sequence_frame_indices(len(frames), 16)
            base_frames.update(frames[i] for i, valid in zip(chosen, mask) if valid)
    except (ValueError, KeyError, TypeError) as error:
        row["reason"] = f"invalid_frame_metadata: {error}"
        return row, plans
    added_frames = set()
    for window in windows:
        added_frames.update(f for f, valid in zip(window["source_frames"], window["frame_mask"])
                            if valid)
        plans.append({
            "context_id": f"{identifier}-context-{window['slot']:03d}",
            "video_id": identifier, "file": source["file"], "source_group": row["source_group"],
            "dataset_split": "train", "fps_proxy": fps,
            "start_frame": window["start_frame"],
            "stop_frame_exclusive": window["stop_frame_exclusive"],
            "source_frames_json": json.dumps(window["source_frames"]),
            "frame_mask_json": json.dumps(window["frame_mask"]),
        })
    row.update(context_windows=len(windows), status="audited", reason="")
    # Annotations are read only after the annotation-free context plan is constructed.
    try:
        for kind, field in (("general", "fake_segments_json"),
                            ("audio", "audio_fake_segments_json"),
                            ("visual", "visual_fake_segments_json")):
            segments = parse_segments(source[field])
            if any(not math.isfinite(a) or not math.isfinite(b) or a < 0 for a, b in segments):
                raise ValueError("Annotation intervals must be finite and nonnegative")
            row[f"baseline_{kind}_hit"] = frames_overlap(base_frames, fps, segments)
            row[f"context_{kind}_hit"] = frames_overlap(added_frames, fps, segments)
            row[f"combined_{kind}_hit"] = frames_overlap(base_frames | added_frames, fps, segments)
        if source["modify_type"] in {"visual_modified", "both_modified"} and not parse_segments(
            source["visual_fake_segments_json"]
        ):
            raise ValueError("Visual-modified source has no visual annotation")
    except (ValueError, KeyError, TypeError) as error:
        row.update(status="annotation_unavailable", reason=str(error))
    return row, plans


def run_coverage_pilot(job: CoveragePilotJob, progress: ProgressCallback | None = None) -> dict:
    if progress:
        progress("verify coverage pilot tables", 0, 0, "")
    contract = _contract(job)
    if coverage_outputs_are_current(job):
        return {"name": job.name, "action": "skipped",
                "summary": json.loads((job.output_dir / "summary.json").read_text())}
    contract_path = job.output_dir / "contract.json"
    if (job.output_dir.exists() and any(job.output_dir.iterdir())
            and (not contract_path.is_file()
                 or json.loads(contract_path.read_text()) != contract
                 or (job.output_dir / "summary.json").exists())):
        raise FileExistsError("Existing pilot output differs; inspect it before using a fresh output")
    sources, videos, grouped = _inputs(job)
    selected = select_pilot_sources(sources, job.max_videos, job.seed)
    atomic_write_json(contract_path, contract)
    coverage, plans = [], []
    for index, source in enumerate(selected, 1):
        video = videos[source["file"]]
        row, windows = _audit_video(source, video, grouped[video["video_id"]])
        coverage.append(row)
        plans.extend(windows)
        if progress:
            progress("audit proposed context coverage", index, len(selected), video["video_id"])
    visual = [r for r in coverage if r["status"] == "audited"
              and r["category"] in {"visual_modified", "both_modified"}]
    baseline = sum(r["baseline_visual_hit"] for r in visual)
    combined = sum(r["combined_visual_hit"] for r in visual)
    gain = (combined - baseline) / len(visual) if visual else None
    adequate = sum(r["status"] == "audited" for r in coverage) >= 100 and len(visual) >= 30
    gate = adequate and gain is not None and gain >= 0.10
    atomic_write_csv(job.output_dir / "context_plan.csv", plans)
    atomic_write_csv(job.output_dir / "video_coverage.csv", coverage)
    summary = {
        "status": "complete", "contract": contract,
        "scope": "Train timing-only proxy; no media, models, training export or new eligibility",
        "selected_videos": len(selected), "videos_audited": sum(r["status"] == "audited"
                                                                   for r in coverage),
        "status_counts": dict(Counter(r["status"] for r in coverage)),
        "selected_categories": dict(Counter(r["category"] for r in coverage)),
        "visual_manipulated_videos_audited": len(visual),
        "baseline_visual_covered": baseline, "combined_visual_covered": combined,
        "visual_coverage_gain": gain, "proposed_context_windows": len(plans),
        "gate": {"adequate_sample": adequate, "minimum_gain": 0.10,
                 "proceed_to_media_qa": gate,
                 "decision": "review_bounded_media_QA" if gate else "stop_and_review"},
        "limitations": ["Timestamp/FPS overlap is not verified pixel or acoustic evidence",
                        "Face/mouth quality outside bilabials is unverified",
                        "Fixed added cap is not a matched total classifier clip budget",
                        "Audio-annotation time overlap is not a learned audio feature",
                        "No predictive improvement/FPR estimate or training authorization"],
        "output_sha256": {name: file_sha256(job.output_dir / name)
                          for name in ("context_plan.csv", "video_coverage.csv")},
    }
    atomic_write_json(job.output_dir / "summary.json", summary)
    return {"name": job.name, "action": "ran", "summary": summary}
