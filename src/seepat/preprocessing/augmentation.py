"""Restartable visual-only measurements alongside immutable VILD traces."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_gzip_json,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    read_gzip_json,
    stable_id,
)
from seepat.live_progress import ProgressCallback
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.preprocessing.vild import load_vild_trace

AUGMENTATION_VERSION = "vild-augmentation-v1"


@dataclass(frozen=True)
class TraceAugmentationJob:
    name: str
    manifest: Path
    extracted_root: Path
    output_dir: Path
    max_videos: int | None = None
    normalized_tolerance: float = 1e-5
    timestamp_tolerance_s: float = 1e-6

    def validate(self) -> None:
        if self.max_videos is not None and (
            not isinstance(self.max_videos, int)
            or isinstance(self.max_videos, bool)
            or self.max_videos < 1
        ):
            raise ValueError("max_videos must be positive")
        for value in (self.normalized_tolerance, self.timestamp_tolerance_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Augmentation tolerances must be finite and nonnegative")


def measurement_contract(job: TraceAugmentationJob) -> dict[str, object]:
    job.validate()
    return {
        "version": AUGMENTATION_VERSION,
        "normalized_tolerance": job.normalized_tolerance,
        "timestamp_tolerance_s": job.timestamp_tolerance_s,
        "mediapipe": version("mediapipe"),
        "opencv": version("opencv-contrib-python"),
        "measurement_code_sha256": file_sha256(Path(__file__).with_name("face.py")),
        "face_bbox_definition": "diagonal of all FaceMesh landmarks in pixel coordinates",
        "min_detection_confidence": 0.5,
    }


def group_video_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    event_ids: set[str] = set()
    for row in rows:
        if not row.get("video_id") or not row.get("event_id"):
            raise ValueError("Manifest requires video_id and event_id")
        if row["event_id"] in event_ids:
            raise ValueError("Manifest contains duplicate event IDs")
        event_ids.add(row["event_id"])
        grouped[row["video_id"]].append(row)
    for group in grouped.values():
        for field in (
            "dataset_split",
            "file",
            "vild_trace_path",
            "vild_trace_sha256",
            "vild_augmentation_path",
            "vild_augmentation_sha256",
        ):
            if len({row.get(field, "") for row in group}) != 1:
                raise ValueError(f"Conflicting {field} within one input video")
    return dict(sorted(grouped.items()))


def trace_for_row(row: dict[str, str]) -> dict[str, object]:
    if not row.get("vild_trace_sha256"):
        raise ValueError("A verified VILD trace hash is required")
    trace = load_vild_trace(Path(row["vild_trace_path"]), row["vild_trace_sha256"])
    for field, trace_field in (
        ("video_id", "video_id"),
        ("dataset_split", "dataset_split"),
        ("file", "source_file"),
    ):
        if row.get(field) != trace.get(trace_field):
            raise ValueError(f"VILD trace identity mismatch: {field}")
    return trace


def validate_measurements(
    trace: dict[str, object],
    measured: list[dict[str, object]],
    normalized_tolerance: float,
    timestamp_tolerance_s: float,
) -> dict[str, object]:
    original = trace["frames"]
    if len(original) != len(measured) or not original:
        raise ValueError("Augmentation frame count differs from the original trace")
    errors: list[float] = []
    for old, new in zip(original, measured, strict=True):
        delta = abs(float(old["timestamp_s"]) - float(new["timestamp_s"]))
        if (
            old["frame_index"] != new["frame_index"]
            or not math.isfinite(delta)
            or delta > timestamp_tolerance_s
        ):
            raise ValueError("Augmentation frame index/timestamp mismatch")
        if old["face_count"] != new["face_count"]:
            raise ValueError("Augmentation face count differs from the original trace")
        old_vild, new_vild = old.get("normalized_vild"), new.get("normalized_vild")
        if (old_vild is None) != (new_vild is None):
            raise ValueError("Augmentation landmark availability mismatch")
        if old_vild is None:
            continue
        error = abs(float(old_vild) - float(new_vild))
        if not math.isfinite(error) or error > normalized_tolerance:
            raise ValueError(f"Normalized VILD mismatch exceeds {normalized_tolerance}: {error}")
        raw, bbox = float(new["raw_vild_px"]), float(new["face_bbox_size_px"])
        if not math.isfinite(raw) or raw < 0 or not math.isfinite(bbox) or bbox <= 0:
            raise ValueError("Augmentation has invalid raw VILD/face size")
        errors.append(error)
    return {
        "frames": len(original),
        "compared_valid_frames": len(errors),
        "maximum_normalized_error": max(errors, default=0.0),
    }


def augmented_trace_for_row(row: dict[str, str]) -> dict[str, object]:
    trace = trace_for_row(row)
    path_text = row.get("vild_augmentation_path", "")
    if not path_text:
        # Legacy normalized-only manifests remain loadable; raw features stay missing.
        return trace
    path = Path(path_text)
    if (
        not row.get("vild_augmentation_sha256")
        or file_sha256(path) != row["vild_augmentation_sha256"]
    ):
        raise ValueError("VILD augmentation hash mismatch")
    extra = read_gzip_json(path)
    if (
        extra.get("artifact_version") != AUGMENTATION_VERSION
        or extra.get("trace_sha256") != row["vild_trace_sha256"]
        or extra.get("video_id") != row["video_id"]
    ):
        raise ValueError("VILD augmentation does not belong to this trace")
    contract = extra["measurement_contract"]
    validate_measurements(
        trace, extra["frames"], contract["normalized_tolerance"], contract["timestamp_tolerance_s"]
    )
    return {
        **trace,
        "frames": [
            {
                **old,
                "raw_vild_px": new.get("raw_vild_px"),
                "face_bbox_size_px": new.get("face_bbox_size_px"),
            }
            for old, new in zip(trace["frames"], extra["frames"], strict=True)
        ],
    }


def verified_manifest_dependencies(manifests: list[Path]) -> dict[str, str]:
    """Hash referenced files once; CSV hashes alone cannot detect corrupt traces."""
    hashes: dict[str, str] = {}
    for manifest in manifests:
        hashes[manifest.as_posix()] = file_sha256(manifest)
        for row in read_csv_rows(manifest):
            for prefix in ("vild_trace", "vild_augmentation"):
                path_text, expected = row.get(f"{prefix}_path"), row.get(f"{prefix}_sha256")
                if not path_text:
                    if prefix == "vild_trace":
                        raise ValueError("Missing VILD trace")
                    continue
                path = Path(path_text)
                key = path.as_posix()
                if key not in hashes:
                    hashes[key] = file_sha256(path)
                if not expected or hashes[key] != expected:
                    raise ValueError(f"{prefix} hash mismatch: {path}")
    return hashes


def recorded_hashes_are_current(hashes: dict[str, str]) -> bool:
    return bool(hashes) and all(
        Path(p).is_file() and file_sha256(Path(p)) == h for p, h in hashes.items()
    )


def augmentation_outputs_are_current(job: TraceAugmentationJob) -> bool:
    try:
        summary = json.loads((job.output_dir / "summary.json").read_text(encoding="utf-8"))
        return (
            summary["status"] == "complete"
            and summary["job"] == json.loads(json.dumps(asdict(job), default=str))
            and summary["measurement_contract"] == measurement_contract(job)
            and recorded_hashes_are_current(summary["input_hashes"])
            and recorded_hashes_are_current(summary["output_hashes"])
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def run_trace_augmentation(
    job: TraceAugmentationJob, progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Decode selected videos only; never invoke audio, MFA, or event preprocessing."""
    if progress is not None:
        progress("verify augmentation cache", 0, 0, "")
    if augmentation_outputs_are_current(job):
        return json.loads((job.output_dir / "summary.json").read_text(encoding="utf-8"))
    contract = measurement_contract(job)
    grouped = group_video_rows(read_csv_rows(job.manifest))
    selected = list(grouped.items())[: job.max_videos]
    if not selected:
        raise ValueError("Augmentation manifest is empty")
    started = perf_counter()
    summary = {
        "job": json.loads(json.dumps(asdict(job), default=str)),
        "measurement_contract": contract,
        "status": "running",
        "videos_requested": len(selected),
        "videos_finished": 0,
        "videos_reused": 0,
        "failures": [],
        "input_hashes": {job.manifest.as_posix(): file_sha256(job.manifest)},
        "output_hashes": {},
    }
    output_rows: list[dict[str, object]] = []
    analyzer = None
    atomic_write_json(job.output_dir / "summary.json", summary)
    try:
        for video_id, rows in selected:
            if progress is not None:
                progress("augment visual traces", summary["videos_finished"], len(selected), video_id)
            row = rows[0]
            try:
                trace = trace_for_row(row)
                video_path = (job.extracted_root / row["file"]).resolve()
                if not video_path.is_relative_to(job.extracted_root.resolve()):
                    raise ValueError("Source video is outside extracted_root")
                source_hash = file_sha256(video_path)
                summary["input_hashes"].update(
                    {
                        video_path.as_posix(): source_hash,
                        row["vild_trace_path"]: row["vild_trace_sha256"],
                    }
                )
                path = job.output_dir / "traces" / f"{stable_id(video_id)}.json.gz"
                record_path = path.with_suffix(".record.json")
                identity = {
                    "video_id": video_id,
                    "trace_sha256": row["vild_trace_sha256"],
                    "source_sha256": source_hash,
                    "measurement_contract": contract,
                }
                reused = False
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    reused = record["identity"] == identity and record["sha256"] == file_sha256(
                        path
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                if not reused:
                    if analyzer is None:
                        analyzer = MouthEventAnalyzer()
                    measured = analyzer.trace_video(video_path, float(trace["timing"]["fps"]))
                    comparison = validate_measurements(
                        trace,
                        measured["frames"],
                        job.normalized_tolerance,
                        job.timestamp_tolerance_s,
                    )
                    atomic_write_gzip_json(
                        path,
                        {
                            "artifact_version": AUGMENTATION_VERSION,
                            **identity,
                            "source_path": video_path.as_posix(),
                            "frames": measured["frames"],
                            "comparison": comparison,
                        },
                    )
                    atomic_write_json(
                        record_path, {"identity": identity, "sha256": file_sha256(path)}
                    )
                else:
                    summary["videos_reused"] += 1
                digest = file_sha256(path)
                summary["output_hashes"][path.as_posix()] = digest
                output_rows.extend(
                    {
                        **event,
                        "vild_augmentation_path": path.as_posix(),
                        "vild_augmentation_sha256": digest,
                    }
                    for event in rows
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                summary["failures"].append({"video_id": video_id, "reason": str(error)})
            summary["videos_finished"] += 1
            summary["elapsed_seconds"] = round(perf_counter() - started, 3)
            atomic_write_json(
                job.output_dir / "progress.json",
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "videos_finished",
                        "videos_requested",
                        "videos_reused",
                        "elapsed_seconds",
                    )
                },
            )
            if progress is None:
                print(f"[{job.name}] traces {summary['videos_finished']}/{len(selected)}", flush=True)
    finally:
        if analyzer is not None:
            analyzer.close()
    summary["status"] = "failed" if summary["failures"] else "complete"
    if not summary["failures"]:
        manifest_path = job.output_dir / "events_augmented.csv"
        atomic_write_csv(manifest_path, output_rows)
        summary["output_hashes"][manifest_path.as_posix()] = file_sha256(manifest_path)
    atomic_write_json(job.output_dir / "summary.json", summary)
    atomic_write_json(
        job.output_dir / "progress.json",
        {
            key: summary[key]
            for key in (
                "status",
                "videos_finished",
                "videos_requested",
                "videos_reused",
                "elapsed_seconds",
            )
        },
    )
    if summary["failures"]:
        raise RuntimeError(
            f"[{job.name}] augmentation failed; review {job.output_dir / 'summary.json'}"
        )
    if progress is not None:
        progress("augment visual traces", len(selected), len(selected),
                 f"reused={summary['videos_reused']}; failures=0")
    return summary
