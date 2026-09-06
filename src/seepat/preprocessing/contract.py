from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from seepat.artifacts import file_sha256, read_csv_rows
from seepat.preprocessing.vild import load_vild_trace

HASHED_VIDEO_ARTIFACTS = (
    ("raw_audio_path", "raw_audio_sha256"),
    ("alignment_audio_path", "alignment_audio_sha256"),
    ("transcription_path", "transcription_sha256"),
    ("alignment_path", "alignment_sha256"),
)
VALID_REFERENCE_STRATEGIES = {
    "mfa_phone_complement",
    "whisper_segment_complement_fallback",
    "unavailable",
}


def _unique_rows(rows: list[dict[str, str]], key: str, source: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            raise ValueError(f"{source} has a row without {key}")
        if value in indexed:
            raise ValueError(f"{source} contains duplicate {key} {value!r}")
        indexed[value] = row
    return indexed


def _resolved(path_text: str, project_root: Path) -> Path:
    path = Path(path_text.replace("\\", "/"))
    return (path if path.is_absolute() else project_root / path).resolve()


def _verify_file(
    row: dict[str, str],
    path_field: str,
    hash_field: str,
    project_root: Path,
) -> None:
    path_text = row.get(path_field, "").strip()
    expected_hash = row.get(hash_field, "").strip()
    if not path_text or not expected_hash:
        raise ValueError(f"{row.get('video_id', 'video')}: missing {path_field} or {hash_field}")
    path = _resolved(path_text, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Recorded artifact is missing: {path}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"Recorded artifact hash does not match: {path}")


def audit_preprocessing_contract(
    output_dir: Path,
    project_root: Path = Path("."),
) -> dict[str, object]:
    summary_path = output_dir / "run_summary.json"
    video_path = output_dir / "video_manifest.csv"
    event_path = output_dir / "bilabial_events.csv"
    trace_index_path = output_dir / "vild_trace_index.csv"
    for path in (summary_path, video_path, event_path, trace_index_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required preprocessing artifact is missing: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("vild_trace_enabled") is not True:
        raise ValueError("Run summary does not record an enabled VILD trace contract")
    if summary.get("vild_trace_index_sha256") != file_sha256(trace_index_path):
        raise ValueError("VILD trace index hash does not match the run summary")

    videos = _unique_rows(read_csv_rows(video_path), "video_id", video_path)
    events = _unique_rows(read_csv_rows(event_path), "event_id", event_path)
    traces = _unique_rows(read_csv_rows(trace_index_path), "video_id", trace_index_path)
    expected_trace_ids = {
        video_id
        for video_id, row in videos.items()
        if row.get("pipeline_status") == "complete" and int(row.get("bilabial_event_count", 0)) > 0
    }
    if set(traces) != expected_trace_ids:
        raise ValueError("VILD trace index does not match completed videos with bilabials")

    events_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events.values():
        if event.get("video_id") not in videos:
            raise ValueError(f"Event has no matching video: {event['event_id']}")
        events_by_video[event["video_id"]].append(event)

    reference_strategies: Counter[str] = Counter()
    reference_windows = 0
    eligible_events = 0
    for video_id, trace_row in traces.items():
        video = videos[video_id]
        for path_field, hash_field in HASHED_VIDEO_ARTIFACTS:
            _verify_file(video, path_field, hash_field, project_root)

        trace_path = _resolved(trace_row["vild_trace_path"], project_root)
        artifact = load_vild_trace(
            trace_path,
            expected_sha256=trace_row["vild_trace_sha256"],
        )
        if artifact.get("video_id") != video_id:
            raise ValueError(f"VILD trace video ID mismatch: {trace_path}")
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError(f"VILD trace has no provenance record: {trace_path}")
        for path_field, hash_field in HASHED_VIDEO_ARTIFACTS:
            if provenance.get(path_field) != video.get(path_field) or provenance.get(
                hash_field
            ) != video.get(hash_field):
                raise ValueError(f"VILD trace provenance mismatch: {trace_path}")
        frames = artifact["frames"]
        if int(trace_row["vild_trace_frame_count"]) != len(frames):
            raise ValueError(f"VILD trace frame count mismatch: {trace_path}")
        probed_frame_count = videos[video_id].get("video_frame_count", "").strip()
        if probed_frame_count and int(probed_frame_count) != len(frames):
            raise ValueError(f"Decoded/probed frame count mismatch: {trace_path}")

        event_windows = {
            str(window["event_id"]): window for window in artifact["bilabial_event_windows"]
        }
        video_events = events_by_video[video_id]
        if set(event_windows) != {event["event_id"] for event in video_events}:
            raise ValueError(f"VILD trace event IDs do not match the event CSV: {trace_path}")
        for event in video_events:
            if _resolved(event["vild_trace_path"], project_root) != trace_path:
                raise ValueError(f"Event points to the wrong VILD trace: {event['event_id']}")
            if event.get("vild_trace_sha256") != trace_row["vild_trace_sha256"]:
                raise ValueError(f"Event has the wrong VILD trace hash: {event['event_id']}")
            if event.get("vild_trace_event_key") != event["event_id"]:
                raise ValueError(f"Event has the wrong VILD trace key: {event['event_id']}")
            try:
                clip_indices = json.loads(event.get("mouth_crop_frame_indices_json", ""))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Event has an invalid mouth-clip frame map: {event['event_id']}"
                ) from error
            if not isinstance(clip_indices, list) or any(
                not isinstance(index, int) for index in clip_indices
            ):
                raise TypeError(
                    f"Event has a non-integer mouth-clip frame map: {event['event_id']}"
                )
            if clip_indices != sorted(set(clip_indices)) or any(
                index < 0 or index >= len(frames) for index in clip_indices
            ):
                raise ValueError(
                    f"Event has an out-of-range mouth-clip frame map: {event['event_id']}"
                )
            trace_indices = event_windows[event["event_id"]].get("mouth_clip_frame_indices")
            if clip_indices != trace_indices:
                raise ValueError(f"Event and trace frame maps differ: {event['event_id']}")
            if event.get("eligibility_status") == "eligible":
                eligible_events += 1
                if not clip_indices:
                    raise ValueError(
                        f"Eligible event has no mouth-clip frame map: {event['event_id']}"
                    )

        reference = artifact.get("non_speech_reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("windows"), list):
            raise TypeError(f"VILD trace has no non-speech reference contract: {trace_path}")
        strategy = str(reference.get("strategy", ""))
        if strategy not in VALID_REFERENCE_STRATEGIES:
            raise ValueError(f"VILD trace has an invalid reference strategy: {trace_path}")
        reference_strategies[strategy] += 1
        reference_windows += len(reference["windows"])

        if int(trace_row["vild_reference_window_count"]) != len(reference["windows"]):
            raise ValueError(f"VILD reference count mismatch: {trace_path}")

    if int(summary.get("vild_trace_videos", -1)) != len(traces):
        raise ValueError("Run summary has the wrong VILD trace count")
    if int(summary.get("vild_reference_windows", -1)) != reference_windows:
        raise ValueError("Run summary has the wrong VILD reference-window count")
    if int(summary.get("bilabial_events", -1)) != len(events):
        raise ValueError("Run summary has the wrong bilabial-event count")

    return {
        "status": "passed",
        "output_dir": str(output_dir),
        "videos": len(videos),
        "trace_videos": len(traces),
        "events": len(events),
        "eligible_events": eligible_events,
        "reference_windows": reference_windows,
        "reference_strategies": dict(sorted(reference_strategies.items())),
        "trace_index_sha256": file_sha256(trace_index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.preprocessing.contract")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = audit_preprocessing_contract(args.output_dir, args.project_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
