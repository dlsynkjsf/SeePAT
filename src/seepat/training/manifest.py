from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
)
from seepat.preprocessing.vild import load_vild_trace

SUPPORTED_LABELS = {"real": 0, "fake": 1}


def _unique_rows(rows: list[dict[str, str]], key: str, source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            raise ValueError(f"Missing {key!r} in {source}")
        if value in indexed:
            raise ValueError(f"Duplicate {key} {value!r} in {source}")
        indexed[value] = row
    return indexed


def _portable_path(value: str) -> str:
    return value.strip().replace("\\", "/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_text(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return str(float(value))


def _event_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        str(row["dataset_split"]),
        str(row["source_group"]),
        str(row["video_id"]),
        float(row["phone_start_s"]),
        str(row["event_id"]),
    )


def _check_source_group_leakage(rows: list[dict[str, object]]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["source_group"])].add(str(row["dataset_split"]))

    leaking = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if leaking:
        examples = ", ".join(
            f"{group} ({'/'.join(splits)})"
            for group, splits in sorted(leaking.items())[:5]
        )
        raise ValueError(f"Source-group leakage across dataset splits: {examples}")


def _omission_reason(event: dict[str, str], video: dict[str, str]) -> str | None:
    if video.get("pipeline_status", "").strip() != "complete":
        return "video_incomplete"
    if video.get("eligibility_status", "").strip() != "eligible":
        return "video_ineligible"
    if event.get("eligibility_status", "").strip() != "eligible":
        return "event_ineligible"
    if event.get("training_label_status", "").strip() != "usable":
        if event.get("event_label", "").strip() == "ambiguous":
            return "ambiguous_boundary"
        return "unusable_training_label"
    if event.get("event_label", "").strip() not in SUPPORTED_LABELS:
        return "unsupported_label"
    if not event.get("mouth_crop_path", "").strip():
        return "missing_clip_path"
    return None


def _training_row(
    event: dict[str, str],
    video: dict[str, str],
    source: dict[str, str],
) -> dict[str, object]:
    file_name = event.get("file", "").strip() or video.get("file", "").strip()
    source_group = (
        source.get("source_group", "").strip()
        or source.get("original", "").strip()
        or file_name
    )
    start = float(event["phone_start_s"])
    end = float(event["phone_end_s"])
    if end < start:
        raise ValueError(f"Event {event['event_id']!r} ends before it starts")

    label = event["event_label"].strip()
    named_splits = {
        name: row.get("split", "").strip()
        for name, row in (("source", source), ("video", video), ("event", event))
        if row.get("split", "").strip()
    }
    if len(set(named_splits.values())) > 1:
        details = ", ".join(f"{name}={split}" for name, split in named_splits.items())
        raise ValueError(f"Event {event['event_id']!r} has conflicting dataset splits: {details}")
    dataset_split = next(iter(named_splits.values()), "")
    if not dataset_split:
        raise ValueError(f"Event {event['event_id']!r} has no dataset split")

    return {
        "event_id": event["event_id"].strip(),
        "video_id": event["video_id"].strip(),
        "file": _portable_path(file_name),
        "dataset_split": dataset_split,
        "source_group": source_group,
        "subject_id": source.get("subject_id", "").strip(),
        "manipulation_modality": event.get("manipulation_modality", "").strip(),
        "phoneme": event.get("phoneme", "").strip(),
        "phone_start_s": str(start),
        "phone_end_s": str(end),
        "video_phone_start_s": _float_text(
            event.get("video_phone_start_s", event["phone_start_s"])
        ),
        "video_phone_end_s": _float_text(
            event.get("video_phone_end_s", event["phone_end_s"])
        ),
        "audio_video_start_offset_s": _float_text(
            event.get("audio_video_start_offset_s", "0")
        ),
        "phone_duration_s": str(round(end - start, 9)),
        "event_label": label,
        "class_id": SUPPORTED_LABELS[label],
        "mouth_clip_path": _portable_path(event["mouth_crop_path"]),
        "mouth_crop_frame_indices_json": event.get(
            "mouth_crop_frame_indices_json", "[]"
        ).strip(),
        "vild_trace_path": _portable_path(event.get("vild_trace_path", "")),
        "vild_trace_sha256": event.get("vild_trace_sha256", "").strip(),
        "vild_trace_event_key": event.get("vild_trace_event_key", "").strip(),
        "normalized_minimum_closure": _float_text(
            event.get("normalized_minimum_closure", "")
        ),
        "closure_time_s": _float_text(event.get("closure_time_s", "")),
        "closure_duration_s": _float_text(event.get("closure_duration_s", "")),
        "closing_velocity": _float_text(event.get("closing_velocity", "")),
        "opening_velocity": _float_text(event.get("opening_velocity", "")),
        "valid_landmark_ratio": _float_text(event.get("valid_landmark_ratio", "")),
        "multiple_face_ratio": _float_text(event.get("multiple_face_ratio", "")),
    }


def prepare_training_manifests(
    source_manifest_path: Path,
    video_manifest_path: Path,
    event_manifest_path: Path,
    output_dir: Path,
    project_root: Path = Path("."),
    require_clips: bool = True,
) -> dict[str, object]:
    source_rows = read_csv_rows(source_manifest_path)
    video_rows = read_csv_rows(video_manifest_path)
    event_rows = read_csv_rows(event_manifest_path)
    sources_by_file = _unique_rows(source_rows, "file", "source manifest")
    videos_by_id = _unique_rows(video_rows, "video_id", "video manifest")

    prepared: list[dict[str, object]] = []
    omitted: Counter[str] = Counter()
    seen_event_ids: set[str] = set()
    verified_trace_hashes: dict[Path, str] = {}
    verified_trace_events: dict[Path, set[str]] = {}

    for event in event_rows:
        event_id = event.get("event_id", "").strip()
        if not event_id:
            raise ValueError("Missing 'event_id' in event manifest")
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate event_id {event_id!r} in event manifest")
        seen_event_ids.add(event_id)

        video_id = event.get("video_id", "").strip()
        if video_id not in videos_by_id:
            raise ValueError(f"Event {event_id!r} has no matching video row")
        video = videos_by_id[video_id]

        event_file = event.get("file", "").strip()
        video_file = video.get("file", "").strip()
        if event_file and video_file and event_file != video_file:
            raise ValueError(f"Event {event_id!r} disagrees with its video file")
        file_name = event_file or video_file
        if file_name not in sources_by_file:
            raise ValueError(f"Event {event_id!r} has no matching source-manifest row")

        reason = _omission_reason(event, video)
        if reason is not None:
            omitted[reason] += 1
            continue

        clip_path = Path(_portable_path(event["mouth_crop_path"]))
        resolved_clip = clip_path if clip_path.is_absolute() else project_root / clip_path
        if require_clips and not resolved_clip.is_file():
            omitted["missing_clip_file"] += 1
            continue

        trace_text = _portable_path(event.get("vild_trace_path", ""))
        if video.get("vild_trace_path", "").strip() and not trace_text:
            omitted["missing_vild_trace_path"] += 1
            continue
        if trace_text:
            trace_path = Path(trace_text)
            resolved_trace = (
                trace_path if trace_path.is_absolute() else project_root / trace_path
            )
            if not resolved_trace.is_file():
                omitted["missing_vild_trace_file"] += 1
                continue
            recorded_trace_hash = event.get("vild_trace_sha256", "").strip()
            if not recorded_trace_hash:
                omitted["missing_vild_trace_hash"] += 1
                continue
            actual_trace_hash = verified_trace_hashes.get(resolved_trace)
            if actual_trace_hash is None:
                actual_trace_hash = file_sha256(resolved_trace)
                verified_trace_hashes[resolved_trace] = actual_trace_hash
            if recorded_trace_hash != actual_trace_hash:
                omitted["vild_trace_hash_mismatch"] += 1
                continue
            trace_event_ids = verified_trace_events.get(resolved_trace)
            if trace_event_ids is None:
                trace_artifact = load_vild_trace(resolved_trace)
                trace_event_ids = {
                    str(window["event_id"])
                    for window in trace_artifact["bilabial_event_windows"]
                }
                verified_trace_events[resolved_trace] = trace_event_ids
            trace_event_key = event.get("vild_trace_event_key", "").strip()
            if trace_event_key != event_id or trace_event_key not in trace_event_ids:
                omitted["vild_trace_event_missing"] += 1
                continue

        prepared.append(_training_row(event, video, sources_by_file[file_name]))

    if not prepared:
        raise ValueError("No usable training events remained after filtering")

    prepared.sort(key=_event_sort_key)
    _check_source_group_leakage(prepared)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "events.csv"
    atomic_write_csv(combined_path, prepared)

    split_paths: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_groups: dict[str, set[str]] = defaultdict(set)
    for row in prepared:
        split = str(row["dataset_split"])
        split_counts[split] += 1
        label_counts[str(row["event_label"])] += 1
        split_label_counts[split][str(row["event_label"])] += 1
        source_groups[split].add(str(row["source_group"]))

    for split in sorted(split_counts):
        split_path = output_dir / f"events_{split}.csv"
        atomic_write_csv(
            split_path,
            [row for row in prepared if row["dataset_split"] == split],
        )
        split_paths[split] = split_path.as_posix()

    summary: dict[str, object] = {
        "input_artifacts": {
            "source_manifest": {
                "path": source_manifest_path.as_posix(),
                "sha256": _sha256(source_manifest_path),
            },
            "video_manifest": {
                "path": video_manifest_path.as_posix(),
                "sha256": _sha256(video_manifest_path),
            },
            "event_manifest": {
                "path": event_manifest_path.as_posix(),
                "sha256": _sha256(event_manifest_path),
            },
        },
        "input_events": len(event_rows),
        "written_events": len(prepared),
        "omitted_events": sum(omitted.values()),
        "omission_reasons": dict(sorted(omitted.items())),
        "split_event_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_label_counts": {
            split: dict(sorted(split_label_counts[split].items()))
            for split in sorted(split_label_counts)
        },
        "source_group_counts": {
            split: len(source_groups[split]) for split in sorted(source_groups)
        },
        "require_clips": require_clips,
        "verified_vild_trace_files": len(verified_trace_hashes),
        "combined_manifest": combined_path.as_posix(),
        "combined_manifest_sha256": _sha256(combined_path),
        "split_manifests": split_paths,
        "split_manifest_sha256": {
            split: _sha256(output_dir / f"events_{split}.csv")
            for split in sorted(split_counts)
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.manifest")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--allow-missing-clips", action="store_true")
    args = parser.parse_args()

    summary = prepare_training_manifests(
        source_manifest_path=args.source_manifest,
        video_manifest_path=args.video_manifest,
        event_manifest_path=args.events,
        output_dir=args.output_dir,
        project_root=args.project_root,
        require_clips=not args.allow_missing_clips,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
