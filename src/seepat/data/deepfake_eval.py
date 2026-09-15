"""Deepfake-Eval-2024 evaluation split adapter (Figure 4.1 dataset node).

The benchmark is evaluated as an external, unseen-generator test pool: its
videos must never influence calibration or threshold selection. This adapter
converts a local Deepfake-Eval-2024 index (JSON or CSV, prepared offline from
the benchmark metadata) into a SeePAT pipeline manifest with the same schema
as the AV++ manifests, so the existing preprocessing, frozen-calibration
scoring, and verdict stages can run unchanged with ``split: test``.

Expected index fields per entry:

* ``file`` (required): media path relative to the configured archive root.
* ``label`` (required): ``real``/``fake`` (case-insensitive) or ``0``/``1``.
* ``modality`` (optional): ``audio``, ``visual`` or ``both`` for fakes.
* ``duration_s`` (optional): bounds the assumed whole-video manipulation
  segment when the benchmark does not provide fake segments.
* ``subject_id``, ``original``, ``video_frames``, ``audio_frames`` (optional).

The output manifest keeps the AV++ column schema so ``_training_row`` and the
event-labeling logic consume it directly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows

DEEPFAKE_EVAL_SOURCE = "Deepfake-Eval-2024"
DEEPFAKE_EVAL_SPLIT = "test"
MODALITY_MAP = {
    "audio": "audio_modified",
    "visual": "visual_modified",
    "both": "both_modified",
}
MANIFEST_FIELDS = (
    "file",
    "original",
    "split",
    "modify_type",
    "audio_model",
    "video_model",
    "fake_segments_json",
    "audio_fake_segments_json",
    "visual_fake_segments_json",
    "video_frames",
    "audio_frames",
    "subject_id",
    "source_group",
    "selection_order",
    "sampling_seed",
    "dataset_source",
)
WHOLE_VIDEO_END_S = 1_000_000_000.0


def _normalize_label(value: object, row_number: int) -> bool:
    text = str(value).strip().lower()
    if text in {"real", "authentic", "0", "false"}:
        return False
    if text in {"fake", "manipulated", "1", "true"}:
        return True
    raise ValueError(f"Index row {row_number} has an unsupported label: {value!r}")


def _read_index(index_path: Path) -> list[dict[str, object]]:
    if index_path.suffix.lower() == ".csv":
        return [dict(row) for row in read_csv_rows(index_path)]
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("videos")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The Deepfake-Eval index must be a non-empty list of records")
    records: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError("Every Deepfake-Eval index record must be a JSON object")
        records.append(entry)
    return records


def build_deepfake_eval_manifest(
    index_path: Path,
    output_manifest: Path,
    summary_path: Path | None = None,
    split: str = DEEPFAKE_EVAL_SPLIT,
    default_modality: str = "both",
    sampling_seed: int = 0,
) -> dict[str, object]:
    if default_modality not in MODALITY_MAP:
        raise ValueError("default_modality must be one of: audio, visual, both")
    if split not in {"test", "external-test"}:
        raise ValueError("Deepfake-Eval must remain a locked test split")
    records = _read_index(index_path)

    rows: list[dict[str, object]] = []
    seen_files: set[str] = set()
    label_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    for row_number, record in enumerate(records, start=1):
        file_name = str(record.get("file", "")).strip()
        if not file_name:
            raise ValueError(f"Index row {row_number} has no 'file'")
        if file_name in seen_files:
            raise ValueError(f"Duplicate Deepfake-Eval file: {file_name}")
        seen_files.add(file_name)
        is_fake = _normalize_label(record.get("label"), row_number)
        modality = str(record.get("modality", "") or "").strip().lower()
        if modality and modality not in MODALITY_MAP:
            raise ValueError(
                f"Index row {row_number} has an unsupported modality: {modality!r}"
            )
        modify_type = (
            MODALITY_MAP[modality or default_modality] if is_fake else "real"
        )
        duration = record.get("duration_s")
        try:
            duration_value = None if duration in (None, "") else float(duration)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Index row {row_number} has an invalid duration_s") from error
        if is_fake:
            segment_end = (
                duration_value if duration_value is not None else WHOLE_VIDEO_END_S
            )
            fake_segments = json.dumps([[0.0, segment_end]])
        else:
            fake_segments = "[]"
        original = str(record.get("original", "")).strip() or file_name
        rows.append(
            {
                "file": file_name,
                "original": original,
                "split": split,
                "modify_type": modify_type,
                "audio_model": "",
                "video_model": "",
                "fake_segments_json": fake_segments,
                "audio_fake_segments_json": "[]",
                "visual_fake_segments_json": "[]",
                "video_frames": int(record.get("video_frames", 0) or 0),
                "audio_frames": int(record.get("audio_frames", 0) or 0),
                "subject_id": str(record.get("subject_id", "") or "").strip(),
                "source_group": original,
                "selection_order": row_number - 1,
                "sampling_seed": sampling_seed,
                "dataset_source": DEEPFAKE_EVAL_SOURCE,
            }
        )
        label_counts["fake" if is_fake else "real"] += 1
        modality_counts[modify_type] += 1

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_manifest, rows)
    summary: dict[str, object] = {
        "source": DEEPFAKE_EVAL_SOURCE,
        "index": {
            "path": index_path.as_posix(),
            "sha256": file_sha256(index_path),
        },
        "manifest": {
            "path": output_manifest.as_posix(),
            "sha256": file_sha256(output_manifest),
        },
        "split": split,
        "records": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "modify_type_counts": dict(sorted(modality_counts.items())),
        "note": (
            "External evaluation split: never use for calibration fitting or "
            "threshold selection."
        ),
    }
    if summary_path is not None:
        atomic_write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.data.deepfake_eval")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--split", default=DEEPFAKE_EVAL_SPLIT)
    parser.add_argument(
        "--default-modality",
        choices=sorted(MODALITY_MAP),
        default="both",
        help="Modality assumed for fake videos without a modality field",
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    args = parser.parse_args()
    summary = build_deepfake_eval_manifest(
        index_path=args.index,
        output_manifest=args.output,
        summary_path=args.summary,
        split=args.split,
        default_modality=args.default_modality,
        sampling_seed=args.sampling_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
