"""Audit an uploaded AV++ subset and create separate, source-disjoint run manifests.

Run from the package root, for example::

    python scripts/prepare_local_subset.py --dataset-root ../Videos_Dataset/SeePAT_dataset_5000_train_1000_val --output-dir data/manifests/local_subset_v1

This standard-library-only utility does not edit or decode uploaded media. It keeps
all Validation rows, excludes overlapping Train originals, and keeps the earliest
selection of a repeated Train original. Only output source_group values change:
standard separator/whitespace normalization is followed by removal of the
documented redundant vox_celeb_2/ prefix immediately before VoxCeleb2/, with
canonical VoxCeleb2 casing. Original paths and edit JSON text remain unchanged.
Historical published cohorts are not rewritten by this new-run cleanup.
Existing identical outputs are reusable; changed outputs require a new directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path, PureWindowsPath

METADATA_COLUMNS = (
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
)
INDEX_COLUMNS = ("split", "file", "source_group", "category", "relative_path", "binary_label")
CATEGORIES = {"real", "audio_modified", "visual_modified", "both_modified"}
EXCLUSION_COLUMNS = (
    "split",
    "file",
    "reason",
    "original_source_group",
    "normalized_source_group",
    "original",
    "retained_file",
)


def canonical_source_group(value: str) -> str:
    """Normalize separators and the documented VoxCeleb2 alias, preserving IDs' case."""
    value = re.sub(r"/+", "/", value.strip().replace("\\", "/"))
    return re.sub(
        r"^(?:vox_celeb_2/)?voxceleb2/", "VoxCeleb2/", value, count=1, flags=re.IGNORECASE
    )


def _read_csv(path: Path, required: tuple[str, ...]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(required).issubset(fields):
            raise ValueError(f"Invalid or incomplete CSV header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV row does not match its header: {path}")
    return fields, rows


def _group(row: dict[str, str]) -> str:
    original = canonical_source_group(row["original"])
    recorded = canonical_source_group(row["source_group"])
    if original and recorded and original != recorded:
        raise ValueError(f"Conflicting original/source_group: {row['file']}")
    result = recorded or original or canonical_source_group(row["file"])
    if not result:
        raise ValueError(f"Missing source identity: {row['file']}")
    return result


def _media_path(root: Path, split: str, relative: str) -> Path:
    # Windows drive/UNC forms must be rejected even when tests run on POSIX.
    parts = relative.replace("\\", "/").split("/")
    if (
        not relative
        or PureWindowsPath(relative).drive
        or relative.startswith(("/", "\\"))
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Unsafe media path: {relative!r}")
    split_root = (root / split).resolve()
    result = split_root.joinpath(*parts).resolve()
    if not result.is_relative_to(split_root):
        raise ValueError(f"Media path escapes its split: {relative!r}")
    if not result.is_file():
        raise ValueError(f"Missing media: {result}")
    if result.stat().st_size == 0:
        raise ValueError(f"Zero-byte media: {result}")
    return result


def _validate_metadata(row: dict[str, str], split: str) -> None:
    if row["split"] != split or row["modify_type"] not in CATEGORIES:
        raise ValueError(f"Invalid split/category: {row['file']}")
    if not row["subject_id"]:
        raise ValueError(f"Missing subject_id: {row['file']}")
    for field in ("video_frames", "audio_frames"):
        try:
            valid = int(row[field]) > 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"Invalid {field}: {row['file']}")
    try:
        if int(row["selection_order"]) < 0:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"Invalid selection_order: {row['file']}") from error
    for field in ("fake_segments_json", "audio_fake_segments_json", "visual_fake_segments_json"):
        try:
            intervals = json.loads(row[field])
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid {field}: {row['file']}") from error
        if not isinstance(intervals, list):
            # This is malformed serialized data, rather than an API argument type.
            raise ValueError(f"Expected JSON array in {field}: {row['file']}")  # noqa: TRY004
        for interval in intervals:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in interval
                )
                or not 0 <= interval[0] < interval[1]
            ):
                raise ValueError(f"Invalid interval in {field}: {row['file']}")


def _csv_bytes(fields: list[str] | tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def prepare_local_subset(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    """Validate inputs and write train.csv, val.csv, exclusions.csv and audit.json.

    No predictions or evaluation outcomes participate in selection. Validation is
    preserved in original order. Train duplicate winners use selection_order,
    then file path, then source row order; retained outputs keep source row order.
    """
    root = dataset_root.resolve()
    destination = output_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory is absent: {root}")
    if destination.is_relative_to(root):
        raise ValueError("Output directory must be outside the uploaded dataset")
    input_names = {"train": "train_subset5000.csv", "val": "val_subset1000.csv"}
    _, index_rows = _read_csv(root / "videos.csv", INDEX_COLUMNS)
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in index_rows:
        key = (row["split"], row["file"])
        if key in index:
            raise ValueError(f"Duplicate videos.csv entry: {key}")
        index[key] = row
    headers: dict[str, list[str]] = {}
    source: dict[str, list[dict[str, str]]] = {}
    identities: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str]] = set()
    media_bytes: Counter[str] = Counter()
    normalized_rows: Counter[str] = Counter()
    for split, name in input_names.items():
        headers[split], source[split] = _read_csv(root / name, METADATA_COLUMNS)
        for row in source[split]:
            _validate_metadata(row, split)
            key = (split, row["file"])
            if key in seen:
                raise ValueError(f"Repeated file in source manifest: {key}")
            seen.add(key)
            group = _group(row)
            identities[key] = group
            normalized_rows[split] += row["source_group"] != group
            entry = index.get(key)
            expected = {
                "category": row["modify_type"],
                "relative_path": f"{split}/{row['file']}",
                "binary_label": str(int(row["modify_type"] != "real")),
            }
            if entry is None or any(entry[field] != value for field, value in expected.items()):
                raise ValueError(f"videos.csv metadata mismatch: {key}")
            index_group = canonical_source_group(
                entry["source_group"] or row["original"] or entry["file"]
            )
            if index_group != group:
                raise ValueError(f"videos.csv source-group mismatch: {key}")
            media_bytes[split] += _media_path(root, split, row["file"]).stat().st_size
    if set(index) != seen:
        raise ValueError("videos.csv includes files absent from the two source manifests")

    val_groups = {identities[("val", row["file"])] for row in source["val"]}
    if len(val_groups) != len(source["val"]):
        raise ValueError(
            "Validation has repeated canonical originals; keep it unchanged for review"
        )
    winners: dict[str, str] = {}
    for row in sorted(source["train"], key=lambda row: (int(row["selection_order"]), row["file"])):
        winners.setdefault(identities[("train", row["file"])], row["file"])
    retained: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    exclusions: list[dict[str, str]] = []
    for split in ("train", "val"):
        for row in source[split]:
            group = identities[(split, row["file"])]
            reason = ""
            retained_file = ""
            if split == "train" and group in val_groups:
                reason = "train_validation_source_overlap"
                retained_file = next(
                    item["file"]
                    for item in source["val"]
                    if identities[("val", item["file"])] == group
                )
            elif split == "train" and winners[group] != row["file"]:
                reason = "duplicate_train_source_group"
                retained_file = winners[group]
            if reason:
                exclusions.append(
                    {
                        "split": split,
                        "file": row["file"],
                        "reason": reason,
                        "original_source_group": row["source_group"],
                        "normalized_source_group": group,
                        "original": row["original"],
                        "retained_file": retained_file,
                    }
                )
            else:
                retained[split].append({**row, "source_group": group})

    outputs = {
        "train.csv": _csv_bytes(headers["train"], retained["train"]),
        "val.csv": _csv_bytes(headers["val"], retained["val"]),
        "exclusions.csv": _csv_bytes(EXCLUSION_COLUMNS, exclusions),
    }
    audit: dict[str, object] = {
        "version": "local-subset-v1",
        "dataset_root": str(root),
        "selection": "metadata source identity only; no model predictions or evaluation outcomes",
        "canonical_rule": "strip surrounding whitespace, convert backslash to slash, collapse "
        "repeated slashes, then remove only leading vox_celeb_2/ directly before VoxCeleb2/; "
        "prefix comparison is case-insensitive; keep canonical VoxCeleb2 casing and identifiers",
        "source_identity_precedence": "recorded source_group, falling back to original then file; "
        "provided original must agree under the same rules; only output source_group is rewritten",
        "duplicate_winner": "lowest selection_order, then file path; retain input row order",
        "validation_rows_preserved": True,
        "media_content_decoded_or_hashed": False,
        "input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("videos.csv", *input_names.values())
        },
        "input_rows": {split: len(rows) for split, rows in source.items()},
        "output_rows": {split: len(rows) for split, rows in retained.items()},
        "input_recorded_source_groups": {
            split: len({row["source_group"] for row in rows}) for split, rows in source.items()
        },
        "input_canonical_source_groups": {
            split: len({identities[(split, row["file"])] for row in rows})
            for split, rows in source.items()
        },
        "input_canonical_source_overlap": len(
            {identities[("train", row["file"])] for row in source["train"]} & val_groups
        ),
        "output_canonical_source_groups": {
            split: len({row["source_group"] for row in rows}) for split, rows in retained.items()
        },
        "normalized_source_group_rows": dict(normalized_rows),
        "input_media_bytes": dict(media_bytes),
        "output_category_counts": {
            split: dict(sorted(Counter(row["modify_type"] for row in rows).items()))
            for split, rows in retained.items()
        },
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "output_canonical_source_overlap": len(
            {row["source_group"] for row in retained["train"]}
            & {row["source_group"] for row in retained["val"]}
        ),
        "subject_identity_separation_asserted": False,
        "output_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
    }
    outputs["audit.json"] = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")
    for name, data in outputs.items():
        path = destination / name
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise FileExistsError(
                f"Changed run output exists; choose a new output directory: {path}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = prepare_local_subset(args.dataset_root, args.output_dir)
    print(
        json.dumps(
            {"output_rows": audit["output_rows"], "exclusions": audit["exclusion_counts"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
