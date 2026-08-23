from __future__ import annotations

import csv
import random
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from seepat.artifacts import atomic_write_csv, atomic_write_json
from seepat.data.inventory import file_sha256

CANARY_FIELDS = """
file, original, split, modify_type, audio_model, video_model,
fake_segments_json, audio_fake_segments_json, visual_fake_segments_json,
video_frames, audio_frames, subject_id
""".replace("\n", " ").strip()


def canonical_source_group(original: object, file_name: object) -> str:
    value = str(original or file_name or "").strip().replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    if not value:
        raise ValueError("A dataset row has neither an original nor a file path")
    return value


def _category_rng(seed: int, category: str) -> random.Random:
    digest = sha256(f"{seed}\0{category}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def _canary_metadata_omission_reason(
    row: sqlite3.Row,
    category: str,
) -> str | None:
    if not row["video_frames"] or int(row["video_frames"]) <= 0:
        return "missing_video_frames"
    if not row["audio_frames"] or int(row["audio_frames"]) <= 0:
        return "missing_audio_frames"
    if category != "real" and row["fake_segments_json"] == "[]":
        return "missing_fake_segments"
    if category in {"audio_modified", "both_modified"} and (
        row["audio_fake_segments_json"] == "[]"
    ):
        return "missing_audio_fake_segments"
    if category in {"visual_modified", "both_modified"} and (
        row["visual_fake_segments_json"] == "[]"
    ):
        return "missing_visual_fake_segments"
    return None


def sample_pilot(
    database_path: Path,
    output_path: Path,
    split: str,
    categories: Sequence[str],
    per_category: int,
    seed: int,
) -> list[dict[str, object]]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Inventory database not found: {database_path}")

    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    used_originals: set[str] = set()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        for category in categories:
            candidates = list(
                connection.execute(
                    """
                    SELECT file, original, split, modify_type, audio_model, video_model,
                           fake_segments_json, audio_fake_segments_json,
                           visual_fake_segments_json, video_frames, audio_frames, subject_id
                    FROM videos
                    WHERE split = ? AND modify_type = ?
                    ORDER BY file
                    """,
                    (split, category),
                )
            )
            rng.shuffle(candidates)
            category_rows: list[dict[str, object]] = []
            for candidate in candidates:
                group_key = str(candidate["original"] or candidate["file"])
                if group_key in used_originals:
                    continue
                row = dict(candidate)
                row["source_group"] = group_key
                row["selection_order"] = len(selected) + len(category_rows)
                row["sampling_seed"] = seed
                category_rows.append(row)
                used_originals.add(group_key)
                if len(category_rows) == per_category:
                    break
            if len(category_rows) != per_category:
                raise ValueError(
                    f"Requested {per_category} unique samples for {category}, "
                    f"but selected {len(category_rows)}"
                )
            selected.extend(category_rows)
    finally:
        connection.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0]) if selected else []
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def sample_training_canary(
    database_path: Path,
    output_path: Path,
    summary_path: Path,
    split: str,
    categories: Sequence[str],
    per_category: int,
    seed: int,
    excluded_splits: Sequence[str] = ("val",),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Inventory database not found: {database_path}")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be a non-empty sequence of unique values")
    if per_category < 1:
        raise ValueError("per_category must be positive")
    if split in excluded_splits:
        raise ValueError("The selected split cannot also be excluded")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    selected: list[dict[str, object]] = []
    used_source_groups: set[str] = set()
    excluded_source_groups: set[str] = set()
    available_group_counts: dict[str, int] = {}
    metadata_rejections: dict[str, Counter[str]] = {}

    try:
        if excluded_splits:
            placeholders = ", ".join("?" for _ in excluded_splits)
            for row in connection.execute(
                f"SELECT file, original FROM videos WHERE split IN ({placeholders})",
                tuple(excluded_splits),
            ):
                excluded_source_groups.add(
                    canonical_source_group(row["original"], row["file"])
                )

        for category in categories:
            rng = _category_rng(seed, category)
            category_source_groups: set[str] = set()
            category_rejections: Counter[str] = Counter()
            reservoir: list[dict[str, object]] = []
            eligible_group_count = 0
            candidates = connection.execute(
                f"""
                SELECT {CANARY_FIELDS}
                FROM videos
                WHERE split = ? AND modify_type = ?
                ORDER BY file
                """,
                (split, category),
            )
            for candidate in candidates:
                omission_reason = _canary_metadata_omission_reason(candidate, category)
                if omission_reason is not None:
                    category_rejections[omission_reason] += 1
                    continue
                source_group = canonical_source_group(
                    candidate["original"],
                    candidate["file"],
                )
                if source_group in category_source_groups:
                    continue
                category_source_groups.add(source_group)
                if (
                    source_group in excluded_source_groups
                    or source_group in used_source_groups
                ):
                    continue

                eligible_group_count += 1
                row = dict(candidate)
                row["source_group"] = source_group
                if len(reservoir) < per_category:
                    reservoir.append(row)
                    continue
                replacement_index = rng.randrange(eligible_group_count)
                if replacement_index < per_category:
                    reservoir[replacement_index] = row

            available_group_counts[category] = eligible_group_count
            metadata_rejections[category] = category_rejections
            if len(reservoir) != per_category:
                raise ValueError(
                    f"Requested {per_category} unique {category} source groups in {split}, "
                    f"but only {len(reservoir)} remained after exclusions"
                )
            rng.shuffle(reservoir)
            for row in reservoir:
                row["selection_order"] = len(selected)
                row["sampling_seed"] = seed
                selected.append(row)
                used_source_groups.add(str(row["source_group"]))
    finally:
        connection.close()

    atomic_write_csv(output_path, selected)
    category_counts = Counter(str(row["modify_type"]) for row in selected)
    summary: dict[str, object] = {
        "purpose": "avpp_train_preprocessing_canary_not_full_training_set",
        "database": database_path.as_posix(),
        "split": split,
        "categories": list(categories),
        "per_category": per_category,
        "sampling_seed": seed,
        "excluded_splits": list(excluded_splits),
        "excluded_source_groups": len(excluded_source_groups),
        "available_source_groups_after_exclusions": available_group_counts,
        "metadata_row_rejections": {
            category: dict(sorted(metadata_rejections[category].items()))
            for category in categories
        },
        "selected_videos": len(selected),
        "selected_source_groups": len(used_source_groups),
        "category_counts": dict(sorted(category_counts.items())),
        "manifest": output_path.as_posix(),
        "manifest_sha256": file_sha256(output_path),
    }
    atomic_write_json(summary_path, summary)
    return selected, summary
