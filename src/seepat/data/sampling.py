from __future__ import annotations

import csv
import random
import sqlite3
from collections.abc import Sequence
from pathlib import Path


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
