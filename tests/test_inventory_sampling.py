from __future__ import annotations

import json
from pathlib import Path

from seepat.data.inventory import build_inventory
from seepat.data.sampling import sample_pilot


def _write_fixture(path: Path) -> None:
    rows = []
    categories = ("real", "audio_modified", "visual_modified", "both_modified")
    for category_index, category in enumerate(categories):
        for item_index in range(4):
            base = f"id{category_index:02d}_{item_index:02d}"
            rows.append(
                {
                    "file": f"vox_celeb_2/{base}/clip/{category}.mp4",
                    "original": f"vox_celeb_2/{base}/clip/real.mp4",
                    "split": "val",
                    "modify_type": category,
                    "audio_model": None,
                    "video_model": None,
                    "fake_segments": [] if category == "real" else [[1.0, 1.5]],
                    "audio_fake_segments": [],
                    "visual_fake_segments": [],
                    "video_frames": 100,
                    "audio_frames": 64_000,
                }
            )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_inventory_and_sampling_are_deterministic(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    database_path = tmp_path / "inventory.sqlite"
    summary_path = tmp_path / "summary.json"
    first_manifest = tmp_path / "pilot_first.csv"
    second_manifest = tmp_path / "pilot_second.csv"
    _write_fixture(metadata_path)

    summary = build_inventory([metadata_path], database_path, summary_path)
    assert summary["total_rows"] == 16

    first = sample_pilot(
        database_path,
        first_manifest,
        "val",
        ("real", "audio_modified", "visual_modified", "both_modified"),
        per_category=2,
        seed=123,
    )
    second = sample_pilot(
        database_path,
        second_manifest,
        "val",
        ("real", "audio_modified", "visual_modified", "both_modified"),
        per_category=2,
        seed=123,
    )

    assert first == second
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert len(first) == 8
    assert len({row["source_group"] for row in first}) == 8
    assert [row["selection_order"] for row in first] == list(range(8))
