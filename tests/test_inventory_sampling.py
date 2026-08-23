from __future__ import annotations

import json
from pathlib import Path

from seepat.data.inventory import build_inventory
from seepat.data.sampling import (
    canonical_source_group,
    sample_pilot,
    sample_training_canary,
)


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


def test_canonical_source_group_normalizes_dataset_paths() -> None:
    assert (
        canonical_source_group(
            "vox_celeb_2/VoxCeleb2//dev/mp4/id01/source.mp4",
            "unused.mp4",
        )
        == "vox_celeb_2/VoxCeleb2/dev/mp4/id01/source.mp4"
    )
    assert canonical_source_group(None, "lrs3\\speaker\\clip.mp4") == (
        "lrs3/speaker/clip.mp4"
    )


def test_training_canary_is_balanced_deterministic_and_validation_safe(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    database_path = tmp_path / "inventory.sqlite"
    inventory_summary_path = tmp_path / "inventory_summary.json"
    categories = ("real", "audio_modified", "visual_modified", "both_modified")
    rows = []
    validation_sources = set()
    for category_index, category in enumerate(categories):
        for item_index in range(6):
            source = f"sources/group-{category_index}-{item_index}/real.mp4"
            rows.append(
                {
                    "file": f"train/group-{category_index}-{item_index}/{category}.mp4",
                    "original": source,
                    "split": "train",
                    "modify_type": category,
                    "fake_segments": [] if category == "real" else [[0.2, 0.4]],
                    "audio_fake_segments": (
                        [[0.2, 0.4]]
                        if category in {"audio_modified", "both_modified"}
                        else []
                    ),
                    "visual_fake_segments": (
                        [[0.2, 0.4]]
                        if category in {"visual_modified", "both_modified"}
                        else []
                    ),
                    "video_frames": 100,
                    "audio_frames": 64_000,
                }
            )
        validation_source = f"sources/group-{category_index}-0/real.mp4"
        validation_sources.add(validation_source)
        rows.append(
            {
                "file": f"val/group-{category_index}-0/{category}.mp4",
                "original": validation_source.replace("/group", "//group"),
                "split": "val",
                "modify_type": category,
                "fake_segments": [],
                "audio_fake_segments": [],
                "visual_fake_segments": [],
                "video_frames": 100,
                "audio_frames": 64_000,
            }
        )
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")
    build_inventory([metadata_path], database_path, inventory_summary_path)

    first_manifest = tmp_path / "canary_first.csv"
    first_summary_path = tmp_path / "canary_first.json"
    second_manifest = tmp_path / "canary_second.csv"
    second_summary_path = tmp_path / "canary_second.json"
    first, first_summary = sample_training_canary(
        database_path=database_path,
        output_path=first_manifest,
        summary_path=first_summary_path,
        split="train",
        categories=categories,
        per_category=2,
        seed=456,
        excluded_splits=("val",),
    )
    second, second_summary = sample_training_canary(
        database_path=database_path,
        output_path=second_manifest,
        summary_path=second_summary_path,
        split="train",
        categories=categories,
        per_category=2,
        seed=456,
        excluded_splits=("val",),
    )

    assert first == second
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert len(first) == 8
    assert len({row["source_group"] for row in first}) == 8
    assert not ({str(row["source_group"]) for row in first} & validation_sources)
    assert first_summary["category_counts"] == {category: 2 for category in categories}
    assert first_summary["selected_source_groups"] == 8
    assert first_summary["manifest_sha256"] == second_summary["manifest_sha256"]
