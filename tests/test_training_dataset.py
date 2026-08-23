from __future__ import annotations

from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv
from seepat.training.dataset import (
    FEATURE_FIELDS,
    MouthEventDataset,
    numeric_feature_values,
    sequence_frame_indices,
)


def test_sequence_frame_indices_samples_full_clip() -> None:
    indices, mask = sequence_frame_indices(frame_count=9, sequence_length=5)

    assert indices == [0, 2, 4, 6, 8]
    assert mask == [True, True, True, True, True]


def test_sequence_frame_indices_repeats_last_frame_for_padding() -> None:
    indices, mask = sequence_frame_indices(frame_count=3, sequence_length=5)

    assert indices == [0, 1, 2, 2, 2]
    assert mask == [True, True, True, False, False]


def test_numeric_feature_values_masks_missing_and_nonfinite_values() -> None:
    row = {field: str(index / 10) for index, field in enumerate(FEATURE_FIELDS)}
    row["closing_velocity"] = ""
    row["opening_velocity"] = "nan"

    values, mask = numeric_feature_values(row)

    assert values[FEATURE_FIELDS.index("closing_velocity")] == 0.0
    assert values[FEATURE_FIELDS.index("opening_velocity")] == 0.0
    assert mask[FEATURE_FIELDS.index("closing_velocity")] is False
    assert mask[FEATURE_FIELDS.index("opening_velocity")] is False


def test_dataset_filters_manifest_by_split_without_loading_clips(tmp_path: Path) -> None:
    manifest = tmp_path / "events.csv"
    base_row = {
        "event_id": "event-1",
        "video_id": "video-1",
        "source_group": "source-1",
        "phoneme": "P",
        "class_id": "0",
        "mouth_clip_path": "clip.mp4",
        **{field: "0.0" for field in FEATURE_FIELDS},
    }
    atomic_write_csv(
        manifest,
        [
            {**base_row, "dataset_split": "train"},
            {**base_row, "event_id": "event-2", "dataset_split": "val"},
        ],
    )

    dataset = MouthEventDataset(manifest, dataset_split="val")

    assert len(dataset) == 1
    assert dataset.rows[0]["event_id"] == "event-2"


def test_dataset_rejects_unknown_split(tmp_path: Path) -> None:
    manifest = tmp_path / "events.csv"
    atomic_write_csv(manifest, [{"dataset_split": "val"}])

    with pytest.raises(ValueError, match="no events"):
        MouthEventDataset(manifest, dataset_split="train")
