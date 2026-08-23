from __future__ import annotations

from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv, read_csv_rows
from seepat.training.manifest import prepare_training_manifests


def _source(file_name: str, split: str = "val", group: str = "source-1") -> dict[str, object]:
    return {
        "file": file_name,
        "split": split,
        "source_group": group,
        "subject_id": "person-1",
    }


def _video(file_name: str, video_id: str = "video-1") -> dict[str, object]:
    return {
        "video_id": video_id,
        "file": file_name,
        "split": "train" if file_name.startswith("train/") else "val",
        "pipeline_status": "complete",
        "eligibility_status": "eligible",
    }


def _event(
    file_name: str,
    clip_path: str,
    event_id: str = "event-1",
    video_id: str = "video-1",
    split: str = "val",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "video_id": video_id,
        "file": file_name,
        "split": split,
        "phoneme": "P",
        "phone_start_s": "1.0",
        "phone_end_s": "1.1",
        "event_label": "fake",
        "training_label_status": "usable",
        "manipulation_modality": "audio_modified",
        "valid_landmark_ratio": "1.0",
        "multiple_face_ratio": "0.0",
        "normalized_minimum_closure": "0.02",
        "closure_time_s": "1.05",
        "closure_duration_s": "0.04",
        "closing_velocity": "-0.2",
        "opening_velocity": "0.3",
        "mouth_crop_path": clip_path,
        "eligibility_status": "eligible",
    }


def test_prepare_training_manifests_filters_and_normalizes_paths(tmp_path: Path) -> None:
    clip = tmp_path / "outputs" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"test clip placeholder")
    source_path = tmp_path / "source.csv"
    video_path = tmp_path / "videos.csv"
    event_path = tmp_path / "events.csv"
    output_dir = tmp_path / "prepared"

    atomic_write_csv(source_path, [_source("val/video.mp4")])
    atomic_write_csv(video_path, [_video("val/video.mp4")])
    usable = _event("val/video.mp4", "outputs\\clip.mp4")
    ambiguous = _event(
        "val/video.mp4",
        "outputs\\clip.mp4",
        event_id="event-2",
    )
    ambiguous["event_label"] = "ambiguous"
    ambiguous["training_label_status"] = "omit_boundary"
    atomic_write_csv(event_path, [usable, ambiguous])

    summary = prepare_training_manifests(
        source_path,
        video_path,
        event_path,
        output_dir,
        project_root=tmp_path,
    )

    rows = read_csv_rows(output_dir / "events_val.csv")
    assert summary["written_events"] == 1
    assert summary["omission_reasons"] == {"ambiguous_boundary": 1}
    assert rows[0]["class_id"] == "1"
    assert rows[0]["mouth_clip_path"] == "outputs/clip.mp4"
    assert float(rows[0]["phone_duration_s"]) == pytest.approx(0.1)


def test_prepare_training_manifests_rejects_source_group_leakage(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    video_path = tmp_path / "videos.csv"
    event_path = tmp_path / "events.csv"
    atomic_write_csv(
        source_path,
        [
            _source("train/video.mp4", split="train", group="shared"),
            _source("val/video.mp4", split="val", group="shared"),
        ],
    )
    atomic_write_csv(
        video_path,
        [
            _video("train/video.mp4", video_id="train-video"),
            _video("val/video.mp4", video_id="val-video"),
        ],
    )
    atomic_write_csv(
        event_path,
        [
            _event(
                "train/video.mp4",
                "train.mp4",
                event_id="train-event",
                video_id="train-video",
                split="train",
            ),
            _event(
                "val/video.mp4",
                "val.mp4",
                event_id="val-event",
                video_id="val-video",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Source-group leakage"):
        prepare_training_manifests(
            source_path,
            video_path,
            event_path,
            tmp_path / "prepared",
            require_clips=False,
        )
