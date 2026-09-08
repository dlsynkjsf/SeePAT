from __future__ import annotations

import json
from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_gzip_json, file_sha256, read_csv_rows
from seepat.preprocessing.vild import VILD_TRACE_VERSION
from seepat.training.calibration import CalibrationOptions, fit_and_score_calibration


def _write_trace(
    directory: Path,
    event_id: str,
    subject: str,
    event_minimum_raw_vild: float,
) -> Path:
    path = directory / f"{event_id}.json.gz"
    frames = []
    for index in range(10):
        timestamp = index / 10
        bbox_size = 100.0 + index
        raw_vild = bbox_size * 0.1 + (0.2 if subject == "subject-sparse" else 0.0)
        if 0.5 <= timestamp <= 0.8:
            raw_vild = event_minimum_raw_vild + (timestamp - 0.5) * 2
        frames.append(
            {
                "timestamp_s": timestamp,
                "face_count": 1,
                "normalized_vild": raw_vild / bbox_size,
                "raw_vild_px": raw_vild,
                "face_bbox_size_px": bbox_size,
            }
        )
    atomic_write_gzip_json(
        path,
        {
            "artifact_version": VILD_TRACE_VERSION,
            "subject_id": subject,
            "frames": frames,
            "bilabial_event_windows": [
                {"event_id": event_id, "window_start_s": 0.5, "window_end_s": 0.8}
            ],
            "non_speech_reference": {
                "windows": [{"start_s": 0.0, "end_s": 0.4}],
            },
        },
    )
    return path


def _row(
    directory: Path,
    event_id: str,
    *,
    split: str = "train",
    class_id: str = "0",
    subject: str = "subject-dense",
    phoneme: str = "p",
    event_minimum_raw_vild: float = 8.0,
) -> dict[str, object]:
    trace_path = _write_trace(directory, event_id, subject, event_minimum_raw_vild)
    return {
        "event_id": event_id,
        "dataset_split": split,
        "class_id": class_id,
        "subject_id": subject,
        "phoneme": phoneme,
        "vild_trace_path": str(trace_path),
        "vild_trace_sha256": file_sha256(trace_path),
        "vild_trace_event_key": event_id,
    }


def test_calibration_uses_train_non_speech_frames_and_scores_event_minima(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    train_rows = [
        _row(tmp_path, f"dense-{index}", event_minimum_raw_vild=8.0 + index * 0.1)
        for index in range(4)
    ] + [
        _row(
            tmp_path,
            f"sparse-{index}",
            subject="subject-sparse",
            phoneme="b",
            event_minimum_raw_vild=8.5 + index * 0.1,
        )
        for index in range(2)
    ] + [
        _row(tmp_path, "fake-is-excluded", class_id="1", event_minimum_raw_vild=100.0)
    ]
    atomic_write_csv(train_path, train_rows)
    validation_path = tmp_path / "validation.csv"
    atomic_write_csv(
        validation_path,
        [
            _row(
                tmp_path,
                "val-dense",
                split="val",
                event_minimum_raw_vild=8.1,
            ),
            _row(
                tmp_path,
                "val-sparse",
                split="val",
                subject="subject-sparse",
                event_minimum_raw_vild=8.6,
            ),
        ],
    )

    summary = fit_and_score_calibration(
        train_path,
        {"train": train_path, "val": validation_path},
        tmp_path / "calibration",
        CalibrationOptions(
            min_subject_reference_frames=16,
            isolation_trees=10,
            random_seed=9,
        ),
    )

    artifact = json.loads((tmp_path / "calibration" / "calibration.json").read_text())
    assert artifact["fit_population"] == "genuine AV++ Train non-speech VILD trace frames only"
    assert artifact["reference_frames"] == 24
    assert artifact["regression"]["dependent_variable"] == "raw_vild_px"
    assert artifact["regression"]["independent_variable"] == "face_bbox_size_px"
    assert list(artifact["regression"]["subjects"]) == ["subject-dense"]
    assert summary["subject_models"] == 1

    scored = read_csv_rows(tmp_path / "calibration" / "events_val_calibrated.csv")
    assert scored[0]["isolation_forest_scope"] == "subject"
    assert scored[1]["isolation_forest_scope"] == "global_sparse_subject_fallback"
    assert float(scored[0]["event_minimum_raw_vild_px"]) == pytest.approx(8.1)
    assert scored[0]["vild_regression_residual_px"]
    assert scored[0]["phoneme_viseme_residual_z"]
    assert scored[0]["isolation_forest_anomaly_score"]


def test_calibration_rejects_train_manifest_without_genuine_train_events(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    atomic_write_csv(train_path, [_row(tmp_path, "fake", class_id="1")])

    with pytest.raises(ValueError, match="genuine events"):
        fit_and_score_calibration(train_path, {"train": train_path}, tmp_path / "out")
