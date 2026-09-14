from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_gzip_json, file_sha256, read_csv_rows
from seepat.evidence import FUSION_EVIDENCE_FIELDS, evidence_feature_values
from seepat.preprocessing.vild import load_vild_trace
from seepat.training import calibration as cal


def make_row(
    directory: Path,
    video: str,
    *,
    split="train",
    label="0",
    baseline=1.0,
    reference_end=2.0,
    minimum=8.0,
    raw=True,
) -> dict[str, str]:
    frames = []
    for i in range(30):
        bbox = 100.0 + i
        value = bbox * 0.1 + baseline * (1 + (i % 3) / 10) if i < 20 else minimum + i / 100
        frame = {
            "frame_index": i,
            "timestamp_s": i / 10,
            "face_count": 1,
            "normalized_vild": value / 40,
        }
        if raw:
            frame.update(raw_vild_px=value, face_bbox_size_px=bbox)
        frames.append(frame)
    trace_path = directory / f"{video}.json.gz"
    atomic_write_gzip_json(
        trace_path,
        {
            "artifact_version": "vild-trace-v1",
            "video_id": video,
            "source_file": f"{video}.mp4",
            "dataset_split": split,
            "subject_id": "same-subject",
            "timing": {"fps": 10.0},
            "frames": frames,
            "bilabial_event_windows": [
                {"event_id": video + "-0", "window_start_s": 2.0, "window_end_s": 2.9},
                {"event_id": video + "-1", "window_start_s": 2.0, "window_end_s": 2.9},
            ],
            "non_speech_reference": {"windows": [{"start_s": 0.0, "end_s": reference_end}]},
        },
    )
    return {
        "video_id": video,
        "event_id": video + "-0",
        "file": f"{video}.mp4",
        "dataset_split": split,
        "source_group": video,
        "subject_id": "same-subject",
        "class_id": label,
        "phoneme": "p",
        "vild_trace_path": str(trace_path),
        "vild_trace_sha256": file_sha256(trace_path),
    }


OPTIONS = cal.CalibrationOptions(isolation_trees=10, random_seed=9)


def test_labels_do_not_select_regression_or_video_references(tmp_path):
    rows = [make_row(tmp_path, "a"), make_row(tmp_path, "b", baseline=3, label="1")]
    population = cal._fit_population(rows, OPTIONS)
    flipped = cal._fit_population([{**row, "class_id": "1"} for row in rows], OPTIONS)
    removed = cal._fit_population(
        [{k: v for k, v in row.items() if k != "class_id"} for row in rows], OPTIONS
    )
    assert population["reference_frames"] == 40
    assert population["regression"] == flipped["regression"] == removed["regression"]
    artifact = {**population, "options": OPTIONS.__dict__}
    first = cal._score_video(rows[:1], artifact, OPTIONS)[0][0]
    changed = cal._score_video([{**rows[0], "class_id": "1"}], artifact, OPTIONS)[0][0]
    assert first["isolation_forest_anomaly_score"] == changed["isolation_forest_anomaly_score"]


def test_same_subject_videos_fit_independently_and_trace_loaded_once_per_pass(
    tmp_path, monkeypatch
):
    a = make_row(tmp_path, "a")
    b = make_row(tmp_path, "b", baseline=5, minimum=9)
    rows = [a, {**a, "event_id": "a-1"}, b]
    train = tmp_path / "train.csv"
    atomic_write_csv(train, rows)
    fit_inputs, loads = [], []
    original_fit, original_load = cal.IsolationForest.fit, cal.augmented_trace_for_row

    def fit(self, values, *args, **kwargs):
        fit_inputs.append(np.asarray(values).copy())
        return original_fit(self, values, *args, **kwargs)

    def load(row):
        loads.append(row["video_id"])
        return original_load(row)

    monkeypatch.setattr(cal.IsolationForest, "fit", fit)
    monkeypatch.setattr(cal, "augmented_trace_for_row", load)
    cal.fit_and_score_calibration(train, {"train": train}, tmp_path / "out", OPTIONS)
    assert len(fit_inputs) == 2
    assert all(values.shape == (20, 1) for values in fit_inputs)
    assert not np.array_equal(*fit_inputs)
    assert loads == ["a", "b", "a", "b"]  # population pass then scoring pass
    scored = read_csv_rows(tmp_path / "out/events_train_calibrated.csv")
    assert len(scored) == 3
    assert {row["isolation_forest_scope"] for row in scored} == {"input_video"}
    assert all(0 <= float(row["isolation_forest_anomaly_score"]) <= 1 for row in scored)


def test_sparse_and_legacy_traces_mask_missing_features_without_fallback(tmp_path):
    dense = make_row(tmp_path, "dense")
    artifact = cal._fit_population([dense], OPTIONS)
    sparse = make_row(tmp_path, "sparse", reference_end=0.2)
    legacy = make_row(tmp_path, "legacy", raw=False)
    for row in (sparse, legacy):
        scored = cal._score_video([row], artifact, OPTIONS)[0][0]
        assert scored["isolation_forest_available"] is False
        assert scored["isolation_forest_anomaly_score"] == ""
        assert (
            scored["isolation_forest_unavailable_reason"] == "insufficient_video_reference_frames"
        )
        _, mask = evidence_feature_values(scored)
        assert not mask[FUSION_EVIDENCE_FIELDS.index("isolation_forest_anomaly_score")]
    assert cal._score_video([legacy], artifact, OPTIONS)[0][0]["vild_regression_residual_px"] == ""


def test_reference_quality_and_unique_frame_selection(tmp_path):
    row = make_row(tmp_path, "a")
    trace = load_vild_trace(Path(row["vild_trace_path"]))
    trace["non_speech_reference"]["windows"] *= 2
    assert len(cal.reference_samples(trace, OPTIONS)) == 20
    trace["frames"][0]["face_count"] = 2
    assert cal.reference_samples(trace, OPTIONS) == []
    trace["frames"][0]["face_count"] = 1
    for frame in trace["frames"][:5]:
        frame["raw_vild_px"] = None
    assert cal.reference_samples(trace, OPTIONS) == []


@pytest.mark.parametrize("split", ["val", "test"])
def test_non_train_cannot_enter_population_fit(tmp_path, split):
    with pytest.raises(ValueError, match="exclusively Train"):
        cal._fit_population(
            [make_row(tmp_path, "a"), make_row(tmp_path, "b", split=split)], OPTIONS
        )


def test_frozen_population_and_resume_cache_and_hash_invalidation(tmp_path, monkeypatch):
    train = tmp_path / "train.csv"
    atomic_write_csv(train, [make_row(tmp_path, "a"), make_row(tmp_path, "b", minimum=9)])
    val = tmp_path / "val.csv"
    atomic_write_csv(val, [make_row(tmp_path, "v", split="val", baseline=20)])
    out = tmp_path / "out"
    cal.fit_and_score_calibration(train, {"train": train, "val": val}, out, OPTIONS)
    digest = file_sha256(out / "calibration.json")
    assert cal.calibration_outputs_are_current(train, {"train": train, "val": val}, out, OPTIONS)
    assert not cal.calibration_outputs_are_current(
        train, {"train": train, "val": val}, out, replace(OPTIONS, isolation_trees=11)
    )
    original_fit = cal.IsolationForest.fit

    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged population and video scores should be reused")

    monkeypatch.setattr(cal.IsolationForest, "fit", unexpected)
    monkeypatch.setattr(cal, "_fit_population", unexpected)
    cal.fit_and_score_calibration(train, {"train": train, "val": val}, out, OPTIONS)
    monkeypatch.setattr(cal.IsolationForest, "fit", original_fit)
    atomic_write_csv(val, [make_row(tmp_path, "v", split="val", baseline=50)])
    cal.fit_and_score_calibration(train, {"train": train, "val": val}, out, OPTIONS)
    assert file_sha256(out / "calibration.json") == digest
    external = tmp_path / "test.csv"
    atomic_write_csv(external, [make_row(tmp_path, "external", split="test", baseline=100)])
    summary = cal.score_manifests_with_calibration(
        out / "calibration.json", {"test": external}, tmp_path / "scored"
    )
    assert summary["mode"] == "score_only"
    assert file_sha256(out / "calibration.json") == digest
    Path(read_csv_rows(val)[0]["vild_trace_path"]).write_bytes(b"corrupt")
    assert not cal.calibration_outputs_are_current(
        train, {"train": train, "val": val}, out, OPTIONS
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        cal.fit_and_score_calibration(train, {"val": val}, out, OPTIONS)


def test_calibration_rejects_corrupt_frozen_artifact(tmp_path):
    train = tmp_path / "train.csv"
    atomic_write_csv(train, [make_row(tmp_path, "a")])
    out = tmp_path / "out"
    cal.fit_and_score_calibration(train, {"train": train}, out, OPTIONS)
    path = out / "calibration.json"
    artifact = json.loads(path.read_text())
    artifact["regression"]["global"]["slope"] += 1
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="hash mismatch"):
        cal.score_manifests_with_calibration(path, {"train": train}, tmp_path / "scored")
