from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_calibration import make_row

from seepat.artifacts import atomic_write_csv, atomic_write_gzip_json, file_sha256, read_csv_rows
from seepat.preprocessing import augmentation as aug
from seepat.preprocessing.vild import load_vild_trace


def test_visual_augmentation_preserves_originals_resumes_and_verifies_hashes(tmp_path, monkeypatch):
    row = make_row(tmp_path, "a", raw=False)
    manifest = tmp_path / "events.csv"
    atomic_write_csv(manifest, [row, {**row, "event_id": "a-1"}])
    (tmp_path / "a.mp4").write_bytes(b"synthetic-video")
    job = aug.TraceAugmentationJob("test", manifest, tmp_path, tmp_path / "aug")
    monkeypatch.setattr(
        aug,
        "measurement_contract",
        lambda job: {
            "version": aug.AUGMENTATION_VERSION,
            "normalized_tolerance": job.normalized_tolerance,
            "timestamp_tolerance_s": job.timestamp_tolerance_s,
        },
    )
    calls = []

    class Analyzer:
        def trace_video(self, path, fps):
            calls.append(path)
            trace = load_vild_trace(Path(row["vild_trace_path"]))
            for frame in trace["frames"]:
                frame.update(raw_vild_px=frame["normalized_vild"] * 40, face_bbox_size_px=100.0)
            return trace

        def close(self):
            pass

    monkeypatch.setattr(aug, "MouthEventAnalyzer", Analyzer)
    original = file_sha256(Path(row["vild_trace_path"]))
    updates = []
    aug.run_trace_augmentation(job, progress=lambda *args: updates.append(args))
    assert updates == [
        ("verify augmentation cache", 0, 0, ""),
        ("augment visual traces", 0, 1, "a"),
        ("augment visual traces", 1, 1, "reused=0; failures=0"),
    ]
    assert len(calls) == 1
    assert file_sha256(Path(row["vild_trace_path"])) == original
    assert aug.augmentation_outputs_are_current(job)
    augmented = read_csv_rows(job.output_dir / "events_augmented.csv")
    assert len(augmented) == 2
    assert aug.augmented_trace_for_row(augmented[0])["frames"][0]["raw_vild_px"] > 0
    aug.run_trace_augmentation(job)
    assert len(calls) == 1
    # A missing final manifest simulates interruption after the individual trace was saved.
    (job.output_dir / "events_augmented.csv").unlink()
    aug.run_trace_augmentation(job)
    assert len(calls) == 1
    Path(augmented[0]["vild_augmentation_path"]).write_bytes(b"corrupt")
    assert not aug.augmentation_outputs_are_current(job)
    with pytest.raises(ValueError, match="hash mismatch"):
        aug.augmented_trace_for_row(augmented[0])
    aug.run_trace_augmentation(job)
    assert len(calls) == 2
    assert not aug.augmentation_outputs_are_current(replace(job, normalized_tolerance=1e-6))
    (tmp_path / "a.mp4").write_bytes(b"changed-source")
    assert not aug.augmentation_outputs_are_current(job)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp_s", 100.0),
        ("frame_index", 99),
        ("face_count", 2),
        ("normalized_vild", 99.0),
        ("normalized_vild", None),
        ("raw_vild_px", float("nan")),
    ],
)
def test_measurement_mismatch_is_rejected(tmp_path, field, value):
    row = make_row(tmp_path, "a")
    trace = load_vild_trace(Path(row["vild_trace_path"]))
    measured = copy.deepcopy(trace["frames"])
    measured[0][field] = value
    with pytest.raises(ValueError):
        aug.validate_measurements(trace, measured, 1e-5, 1e-6)


def test_mismatched_trace_identity_is_rejected(tmp_path):
    row = make_row(tmp_path, "a")
    with pytest.raises(ValueError, match="identity mismatch"):
        aug.trace_for_row({**row, "dataset_split": "test"})


def test_failed_augmentation_reports_error_and_never_rewrites_v1(tmp_path, monkeypatch):
    row = make_row(tmp_path, "a")
    trace_path = Path(row["vild_trace_path"])
    trace = load_vild_trace(trace_path)
    trace["frames"][0]["normalized_vild"] = None
    atomic_write_gzip_json(trace_path, trace)
    # Stale manifest hash must fail before any decoding.
    manifest = tmp_path / "events.csv"
    atomic_write_csv(manifest, [row])
    job = aug.TraceAugmentationJob("failure", manifest, tmp_path, tmp_path / "out")
    monkeypatch.setattr(aug, "measurement_contract", lambda job: {})
    with pytest.raises(RuntimeError, match="augmentation failed"):
        aug.run_trace_augmentation(job)
    report = json.loads((job.output_dir / "summary.json").read_text())
    assert report["status"] == "failed"
    assert len(report["failures"]) == 1
    assert not (job.output_dir / "events_augmented.csv").exists()
