from __future__ import annotations

import pytest

from seepat.artifacts import atomic_write_gzip_json, file_sha256
from seepat.preprocessing.vild import (
    VILD_TRACE_VERSION,
    event_vild_frames,
    load_vild_trace,
    non_speech_reference_windows,
    summarize_reference_windows,
)


def test_non_speech_windows_are_label_independent_and_evenly_bounded() -> None:
    windows = non_speech_reference_windows(
        duration_s=5.0,
        speech_intervals=[(1.0, 2.0), (3.0, 4.0)],
        window_seconds=0.5,
        max_windows=4,
        speech_margin_seconds=0.1,
    )

    assert [window["reference_id"] for window in windows] == [
        "reference-000",
        "reference-001",
        "reference-002",
    ]
    assert all(
        float(window["end_s"]) - float(window["start_s"]) == pytest.approx(0.5)
        for window in windows
    )
    assert all(
        float(window["end_s"]) <= 0.9
        or 2.1 <= float(window["start_s"]) < float(window["end_s"]) <= 2.9
        or float(window["start_s"]) >= 4.1
        for window in windows
    )


def test_non_speech_windows_require_observed_speech_segments() -> None:
    assert (
        non_speech_reference_windows(
            duration_s=5.0,
            speech_intervals=[],
            window_seconds=0.5,
            max_windows=8,
            speech_margin_seconds=0.1,
        )
        == []
    )


def test_reference_window_summary_preserves_missing_frame_evidence() -> None:
    frames = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "face_count": 1,
            "normalized_vild": 0.1,
        },
        {
            "frame_index": 1,
            "timestamp_s": 0.1,
            "face_count": 2,
            "normalized_vild": None,
        },
        {
            "frame_index": 2,
            "timestamp_s": 0.2,
            "face_count": 1,
            "normalized_vild": 0.3,
        },
    ]
    summaries = summarize_reference_windows(
        frames,
        [{"reference_id": "reference-000", "start_s": 0.0, "end_s": 0.3}],
    )

    summary = summaries[0]
    assert summary["attempted_frames"] == 3
    assert summary["valid_frames"] == 2
    assert summary["valid_landmark_ratio"] == pytest.approx(2 / 3)
    assert summary["multiple_face_ratio"] == pytest.approx(1 / 3)
    assert summary["normalized_vild_median"] == pytest.approx(0.2)
    assert summary["normalized_vild_iqr"] == pytest.approx(0.1)
    assert summary["mean_absolute_vild_delta"] == pytest.approx(0.2)


def test_trace_loader_verifies_hash_version_and_event_lookup(tmp_path) -> None:
    path = tmp_path / "trace.json.gz"
    artifact = {
        "artifact_version": VILD_TRACE_VERSION,
        "frames": [
            {"timestamp_s": 0.0, "normalized_vild": 0.2},
            {"timestamp_s": 0.1, "normalized_vild": 0.1},
            {"timestamp_s": 0.2, "normalized_vild": 0.3},
        ],
        "bilabial_event_windows": [
            {"event_id": "event-1", "window_start_s": 0.05, "window_end_s": 0.2}
        ],
    }
    atomic_write_gzip_json(path, artifact)

    loaded = load_vild_trace(path, expected_sha256=file_sha256(path))

    assert [frame["timestamp_s"] for frame in event_vild_frames(loaded, "event-1")] == [
        0.1,
        0.2,
    ]
    with pytest.raises(ValueError, match="hash does not match"):
        load_vild_trace(path, expected_sha256="incorrect")
