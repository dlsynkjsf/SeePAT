"""Synthetic checks for the evidence-path review; no saved project inputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seepat.evidence import FUSION_EVIDENCE_FIELDS, evidence_feature_values
from seepat.preprocessing.face import MouthEventAnalyzer
from seepat.training import calibration
from seepat.training.dataset import numeric_feature_values


def event_summary(monkeypatch, openings):
    """Run the actual summary code with an in-memory capture and inert writer."""
    frames = iter(np.zeros((4, 4, 3), dtype=np.uint8) for _ in openings)
    measurements = iter(openings)
    capture = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=lambda: (True, next(frames)),
        release=lambda: None,
    )
    writer = SimpleNamespace(write=lambda *_: None, release=lambda: None)
    analyzer = MouthEventAnalyzer.__new__(MouthEventAnalyzer)
    analyzer.cv2 = SimpleNamespace(
        VideoCapture=lambda *_: capture,
        VideoWriter=lambda *_: writer,
        VideoWriter_fourcc=lambda *_: 0,
        CAP_PROP_POS_FRAMES=0,
        resize=lambda image, _: image,
    )
    analyzer._measure_frame = lambda _: (
        1, next(measurements) / 100, {0: (1, 1), 1: (3, 3)}, None, None,
    )
    # No real file, codec, face detector or model is opened by this fixture.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
        patch.setattr(Path, "replace", lambda *_args, **_kwargs: None)
        return analyzer.analyze(
            Path("synthetic-only.mp4"), 1.0, 1.2, "p", 10.0, 0.2, 0.2,
            Path("synthetic-output.mp4"), None,
        )


def calibrated_evidence(monkeypatch, openings, summary):
    frames = [
        {"frame_index": index + 8, "timestamp_s": (index + 8) / 10,
         "face_count": 1, "raw_vild_px": value,
         "normalized_vild": value / 100, "face_bbox_size_px": 100.0}
        for index, value in enumerate(openings)
    ]
    trace = {
        "frames": frames,
        "bilabial_event_windows": [
            {"event_id": "synthetic-0", "window_start_s": 0.8, "window_end_s": 1.4},
        ],
        "non_speech_reference": {"windows": []},
    }
    row = {
        **{key: str(value) for key, value in summary.items()},
        "event_id": "synthetic-0", "video_id": "synthetic", "phoneme": "p",
        "video_phone_start_s": "1.0", "phone_duration_s": "0.2",
    }
    population = {
        "regression": {"global": {"intercept": 0.0, "slope": 0.0}},
        "phoneme_viseme_expectations": {
            "p": {"mean_residual_px": 1.0, "scale_residual_px": 1.0},
        },
    }
    with monkeypatch.context() as patch:
        patch.setattr(calibration, "augmented_trace_for_row", lambda _: trace)
        scored, _ = calibration._score_video(
            [row], population, calibration.DEFAULT_CALIBRATION_OPTIONS,
        )
    return scored[0], evidence_feature_values(scored[0])


def test_different_phoneme_motion_can_share_all_current_summary_inputs(monkeypatch):
    first = [2.0, 0.5, 3.0, 4.0, 3.0, 5.0, 2.0]
    second = [2.0, 0.5, 1.0, 1.1, 1.2, 5.0, 2.0]
    summaries = [event_summary(monkeypatch, values) for values in (first, second)]
    assert summaries[0] == summaries[1]
    scored_a, evidence_a = calibrated_evidence(monkeypatch, first, summaries[0])
    scored_b, evidence_b = calibrated_evidence(monkeypatch, second, summaries[1])
    assert evidence_a == evidence_b
    assert numeric_feature_values(scored_a) == numeric_feature_values(scored_b)
    assert min(first[2:5]) == 3.0 and min(second[2:5]) == 1.0
    assert summaries[0]["closure_time_s"] == 0.9  # Outside the aligned [1.0, 1.2].
    assert not evidence_a[1][FUSION_EVIDENCE_FIELDS.index("isolation_forest_anomaly_score")]


def test_current_evidence_retains_a_relative_closure_anchor():
    row = {"closure_time_s": "0.9", "video_phone_start_s": "1.0"}
    shifted = {**row, "video_phone_start_s": "1.1"}
    index = FUSION_EVIDENCE_FIELDS.index("closure_offset_s")
    first, mask = evidence_feature_values(row)
    second, shifted_mask = evidence_feature_values(shifted)
    assert mask[index] and shifted_mask[index]
    assert first[index] == pytest.approx(-0.1)
    assert second[index] == pytest.approx(-0.2)


def test_missing_opening_stays_distinct_from_observed_zero():
    index = FUSION_EVIDENCE_FIELDS.index("normalized_minimum_closure")
    observed, observed_mask = evidence_feature_values({"normalized_minimum_closure": "0"})
    missing, missing_mask = evidence_feature_values({"normalized_minimum_closure": "nan"})
    assert observed[index] == missing[index] == 0.0
    assert observed_mask[index] is True and missing_mask[index] is False
