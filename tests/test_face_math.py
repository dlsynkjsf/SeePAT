from __future__ import annotations

import pytest

from seepat.preprocessing.face import (
    event_frame_bounds,
    normalized_vild,
    timestamp_to_frame,
)


def test_timestamp_to_frame_uses_explicit_floor_and_ceiling() -> None:
    assert timestamp_to_frame(1.01, fps=25.0) == 25
    assert timestamp_to_frame(1.01, fps=25.0, round_up=True) == 26
    assert timestamp_to_frame(-0.1, fps=25.0) == 0


def test_timestamp_to_frame_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        timestamp_to_frame(1.0, fps=0.0)


def test_event_frame_bounds_clamp_at_video_start_and_include_window() -> None:
    assert event_frame_bounds(
        phone_start_s=0.1,
        phone_end_s=0.2,
        fps=25.0,
        window_before_s=0.2,
        window_after_s=0.3,
    ) == (0, 13)


def test_event_frame_bounds_reject_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        event_frame_bounds(1.0, 0.9, 25.0, 0.2, 0.2)
    with pytest.raises(ValueError, match="cannot be negative"):
        event_frame_bounds(1.0, 1.1, 25.0, -0.1, 0.2)


def test_normalized_vild_is_scale_invariant() -> None:
    small = normalized_vild((5.0, 4.0), (5.0, 6.0), (0.0, 5.0), (10.0, 5.0))
    large = normalized_vild((10.0, 8.0), (10.0, 12.0), (0.0, 10.0), (20.0, 10.0))

    assert small == pytest.approx(0.2)
    assert large == pytest.approx(small)


def test_normalized_vild_rejects_degenerate_mouth_width() -> None:
    assert normalized_vild((0.0, 0.0), (0.0, 1.0), (2.0, 2.0), (2.0, 2.0)) is None
