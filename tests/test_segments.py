import pytest

from seepat.data.segments import event_overlaps_any, intervals_overlap


def test_overlap_uses_positive_duration() -> None:
    assert intervals_overlap(1.0, 1.5, 1.4, 2.0)
    assert not intervals_overlap(1.0, 1.5, 1.5, 2.0)


def test_event_overlap_checks_all_segments() -> None:
    assert event_overlaps_any(4.3, 4.5, [[1.0, 2.0], [4.4, 4.64]])
    assert not event_overlaps_any(2.1, 3.0, [[1.0, 2.0], [4.4, 4.64]])


def test_invalid_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        intervals_overlap(2.0, 1.0, 0.0, 1.0)

