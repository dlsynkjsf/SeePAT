from __future__ import annotations

from collections.abc import Iterable


def intervals_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    if first_end < first_start or second_end < second_start:
        raise ValueError("Interval end must be greater than or equal to interval start")
    return max(first_start, second_start) < min(first_end, second_end)


def event_overlaps_any(
    event_start: float,
    event_end: float,
    segments: Iterable[Iterable[float]],
) -> bool:
    return any(
        intervals_overlap(event_start, event_end, float(segment_start), float(segment_end))
        for segment_start, segment_end in segments
    )

