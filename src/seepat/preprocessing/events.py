from __future__ import annotations

import json
from collections.abc import Sequence


def parse_segments(value: str | Sequence[Sequence[float]] | None) -> list[tuple[float, float]]:
    if value is None or value == "":
        return []
    decoded = json.loads(value) if isinstance(value, str) else value
    segments: list[tuple[float, float]] = []
    for segment in decoded:
        if len(segment) != 2:
            raise ValueError(f"Expected [start, end] segment, received: {segment!r}")
        start, end = float(segment[0]), float(segment[1])
        if end < start:
            raise ValueError(f"Segment end precedes start: {segment!r}")
        segments.append((start, end))
    return segments


def classify_event_label(
    event_start_s: float,
    event_end_s: float,
    manipulated_segments: Sequence[tuple[float, float]],
    boundary_tolerance_s: float,
) -> str:
    overlaps_boundary = False
    for segment_start, segment_end in manipulated_segments:
        overlap = min(event_end_s, segment_end) - max(event_start_s, segment_start)
        if overlap <= 0:
            continue
        overlaps_boundary = True
        safely_inside = (
            event_start_s >= segment_start + boundary_tolerance_s
            and event_end_s <= segment_end - boundary_tolerance_s
        )
        if safely_inside:
            return "fake"
    return "ambiguous" if overlaps_boundary else "real"
