from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def event_evidence_status(
    face_result: dict[str, object], min_valid_landmark_ratio: float
) -> tuple[str, str]:
    """Evaluate evidence quality without consulting labels or predictions."""
    valid_ratio = float(face_result["valid_landmark_ratio"])
    multiple_ratio = float(face_result["multiple_face_ratio"])
    if multiple_ratio > 0.5:
        return "ineligible", "multiple_faces"
    if face_result["normalized_minimum_closure"] is None:
        return "ineligible", "face_missing"
    if valid_ratio < min_valid_landmark_ratio:
        return "ineligible", "landmarks_unstable"
    return "eligible", "eligible"


def video_evidence_status(
    events: Sequence[dict[str, object]], minimum_events: int
) -> tuple[str, str, int]:
    eligible_events = sum(event["eligibility_status"] == "eligible" for event in events)
    if len(events) < minimum_events:
        return "ineligible", "insufficient_bilabials", eligible_events
    if eligible_events < minimum_events:
        reasons = [
            str(event["exclusion_reason"])
            for event in events
            if event["eligibility_status"] != "eligible"
        ]
        reason = Counter(reasons).most_common(1)[0][0] if reasons else "face_missing"
        return "ineligible", reason, eligible_events
    return "eligible", "eligible", eligible_events
