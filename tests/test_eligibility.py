from __future__ import annotations

from seepat.preprocessing.eligibility import (
    event_evidence_status,
    video_evidence_status,
)


def test_event_eligibility_uses_evidence_quality_only() -> None:
    good_face = {
        "valid_landmark_ratio": 0.9,
        "multiple_face_ratio": 0.0,
        "normalized_minimum_closure": 0.1,
    }
    for changes, expected in [
        ({}, ("eligible", "eligible")),
        ({"multiple_face_ratio": 0.75}, ("ineligible", "multiple_faces")),
        ({"normalized_minimum_closure": None}, ("ineligible", "face_missing")),
        ({"valid_landmark_ratio": 0.5}, ("ineligible", "landmarks_unstable")),
    ]:
        assert event_evidence_status({**good_face, **changes}, 0.8) == expected


def test_video_eligibility_requires_enough_usable_events() -> None:
    events = [
        {"eligibility_status": "eligible", "exclusion_reason": "eligible"},
        {"eligibility_status": "ineligible", "exclusion_reason": "face_missing"},
    ]

    for selected, minimum, expected in [
        (events, 1, ("eligible", "eligible", 1)),
        (events, 2, ("ineligible", "face_missing", 1)),
        (events[:1], 2, ("ineligible", "insufficient_bilabials", 1)),
    ]:
        assert video_evidence_status(selected, minimum_events=minimum) == expected
