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
    multiple_faces = {**good_face, "multiple_face_ratio": 0.75}
    missing_face = {**good_face, "normalized_minimum_closure": None}
    unstable = {**good_face, "valid_landmark_ratio": 0.5}

    assert event_evidence_status(good_face, 0.8) == ("eligible", "eligible")
    assert event_evidence_status(multiple_faces, 0.8) == (
        "ineligible",
        "multiple_faces",
    )
    assert event_evidence_status(missing_face, 0.8) == ("ineligible", "face_missing")
    assert event_evidence_status(unstable, 0.8) == (
        "ineligible",
        "landmarks_unstable",
    )


def test_video_eligibility_requires_enough_usable_events() -> None:
    events = [
        {"eligibility_status": "eligible", "exclusion_reason": "eligible"},
        {"eligibility_status": "ineligible", "exclusion_reason": "face_missing"},
    ]

    assert video_evidence_status(events, minimum_events=1) == (
        "eligible",
        "eligible",
        1,
    )
    assert video_evidence_status(events, minimum_events=2) == (
        "ineligible",
        "face_missing",
        1,
    )
    assert video_evidence_status(events[:1], minimum_events=2) == (
        "ineligible",
        "insufficient_bilabials",
        1,
    )
