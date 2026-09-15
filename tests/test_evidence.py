from __future__ import annotations

import pytest

from seepat.evidence import (
    EVIDENCE_LABELS,
    EVIDENCE_VERSION,
    FUSION_EVIDENCE_FIELDS,
    closure_offset_s,
    evidence_coverage,
    evidence_feature_values,
)


def test_evidence_version_and_labels_cover_every_field() -> None:
    assert EVIDENCE_VERSION == "fusion-evidence-v2"
    assert set(EVIDENCE_LABELS) == set(FUSION_EVIDENCE_FIELDS)
    assert len(FUSION_EVIDENCE_FIELDS) == len(set(FUSION_EVIDENCE_FIELDS))


def test_evidence_feature_values_parse_calibrated_row() -> None:
    row = {
        "vild_regression_residual_px": "1.5",
        "phoneme_viseme_residual_z": "-2.25",
        "isolation_forest_anomaly_score": "0.62",
        "normalized_minimum_closure": "0.0021",
        "closure_duration_s": "0.2",
        "phone_duration_s": "0.08",
        "closure_time_s": "9.72",
        "video_phone_start_s": "9.73",
    }
    values, mask = evidence_feature_values(row)

    assert mask == [True] * len(FUSION_EVIDENCE_FIELDS)
    assert values[FUSION_EVIDENCE_FIELDS.index("vild_regression_residual_px")] == 1.5
    offset_index = FUSION_EVIDENCE_FIELDS.index("closure_offset_s")
    assert values[offset_index] == closure_offset_s(row)
    assert values[offset_index] == pytest.approx(-0.01)


def test_evidence_feature_values_mask_missing_and_nonfinite_values() -> None:
    row = {
        "vild_regression_residual_px": "nan",
        "phoneme_viseme_residual_z": "",
        "isolation_forest_anomaly_score": "0.5",
        "normalized_minimum_closure": "0.002",
        "closure_duration_s": "0.1",
        "phone_duration_s": "inf",
        "closure_time_s": "9.7",
    }
    values, mask = evidence_feature_values(row)

    for index, field in enumerate(FUSION_EVIDENCE_FIELDS):
        if field in {"vild_regression_residual_px", "phoneme_viseme_residual_z", "phone_duration_s"}:
            assert mask[index] is False
            assert values[index] == 0.0
        elif field != "closure_offset_s":
            assert mask[index] is True


def test_closure_offset_requires_both_timing_fields() -> None:
    assert closure_offset_s({"closure_time_s": "1.0"}) is None
    assert closure_offset_s({"video_phone_start_s": "0.5"}) is None
    assert closure_offset_s({"closure_time_s": "1.0", "video_phone_start_s": "0.5"}) == 0.5


def test_evidence_coverage_reports_field_fractions() -> None:
    rows = [
        {
            "closure_duration_s": "0.2",
            "phone_duration_s": "0.1",
        },
        {
            "closure_duration_s": "0.2",
        },
    ]
    coverage = evidence_coverage(rows)

    assert coverage["closure_duration_s"] == 1.0
    assert coverage["phone_duration_s"] == 0.5
    assert coverage["vild_regression_residual_px"] == 0.0
    assert coverage["closure_offset_s"] == 0.0
    assert evidence_coverage([]) == {field: 0.0 for field in FUSION_EVIDENCE_FIELDS}
