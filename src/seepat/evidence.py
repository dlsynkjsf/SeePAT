"""Frozen contract for the biological-alignment evidence of SeePAT.

The Audio-Visual Temporal Fusion block of Figure 4.1 combines the Video
Swin-Base temporal features with the calibrated biological alignment:
VILD normalization (correlation and regression), dynamic calibration
(Isolation Forest) and phoneme-viseme mapping. This module defines the
ordered evidence vector the fusion model consumes, the missing-data
handling, and the human-readable labels used by the XAI forensic trace.

Field order is part of the contract: changing it changes the model input
width and invalidates trained fusion checkpoints.
"""

from __future__ import annotations

import math

EVIDENCE_VERSION = "fusion-evidence-v1"

#: Columns of the calibrated manifest consumed by the fusion model, in order.
#: The first three are written by the numerical calibration stage; the rest are
#: base bilabial-event measurements. ``closure_offset_s`` is derived below.
FUSION_EVIDENCE_FIELDS = (
    "vild_regression_residual_px",
    "phoneme_viseme_residual_z",
    "isolation_forest_anomaly_score",
    "normalized_minimum_closure",
    "closure_duration_s",
    "phone_duration_s",
    "closure_offset_s",
)

#: Fields computed from other columns instead of being read directly.
DERIVED_EVIDENCE_FIELDS = ("closure_offset_s",)

EVIDENCE_LABELS = {
    "vild_regression_residual_px": "VILD regression residual (px)",
    "phoneme_viseme_residual_z": "phoneme-viseme residual (z)",
    "isolation_forest_anomaly_score": "Isolation Forest anomaly score",
    "normalized_minimum_closure": "normalized minimum closure",
    "closure_duration_s": "visual closure duration (s)",
    "phone_duration_s": "bilabial phoneme duration (s)",
    "closure_offset_s": "closure offset from phoneme start (s)",
}


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def closure_offset_s(row: dict[str, str]) -> float | None:
    """Time from the MFA bilabial anchor to the visual closure minimum.

    ``closure_time_s`` is on the video timeline; ``video_phone_start_s`` is the
    MFA phoneme start shifted by the measured audio/video offset. Both are
    required for the derived timing evidence to be considered available.
    """
    closure_time = _float_or_none(row.get("closure_time_s", ""))
    phone_start = _float_or_none(row.get("video_phone_start_s", ""))
    if closure_time is None or phone_start is None:
        return None
    return closure_time - phone_start


def evidence_feature_values(row: dict[str, str]) -> tuple[list[float], list[bool]]:
    """Return the ordered fusion evidence values and their availability mask.

    Missing or non-finite values are zeroed and masked exactly like the base
    numerical features handled by :func:`seepat.training.dataset.numeric_feature_values`.
    """
    values: list[float] = []
    mask: list[bool] = []
    for field in FUSION_EVIDENCE_FIELDS:
        if field == "closure_offset_s":
            number = closure_offset_s(row)
        else:
            number = _float_or_none(row.get(field, ""))
        if number is None:
            values.append(0.0)
            mask.append(False)
        else:
            values.append(number)
            mask.append(True)
    return values, mask


def evidence_coverage(rows: list[dict[str, str]]) -> dict[str, float]:
    """Fraction of rows with each evidence field available, for run audits."""
    totals = [0] * len(FUSION_EVIDENCE_FIELDS)
    for row in rows:
        for index, available in enumerate(evidence_feature_values(row)[1]):
            totals[index] += int(available)
    denominator = len(rows) if rows else 1
    return {
        field: round(totals[index] / denominator, 6)
        for index, field in enumerate(FUSION_EVIDENCE_FIELDS)
    }
