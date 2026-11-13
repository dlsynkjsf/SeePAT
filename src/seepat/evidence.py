"""Ordered fusion measurements, availability masks, and report labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

from seepat.artifacts import file_sha256, read_csv_rows

EVIDENCE_VERSION = "fusion-evidence-v2"
CALIBRATED_INPUT_VERSION = "vild-calibration-v3"
CLOSURE_OFFSET_DEFINITION = (
    "closure_time_s - video_phone_start_s; seconds; positive means visual minimum "
    "after aligned phoneme start; descriptive timing, not an anomaly probability"
)

# Keep this order: trained checkpoints expect each measurement in the same position.
FUSION_EVIDENCE_FIELDS = (
    "vild_regression_residual_px",
    "phoneme_viseme_residual_z",
    "isolation_forest_anomaly_score",
    "normalized_minimum_closure",
    "closure_duration_s",
    "phone_duration_s",
    "closure_offset_s",
)

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


def finite_float(value: object) -> float | None:
    """Read an optional measurement; missing, invalid, and infinite values stay missing."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def closure_offset_s(row: dict[str, str]) -> float | None:
    """Seconds from the aligned video phoneme start to the visual closure minimum.

    Both times use the video clock. Positive means the closure comes later;
    this describes timing, not the probability of manipulation.
    """
    closure_time = finite_float(row.get("closure_time_s", ""))
    phone_start = finite_float(row.get("video_phone_start_s", ""))
    if closure_time is None or phone_start is None:
        return None
    return closure_time - phone_start


def calibrated_manifest_contract(
    path: Path,
    rows: list[dict[str, str]] | None = None,
    *, allow_empty: bool = False,
) -> dict[str, object]:
    """Verify fusion's calibrated CSV and bind its frozen population artifact.

    Full trace/source integrity remains the preceding workflow stages' job.
    Fusion consumes the exported numbers and their explicit missingness.
    """
    rows = read_csv_rows(path) if rows is None else rows
    required = (set(FUSION_EVIDENCE_FIELDS) - set(DERIVED_EVIDENCE_FIELDS)) | {
        "closure_time_s",
        "video_phone_start_s",
        "calibration_version",
        "isolation_forest_available",
        "isolation_forest_unavailable_reason",
        "isolation_forest_scope",
    }
    if (not rows and not allow_empty) or any(required - row.keys() for row in rows):
        raise ValueError("Fusion requires a calibrated manifest with all evidence columns")
    for row in rows:
        if row["calibration_version"] != CALIBRATED_INPUT_VERSION:
            raise ValueError("Unsupported fusion calibration version")
        available = row["isolation_forest_available"].lower()
        score = finite_float(row["isolation_forest_anomaly_score"])
        if available not in {"true", "false"} or (available == "true") != (score is not None):
            raise ValueError("Calibrated anomaly score and availability disagree")
        if score is not None:
            if not 0 <= score <= 1 or row["isolation_forest_scope"] != "input_video":
                raise ValueError("Fusion requires per-input-video anomaly scores in [0, 1]")
            if row["isolation_forest_unavailable_reason"]:
                raise ValueError("Available anomaly score has an unavailable reason")
        elif (
            row["isolation_forest_anomaly_score"] or not row["isolation_forest_unavailable_reason"]
        ):
            raise ValueError("Missing anomaly scores must be blank and have a reason")
    summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    matches = [
        name
        for name, recorded in summary["scored_manifests"].items()
        if Path(recorded).resolve() == path.resolve()
    ]
    if len(matches) != 1 or summary["scored_manifest_sha256"][matches[0]] != file_sha256(path):
        raise ValueError("Calibrated manifest hash does not match its summary")
    population_path = Path(summary["calibration"])
    population = json.loads(population_path.read_text(encoding="utf-8"))
    if population["calibration_version"] != CALIBRATED_INPUT_VERSION or summary[
        "calibration_sha256"
    ] != file_sha256(population_path):
        raise ValueError("Frozen calibration artifact hash/version mismatch")
    return {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_fields": list(FUSION_EVIDENCE_FIELDS),
        "mask_encoding": "one availability flag per feature",
        "calibration_version": CALIBRATED_INPUT_VERSION,
        "calibration_sha256": summary["calibration_sha256"],
        "closure_offset_definition": CLOSURE_OFFSET_DEFINITION,
    }


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
            number = finite_float(row.get(field, ""))
        if number is None:
            # The mask tells the model this zero is a placeholder, not a measurement.
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
