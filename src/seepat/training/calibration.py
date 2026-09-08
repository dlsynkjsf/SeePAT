"""Thesis-aligned, Train-only VILD calibration.

Regression removes camera-scale variation from raw pixel VILD using the face
bounding-box diagonal. Isolation Forest then learns each subject's non-speech
residual baseline and scores active bilabial event minima. No label is used
while scoring an event.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.preprocessing.vild import event_vild_frames, load_vild_trace

CALIBRATION_VERSION = "vild-calibration-v2"


@dataclass(frozen=True)
class CalibrationOptions:
    """Evidence requirements and deterministic Isolation Forest parameters."""

    min_subject_reference_frames: int = 16
    isolation_trees: int = 100
    random_seed: int = 20260908

    def validate(self) -> None:
        if self.min_subject_reference_frames < 2:
            raise ValueError("min_subject_reference_frames must be at least 2")
        if self.isolation_trees < 1:
            raise ValueError("isolation_trees must be positive")


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normal_train_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normal = [
        row
        for row in rows
        if row.get("dataset_split", "").strip() == "train"
        and row.get("class_id", "").strip() == "0"
    ]
    if not normal:
        raise ValueError("Calibration requires genuine events with dataset_split='train'")
    return normal


def _trace_for_row(row: dict[str, str]) -> dict[str, object]:
    path_text = row.get("vild_trace_path", "").strip()
    if not path_text:
        raise ValueError(f"Event {row.get('event_id', '<unknown>')} has no VILD trace path")
    trace_path = Path(path_text)
    if not trace_path.is_file():
        raise FileNotFoundError(f"VILD trace does not exist: {trace_path}")
    expected_hash = row.get("vild_trace_sha256", "").strip() or None
    return load_vild_trace(trace_path, expected_sha256=expected_hash)


def _trace_reference_samples(
    normal_rows: list[dict[str, str]],
) -> list[tuple[str, float, float]]:
    """Return unique genuine-Train non-speech (subject, bbox, raw VILD) samples."""
    samples: list[tuple[str, float, float]] = []
    seen_traces: set[tuple[str, str]] = set()
    for row in normal_rows:
        trace_path = row.get("vild_trace_path", "").strip()
        trace_hash = row.get("vild_trace_sha256", "").strip()
        key = (trace_path, trace_hash)
        if key in seen_traces:
            continue
        seen_traces.add(key)
        trace = _trace_for_row(row)
        subject = str(trace.get("subject_id") or row.get("subject_id", "")).strip()
        if not subject:
            continue
        reference = trace.get("non_speech_reference")
        frames = trace.get("frames")
        if not isinstance(reference, dict) or not isinstance(frames, list):
            raise TypeError(f"VILD trace is missing reference frames: {trace_path}")
        windows = reference.get("windows")
        if not isinstance(windows, list):
            raise TypeError(f"VILD trace has invalid reference windows: {trace_path}")
        intervals = [
            (float(window["start_s"]), float(window["end_s"]))
            for window in windows
            if isinstance(window, dict)
        ]
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            timestamp = _float_or_none(frame.get("timestamp_s"))
            raw_vild = _float_or_none(frame.get("raw_vild_px"))
            bbox_size = _float_or_none(frame.get("face_bbox_size_px"))
            if (
                timestamp is not None
                and raw_vild is not None
                and bbox_size is not None
                and bbox_size > 0
                and any(start <= timestamp < end for start, end in intervals)
            ):
                samples.append((subject, bbox_size, raw_vild))
    if len(samples) < 2:
        raise ValueError(
            "Calibration requires at least two valid raw-VILD non-speech reference frames "
            "from genuine Train traces; regenerate VILD traces with vild-trace-v2"
        )
    return samples


def _linear_model(samples: list[tuple[str, float, float]]) -> dict[str, float | int | None]:
    x_values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    y_values = np.asarray([sample[2] for sample in samples], dtype=np.float64)
    if len(samples) < 2 or np.std(x_values) <= 1e-12:
        return {
            "samples": len(samples),
            "intercept": float(np.mean(y_values)),
            "slope": 0.0,
            "pearson_r": None,
        }
    slope, intercept = np.polyfit(x_values, y_values, 1)
    pearson_r = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if np.std(y_values) > 1e-12
        else None
    )
    return {
        "samples": len(samples),
        "intercept": float(intercept),
        "slope": float(slope),
        "pearson_r": pearson_r,
    }


def _residual(bbox_size: float, raw_vild: float, model: dict[str, float | int | None]) -> float:
    return raw_vild - (float(model["intercept"]) + float(model["slope"]) * bbox_size)


def _fit_models(
    samples: list[tuple[str, float, float]], options: CalibrationOptions
) -> tuple[dict[str, object], dict[str, object]]:
    global_regression = _linear_model(samples)
    grouped: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[0]].append(sample)
    subject_regressions = {
        subject: _linear_model(subject_samples)
        for subject, subject_samples in sorted(grouped.items())
        if len(subject_samples) >= options.min_subject_reference_frames
    }

    def fit_forest(values: list[float], seed: int) -> IsolationForest:
        return IsolationForest(
            n_estimators=options.isolation_trees,
            contamination="auto",
            random_state=seed,
        ).fit(np.asarray(values, dtype=np.float64).reshape(-1, 1))

    global_residuals = [_residual(bbox, raw, global_regression) for _, bbox, raw in samples]
    subject_forests: dict[str, IsolationForest] = {}
    for index, (subject, regression) in enumerate(sorted(subject_regressions.items()), start=1):
        subject_samples = grouped[subject]
        subject_forests[subject] = fit_forest(
            [_residual(bbox, raw, regression) for _, bbox, raw in subject_samples],
            options.random_seed + index,
        )
    artifact = {
        "regression": {
            "dependent_variable": "raw_vild_px",
            "independent_variable": "face_bbox_size_px",
            "normalization": "raw_vild_px - predicted_raw_vild_px",
            "global": global_regression,
            "subjects": subject_regressions,
        },
        "reference_frames": len(samples),
        "reference_subjects": sorted(grouped),
        "subject_model_min_reference_frames": options.min_subject_reference_frames,
        "sparse_subject_fallback": "global genuine-Train non-speech reference model",
    }
    return artifact, {
        "global": fit_forest(global_residuals, options.random_seed),
        "subjects": subject_forests,
    }


def _event_minimum(row: dict[str, str]) -> tuple[float, float] | None:
    trace = _trace_for_row(row)
    event_id = row.get("vild_trace_event_key", "").strip() or row.get("event_id", "")
    frames = event_vild_frames(trace, event_id)
    candidates = [
        (_float_or_none(frame.get("raw_vild_px")), _float_or_none(frame.get("face_bbox_size_px")))
        for frame in frames
        if isinstance(frame, dict)
    ]
    valid = [
        (raw, bbox)
        for raw, bbox in candidates
        if raw is not None and bbox is not None and bbox > 0
    ]
    if not valid:
        return None
    raw_vild, bbox_size = min(valid, key=lambda value: value[0])
    return bbox_size, raw_vild


def _phoneme_expectations(
    normal_rows: list[dict[str, str]], regression: dict[str, object]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    global_model = regression["global"]
    assert isinstance(global_model, dict)
    for row in normal_rows:
        minimum = _event_minimum(row)
        phoneme = row.get("phoneme", "").strip().lower()
        if minimum is None or not phoneme:
            continue
        bbox, raw = minimum
        grouped[phoneme].append(_residual(bbox, raw, global_model))
    return {
        phoneme: {
            "count": len(values),
            "mean_residual_px": float(np.mean(values)),
            "scale_residual_px": float(np.std(values)) if np.std(values) > 1e-12 else 1.0,
        }
        for phoneme, values in sorted(grouped.items())
    }


def _score_rows(
    rows: list[dict[str, str]], artifact: dict[str, object], forests: dict[str, object]
) -> list[dict[str, object]]:
    regression = artifact["regression"]
    expectations = artifact["phoneme_viseme_expectations"]
    assert isinstance(regression, dict) and isinstance(expectations, dict)
    global_model = regression["global"]
    subject_models = regression["subjects"]
    global_forest = forests["global"]
    subject_forests = forests["subjects"]
    assert isinstance(global_model, dict) and isinstance(subject_models, dict)
    assert isinstance(global_forest, IsolationForest) and isinstance(subject_forests, dict)
    output: list[dict[str, object]] = []
    for row in rows:
        scored: dict[str, object] = dict(row)
        minimum = _event_minimum(row)
        subject = row.get("subject_id", "").strip()
        model = subject_models.get(subject)
        forest = subject_forests.get(subject)
        scope = (
            "subject"
            if isinstance(model, dict) and isinstance(forest, IsolationForest)
            else "global_sparse_subject_fallback"
        )
        if not isinstance(model, dict) or not isinstance(forest, IsolationForest):
            model, forest = global_model, global_forest
        if minimum is None:
            scored.update(
                {
                    "vild_regression_prediction_px": "",
                    "vild_regression_residual_px": "",
                    "phoneme_viseme_residual_z": "",
                    "isolation_forest_anomaly_score": "",
                    "isolation_forest_scope": scope,
                }
            )
            output.append(scored)
            continue
        bbox, raw = minimum
        predicted = float(model["intercept"]) + float(model["slope"]) * bbox
        residual = raw - predicted
        phoneme_stats = expectations.get(row.get("phoneme", "").strip().lower())
        viseme_z = ""
        if isinstance(phoneme_stats, dict):
            viseme_z = round(
                (residual - float(phoneme_stats["mean_residual_px"]))
                / float(phoneme_stats["scale_residual_px"]),
                9,
            )
        scored.update(
            {
                "event_minimum_raw_vild_px": round(raw, 9),
                "event_minimum_face_bbox_size_px": round(bbox, 9),
                "vild_regression_prediction_px": round(predicted, 9),
                "vild_regression_residual_px": round(residual, 9),
                "phoneme_viseme_residual_z": viseme_z,
                # sklearn score_samples is the negative of the paper's path-length score.
                "isolation_forest_anomaly_score": round(
                    float(-forest.score_samples(np.asarray([[residual]]))[0]), 9
                ),
                "isolation_forest_scope": scope,
            }
        )
        output.append(scored)
    return output


def fit_and_score_calibration(
    train_manifest: Path,
    score_manifests: dict[str, Path],
    output_dir: Path,
    options: CalibrationOptions = CalibrationOptions(),
) -> dict[str, object]:
    """Fit on genuine Train non-speech frames, then score active event minima."""
    options.validate()
    if not score_manifests:
        raise ValueError("At least one scoring manifest is required")
    train_rows = read_csv_rows(train_manifest)
    normal_rows = _normal_train_rows(train_rows)
    reference_samples = _trace_reference_samples(normal_rows)
    fitted, forests = _fit_models(reference_samples, options)
    regression = fitted["regression"]
    assert isinstance(regression, dict)
    artifact: dict[str, object] = {
        "calibration_version": CALIBRATION_VERSION,
        "fit_population": "genuine AV++ Train non-speech VILD trace frames only",
        "active_speech_scoring": "minimum raw VILD frame in each bilabial event window",
        **fitted,
        "phoneme_viseme_expectations": _phoneme_expectations(normal_rows, regression),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "isolation_forests.joblib"
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    joblib.dump(forests, temporary)
    temporary.replace(model_path)
    artifact["train_manifest"] = train_manifest.as_posix()
    artifact["train_manifest_sha256"] = file_sha256(train_manifest)
    artifact["isolation_forest_models"] = model_path.as_posix()
    artifact["isolation_forest_models_sha256"] = file_sha256(model_path)
    artifact_path = output_dir / "calibration.json"
    atomic_write_json(artifact_path, artifact)

    scored_paths: dict[str, str] = {}
    score_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for name, manifest_path in sorted(score_manifests.items()):
        if not name.strip():
            raise ValueError("Scoring manifest names cannot be empty")
        path = output_dir / f"events_{name}_calibrated.csv"
        atomic_write_csv(path, _score_rows(read_csv_rows(manifest_path), artifact, forests))
        scored_paths[name] = path.as_posix()
        score_hashes[name] = file_sha256(path)
        input_hashes[name] = file_sha256(manifest_path)
    summary = {
        "calibration_version": CALIBRATION_VERSION,
        "calibration": artifact_path.as_posix(),
        "calibration_sha256": file_sha256(artifact_path),
        "train_manifest": train_manifest.as_posix(),
        "train_manifest_sha256": artifact["train_manifest_sha256"],
        "score_manifest_sha256": input_hashes,
        "scored_manifests": scored_paths,
        "scored_manifest_sha256": score_hashes,
        "reference_frames": artifact["reference_frames"],
        "subject_models": len(forests["subjects"]),
        "sparse_subject_fallback": artifact["sparse_subject_fallback"],
        "options": {
            "min_subject_reference_frames": options.min_subject_reference_frames,
            "isolation_trees": options.isolation_trees,
            "random_seed": options.random_seed,
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.calibration")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--score-manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-subject-reference-frames", type=int, default=16)
    parser.add_argument("--isolation-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    manifests: dict[str, Path] = {}
    for item in args.score_manifest:
        name, separator, text_path = item.partition("=")
        if not separator or not name or not text_path or name in manifests:
            raise ValueError("--score-manifest must be unique NAME=PATH values")
        manifests[name] = Path(text_path)
    summary = fit_and_score_calibration(
        args.train_manifest,
        manifests,
        args.output_dir,
        CalibrationOptions(
            args.min_subject_reference_frames,
            args.isolation_trees,
            args.seed,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
