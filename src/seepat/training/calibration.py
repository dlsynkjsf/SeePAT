"""Train-frozen scale regression and input-video non-speech Isolation Forests."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.live_progress import ProgressCallback
from seepat.preprocessing.augmentation import (
    augmented_trace_for_row,
    group_video_rows,
    recorded_hashes_are_current,
    verified_manifest_dependencies,
)
from seepat.preprocessing.vild import event_vild_frames

CALIBRATION_VERSION = "vild-calibration-v3"


@dataclass(frozen=True)
class CalibrationOptions:
    min_video_reference_frames: int = 16
    isolation_trees: int = 100
    random_seed: int = 20260908
    min_reference_valid_ratio: float = 0.8

    def validate(self) -> None:
        if self.min_video_reference_frames < 2:
            raise ValueError("min_video_reference_frames must be at least 2")
        if self.isolation_trees < 1:
            raise ValueError("isolation_trees must be positive")
        if not 0 <= self.min_reference_valid_ratio <= 1:
            raise ValueError("min_reference_valid_ratio must be between 0 and 1")


DEFAULT_CALIBRATION_OPTIONS = CalibrationOptions()


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_frame(frame: dict[str, object]) -> bool:
    raw = _float_or_none(frame.get("raw_vild_px"))
    bbox = _float_or_none(frame.get("face_bbox_size_px"))
    return (
        frame.get("face_count") == 1
        and _float_or_none(frame.get("normalized_vild")) is not None
        and raw is not None
        and raw >= 0
        and bbox is not None
        and bbox > 0
    )


def reference_samples(
    trace: dict[str, object],
    options: CalibrationOptions,
) -> list[tuple[float, float]]:
    """Select valid, unique frames in this video's eligible windows, without labels."""
    frames = trace["frames"]
    selected: set[int] = set()
    for window in trace["non_speech_reference"]["windows"]:
        indices = [
            i
            for i, frame in enumerate(frames)
            if float(window["start_s"]) <= float(frame["timestamp_s"]) < float(window["end_s"])
        ]
        if not indices or any(frames[i].get("face_count", 0) > 1 for i in indices):
            continue
        valid = [i for i in indices if _valid_frame(frames[i])]
        if len(valid) / len(indices) >= options.min_reference_valid_ratio:
            selected.update(valid)
    return [
        (float(frames[i]["face_bbox_size_px"]), float(frames[i]["raw_vild_px"]))
        for i in sorted(selected)
    ]


def _linear_model(samples: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if len(samples) < 2:
        raise ValueError(
            "Calibration needs at least two eligible Train raw-VILD reference frames; "
            "run visual trace augmentation, not full preprocessing"
        )
    x, y = np.asarray(samples, dtype=np.float64).T
    if np.std(x) <= 1e-12:
        slope, intercept, pearson = 0.0, float(np.mean(y)), None
    else:
        slope, intercept = np.polyfit(x, y, 1)
        pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else None
    return {
        "samples": len(samples),
        "intercept": float(intercept),
        "slope": float(slope),
        "pearson_r": pearson,
    }


def _residual(bbox: float, raw: float, model: dict[str, object]) -> float:
    return raw - (float(model["intercept"]) + float(model["slope"]) * bbox)


def _event_minimum(trace: dict[str, object], row: dict[str, str]) -> dict[str, object] | None:
    frames = event_vild_frames(trace, row.get("vild_trace_event_key") or row["event_id"])
    # Earliest frame breaks ties deterministically, matching the event-window timeline.
    return min(
        (frame for frame in frames if _valid_frame(frame)),
        key=lambda frame: (float(frame["raw_vild_px"]), float(frame["timestamp_s"])),
        default=None,
    )


def _fit_population(
    train_rows: list[dict[str, str]], options: CalibrationOptions,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if not train_rows or any(row.get("dataset_split") != "train" for row in train_rows):
        raise ValueError("Population fitting requires an exclusively Train manifest")
    samples: list[tuple[float, float]] = []
    genuine_minima: dict[str, list[tuple[float, float]]] = defaultdict(list)
    reference_videos = 0
    groups = group_video_rows(train_rows)
    for index, (video_id, rows) in enumerate(groups.items()):
        if progress is not None:
            progress("fit Train population", index, len(groups), video_id)
        # One decompression per video in this pass, regardless of event count.
        trace = augmented_trace_for_row(rows[0])
        references = reference_samples(trace, options)
        samples.extend(references)
        reference_videos += bool(references)
        for row in rows:
            # Labels are allowed ONLY for genuine-Train phoneme expectations.
            if row.get("class_id") != "0":
                continue
            minimum = _event_minimum(trace, row)
            if minimum is not None:
                genuine_minima[row["phoneme"].lower()].append(
                    (float(minimum["face_bbox_size_px"]), float(minimum["raw_vild_px"]))
                )
    if progress is not None:
        progress("fit Train population", len(groups), len(groups), "")
    regression = _linear_model(samples)
    expectations = {}
    for phoneme, minima in sorted(genuine_minima.items()):
        values = [_residual(bbox, raw, regression) for bbox, raw in minima]
        scale = float(np.std(values))
        # A single/constant observation cannot establish a standardized deviation.
        expectations[phoneme] = {
            "count": len(values),
            "mean_residual_px": float(np.mean(values)),
            "scale_residual_px": scale if scale > 1e-12 else None,
        }
    return {
        "regression": {
            "dependent_variable": "raw_vild_px",
            "independent_variable": "face_bbox_size_px",
            "normalization": "raw_vild_px - predicted_raw_vild_px",
            "global": regression,
        },
        "reference_frames": len(samples),
        "reference_videos": reference_videos,
        "phoneme_viseme_expectations": expectations,
    }


def _score_video(
    rows: list[dict[str, str]],
    artifact: dict[str, object],
    options: CalibrationOptions,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    trace = augmented_trace_for_row(rows[0])
    samples = reference_samples(trace, options)
    regression = artifact["regression"]["global"]
    forest = None
    reason = "insufficient_video_reference_frames"
    if len(samples) >= options.min_video_reference_frames:
        forest = IsolationForest(
            n_estimators=options.isolation_trees,
            contamination="auto",
            random_state=options.random_seed,
            n_jobs=1,
        ).fit(
            np.asarray([_residual(bbox, raw, regression) for bbox, raw in samples]).reshape(-1, 1)
        )
        reason = ""
    output = []
    for row in rows:
        minimum = _event_minimum(trace, row)
        scored = dict(row)
        scored.update(
            {
                name: ""
                for name in (
                    "event_minimum_raw_vild_px",
                    "event_minimum_face_bbox_size_px",
                    "event_minimum_frame_index",
                    "event_minimum_timestamp_s",
                    "vild_regression_prediction_px",
                    "vild_regression_residual_px",
                    "phoneme_viseme_residual_z",
                    "isolation_forest_anomaly_score",
                )
            }
        )
        scored.update(
            {
                "isolation_forest_scope": "input_video" if forest is not None else "unavailable",
                "isolation_forest_reference_frames": len(samples),
                "isolation_forest_available": False,
                "isolation_forest_unavailable_reason": reason or "no_valid_event_measurements",
                "calibration_version": CALIBRATION_VERSION,
            }
        )
        if minimum is not None:
            bbox, raw = float(minimum["face_bbox_size_px"]), float(minimum["raw_vild_px"])
            residual = _residual(bbox, raw, regression)
            stats = artifact["phoneme_viseme_expectations"].get(row.get("phoneme", "").lower())
            scored.update(
                {
                    "event_minimum_raw_vild_px": raw,
                    "event_minimum_face_bbox_size_px": bbox,
                    "event_minimum_frame_index": minimum["frame_index"],
                    "event_minimum_timestamp_s": minimum["timestamp_s"],
                    "vild_regression_prediction_px": round(raw - residual, 9),
                    "vild_regression_residual_px": round(residual, 9),
                }
            )
            if stats and stats["scale_residual_px"] is not None:
                scored["phoneme_viseme_residual_z"] = round(
                    (residual - stats["mean_residual_px"]) / stats["scale_residual_px"], 9
                )
            if forest is not None:
                scored.update(
                    {
                        # sklearn returns the negative of the paper's path-length anomaly score.
                        "isolation_forest_anomaly_score": round(
                            float(-forest.score_samples([[residual]])[0]), 9
                        ),
                        "isolation_forest_available": True,
                        "isolation_forest_unavailable_reason": "",
                    }
                )
        output.append(scored)
    return output, {
        "video_id": rows[0]["video_id"],
        "reference_frames": len(samples),
        "forest_fitted": forest is not None,
        "unavailable_reason": reason,
    }


def _validate_manifests(manifests: dict[str, Path]) -> None:
    if not manifests or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in manifests):
        raise ValueError("Scoring manifests need safe, nonempty names")


def _score_manifests(
    artifact: dict[str, object],
    manifests: dict[str, Path],
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    options = CalibrationOptions(**artifact["options"])
    options.validate()
    scored_paths, hashes, diagnostics = {}, {}, {}
    artifact_key = stable_id(json.dumps(artifact, sort_keys=True), 64)
    for name, manifest in sorted(manifests.items()):
        rows = read_csv_rows(manifest)
        by_event, videos = {}, []
        grouped = group_video_rows(rows)
        if not grouped:
            raise ValueError(f"Scoring manifest is empty: {manifest}")
        for index, (video_id, group) in enumerate(grouped.items(), 1):
            if progress is not None:
                progress(f"calibrate {name} videos", index - 1, len(grouped), video_id)
            key = stable_id(
                json.dumps(
                    {"artifact": artifact_key, "rows": group, "sklearn": sklearn.__version__},
                    sort_keys=True,
                ),
                64,
            )
            path = output_dir / "video_scores" / name / f"{stable_id(video_id)}.json"
            record_path = path.with_suffix(".record.json")
            cached = None
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record["key"] == key and record["sha256"] == file_sha256(path):
                    cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, KeyError):
                pass
            if cached is None:
                scored, diagnostic = _score_video(group, artifact, options)
                cached = {"rows": scored, "diagnostic": diagnostic}
                atomic_write_json(path, cached)
                atomic_write_json(record_path, {"key": key, "sha256": file_sha256(path)})
            by_event.update({row["event_id"]: row for row in cached["rows"]})
            videos.append(cached["diagnostic"])
            atomic_write_json(
                output_dir / "progress.json",
                {
                    "status": "running",
                    "manifest": name,
                    "videos_finished": index,
                    "videos_requested": len(grouped),
                },
            )
        if progress is not None:
            progress(f"calibrate {name} videos", len(grouped), len(grouped), "")
        path = output_dir / f"events_{name}_calibrated.csv"
        atomic_write_csv(path, [by_event[row["event_id"]] for row in rows])
        scored_paths[name], hashes[name] = path.as_posix(), file_sha256(path)
        diagnostics[name] = videos
    audit_path = output_dir / "video_calibration_audit.json"
    atomic_write_json(audit_path, diagnostics)
    atomic_write_json(output_dir / "progress.json", {"status": "complete"})
    return {
        "scored_manifests": scored_paths,
        "scored_manifest_sha256": hashes,
        "audit": audit_path.as_posix(),
        "audit_sha256": file_sha256(audit_path),
    }


def fit_and_score_calibration(
    train_manifest: Path,
    score_manifests: dict[str, Path],
    output_dir: Path,
    options: CalibrationOptions = DEFAULT_CALIBRATION_OPTIONS,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    options.validate()
    _validate_manifests(score_manifests)
    dependencies = verified_manifest_dependencies([train_manifest, *score_manifests.values()])
    train_rows = read_csv_rows(train_manifest)
    population_key = {
        "calibration_version": CALIBRATION_VERSION,
        "options": asdict(options),
        "train_manifest": train_manifest.as_posix(),
        "train_manifest_sha256": file_sha256(train_manifest),
        "train_dependencies": verified_manifest_dependencies([train_manifest]),
        "sklearn_version": sklearn.__version__,
    }
    path = output_dir / "calibration.json"
    artifact = None
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        record = json.loads(path.with_suffix(".record.json").read_text(encoding="utf-8"))
        if record["sha256"] == file_sha256(path) and all(
            candidate.get(k) == v for k, v in population_key.items()
        ):
            artifact = candidate
    except (OSError, ValueError, TypeError, KeyError):
        pass
    if artifact is None:
        artifact = {
            **population_key,
            **_fit_population(train_rows, options, progress),
            "fit_population": "label-independent eligible Train non-speech frames",
            "phoneme_expectation_population": "genuine Train events only",
            "isolation_forest_population": "each input video's own non-speech frames",
            "sparse_video_fallback": None,
            "active_speech_scoring": "earliest minimum raw-VILD frame in each event window",
        }
        atomic_write_json(path, artifact)
        atomic_write_json(path.with_suffix(".record.json"), {"sha256": file_sha256(path)})
    summary = {
        **_score_manifests(artifact, score_manifests, output_dir, progress),
        "calibration_version": CALIBRATION_VERSION,
        "calibration": path.as_posix(),
        "calibration_sha256": file_sha256(path),
        "input_hashes": dependencies,
        "train_manifest_sha256": file_sha256(train_manifest),
        "options": asdict(options),
        "reference_frames": artifact["reference_frames"],
        "sklearn_version": sklearn.__version__,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def score_manifests_with_calibration(
    calibration_path: Path,
    score_manifests: dict[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Freeze Train parameters; fit only each new input's own resting forest."""
    _validate_manifests(score_manifests)
    artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
    if artifact.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("Unsupported calibration artifact version")
    record = json.loads(calibration_path.with_suffix(".record.json").read_text(encoding="utf-8"))
    if record["sha256"] != file_sha256(calibration_path):
        raise ValueError("Calibration artifact hash mismatch")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the frozen calibration's scikit-learn version")
    dependencies = verified_manifest_dependencies(list(score_manifests.values()))
    summary = {
        **_score_manifests(artifact, score_manifests, output_dir),
        "mode": "score_only",
        "strategy": "frozen Train parameters; independent input-video forest fitting",
        "calibration_version": CALIBRATION_VERSION,
        "calibration_sha256": file_sha256(calibration_path),
        "input_hashes": dependencies,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def calibration_outputs_are_current(
    train_manifest: Path,
    score_manifests: dict[str, Path],
    output_dir: Path,
    options: CalibrationOptions,
) -> bool:
    try:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        record = json.loads((output_dir / "calibration.record.json").read_text(encoding="utf-8"))
        return (
            summary["calibration_version"] == CALIBRATION_VERSION
            and summary["sklearn_version"] == sklearn.__version__
            and summary["options"] == asdict(options)
            and summary["input_hashes"]
            == verified_manifest_dependencies([train_manifest, *score_manifests.values()])
            and set(summary["scored_manifests"]) == set(score_manifests)
            and summary["calibration_sha256"] == file_sha256(output_dir / "calibration.json")
            and record["sha256"] == summary["calibration_sha256"]
            and summary["audit_sha256"] == file_sha256(Path(summary["audit"]))
            and recorded_hashes_are_current(
                {
                    summary["scored_manifests"][name]: digest
                    for name, digest in summary["scored_manifest_sha256"].items()
                }
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.calibration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-manifest", type=Path)
    mode.add_argument("--calibration", type=Path)
    parser.add_argument("--score-manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-video-reference-frames", type=int, default=16)
    parser.add_argument("--isolation-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    manifests = {}
    for item in args.score_manifest:
        name, separator, value = item.partition("=")
        if not separator or not value or name in manifests:
            raise ValueError("--score-manifest must be unique NAME=PATH values")
        manifests[name] = Path(value)
    if args.calibration:
        summary = score_manifests_with_calibration(args.calibration, manifests, args.output_dir)
    else:
        summary = fit_and_score_calibration(
            args.train_manifest,
            manifests,
            args.output_dir,
            CalibrationOptions(args.min_video_reference_frames, args.isolation_trees, args.seed),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
