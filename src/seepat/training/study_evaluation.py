"""Frozen-threshold fold evaluation and paired, descriptive ablation reports."""
from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.training.dataset import MouthEventDataset, video_class_id
from seepat.training.metrics import aggregate_video_probabilities, binary_classification_metrics

EVALUATION_VERSION = "ablation-evaluation-v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_metrics(records: list[dict], threshold: float) -> dict:
    _, labels, probabilities = aggregate_video_probabilities(
        [str(r["video_id"]) for r in records], [int(r["video_label"]) for r in records],
        [float(r["manipulated_probability"]) for r in records],
    )
    if set(labels) != {0, 1}:
        raise ValueError("Fold video metrics require both classes; do not silently omit a fold")
    return binary_classification_metrics(labels, probabilities, threshold)


def evaluate_model(job, fold: dict, training, progress=None) -> dict:
    import torch

    from seepat.evidence import calibrated_manifest_contract
    from seepat.training.train import FUSION_MODEL, _device
    from seepat.verdict import (
        _score_events,
        evaluation_inputs,
        load_evaluation_model,
        select_threshold,
    )
    from seepat.workflow import model_training_outputs_are_current

    _verify_fold(fold)
    if any(file_sha256(path) != fold["hashes"][fold["manifests"][name]]
           for name, path in (("train", training.train_manifest), ("val", training.validation_manifest))):
        raise ValueError("Evaluation job must use this fold's exact Train and Validation manifests")
    if not model_training_outputs_are_current(training):
        raise ValueError("Complete the matching fold experiment before evaluation")
    run = _read(training.output_dir / "run.json")
    checkpoint = Path(run["best_checkpoint"])
    # This preserves the production Validation-only threshold gate.
    contract, provenance = evaluation_inputs(training.validation_manifest, checkpoint, "val")
    if contract.get("train_manifest_sha256") != file_sha256(training.train_manifest):
        raise ValueError("Selected checkpoint was not fitted on this fold's training subset")
    heldout = Path(fold["manifests"]["heldout"])
    dataset = MouthEventDataset(
        heldout, job.project_root, dataset_split="train", sequence_length=contract["sequence_length"],
        image_size=contract["image_size"], require_calibration=training.model == FUSION_MODEL,
    )
    if training.model == FUSION_MODEL and calibrated_manifest_contract(heldout) != contract["fusion_inputs"]:
        raise ValueError("Held-out fold differs from the checkpoint's frozen calibration")
    directory = training.output_dir / "fold_evaluation"
    key = {"version": EVALUATION_VERSION, "study_sha256": file_sha256(job.output_dir / "study.json"),
           "heldout_sha256": file_sha256(heldout), **provenance}
    output = directory / "metrics.json"
    output_record = directory / "metrics.record.json"
    try:
        previous = _read(output)
        if _read(output_record)["sha256"] == file_sha256(output) and previous["provenance"] == key and all(
            file_sha256(Path(path)) == digest for path, digest in previous["artifacts"].items()
        ):
            return previous
    except (OSError, ValueError, KeyError, TypeError):
        pass
    threshold_path = directory / "threshold.json"
    threshold_record = directory / "threshold.record.json"
    threshold = None
    try:
        cached = _read(threshold_path)
        if (_read(threshold_record)["sha256"] == file_sha256(threshold_path)
                and cached["provenance"] == provenance and cached["predictions"]["sha256"] == file_sha256(
                    Path(cached["predictions"]["path"]))):
            threshold = cached
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if threshold is None:
        threshold = select_threshold(
            training.validation_manifest, checkpoint, threshold_path,
            project_root=job.project_root, device_name=job.device, progress=progress,
        )
        atomic_write_json(threshold_record, {"sha256": file_sha256(threshold_path)})
    device = _device(job.device)
    model, _ = load_evaluation_model(checkpoint, device)
    records = _score_events(model, dataset, device, 1, progress)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictions = directory / "heldout_events.csv"
    atomic_write_csv(predictions, records)
    result = {"fold": fold["fold"], "model": training.model, "provenance": key,
              "scope": "held-out AV++ Train groups; not external-test performance",
              "event_metrics": binary_classification_metrics(
                  [int(r["label"]) for r in records],
                  [float(r["manipulated_probability"]) for r in records], float(threshold["threshold"])),
              "video_metrics": _video_metrics(records, float(threshold["threshold"])),
              "artifacts": {p.as_posix(): file_sha256(p) for p in (
                  predictions, threshold_path, threshold_record, Path(threshold["predictions"]["path"]))}}
    atomic_write_json(output, result)
    atomic_write_json(output_record, {"sha256": file_sha256(output)})
    return result


def _verify_fold(fold: dict) -> None:
    """Validate the held-out boundary even for direct Python evaluation callers."""
    from seepat.training.study import _disjoint, _validate_rows

    for name, split in (("train", "train"), ("val", "val"), ("heldout", "train")):
        path = Path(fold["manifests"][name])
        if file_sha256(path) != fold["hashes"][path.as_posix()]:
            raise ValueError("Fold evaluation manifest hash mismatch")
        rows = read_csv_rows(path)
        _validate_rows(rows, split)
    fitting = read_csv_rows(Path(fold["manifests"]["train"]))
    heldout = read_csv_rows(Path(fold["manifests"]["heldout"]))
    validation = read_csv_rows(Path(fold["manifests"]["val"]))
    _disjoint(fitting, heldout)
    _disjoint(fitting + heldout, validation)


def matched_geometry(rows: list[dict]) -> tuple[list[dict], dict]:
    """Availability alone selects the paired cohort, never labels or predictions."""
    paired = []
    missing = {"static": 0, "dynamic": 0}
    for row in rows:
        try:
            static = float(row["normalized_minimum_closure"])
            static_ok = math.isfinite(static) and static >= 0
        except (KeyError, TypeError, ValueError):
            static, static_ok = 0., False
        try:
            dynamic = float(row["isolation_forest_anomaly_score"])
            dynamic_ok = (math.isfinite(dynamic) and 0 <= dynamic <= 1
                          and str(row["isolation_forest_available"]).lower() == "true")
        except (KeyError, TypeError, ValueError):
            dynamic, dynamic_ok = 0., False
        missing["static"] += not static_ok
        missing["dynamic"] += not dynamic_ok
        if static_ok and dynamic_ok:
            # Fixed monotone map permits the existing threshold/metric helpers.
            # These are geometry/anomaly scores, NOT calibrated probabilities.
            paired.append({"event_id": row["event_id"], "video_id": row["video_id"],
                           "label": int(row["class_id"]),
                           "video_label": video_class_id(row["manipulation_modality"]),
                           "static": static / (1 + static), "dynamic": dynamic})
    cohort = {"input_events": len(rows), "paired_events": len(paired),
              "input_videos": len({r["video_id"] for r in rows}),
              "paired_videos": len({r["video_id"] for r in paired}), "missing_events": missing,
              "input_event_classes": dict(Counter(str(r["class_id"]) for r in rows)),
              "paired_event_classes": dict(Counter(str(r["label"]) for r in paired)),
              "input_video_classes": dict(Counter(str(v) for v in {
                  r["video_id"]: video_class_id(r["manipulation_modality"]) for r in rows}.values())),
              "paired_video_classes": dict(Counter(str(v) for v in {
                  r["video_id"]: r["video_label"] for r in paired}.values()))}
    return paired, cohort


def calibration_comparison(validation: list[dict], heldout: list[dict]) -> list[dict]:
    from seepat.verdict import choose_threshold

    paired_val, val_coverage = matched_geometry(validation)
    paired_heldout, heldout_coverage = matched_geometry(heldout)
    results = []
    for method in ("static", "dynamic"):
        val_records = [{**r, "manipulated_probability": r[method]} for r in paired_val]
        held_records = [{**r, "manipulated_probability": r[method]} for r in paired_heldout]
        _, labels, scores = aggregate_video_probabilities(
            [r["video_id"] for r in val_records], [r["video_label"] for r in val_records],
            [r["manipulated_probability"] for r in val_records],
        )
        selection = choose_threshold(labels, scores)
        threshold = float(selection["threshold"])
        results.append({"model": f"geometry_{method}", "selection": selection,
                        "static_threshold_normalized_vild": (
                            threshold / (1 - threshold) if method == "static" and threshold < 1 else None),
                        "video_metrics": _video_metrics(held_records, threshold),
                        "event_metrics": binary_classification_metrics(
                            [r["label"] for r in held_records], [r[method] for r in paired_heldout], threshold),
                        "coverage": {"validation": val_coverage, "heldout": heldout_coverage},
                        "scope": "Matched available events; geometry control, not a fusion intervention"})
    return results


def evaluate_calibration(job, fold: dict) -> list[dict]:
    from seepat.evidence import calibrated_manifest_contract

    _verify_fold(fold)
    paths = {key: Path(fold["manifests"][key]) for key in ("val", "heldout")}
    if calibrated_manifest_contract(paths["val"]) != calibrated_manifest_contract(paths["heldout"]):
        raise ValueError("Calibration comparison must share one fold-fitted population")
    results = calibration_comparison(read_csv_rows(paths["val"]), read_csv_rows(paths["heldout"]))
    provenance = {"version": EVALUATION_VERSION, "study_sha256": file_sha256(job.output_dir / "study.json"),
                  "input_hashes": {p.as_posix(): file_sha256(p) for p in paths.values()},
                  "static_score": "normalized_minimum_closure / (1 + normalized_minimum_closure)",
                  "dynamic_score": "per-input-video Isolation Forest anomaly score",
                  "direction": "higher scores predict manipulation", "aggregation": "maximum event score",
                  "threshold_fit": "official Validation paired cohort; balanced accuracy; lowest threshold ties"}
    for result in results:
        result.update(fold=fold["fold"], provenance=provenance)
    atomic_write_json(job.output_dir / f"fold_{fold['fold']}" / "calibration_comparison.json", {"results": results})
    return results


def paired_statistics(first: list[float], second: list[float]) -> dict:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Paired statistics need at least two matching folds")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Paired fold measurements must be finite")
    nonzero = differences[differences != 0]
    if len(nonzero) > 15:
        raise ValueError("Exact signed-rank enumeration is bounded to 15 nonzero pairs")
    # Exact sign enumeration also handles tied absolute differences and zero removal.
    ranks = stats.rankdata(np.abs(nonzero))
    observed = abs(float(np.sum(np.sign(nonzero) * ranks)))
    permutations = [abs(float(np.dot(signs, ranks))) for signs in product((-1, 1), repeat=len(nonzero))]
    p_value = sum(value >= observed for value in permutations) / len(permutations)
    varying = bool(np.ptp(differences) > 0)
    normality = stats.shapiro(differences) if len(first) >= 3 and varying else None
    t_result = stats.ttest_rel(first, second) if varying else None
    return {"pairs": len(first), "differences_first_minus_second": differences.tolist(),
            "nonzero_pairs": len(nonzero), "mean_difference": float(np.mean(differences)),
            "shapiro_statistic": float(normality.statistic) if normality is not None else None,
            "shapiro_p": float(normality.pvalue) if normality is not None else None,
            "paired_t_statistic": float(t_result.statistic) if t_result is not None else None,
            "paired_t_p": float(t_result.pvalue) if t_result is not None else None,
            "signed_rank_statistic": float((np.sum(ranks) - observed) / 2),
            "exact_two_sided_signed_rank_p": p_value,
            "minimum_two_sided_p": min(1., 2 / (2 ** len(nonzero))),
            "interpretation": "Descriptive diagnostics only; overlapping training folds are dependent. "
            "At five nonzero pairs the minimum two-sided signed-rank p is 0.0625. "
            "No automatic significance claim or post-hoc test switching."}


def write_comparison(prefix: Path, results: list[dict], **kwargs) -> list[Path]:
    from seepat.training.study_reporting import write_report

    return write_report(prefix, results, **kwargs)
