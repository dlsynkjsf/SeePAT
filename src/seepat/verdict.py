"""Binary event classification, maximum event aggregation and deepfake verdict.

Implements the Figure 4.1 classification chain: every bilabial mouth event is
scored by the trained model (event-level manipulation probability), video
probabilities use maximum event aggregation, and a *frozen* threshold turns the
probability into the deepfake verdict.

Reproducibility rules:

* Production decisions require a provenance-checked Validation threshold artifact.
  An explicit threshold is allowed only in labelled engineering-preflight mode.
* This development entry point currently evaluates AV++ Validation only.
  External evaluation remains locked until the final experiment is frozen.
* Calibration artifacts are read-only upstream inputs; this module never fits
  or refits anything.
* Videos that produced no scorable bilabial events are reported as
  ``not_evaluated`` instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.evidence import EVIDENCE_VERSION, FUSION_EVIDENCE_FIELDS
from seepat.live_progress import ProgressCallback
from seepat.training.dataset import MouthEventDataset
from seepat.training.metrics import (
    aggregate_video_probabilities,
    binary_classification_metrics,
    choose_balanced_accuracy_threshold,
)
from seepat.training.train import FUSION_MODEL, build_event_classifier, training_version_for_model

VERDICT_VERSION = "verdict-v2"
THRESHOLD_VERSION = "threshold-v2"
VALIDATION_SOURCE = "AV-Deepfake1M++"
AGGREGATION = "maximum event probability"
SELECTION_METRIC = "balanced_accuracy"
THRESHOLD_TIE_BREAK = "lowest threshold among equal balanced accuracies"
VERDICT_MANIPULATED = "manipulated"
VERDICT_AUTHENTIC = "authentic"
VERDICT_NOT_EVALUATED = "not_evaluated"


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def choose_threshold(
    labels: list[int],
    probabilities: list[float],
) -> dict[str, object]:
    """Backward-compatible public threshold helper."""
    return dict(choose_balanced_accuracy_threshold(labels, probabilities))


def load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_type") != "seepat_evaluation_model":
        raise ValueError("The checkpoint is not a SeePAT evaluation model")
    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, dict) or not contract.get("model_name"):
        raise ValueError("The checkpoint is missing its model contract")
    if contract.get("training_version") != training_version_for_model(str(contract["model_name"])):
        raise ValueError("Unsupported checkpoint training version")
    model = build_event_classifier(
        model_name=str(contract["model_name"]),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, contract


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, list[float | None]]:
    videos = batch["video"].to(device, non_blocking=True)
    kwargs: dict[str, torch.Tensor] = {}
    if bool(getattr(model, "uses_frame_mask", False)):
        kwargs["frame_mask"] = batch["frame_mask"].to(
            device=device,
            dtype=torch.bool,
        )
    if bool(getattr(model, "uses_evidence_features", False)):
        kwargs["features"] = batch["evidence_features"].to(device=device)
        kwargs["feature_mask"] = batch["evidence_feature_mask"].to(
            device=device,
            dtype=torch.bool,
        )
    logits = model(videos, **kwargs) if kwargs else model(videos)
    if logits.shape != (videos.shape[0], 2) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite binary event logits")
    return logits, [None] * videos.shape[0]


def _score_events(
    model: torch.nn.Module,
    dataset: MouthEventDataset,
    device: torch.device,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> list[dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows_by_event = {row["event_id"]: row for row in dataset.rows}
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if progress is not None:
                progress("score verdict events", batch_index, len(loader), str(batch["video_id"][0]))
            logits, sync_scores = _forward_batch(model, batch, device)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
            event_ids = list(batch["event_id"])
            for index, event_id in enumerate(event_ids):
                row = rows_by_event.get(str(event_id), {})
                evidence_mask = batch["evidence_feature_mask"][index]
                record = {
                    "event_id": event_id,
                    "video_id": batch["video_id"][index],
                    "subject_id": row.get("subject_id", ""),
                    "phoneme": row.get("phoneme", ""),
                    "label": int(batch["label"][index].item()),
                    "video_label": int(batch["video_label"][index].item()),
                    "manipulated_probability": float(probabilities[index]),
                    "sync_gap_score": (
                        round(float(sync_scores[index]), 9)
                        if sync_scores[index] is not None
                        else ""
                    ),
                    "evidence_available": int(evidence_mask.sum().item()),
                    "evidence_total": int(evidence_mask.numel()),
                }
                for field_index, field in enumerate(FUSION_EVIDENCE_FIELDS):
                    available = bool(evidence_mask[field_index])
                    record[f"{field}_available"] = available
                    record[field] = (
                        float(batch["evidence_features"][index, field_index]) if available else ""
                    )
                records.append(record)
    if progress is not None:
        progress("score verdict events", len(loader), len(loader), "")
    return records


def _requested_video_ids(source_manifest: Path) -> list[str]:
    rows = read_csv_rows(source_manifest)
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        if row.get("split", "val") != "val" or row.get("dataset_source", VALIDATION_SOURCE) != VALIDATION_SOURCE:
            raise ValueError("Requested-video inventory must belong to AV++ Validation")
        file_name = row.get("file", "").strip()
        if not file_name:
            raise ValueError("Requested-video inventory contains a missing file")
        video_id = stable_id(file_name)
        if video_id not in seen:
            seen.add(video_id)
            ordered.append(video_id)
    return ordered


def _load_threshold(
    threshold: float | None,
    threshold_artifact: Path | None,
    provenance: dict[str, object],
    engineering_preflight: bool = False,
) -> dict[str, object]:
    if threshold_artifact is not None:
        if threshold is not None:
            raise ValueError("Use either --threshold or --threshold-artifact, not both")
        artifact = json.loads(threshold_artifact.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise TypeError("Threshold artifact must be a JSON object")
        if artifact.get("threshold_version") != THRESHOLD_VERSION:
            raise ValueError("Threshold artifact has an unsupported version")
        value = _float_or_none(artifact.get("threshold"))
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError("Threshold artifact does not contain a valid threshold")
        if (
            artifact.get("split") != "val"
            or artifact.get("dataset_source") != VALIDATION_SOURCE
            or artifact.get("aggregation") != AGGREGATION
            or artifact.get("selection_metric") != SELECTION_METRIC
            or artifact.get("tie_break") != THRESHOLD_TIE_BREAK
            or artifact.get("provenance") != provenance
            or artifact.get("selection", {}).get("threshold") != value
        ):
            raise ValueError("Threshold provenance does not match checkpoint, inputs, or aggregation")
        predictions = artifact.get("predictions", {})
        if (
            not predictions.get("path")
            or predictions.get("sha256") != file_sha256(Path(predictions["path"]))
        ):
            raise ValueError("Threshold selection prediction hash mismatch")
        return {
            "value": value,
            "source": "artifact",
            "artifact": threshold_artifact.as_posix(),
            "artifact_sha256": file_sha256(threshold_artifact),
            "selection": artifact.get("selection"),
            "provenance": artifact["provenance"],
            "predictions": predictions,
        }
    if threshold is None:
        raise ValueError(
            "A frozen threshold is required: pass --threshold or --threshold-artifact "
            "(select the threshold on Validation only)"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not engineering_preflight:
        raise ValueError("Explicit thresholds are only allowed for an engineering preflight")
    return {"value": threshold, "source": "engineering_preflight", "artifact": None}


def _validate_scope(split: str | None, dataset_source: str) -> None:
    if split != "val" or dataset_source != VALIDATION_SOURCE:
        raise ValueError(
            "Only AV-Deepfake1M++ Validation (split='val') is enabled; "
            "external/test evaluation remains locked and cannot fit thresholds"
        )


def evaluation_inputs(
    manifest_path: Path, checkpoint_path: Path, split: str | None,
    dataset_source: str = VALIDATION_SOURCE,
    sequence_length: int | None = None, image_size: int | None = None,
    engineering_preflight: bool = False,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Validate provenance before constructing a model or decoding any clip."""
    _validate_scope(split, dataset_source)
    rows = read_csv_rows(manifest_path)
    if any(row.get("dataset_split") != split for row in rows):
        raise ValueError("Evaluation manifests must contain only the declared Validation split")
    if any(row.get("dataset_source", VALIDATION_SOURCE) != VALIDATION_SOURCE for row in rows):
        raise ValueError("External dataset rows cannot be used as AV++ Validation")
    ids = [row.get("event_id") for row in rows]
    if any(not event_id for event_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Evaluation event IDs must be nonempty and unique")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    contract = checkpoint.get("resume_contract", {})
    if checkpoint.get("checkpoint_type") != "seepat_evaluation_model":
        raise ValueError("Expected a SeePAT evaluation checkpoint")
    model_name = str(contract.get("model_name", ""))
    if contract.get("training_version") != training_version_for_model(model_name):
        raise ValueError("Unsupported checkpoint training version")
    options = checkpoint.get("options", {})
    if not engineering_preflight and (
        "max_train_batches" not in options or "max_validation_batches" not in options
        or options["max_train_batches"] is not None
        or options["max_validation_batches"] is not None
    ):
        raise ValueError("A completed model experiment is required; preflight checkpoints cannot select thresholds")
    if not engineering_preflight:
        run = json.loads((checkpoint_path.parent / "run.json").read_text(encoding="utf-8"))
        if (
            run.get("status") != "complete" or run.get("run_type") != "training_experiment"
            or run.get("resume_contract") != contract or int(run.get("global_step", 0)) < 1
            or Path(str(run.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve()
        ):
            raise ValueError("Checkpoint must be the selected output of a completed model experiment")
    for key, requested in (("sequence_length", sequence_length), ("image_size", image_size)):
        if not isinstance(contract.get(key), int) or contract[key] < 1:
            raise ValueError(f"Checkpoint has no valid {key}")
        if requested is not None and requested != contract[key]:
            raise ValueError(f"Evaluation {key} differs from checkpoint")
    manifest_hash = file_sha256(manifest_path)
    if contract.get("validation_manifest_sha256") != manifest_hash:
        raise ValueError("Evaluation manifest differs from the checkpoint's Validation manifest")
    fusion_inputs = None
    if model_name == FUSION_MODEL:
        from seepat.evidence import calibrated_manifest_contract

        fusion_inputs = calibrated_manifest_contract(manifest_path, rows, allow_empty=True)
        if contract.get("fusion_inputs") != fusion_inputs:
            raise ValueError("Evaluation calibration differs from the frozen checkpoint inputs")
    provenance = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_manifest_sha256": manifest_hash,
        "model_name": model_name, "training_version": contract["training_version"],
        "sequence_length": contract["sequence_length"], "image_size": contract["image_size"],
        "evidence_version": EVIDENCE_VERSION, "fusion_inputs": fusion_inputs,
    }
    return contract, provenance


def _evaluation_dataset(manifest_path: Path, project_root: Path, contract: dict[str, Any]):
    return MouthEventDataset(
        manifest_path=manifest_path, project_root=project_root, dataset_split="val",
        sequence_length=contract["sequence_length"], image_size=contract["image_size"],
        require_calibration=contract["model_name"] == FUSION_MODEL, allow_empty=True,
    )


def select_threshold(
    manifest_path: Path, checkpoint_path: Path, output_path: Path,
    split: str = "val", dataset_source: str = VALIDATION_SOURCE,
    project_root: Path = Path("."), device_name: str = "auto", batch_size: int = 1,
    sequence_length: int | None = None, image_size: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    contract, provenance = evaluation_inputs(
        manifest_path, checkpoint_path, split, dataset_source, sequence_length, image_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    dataset = _evaluation_dataset(manifest_path, project_root, contract)
    if not len(dataset):
        raise ValueError("Threshold selection requires eligible Validation events from both classes")
    from seepat.training.dataset import video_class_id

    if {video_class_id(row["manipulation_modality"]) for row in dataset.rows} != {0, 1}:
        raise ValueError("Threshold selection requires both classes in Validation")
    model, _ = load_evaluation_model(checkpoint_path, device)
    records = _score_events(model, dataset, device, batch_size, progress)
    _, labels, probabilities = aggregate_video_probabilities(
        [str(r["video_id"]) for r in records], [int(r["video_label"]) for r in records],
        [float(r["manipulated_probability"]) for r in records],
    )
    selection = choose_threshold(labels, probabilities)
    prediction_path = output_path.with_name(output_path.stem + "_events.csv")
    if any(path.resolve() in {manifest_path.resolve(), checkpoint_path.resolve()} for path in (output_path, prediction_path)):
        raise ValueError("Threshold output must not overwrite its inputs")
    atomic_write_csv(prediction_path, records)
    artifact = {
        "threshold_version": THRESHOLD_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "threshold": selection["threshold"], "selection": selection,
        "split": split, "dataset_source": dataset_source, "aggregation": AGGREGATION,
        "selection_metric": SELECTION_METRIC, "tie_break": THRESHOLD_TIE_BREAK,
        "provenance": provenance,
        "predictions": {"path": prediction_path.as_posix(), "sha256": file_sha256(prediction_path)},
    }
    atomic_write_json(output_path, artifact)
    return artifact


def run_verdict(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    threshold: float | None = None,
    threshold_artifact: Path | None = None,
    split: str | None = None,
    source_manifest: Path | None = None,
    project_root: Path = Path("."),
    device_name: str = "auto",
    batch_size: int = 1,
    sequence_length: int | None = None,
    image_size: int | None = None,
    calibration_summary: Path | None = None,
    dataset_source: str = VALIDATION_SOURCE,
    engineering_preflight: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    else:
        device = torch.device(device_name)

    if threshold is None and threshold_artifact is None:
        raise ValueError("A frozen threshold is required")
    contract, provenance = evaluation_inputs(
        manifest_path, checkpoint_path, split, dataset_source, sequence_length, image_size,
        engineering_preflight,
    )
    threshold_record = _load_threshold(threshold, threshold_artifact, provenance, engineering_preflight)
    frozen_threshold = float(threshold_record["value"])
    protected = {p.resolve() for p in (manifest_path, checkpoint_path, threshold_artifact, source_manifest, calibration_summary) if p is not None}
    if any((output_dir / name).resolve() in protected for name in ("event_predictions.csv", "video_verdicts.csv", "evaluation.json")):
        raise ValueError("Verdict output must not overwrite its inputs")

    dataset = _evaluation_dataset(manifest_path, project_root, contract)
    records = []
    if threshold_record.get("predictions"):
        records = read_csv_rows(Path(threshold_record["predictions"]["path"]))
        if [row["event_id"] for row in records] != [row["event_id"] for row in dataset.rows]:
            raise ValueError("Threshold prediction inventory does not match Validation events")
    elif len(dataset):
        model, _ = load_evaluation_model(checkpoint_path, device)
        records = _score_events(model, dataset, device, batch_size, progress)

    video_ids = [str(record["video_id"]) for record in records]
    video_labels = [int(record["video_label"]) for record in records]
    probabilities = [float(record["manipulated_probability"]) for record in records]
    ordered_ids, ordered_labels, aggregated = aggregate_video_probabilities(
        video_ids,
        video_labels,
        probabilities,
    ) if records else ([], [], [])
    evaluated = {
        video_id: (label, probability)
        for video_id, label, probability in zip(
            ordered_ids,
            ordered_labels,
            aggregated,
            strict=True,
        )
    }

    verdict_rows: list[dict[str, object]] = []
    event_counts = Counter(str(record["video_id"]) for record in records)
    ordering = sorted(evaluated)
    if source_manifest is not None:
        requested = _requested_video_ids(source_manifest)
        if set(evaluated) - set(requested):
            raise ValueError("Scored videos are absent from the requested-video inventory")
        missing = [video_id for video_id in requested if video_id not in evaluated]
        ordering = ordering + sorted(missing)
    for video_id in ordering:
        if video_id in evaluated:
            label, probability = evaluated[video_id]
            verdict_rows.append(
                {
                    "video_id": video_id,
                    "event_count": event_counts[video_id],
                    "maximum_probability": probability,
                    "threshold": frozen_threshold,
                    "verdict": (
                        VERDICT_MANIPULATED
                        if probability >= frozen_threshold
                        else VERDICT_AUTHENTIC
                    ),
                    "label": label,
                }
            )
        else:
            verdict_rows.append(
                {
                    "video_id": video_id,
                    "event_count": 0,
                    "maximum_probability": "",
                    "threshold": frozen_threshold,
                    "verdict": VERDICT_NOT_EVALUATED,
                    "label": "",
                }
            )

    metrics: dict[str, object] | None = None
    if len({label for label in ordered_labels}) > 1:
        metrics = {
            "videos": binary_classification_metrics(
                ordered_labels,
                aggregated,
                frozen_threshold,
            ),
            "events": binary_classification_metrics(
                [int(record["label"]) for record in records],
                probabilities,
                frozen_threshold,
            ),
            "video_aggregation": "maximum event probability",
        }
    not_evaluated = sum(
        1 for row in verdict_rows if row["verdict"] == VERDICT_NOT_EVALUATED
    )
    sync_gap_scored = sum(1 for record in records if record["sync_gap_score"] != "")

    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "event_predictions.csv"
    verdict_path = output_dir / "video_verdicts.csv"
    atomic_write_csv(event_path, records)
    atomic_write_csv(verdict_path, verdict_rows)
    summary: dict[str, object] = {
        "verdict_version": VERDICT_VERSION,
        "mode": "engineering_preflight" if engineering_preflight else "validation",
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "dataset_source": dataset_source,
        "provenance": provenance,
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": file_sha256(checkpoint_path),
        },
        "model_name": contract.get("model_name"),
        "model": contract.get("model"),
        "training_version": contract.get("training_version"),
        "threshold": threshold_record,
        "aggregation": "maximum event probability",
        "event_count": len(records),
        "video_count": len(verdict_rows),
        "not_evaluated_videos": not_evaluated,
        "sync_gap_scores": sync_gap_scored > 0,
        "sync_gap_scored_events": sync_gap_scored,
        "evidence_coverage": dataset.evidence_coverage(),
        "metrics": metrics,
        "outputs": {
            "events": event_path.as_posix(),
            "videos": verdict_path.as_posix(),
        },
        "output_sha256": {"events": file_sha256(event_path), "videos": file_sha256(verdict_path)},
    }
    if calibration_summary is not None:
        summary["calibration_summary"] = {
            "path": calibration_summary.as_posix(),
            "sha256": file_sha256(calibration_summary),
        }
    summary["source_manifest"] = (
        {"path": source_manifest.as_posix(), "sha256": file_sha256(source_manifest)}
        if source_manifest is not None else None
    )
    atomic_write_json(output_dir / "evaluation.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="seepat-verdict")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Explicit threshold for --engineering-preflight only",
    )
    parser.add_argument("--threshold-artifact", type=Path)
    parser.add_argument("--split", default="val", help="Only AV++ val is currently enabled")
    parser.add_argument("--dataset-source", default=VALIDATION_SOURCE)
    parser.add_argument("--engineering-preflight", action="store_true")
    parser.add_argument("--select-threshold", action="store_true", help="Fit threshold.json on Validation")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional pipeline manifest used to report videos without events",
    )
    parser.add_argument("--calibration-summary", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()

    if args.select_threshold:
        if args.threshold is not None or args.threshold_artifact or args.engineering_preflight:
            parser.error("Threshold selection cannot use a supplied threshold or preflight mode")
        result = select_threshold(
            args.manifest, args.checkpoint, args.output_dir / "threshold.json",
            split=args.split, dataset_source=args.dataset_source, project_root=args.project_root,
            device_name=args.device, batch_size=args.batch_size,
            sequence_length=args.sequence_length, image_size=args.image_size,
        )
        print(json.dumps(result, indent=2))
        return
    summary = run_verdict(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        threshold_artifact=args.threshold_artifact,
        split=args.split,
        source_manifest=args.source_manifest,
        project_root=args.project_root,
        device_name=args.device,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        calibration_summary=args.calibration_summary,
        dataset_source=args.dataset_source,
        engineering_preflight=args.engineering_preflight,
    )
    print(json.dumps(summary, indent=2))


def threshold_main() -> None:
    """Select the frozen threshold on a labelled Validation manifest."""
    parser = argparse.ArgumentParser(prog="seepat-threshold")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-source", default=VALIDATION_SOURCE)
    parser.add_argument(
        "--aggregation",
        choices=("video",),
        default="video",
        help="Score videos with maximum event aggregation or raw events",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()

    artifact = select_threshold(
        args.manifest, args.checkpoint, args.output,
        split=args.split, dataset_source=args.dataset_source, project_root=args.project_root,
        device_name=args.device, batch_size=args.batch_size,
        sequence_length=args.sequence_length, image_size=args.image_size,
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
