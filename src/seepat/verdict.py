"""Binary event classification, maximum event aggregation and deepfake verdict.

Implements the Figure 4.1 classification chain: every bilabial mouth event is
scored by the trained model (event-level manipulation probability), video
probabilities use maximum event aggregation, and a *frozen* threshold turns the
probability into the deepfake verdict.

Reproducibility rules:

* The threshold must come from ``--threshold`` or a ``--threshold-artifact``
  produced on the Validation split. It is never selected on the data being
  evaluated.
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
from seepat.evidence import FUSION_EVIDENCE_FIELDS, closure_offset_s
from seepat.training.dataset import MouthEventDataset
from seepat.training.metrics import (
    aggregate_video_probabilities,
    binary_classification_metrics,
)
from seepat.training.train import build_event_classifier

VERDICT_VERSION = "verdict-v1"
THRESHOLD_VERSION = "threshold-v1"
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
    """Pick the threshold with the best balanced accuracy.

    Balanced accuracy is ``(recall + specificity) / 2``, which is insensitive
    to class imbalance. Ties prefer the lower threshold (higher recall), and
    both classes must be present.
    """
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold selection requires both classes in the data")

    grouped: dict[float, list[int]] = {}
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in {0, 1}:
            raise ValueError("binary labels must be 0 or 1")
        grouped.setdefault(probability, []).append(label)

    true_positives = 0
    false_positives = 0
    best_threshold = 1.0
    best_balanced_accuracy = 0.5  # all-negative predictions
    for probability in sorted(grouped, reverse=True):
        for label in grouped[probability]:
            if label == 1:
                true_positives += 1
            else:
                false_positives += 1
        balanced_accuracy = 0.5 * (
            true_positives / positives
            + (negatives - false_positives) / negatives
        )
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = probability
    return {
        "threshold": round(best_threshold, 9),
        "balanced_accuracy": round(best_balanced_accuracy, 6),
        "samples": len(labels),
        "positives": positives,
        "negatives": negatives,
    }


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
    if hasattr(model, "forward_with_sync_gap"):
        logits, sync_gap = model.forward_with_sync_gap(videos, **kwargs)
        sync_scores: list[float | None] = sync_gap.float().cpu().tolist()
    else:
        logits = model(videos, **kwargs) if kwargs else model(videos)
        sync_scores = [None] * videos.shape[0]
    return logits, sync_scores


def _score_events(
    model: torch.nn.Module,
    dataset: MouthEventDataset,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows_by_event = {row["event_id"]: row for row in dataset.rows}
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
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
                    "manipulated_probability": round(float(probabilities[index]), 9),
                    "sync_gap_score": (
                        round(float(sync_scores[index]), 9)
                        if sync_scores[index] is not None
                        else ""
                    ),
                    "evidence_available": int(bool(evidence_mask.sum().item())),
                    "evidence_total": int(evidence_mask.numel()),
                }
                for field in FUSION_EVIDENCE_FIELDS:
                    if field == "closure_offset_s":
                        offset = closure_offset_s(row)
                        record[field] = "" if offset is None else round(offset, 9)
                    else:
                        record[field] = row.get(field, "")
                records.append(record)
    return records


def _requested_video_ids(source_manifest: Path) -> list[str]:
    rows = read_csv_rows(source_manifest)
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        file_name = row.get("file", "").strip()
        if not file_name:
            continue
        video_id = stable_id(file_name)
        if video_id not in seen:
            seen.add(video_id)
            ordered.append(video_id)
    return ordered


def _load_threshold(
    threshold: float | None,
    threshold_artifact: Path | None,
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
        return {
            "value": value,
            "source": "artifact",
            "artifact": threshold_artifact.as_posix(),
            "artifact_sha256": file_sha256(threshold_artifact),
            "selection": artifact.get("selection"),
        }
    if threshold is None:
        raise ValueError(
            "A frozen threshold is required: pass --threshold or --threshold-artifact "
            "(select the threshold on Validation only)"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    return {"value": threshold, "source": "explicit", "artifact": None}


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
    sequence_length: int = 16,
    image_size: int = 224,
    calibration_summary: Path | None = None,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    else:
        device = torch.device(device_name)

    threshold_record = _load_threshold(threshold, threshold_artifact)
    frozen_threshold = float(threshold_record["value"])

    model, contract = load_evaluation_model(checkpoint_path, device)
    dataset = MouthEventDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        dataset_split=split,
        sequence_length=sequence_length,
        image_size=image_size,
    )
    records = _score_events(model, dataset, device, batch_size)

    video_ids = [str(record["video_id"]) for record in records]
    video_labels = [int(record["video_label"]) for record in records]
    probabilities = [float(record["manipulated_probability"]) for record in records]
    ordered_ids, ordered_labels, aggregated = aggregate_video_probabilities(
        video_ids,
        video_labels,
        probabilities,
    )
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
        missing = [video_id for video_id in requested if video_id not in evaluated]
        ordering = ordering + sorted(missing)
    for video_id in ordering:
        if video_id in evaluated:
            label, probability = evaluated[video_id]
            verdict_rows.append(
                {
                    "video_id": video_id,
                    "event_count": event_counts[video_id],
                    "maximum_probability": round(probability, 9),
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
        "mode": "evaluation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
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
    }
    if calibration_summary is not None:
        summary["calibration_summary"] = {
            "path": calibration_summary.as_posix(),
            "sha256": file_sha256(calibration_summary),
        }
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
        help="Frozen decision threshold selected on the Validation split",
    )
    parser.add_argument("--threshold-artifact", type=Path)
    parser.add_argument("--split", help="dataset_split value to evaluate, e.g. val or test")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional pipeline manifest used to report videos without events",
    )
    parser.add_argument("--calibration-summary", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

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
    )
    print(json.dumps(summary, indent=2))


def threshold_main() -> None:
    """Select the frozen threshold on a labelled Validation manifest."""
    parser = argparse.ArgumentParser(prog="seepat-threshold")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--aggregation",
        choices=("video", "event"),
        default="video",
        help="Score videos with maximum event aggregation or raw events",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    else:
        device = torch.device(args.device)

    model, contract = load_evaluation_model(args.checkpoint, device)
    dataset = MouthEventDataset(
        manifest_path=args.manifest,
        project_root=args.project_root,
        dataset_split=args.split,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
    )
    records = _score_events(model, dataset, device, args.batch_size)
    if args.aggregation == "video":
        ordered_ids, labels, probabilities = aggregate_video_probabilities(
            [str(record["video_id"]) for record in records],
            [int(record["video_label"]) for record in records],
            [float(record["manipulated_probability"]) for record in records],
        )
        samples = len(ordered_ids)
    else:
        labels = [int(record["label"]) for record in records]
        probabilities = [float(record["manipulated_probability"]) for record in records]
        samples = len(records)
    selection = choose_threshold(labels, probabilities)
    artifact = {
        "threshold_version": THRESHOLD_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "threshold": selection["threshold"],
        "aggregation": args.aggregation,
        "split": args.split,
        "samples": samples,
        "positives": selection["positives"],
        "negatives": selection["negatives"],
        "balanced_accuracy": selection["balanced_accuracy"],
        "manifest": {
            "path": args.manifest.as_posix(),
            "sha256": file_sha256(args.manifest),
        },
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": file_sha256(args.checkpoint),
        },
        "model_name": contract.get("model_name"),
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
