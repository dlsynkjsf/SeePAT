from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from seepat.artifacts import atomic_write_json
from seepat.live_progress import ProgressCallback
from seepat.models.cnn_temporal import (
    EfficientNetTempCNNEventClassifier,
    TempCNNOnlyEventClassifier,
)
from seepat.models.fusion import FUSION_MODEL_NAME, HybridFusionEventClassifier
from seepat.models.swin_baseline import SwinBaseEventClassifier, parameter_counts
from seepat.training.dataset import MouthEventDataset
from seepat.training.metrics import (
    aggregate_video_probabilities,
    binary_classification_metrics,
    choose_balanced_accuracy_threshold,
)
from seepat.training.video_objective import (
    VIDEO_MAX,
    VIDEO_SUPERVISION_CONTRACT,
    VideoBag,
    scored_video_bags,
    select_video_bags,
    train_video_epoch,
    video_bags,
    video_loader,
)

SWIN_BASE_MODEL = "swin3d_b"
CNN_TEMPORAL_MODEL = "efficientnet_v2_s_tempcnn"
FUSION_MODEL = FUSION_MODEL_NAME
EFFICIENTNET_MODEL = "efficientnet_v2_s_only"
TEMPCNN_MODEL = "tempcnn_gray16"
VISUAL_FUSION_MODEL = "swin3d_b_visual_fusion"
SUPPORTED_MODELS = (
    SWIN_BASE_MODEL, CNN_TEMPORAL_MODEL, FUSION_MODEL,
    EFFICIENTNET_MODEL, TEMPCNN_MODEL, VISUAL_FUSION_MODEL,
)
MODEL_CONTRACT_NAMES = {
    SWIN_BASE_MODEL: "torchvision.swin3d_b",
    CNN_TEMPORAL_MODEL: "torchvision.efficientnet_v2_s+tempcnn",
    FUSION_MODEL: "seepat.hybrid_fusion.swin3d_b_tempcnn_evidence",
    EFFICIENTNET_MODEL: "torchvision.efficientnet_v2_s+masked_mean",
    TEMPCNN_MODEL: "seepat.tempcnn+fixed_grayscale_16x16",
    VISUAL_FUSION_MODEL: "seepat.hybrid_fusion.swin3d_b_tempcnn_no_evidence",
}
MODEL_TRAINING_VERSIONS = {
    SWIN_BASE_MODEL: "swin-baseline-v1",
    CNN_TEMPORAL_MODEL: "cnn-temporal-v1",
    FUSION_MODEL: "hybrid-fusion-v3",
    EFFICIENTNET_MODEL: "efficientnet-only-v1",
    TEMPCNN_MODEL: "tempcnn-gray16-v1",
    VISUAL_FUSION_MODEL: "visual-fusion-v1",
}
# Backward-compatible public name for existing Swin run records and callers.
TRAINING_VERSION = MODEL_TRAINING_VERSIONS[SWIN_BASE_MODEL]
VIDEO_F1_SELECTION = "validation_video_f1"
VIDEO_BALANCED_ACCURACY_SELECTION = "validation_video_balanced_accuracy"


def model_contract_name(model_name: str) -> str:
    try:
        return MODEL_CONTRACT_NAMES[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported training model: {model_name!r}") from error


def training_version_for_model(model_name: str) -> str:
    try:
        return MODEL_TRAINING_VERSIONS[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported training model: {model_name!r}") from error


def build_event_classifier(
    model_name: str,
    pretrained: bool,
    freeze_backbone: bool,
    unfreeze_final_backbone_stages: bool = False,
) -> nn.Module:
    if unfreeze_final_backbone_stages and model_name not in {
        FUSION_MODEL,
        VISUAL_FUSION_MODEL,
    }:
        raise ValueError("Final-stage unfreezing is supported only by fusion models")
    if model_name == SWIN_BASE_MODEL:
        return SwinBaseEventClassifier(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if model_name in {CNN_TEMPORAL_MODEL, EFFICIENTNET_MODEL}:
        return EfficientNetTempCNNEventClassifier(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            temporal=model_name == CNN_TEMPORAL_MODEL,
        )
    if model_name == TEMPCNN_MODEL:
        if pretrained or freeze_backbone:
            raise ValueError("TempCNN alone has no pretrained backbone; disable pretrained/freeze_backbone")
        return TempCNNOnlyEventClassifier()
    if model_name in {FUSION_MODEL, VISUAL_FUSION_MODEL}:
        return HybridFusionEventClassifier(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_final_backbone_stages=unfreeze_final_backbone_stages,
            use_evidence=model_name == FUSION_MODEL,
        )
    raise ValueError(f"Unsupported training model: {model_name!r}")


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 10
    batch_size: int = 1
    workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    sequence_length: int = 16
    image_size: int = 224
    seed: int = 20260823
    amp: bool = True
    freeze_backbone: bool = False
    unfreeze_final_backbone_stages: bool = False
    backbone_learning_rate: float | None = None
    class_weighting: str = "balanced"
    positive_class_weight_ratio: float | None = None
    loss_function: str = "cross_entropy"
    focal_gamma: float = 2.0
    supervision: str = "event"
    selection_metric: str = VIDEO_F1_SELECTION
    early_stopping_patience: int = 3
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def validate(self) -> None:
        integer_values = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "sequence_length": self.sequence_length,
            "image_size": self.image_size,
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0:
            raise ValueError("workers must not be negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must not be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if self.unfreeze_final_backbone_stages and not self.freeze_backbone:
            raise ValueError("Final-stage unfreezing requires freeze_backbone")
        if self.unfreeze_final_backbone_stages and self.backbone_learning_rate is None:
            raise ValueError("Final-stage unfreezing requires backbone_learning_rate")
        if self.backbone_learning_rate is not None and (
            not self.unfreeze_final_backbone_stages
            or self.backbone_learning_rate <= 0
        ):
            raise ValueError(
                "backbone_learning_rate must be positive and requires final-stage unfreezing"
            )
        if self.class_weighting not in {"balanced", "balanced_global", "none"}:
            raise ValueError("class_weighting must be 'balanced', 'balanced_global', or 'none'")
        if self.positive_class_weight_ratio is not None and (
            self.class_weighting != "balanced_global"
            or self.positive_class_weight_ratio <= 0
        ):
            raise ValueError(
                "positive_class_weight_ratio must be positive and requires balanced_global"
            )
        if self.loss_function not in {"cross_entropy", "focal"}:
            raise ValueError("loss_function must be 'cross_entropy' or 'focal'")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must not be negative")
        if self.supervision not in {"event", VIDEO_MAX}:
            raise ValueError("supervision must be 'event' or 'video_max'")
        if self.supervision == VIDEO_MAX and (
            self.batch_size != 1 or self.amp or not self.freeze_backbone
            or self.unfreeze_final_backbone_stages or self.class_weighting != "balanced_global"
            or self.positive_class_weight_ratio is not None or self.loss_function != "cross_entropy"
        ):
            raise ValueError(
                "video_max requires batch size one, FP32, frozen encoders, "
                "balanced_global video weights, no positive ratio override and cross_entropy"
            )
        if self.selection_metric not in {
            VIDEO_F1_SELECTION,
            VIDEO_BALANCED_ACCURACY_SELECTION,
        }:
            raise ValueError(
                "selection_metric must be validation_video_f1 or "
                "validation_video_balanced_accuracy"
            )
        batch_limits = (self.max_train_batches, self.max_validation_batches)
        if (batch_limits[0] is None) != (batch_limits[1] is None):
            raise ValueError(
                "max_train_batches and max_validation_batches must be set together"
            )
        if any(value is not None and value < 1 for value in batch_limits):
            raise ValueError("Preflight batch limits must be positive")
        if self.supervision == VIDEO_MAX and any(value == 1 for value in batch_limits):
            raise ValueError("Video readiness needs at least two complete videos per split")


def source_group_overlap(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> set[str]:
    train_groups = {row["source_group"] for row in train_rows}
    validation_groups = {row["source_group"] for row in validation_rows}
    return train_groups & validation_groups


def _balanced_class_weights(
    rows: list[dict[str, str]],
    device: torch.device,
    positive_ratio: float | None = None,
) -> torch.Tensor:
    counts = [0, 0]
    for row in rows:
        class_id = int(row["class_id"])
        if class_id not in {0, 1}:
            raise ValueError("Training rows must use binary class ids")
        counts[class_id] += 1
    if 0 in counts:
        raise ValueError("Balanced class weighting requires both classes in training data")
    total = sum(counts)
    if positive_ratio is not None:
        negative_weight = total / (counts[0] + counts[1] * positive_ratio)
        return torch.tensor(
            [negative_weight, negative_weight * positive_ratio],
            dtype=torch.float32,
            device=device,
        )
    return torch.tensor(
        [total / (2 * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )


class _FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None,
        gamma: float,
        reduction: str,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
        )
        probabilities = F.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        loss = (1 - probabilities).pow(self.gamma) * cross_entropy
        return loss.mean() if self.reduction == "mean" else loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: Dataset[Any],
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
    balanced_limit: int | None = None,
    require_video_classes: bool = False,
) -> DataLoader[Any]:
    if balanced_limit is not None:
        pools = [[index for index, row in enumerate(dataset.rows) if int(row["class_id"]) == label]
                 for label in (0, 1)]
        if balanced_limit < 2 or not all(pools):
            raise ValueError("Balanced readiness sampling requires both classes and at least two events")
        randomizer = random.Random(seed)
        for pool in pools:
            randomizer.shuffle(pool)
        indices: list[int] = []
        if require_video_classes:
            from seepat.training.dataset import video_class_id

            genuine = next(
                (
                    index for index in pools[0]
                    if video_class_id(dataset.rows[index].get("manipulation_modality", "real")) == 0
                ),
                None,
            )
            manipulated = pools[1][0] if pools[1] else None
            if genuine is None or manipulated is None:
                raise ValueError(
                    "Balanced-accuracy readiness requires genuine and manipulated videos"
                )
            indices.extend((genuine, manipulated))
            pools[0].remove(genuine)
            pools[1].remove(manipulated)
        for index in range(min(balanced_limit, len(dataset))):
            if len(indices) >= balanced_limit:
                break
            label = index % 2
            if not pools[label]:
                label = 1 - label
            indices.append(pools[label].pop())
        dataset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def _forward_logits(
    model: nn.Module,
    videos: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    kwargs: dict[str, torch.Tensor] = {}
    if bool(getattr(model, "uses_frame_mask", False)):
        frame_mask = batch.get("frame_mask")
        if not isinstance(frame_mask, torch.Tensor):
            raise TypeError("A mask-aware model requires a tensor frame_mask in every batch")
        kwargs["frame_mask"] = frame_mask.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
    if bool(getattr(model, "uses_evidence_features", False)):
        evidence = batch.get("evidence_features")
        evidence_mask = batch.get("evidence_feature_mask")
        if not isinstance(evidence, torch.Tensor) or not isinstance(
            evidence_mask, torch.Tensor
        ):
            raise TypeError(
                "An evidence-aware model requires evidence_features and "
                "evidence_feature_mask tensors in every batch"
            )
        kwargs["features"] = evidence.to(device=device, non_blocking=True)
        kwargs["feature_mask"] = evidence_mask.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
    if not kwargs:
        return model(videos)
    return model(videos, **kwargs)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
    progress: ProgressCallback | None = None,
    phase: str = "validate model",
    selection_metric: str = VIDEO_F1_SELECTION,
    video_plan: list[VideoBag] | None = None,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    event_count = 0
    event_labels: list[int] = []
    event_probabilities: list[float] = []
    video_ids: list[str] = []
    video_labels: list[int] = []
    batch_count = min(len(loader), max_batches) if max_batches is not None else len(loader)
    if video_plan is not None:
        if amp_enabled or max_batches is not None:
            raise ValueError("Complete-video validation cannot use AMP or truncate event batches")
        batch_count = len(video_plan)

    with torch.inference_mode():
        if video_plan is not None:
            for bag, scored in zip(video_plan, scored_video_bags(
                model, loader, video_plan, device, _forward_logits, progress, phase,
            ), strict=True):
                loss = loss_function(scored["logits"], scored["label"]).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Validation video loss is not finite")
                loss_sum += float(loss.item())
                labels = scored["event_labels"]
                event_count += len(labels)
                event_labels.extend(labels)
                event_probabilities.extend(scored["event_probabilities"])
                video_ids.extend([bag.video_id] * len(labels))
                video_labels.extend([bag.label] * len(labels))
        for index, batch in enumerate(islice(loader, batch_count) if video_plan is None else ()):
            if progress is not None:
                progress(phase, index, batch_count, str(batch["video_id"][0]))
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = _forward_logits(model, videos, batch, device)
                loss = loss_function(logits, labels).mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Validation loss is not finite")
            batch_size = labels.shape[0]
            loss_sum += float(loss.item()) * batch_size
            event_count += batch_size
            event_labels.extend(labels.cpu().tolist())
            event_probabilities.extend(F.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
            video_ids.extend(batch["video_id"])
            video_labels.extend(batch["video_label"].tolist())

    if progress is not None:
        progress(phase, batch_count, batch_count, "")
    _, aggregated_labels, aggregated_probabilities = aggregate_video_probabilities(
        video_ids,
        video_labels,
        event_probabilities,
    )
    video_metrics = binary_classification_metrics(
        aggregated_labels,
        aggregated_probabilities,
    )
    if selection_metric == VIDEO_BALANCED_ACCURACY_SELECTION:
        threshold_selection = choose_balanced_accuracy_threshold(
            aggregated_labels,
            aggregated_probabilities,
        )
        selected_video_metrics = binary_classification_metrics(
            aggregated_labels,
            aggregated_probabilities,
            float(threshold_selection["threshold"]),
        )
        selection = {
            "metric": selection_metric,
            "value": threshold_selection["balanced_accuracy"],
            **threshold_selection,
        }
    else:
        selected_video_metrics = video_metrics
        selection = {
            "metric": VIDEO_F1_SELECTION,
            "value": video_metrics["f1"],
            "threshold": 0.5,
        }
    return {
        "loss": loss_sum / (len(video_plan) if video_plan is not None else event_count),
        "batches": batch_count,
        "events_processed": event_count,
        "events": binary_classification_metrics(event_labels, event_probabilities),
        "videos": video_metrics,
        "selection": selection,
        "videos_at_selection_threshold": selected_video_metrics,
        "video_aggregation": "maximum event fake probability",
        **({"loss_unit": "video", "complete_video_bags": True,
            "videos_processed": len(video_plan),
            "video_bags": [{"video_id": bag.video_id, "label": bag.label,
                            "events": len(bag.indices)} for bag in video_plan]}
           if video_plan is not None else {}),
    }


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selection_value: float,
    best_selection_fpr: float | None,
    best_epoch: int,
    epochs_without_improvement: int,
    history: list[dict[str, object]],
    options: TrainingOptions,
    resume_contract: dict[str, object],
) -> dict[str, object]:
    return {
        "checkpoint_type": "seepat_resumable_training",
        "completed_epoch": epoch,
        "global_step": global_step,
        "best_selection_metric": options.selection_metric,
        "best_selection_value": best_selection_value,
        "best_selection_fpr": best_selection_fpr,
        **(
            {"best_video_f1": best_selection_value}
            if options.selection_metric == VIDEO_F1_SELECTION
            else {}
        ),
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "options": asdict(options),
        "resume_contract": resume_contract,
        "rng_state": _rng_state(),
    }


def _percent(value: object) -> str:
    return f"{float(value):.2%}"


def _print_epoch_summary(
    epoch: int,
    epochs: int,
    updates: int,
    skipped: int,
    train_loss: float,
    validation: dict[str, object],
    improved: bool,
) -> None:
    selection = validation["selection"]
    videos = validation["videos_at_selection_threshold"]
    balanced_accuracy = (float(videos["recall"]) + float(videos["specificity"])) / 2
    print("\n" + "=" * 72)
    print(f"EPOCH {epoch}/{epochs}  |  {'NEW BEST' if improved else 'no improvement'}")
    if validation.get("loss_unit") == "video":
        print("Objective: complete-video maximum | event metrics are diagnostic, not localization")
    print(
        f"Updates {updates:,}  |  skipped {skipped:,}  |  "
        f"train loss {train_loss:.6f}  |  val loss {float(validation['loss']):.6f}"
    )
    print(
        f"Threshold {float(selection['threshold']):.6f}  |  "
        f"balanced accuracy {_percent(balanced_accuracy)}  |  F1 {_percent(videos['f1'])}"
    )
    print(
        f"Recall {_percent(videos['recall'])}  |  "
        f"specificity {_percent(videos['specificity'])}  |  "
        f"FPR {_percent(videos['false_positive_rate'])}"
    )
    print(f"Selection: {selection['metric']} = {_percent(selection['value'])}")
    print("=" * 72)


def _print_training_summary(run: dict[str, object]) -> None:
    reason = "early stopping" if run["stopped_early"] else "epoch target reached"
    print("\n" + "#" * 72)
    print("TRAINING COMPLETE")
    print(
        f"Epochs {run['completed_epochs']}/{run['requested_epochs']} ({reason})  |  "
        f"best epoch {run['best_epoch']}"
    )
    print(
        f"Best {run['selection_metric']} = {_percent(run['best_selection_value'])}  |  "
        f"FPR {_percent(run['best_selection_fpr'])}"
    )
    print(
        f"Optimizer updates {run['global_step']:,}  |  "
        f"elapsed {float(run['elapsed_seconds']) / 3600:.2f} h"
    )
    print("#" * 72)


def train_model(
    model: nn.Module,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    output_dir: Path,
    options: TrainingOptions,
    device: torch.device,
    resume_contract: dict[str, object],
    resume_from: Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    options.validate()
    train_rows = getattr(train_dataset, "rows", None)
    validation_rows = getattr(validation_dataset, "rows", None)
    if not isinstance(train_rows, list) or not isinstance(validation_rows, list):
        raise TypeError("Training and validation datasets must expose manifest rows")
    overlapping_groups = source_group_overlap(train_rows, validation_rows)
    if overlapping_groups:
        examples = ", ".join(sorted(overlapping_groups)[:5])
        raise ValueError(f"Source-group leakage between train and validation: {examples}")
    expected_supervision = VIDEO_SUPERVISION_CONTRACT if options.supervision == VIDEO_MAX else None
    if resume_contract.get("supervision") is not None and resume_contract["supervision"] != expected_supervision:
        raise ValueError("Declared supervision contract differs from training options")

    resume_contract = {
        **resume_contract,
        **({"supervision": dict(VIDEO_SUPERVISION_CONTRACT)}
           if options.supervision == VIDEO_MAX else {}),
        **({"seed": options.seed} if options.supervision == VIDEO_MAX else {}),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": options.learning_rate,
            "backbone_learning_rate": options.backbone_learning_rate,
            "weight_decay": options.weight_decay,
            "gradient_accumulation_steps": options.gradient_accumulation_steps,
            "step_counting": "completed_optimizer_updates",
        },
        "selection": {
            "metric": options.selection_metric,
            "early_stopping_patience": options.early_stopping_patience,
        },
        "batch_limits": {
            "train": options.max_train_batches,
            "validation": options.max_validation_batches,
        },
    }

    _seed_everything(options.seed)
    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters")
    optimizer_parameters: object = trainable_parameters
    if options.backbone_learning_rate is not None:
        branches = (getattr(model, "backbone", None), getattr(model, "frame_backbone", None))
        backbone_parameters = [
            parameter
            for branch in branches
            if isinstance(branch, nn.Module)
            for parameter in branch.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter for parameter in trainable_parameters if id(parameter) not in backbone_ids
        ]
        if not backbone_parameters or not head_parameters:
            raise ValueError("Differential learning rates require trainable backbone and head parameters")
        optimizer_parameters = [
            {"params": head_parameters, "lr": options.learning_rate},
            {"params": backbone_parameters, "lr": options.backbone_learning_rate},
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    amp_enabled = options.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_bags = video_bags(train_rows) if options.supervision == VIDEO_MAX else None
    validation_bags = video_bags(validation_rows) if options.supervision == VIDEO_MAX else None
    weighting_rows = (
        [{"class_id": str(bag.label)} for bag in train_bags]
        if train_bags is not None else train_rows
    )
    class_weights = (
        _balanced_class_weights(
            weighting_rows,
            device,
            positive_ratio=options.positive_class_weight_ratio,
        )
        if options.class_weighting in {"balanced", "balanced_global"}
        else None
    )
    # Global Train weights must survive batch size one and gradient accumulation.
    # Retain the old batch-normalized mode for historical checkpoint replay.
    reduction = "none" if options.class_weighting == "balanced_global" else "mean"
    loss_function: nn.Module = (
        _FocalLoss(class_weights, options.focal_gamma, reduction)
        if options.loss_function == "focal"
        else nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    start_epoch = 1
    global_step = 0
    best_selection_value = -1.0
    best_selection_fpr: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    if resume_from is not None:
        if progress is not None:
            progress("restore model checkpoint", 0, 0, resume_from.as_posix())
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_type") != "seepat_resumable_training":
            raise ValueError("The resume file is not a SeePAT training checkpoint")
        if checkpoint.get("resume_contract") != resume_contract:
            raise ValueError("The checkpoint does not match the current model and manifests")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_selection_value = float(
            checkpoint.get("best_selection_value", checkpoint.get("best_video_f1", -1.0))
        )
        stored_fpr = checkpoint.get("best_selection_fpr")
        best_selection_fpr = float(stored_fpr) if stored_fpr is not None else None
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        _restore_rng_state(checkpoint["rng_state"])
    if start_epoch > options.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}; "
            f"requested total epochs is {options.epochs}"
        )

    def count_optimizer_step(*_args: Any, **_kwargs: Any) -> None:
        nonlocal global_step
        global_step += 1

    # GradScaler can skip optimizer.step() when gradients overflow.
    optimizer.register_step_post_hook(count_optimizer_step)

    started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    last_completed_epoch = start_epoch - 1
    stopped_early = False
    limited_run = options.max_train_batches is not None
    train_batches_per_epoch = (len(train_dataset) + options.batch_size - 1) // options.batch_size
    validation_batches_per_epoch = (
        len(validation_dataset) + options.batch_size - 1
    ) // options.batch_size
    if train_bags is not None:
        train_batches_per_epoch = len(train_bags)
        validation_batches_per_epoch = len(validation_bags)
    if options.max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, options.max_train_batches)
    if options.max_validation_batches is not None:
        validation_batches_per_epoch = min(
            validation_batches_per_epoch,
            options.max_validation_batches,
        )
    processed_train_events = 0
    processed_validation_events = 0
    evidence_audit: dict[str, object] | None = None
    if bool(getattr(model, "uses_evidence_features", False)):
        train_coverage = getattr(train_dataset, "evidence_coverage", None)
        validation_coverage = getattr(validation_dataset, "evidence_coverage", None)
        if callable(train_coverage) and callable(validation_coverage):
            evidence_audit = {
                "fields": list(getattr(model, "evidence_fields", ())),
                "train": train_coverage(),
                "validation": validation_coverage(),
            }
    run_record: dict[str, object] = {
        "status": "running",
        "run_type": "engineering_preflight" if limited_run else "training_experiment",
        "started_at_utc": started_at,
        "device": str(device),
        "amp_requested": options.amp,
        "amp_enabled": amp_enabled,
        "resumed_from": resume_from.as_posix() if resume_from else None,
        "resume_contract": resume_contract,
        "options": asdict(options),
        "parameters": parameter_counts(model),
        "optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "data": {
            "preflight_sampling": (
                "balanced_complete_videos" if limited_run and train_bags is not None
                else "balanced_events" if limited_run and options.class_weighting == "balanced_global"
                else "original_loader_order"
            ),
            **({"batch_unit": "complete_video", "train_videos": len(train_bags),
                "validation_videos": len(validation_bags)} if train_bags is not None else {}),
            "train_events": len(train_dataset),
            "validation_events": len(validation_dataset),
            "train_source_groups": len({row["source_group"] for row in train_rows}),
            "validation_source_groups": len(
                {row["source_group"] for row in validation_rows}
            ),
            "train_batches_per_epoch": train_batches_per_epoch,
            "validation_batches_per_epoch": validation_batches_per_epoch,
            "evidence_coverage": evidence_audit,
        },
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    atomic_write_json(output_dir / "run.json", run_record)
    if limited_run:
        print(
            "Engineering preflight: "
            f"{train_batches_per_epoch} Train and {validation_batches_per_epoch} Validation "
            f"{'complete videos' if train_bags is not None else 'batches'} per epoch"
        )
    if train_bags is not None:
        print("Video-aware training: one weighted loss per complete video; event scores are not localization.")

    try:
        validation_plan = (
            select_video_bags(validation_bags, options.seed, False, options.max_validation_batches)
            if validation_bags is not None else None
        )
        validation_loader = video_loader(
            validation_dataset, validation_plan, options.workers, device, options.seed,
        ) if validation_plan is not None else _loader(
            validation_dataset,
            options.batch_size,
            options.workers,
            device,
            shuffle=False,
            seed=options.seed,
            balanced_limit=(validation_batches_per_epoch * options.batch_size
                            if limited_run and options.class_weighting == "balanced_global" else None),
            require_video_classes=(
                limited_run
                and options.selection_metric == VIDEO_BALANCED_ACCURACY_SELECTION
            ),
        )
        for epoch in range(start_epoch, options.epochs + 1):
            train_plan = (
                select_video_bags(train_bags, options.seed + epoch, True, options.max_train_batches)
                if train_bags is not None else None
            )
            train_loader = video_loader(
                train_dataset, train_plan, options.workers, device, options.seed + epoch,
            ) if train_plan is not None else _loader(
                train_dataset,
                options.batch_size,
                options.workers,
                device,
                shuffle=True,
                seed=options.seed + epoch,
                balanced_limit=(train_batches_per_epoch * options.batch_size
                                if limited_run and options.class_weighting == "balanced_global" else None),
            )
            model.train()
            if options.freeze_backbone and hasattr(model, "backbone"):
                model.backbone.eval()
                for module in getattr(model, "trainable_backbone_modules", ()):
                    module.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            event_count = 0
            epoch_start_step = global_step
            epoch_skipped_steps = 0
            train_labels: list[int] = []
            train_probabilities: list[float] = []
            video_train = None
            if train_plan is not None:
                video_train = train_video_epoch(
                    model, train_loader, train_plan, device, _forward_logits, loss_function,
                    optimizer, options.gradient_accumulation_steps, trainable_parameters,
                    progress, f"train videos epoch {epoch}/{options.epochs}",
                )
                batch = video_train.pop("first_batch")
                if evidence_audit is not None and "first_train_batch" not in run_record["data"]:
                    run_record["data"]["first_train_batch"] = {
                        "video_shape": list(batch["video"].shape),
                        "frame_mask_shape": list(batch["frame_mask"].shape),
                        "evidence_shape": list(batch["evidence_features"].shape),
                        "evidence_mask": batch["evidence_feature_mask"].tolist(),
                    }
                updates = video_train.pop("optimizer_steps")
                if updates != global_step - epoch_start_step:
                    raise RuntimeError("Video optimizer update counts disagree")
                run_record["optimizer_steps"] += updates
                event_count = video_train["events_processed"]

            for batch_index, batch in enumerate(
                islice(train_loader, train_batches_per_epoch) if train_plan is None else (),
                start=1,
            ):
                if progress is not None:
                    progress(f"train epoch {epoch}/{options.epochs}", batch_index - 1,
                             train_batches_per_epoch, str(batch["video_id"][0]))
                if evidence_audit is not None and "first_train_batch" not in run_record["data"]:
                    run_record["data"]["first_train_batch"] = {
                        "video_shape": list(batch["video"].shape),
                        "frame_mask_shape": list(batch["frame_mask"].shape),
                        "evidence_shape": list(batch["evidence_features"].shape),
                        "evidence_mask": batch["evidence_feature_mask"].tolist(),
                    }
                videos = batch["video"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = _forward_logits(model, videos, batch, device)
                    batch_loss = loss_function(logits, labels).mean()
                    accumulation_size = options.gradient_accumulation_steps
                    backward_loss = batch_loss / accumulation_size
                    if options.class_weighting == "balanced_global":
                        group_start = ((batch_index - 1) // accumulation_size) * accumulation_size
                        group_end = min(group_start + accumulation_size, train_batches_per_epoch)
                        group_samples = min(len(train_dataset), group_end * options.batch_size) - group_start * options.batch_size
                        backward_loss = batch_loss * labels.numel() / group_samples
                if not torch.isfinite(batch_loss).item():
                    raise FloatingPointError("Training loss is not finite")
                scaler.scale(backward_loss).backward()
                final_batch = batch_index == train_batches_per_epoch
                if (
                    batch_index % options.gradient_accumulation_steps == 0
                    or final_batch
                ):
                    step_before = global_step
                    if not scaler.is_enabled():
                        # Validate full-precision gradients without clipping their norm.
                        nn.utils.clip_grad_norm_(
                            trainable_parameters, float("inf"), error_if_nonfinite=True,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updated = global_step > step_before
                    run_record["optimizer_steps"] += int(updated)
                    run_record["skipped_optimizer_steps"] += int(not updated)
                    epoch_skipped_steps += int(not updated)

                batch_size = labels.shape[0]
                loss_sum += float(batch_loss.item()) * batch_size
                event_count += batch_size
                train_labels.extend(labels.detach().cpu().tolist())
                train_probabilities.extend(
                    F.softmax(logits.detach().float(), dim=1)[:, 1].cpu().tolist()
                )
            processed_train_events += event_count
            train_loss = video_train["loss"] if video_train is not None else loss_sum / event_count
            if progress is not None:
                train_phase = "train videos epoch" if train_plan is not None else "train epoch"
                progress(f"{train_phase} {epoch}/{options.epochs}", train_batches_per_epoch,
                         train_batches_per_epoch, "")

            epoch_optimizer_steps = global_step - epoch_start_step
            if epoch_optimizer_steps == 0:
                raise RuntimeError(
                    "No optimizer updates completed this epoch; AMP may have skipped "
                    "all steps after gradient overflow. This run is not a successful "
                    "training check. For the bounded preflight, use amp: false in a "
                    "new output directory."
                )

            validation = _evaluate(
                model,
                validation_loader,
                loss_function,
                device,
                amp_enabled,
                max_batches=options.max_validation_batches if validation_plan is None else None,
                progress=progress,
                phase=f"validate epoch {epoch}/{options.epochs}",
                selection_metric=options.selection_metric,
                video_plan=validation_plan,
            )
            processed_validation_events += int(validation["events_processed"])
            epoch_record: dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_steps": epoch_optimizer_steps,
                "skipped_optimizer_steps": epoch_skipped_steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "train": video_train if video_train is not None else {
                    "loss": train_loss,
                    "batches": train_batches_per_epoch,
                    "events_processed": event_count,
                    "events": binary_classification_metrics(
                        train_labels,
                        train_probabilities,
                    ),
                },
                "validation": validation,
            }
            history.append(epoch_record)
            if limited_run:
                print(
                    f"Epoch {epoch}/{options.epochs}: "
                    f"{train_batches_per_epoch} Train and {validation['batches']} Validation "
                    f"{'complete videos' if train_plan is not None else 'batches'} complete"
                )
            current_selection_value = float(validation["selection"]["value"])  # type: ignore[index]
            current_selection_fpr = float(  # type: ignore[index]
                validation["videos_at_selection_threshold"]["false_positive_rate"]
            )
            improved = current_selection_value > best_selection_value or (
                options.selection_metric == VIDEO_BALANCED_ACCURACY_SELECTION
                and current_selection_value == best_selection_value
                and (
                    best_selection_fpr is None
                    or current_selection_fpr < best_selection_fpr
                )
            )
            if improved:
                best_selection_value = current_selection_value
                best_selection_fpr = current_selection_fpr
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            last_completed_epoch = epoch
            _print_epoch_summary(
                epoch,
                options.epochs,
                epoch_optimizer_steps,
                epoch_skipped_steps,
                train_loss,
                validation,
                improved,
            )

            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_selection_value=best_selection_value,
                best_selection_fpr=best_selection_fpr,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                options=options,
                resume_contract=resume_contract,
            )
            if progress is not None:
                progress(f"save checkpoint epoch {epoch}/{options.epochs}", 0, 0, "")
            _atomic_torch_save(output_dir / "checkpoint_last.pt", checkpoint)
            if improved:
                _atomic_torch_save(
                    output_dir / "checkpoint_best.pt",
                    {
                        "checkpoint_type": "seepat_evaluation_model",
                        "completed_epoch": epoch,
                        "selection_metric": options.selection_metric,
                        "selection_metric_value": current_selection_value,
                        "model_state": model.state_dict(),
                        "options": asdict(options),
                        "resume_contract": resume_contract,
                    },
                )
            atomic_write_json(output_dir / "history.json", history)
            if (
                options.early_stopping_patience > 0
                and epochs_without_improvement >= options.early_stopping_patience
            ):
                stopped_early = True
                break

    except Exception as error:
        run_record.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "global_step": global_step,
            }
        )
        atomic_write_json(output_dir / "run.json", run_record)
        raise

    elapsed_seconds = perf_counter() - started
    processed_events = processed_train_events + processed_validation_events
    run_record.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "requested_epochs": options.epochs,
            "completed_epochs": last_completed_epoch,
            "stopped_early": stopped_early,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "selection_metric": options.selection_metric,
            "best_selection_value": best_selection_value,
            "best_selection_fpr": best_selection_fpr,
            **(
                {"best_validation_video_f1": best_selection_value}
                if options.selection_metric == VIDEO_F1_SELECTION
                else {"best_validation_video_balanced_accuracy": best_selection_value}
            ),
            "processed_train_events": processed_train_events,
            "processed_validation_events": processed_validation_events,
            "processed_events_per_second": (
                round(processed_events / elapsed_seconds, 6) if elapsed_seconds else None
            ),
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "history": (output_dir / "history.json").as_posix(),
            "last_checkpoint": (output_dir / "checkpoint_last.pt").as_posix(),
            "best_checkpoint": (output_dir / "checkpoint_best.pt").as_posix(),
        }
    )
    atomic_write_json(output_dir / "run.json", run_record)
    _print_training_summary(run_record)
    return run_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    expected_contract: dict[str, object],
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    contract = checkpoint.get("resume_contract")
    if checkpoint.get("checkpoint_type") != "seepat_evaluation_model" or not isinstance(
        contract, dict
    ):
        raise ValueError("Initial checkpoint must be a SeePAT evaluation model")
    immutable = (
        "training_version",
        "model",
        "model_name",
        "pretrained",
        "sequence_length",
        "image_size",
        "evidence_fields",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "fusion_inputs",
    )
    if any(contract.get(key) != expected_contract.get(key) for key in immutable):
        raise ValueError("Initial checkpoint does not match the model, manifests, or evidence")
    run = json.loads((checkpoint_path.parent / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("run_type") != "training_experiment"
        or run.get("resume_contract") != contract
        or Path(str(run.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve()
    ):
        raise ValueError("Initial checkpoint must be the selected output of a completed experiment")
    model.load_state_dict(checkpoint["model_state"])


def train_from_manifests(
    train_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    project_root: Path,
    options: TrainingOptions,
    device_name: str,
    pretrained: bool,
    model_name: str = SWIN_BASE_MODEL,
    resume_from: Path | None = None,
    initialize_from: Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    options.validate()
    if options.supervision == VIDEO_MAX and model_name != FUSION_MODEL:
        raise ValueError("Production video_max supervision is supported only by full fusion")
    device = _device(device_name)
    torch.hub.set_dir(str(project_root / ".cache" / "torch"))
    train_dataset = MouthEventDataset(
        manifest_path=train_manifest,
        project_root=project_root,
        dataset_split="train",
        sequence_length=options.sequence_length,
        image_size=options.image_size,
        require_calibration=model_name == FUSION_MODEL,
    )
    validation_dataset = MouthEventDataset(
        manifest_path=validation_manifest,
        project_root=project_root,
        dataset_split="val",
        sequence_length=options.sequence_length,
        image_size=options.image_size,
        require_calibration=model_name == FUSION_MODEL,
    )
    if train_dataset.calibration_contract != validation_dataset.calibration_contract:
        raise ValueError("Fusion Train and Validation must share the same frozen calibration")
    if source_group_overlap(train_dataset.rows, validation_dataset.rows):
        raise ValueError("Source-group leakage between train and validation")
    if progress is not None:
        progress("initialize model", 0, 0, model_name)
    _seed_everything(options.seed)
    model = build_event_classifier(
        model_name=model_name,
        pretrained=pretrained and resume_from is None and initialize_from is None,
        freeze_backbone=options.freeze_backbone,
        unfreeze_final_backbone_stages=options.unfreeze_final_backbone_stages,
    )
    resume_contract = {
        "training_version": training_version_for_model(model_name),
        "model": model_contract_name(model_name),
        "model_name": model_name,
        "pretrained": pretrained,
        "freeze_backbone": options.freeze_backbone,
        "unfreeze_final_backbone_stages": options.unfreeze_final_backbone_stages,
        "sequence_length": options.sequence_length,
        "image_size": options.image_size,
        "class_weighting": options.class_weighting,
        "evidence_fields": list(getattr(model, "evidence_fields", ())),
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifest_sha256": _sha256(validation_manifest),
        **({"supervision": dict(VIDEO_SUPERVISION_CONTRACT)}
           if options.supervision == VIDEO_MAX else {}),
        **({"seed": options.seed} if options.supervision == VIDEO_MAX else {}),
    }
    if options.positive_class_weight_ratio is not None:
        resume_contract["positive_class_weight_ratio"] = options.positive_class_weight_ratio
    if model_name == FUSION_MODEL:
        resume_contract["fusion_inputs"] = train_dataset.calibration_contract
    if initialize_from is not None:
        resume_contract["initial_checkpoint"] = {
            "path": initialize_from.as_posix(),
            "sha256": _sha256(initialize_from),
        }
        if resume_from is None:
            _initialize_model_from_checkpoint(model, initialize_from, resume_contract)
    return train_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        options=options,
        device=device,
        resume_contract=resume_contract,
        resume_from=resume_from,
        progress=progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.train")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=SWIN_BASE_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit Train batches per epoch and mark the run as a preflight",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Limit Validation batches per epoch and mark the run as a preflight",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--unfreeze-final-backbone-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "balanced_global", "none"),
        default="balanced",
    )
    parser.add_argument("--positive-class-weight-ratio", type=float)
    parser.add_argument(
        "--loss-function",
        choices=("cross_entropy", "focal"),
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--supervision", choices=("event", VIDEO_MAX), default="event")
    parser.add_argument(
        "--selection-metric",
        choices=(VIDEO_F1_SELECTION, VIDEO_BALANCED_ACCURACY_SELECTION),
        default=VIDEO_F1_SELECTION,
    )
    args = parser.parse_args()

    options = TrainingOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        seed=args.seed,
        amp=args.amp,
        freeze_backbone=args.freeze_backbone,
        unfreeze_final_backbone_stages=args.unfreeze_final_backbone_stages,
        backbone_learning_rate=args.backbone_learning_rate,
        class_weighting=args.class_weighting,
        positive_class_weight_ratio=args.positive_class_weight_ratio,
        loss_function=args.loss_function,
        focal_gamma=args.focal_gamma,
        supervision=args.supervision,
        selection_metric=args.selection_metric,
        early_stopping_patience=args.early_stopping_patience,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_val_batches,
    )
    report = train_from_manifests(
        train_manifest=args.train_manifest,
        validation_manifest=args.val_manifest,
        output_dir=args.output_dir,
        project_root=args.project_root,
        options=options,
        device_name=args.device,
        pretrained=args.pretrained,
        model_name=args.model,
        resume_from=args.resume_from,
        initialize_from=args.initialize_from,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
