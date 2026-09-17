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
) -> nn.Module:
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
    class_weighting: str = "balanced"
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
        if self.class_weighting not in {"balanced", "balanced_global", "none"}:
            raise ValueError("class_weighting must be 'balanced', 'balanced_global', or 'none'")
        batch_limits = (self.max_train_batches, self.max_validation_batches)
        if (batch_limits[0] is None) != (batch_limits[1] is None):
            raise ValueError(
                "max_train_batches and max_validation_batches must be set together"
            )
        if any(value is not None and value < 1 for value in batch_limits):
            raise ValueError("Preflight batch limits must be positive")


def source_group_overlap(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> set[str]:
    train_groups = {row["source_group"] for row in train_rows}
    validation_groups = {row["source_group"] for row in validation_rows}
    return train_groups & validation_groups


def _balanced_class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = [0, 0]
    for row in rows:
        class_id = int(row["class_id"])
        if class_id not in {0, 1}:
            raise ValueError("Training rows must use binary class ids")
        counts[class_id] += 1
    if 0 in counts:
        raise ValueError("Balanced class weighting requires both classes in training data")
    total = sum(counts)
    return torch.tensor(
        [total / (2 * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )


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
) -> DataLoader[Any]:
    if balanced_limit is not None:
        pools = [[index for index, row in enumerate(dataset.rows) if int(row["class_id"]) == label]
                 for label in (0, 1)]
        if balanced_limit < 2 or not all(pools):
            raise ValueError("Balanced readiness sampling requires both classes and at least two events")
        randomizer = random.Random(seed)
        for pool in pools:
            randomizer.shuffle(pool)
        indices = []
        for index in range(min(balanced_limit, len(dataset))):
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
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    event_count = 0
    event_labels: list[int] = []
    event_probabilities: list[float] = []
    video_ids: list[str] = []
    video_labels: list[int] = []
    batch_count = min(len(loader), max_batches) if max_batches is not None else len(loader)

    with torch.inference_mode():
        for index, batch in enumerate(islice(loader, batch_count)):
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
    return {
        "loss": loss_sum / event_count,
        "batches": batch_count,
        "events_processed": event_count,
        "events": binary_classification_metrics(event_labels, event_probabilities),
        "videos": binary_classification_metrics(
            aggregated_labels,
            aggregated_probabilities,
        ),
        "video_aggregation": "maximum event fake probability",
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
    best_video_f1: float,
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
        "best_video_f1": best_video_f1,
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

    resume_contract = {
        **resume_contract,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": options.learning_rate,
            "weight_decay": options.weight_decay,
            "gradient_accumulation_steps": options.gradient_accumulation_steps,
            "step_counting": "completed_optimizer_updates",
        },
        "selection": {
            "metric": "validation_video_f1",
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
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    amp_enabled = options.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    class_weights = (
        _balanced_class_weights(train_rows, device)
        if options.class_weighting in {"balanced", "balanced_global"}
        else None
    )
    # Global Train weights must survive batch size one and gradient accumulation.
    # Retain the old batch-normalized mode for historical checkpoint replay.
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none" if options.class_weighting == "balanced_global" else "mean",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    start_epoch = 1
    global_step = 0
    best_video_f1 = -1.0
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
        best_video_f1 = float(checkpoint["best_video_f1"])
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
                "balanced_events" if limited_run and options.class_weighting == "balanced_global"
                else "original_loader_order"
            ),
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
            f"{train_batches_per_epoch} Train batches and "
            f"{validation_batches_per_epoch} Validation batches per epoch"
        )

    try:
        validation_loader = _loader(
            validation_dataset,
            options.batch_size,
            options.workers,
            device,
            shuffle=False,
            seed=options.seed,
            balanced_limit=(validation_batches_per_epoch * options.batch_size
                            if limited_run and options.class_weighting == "balanced_global" else None),
        )
        for epoch in range(start_epoch, options.epochs + 1):
            train_loader = _loader(
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
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            event_count = 0
            epoch_start_step = global_step
            epoch_skipped_steps = 0
            train_labels: list[int] = []
            train_probabilities: list[float] = []

            for batch_index, batch in enumerate(
                islice(train_loader, train_batches_per_epoch),
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
            if progress is not None:
                progress(f"train epoch {epoch}/{options.epochs}", train_batches_per_epoch,
                         train_batches_per_epoch, "")

            epoch_optimizer_steps = global_step - epoch_start_step
            print(
                f"Epoch {epoch}/{options.epochs}: {epoch_optimizer_steps} optimizer updates, "
                f"{epoch_skipped_steps} skipped"
            )
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
                max_batches=options.max_validation_batches,
                progress=progress,
                phase=f"validate epoch {epoch}/{options.epochs}",
            )
            processed_validation_events += int(validation["events_processed"])
            epoch_record: dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_steps": epoch_optimizer_steps,
                "skipped_optimizer_steps": epoch_skipped_steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": {
                    "loss": loss_sum / event_count,
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
                    f"{train_batches_per_epoch} Train batches and "
                    f"{validation['batches']} Validation batches complete"
                )
            current_video_f1 = float(validation["videos"]["f1"])  # type: ignore[index]
            improved = current_video_f1 > best_video_f1
            if improved:
                best_video_f1 = current_video_f1
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            last_completed_epoch = epoch

            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_video_f1=best_video_f1,
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
                        "selection_metric": "validation_video_f1",
                        "selection_metric_value": current_video_f1,
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
            "best_validation_video_f1": best_video_f1,
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
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
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
        pretrained=pretrained and resume_from is None,
        freeze_backbone=options.freeze_backbone,
    )
    resume_contract = {
        "training_version": training_version_for_model(model_name),
        "model": model_contract_name(model_name),
        "model_name": model_name,
        "pretrained": pretrained,
        "freeze_backbone": options.freeze_backbone,
        "sequence_length": options.sequence_length,
        "image_size": options.image_size,
        "class_weighting": options.class_weighting,
        "evidence_fields": list(getattr(model, "evidence_fields", ())),
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifest_sha256": _sha256(validation_manifest),
    }
    if model_name == FUSION_MODEL:
        resume_contract["fusion_inputs"] = train_dataset.calibration_contract
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
        "--class-weighting",
        choices=("balanced", "balanced_global", "none"),
        default="balanced",
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
        class_weighting=args.class_weighting,
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
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
