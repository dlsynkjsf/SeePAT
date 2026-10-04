"""Complete-video maximum-score supervision; stored event targets stay unchanged."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from seepat.live_progress import ProgressCallback
from seepat.training.dataset import video_class_id
from seepat.training.metrics import binary_classification_metrics

VIDEO_MAX = "video_max"
VIDEO_SUPERVISION_CONTRACT = {
    "version": "video-max-v1",
    "target": "video manipulation category",
    "loss_unit": "complete video",
    "aggregation": "maximum positive logit margin",
    "tie_break": "first event in manifest order",
    "class_weight_unit": "unique Train video",
    "accumulation_unit": "complete video",
    "event_score_semantics": "video evidence contribution, not manipulation localization",
}


@dataclass(frozen=True)
class VideoBag:
    video_id: str
    label: int
    indices: tuple[int, ...]
    event_ids: tuple[str, ...]


def video_bags(rows: list[dict[str, str]]) -> list[VideoBag]:
    groups: dict[str, list[int]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    event_ids: set[str] = set()
    for index, row in enumerate(rows):
        identifier = row.get("video_id", "")
        event_id = row.get("event_id", "")
        source = row.get("source_group", "")
        modality = row.get("manipulation_modality", "")
        label = video_class_id(modality)
        if not identifier or not source or not event_id or event_id in event_ids:
            raise ValueError("Video bags require unique event IDs and nonempty video/source IDs")
        if identifier in metadata and metadata[identifier] != (source, modality):
            raise ValueError("Events of a video have inconsistent source group or video label")
        if row.get("class_id") not in {"0", "1"} or (label == 0 and row["class_id"] != "0"):
            raise ValueError("Invalid event label in a video bag")
        event_ids.add(event_id)
        metadata[identifier] = (source, modality)
        groups.setdefault(identifier, []).append(index)
    return [
        VideoBag(
            identifier,
            video_class_id(metadata[identifier][1]),
            tuple(indices),
            tuple(rows[index]["event_id"] for index in indices),
        )
        for identifier, indices in groups.items()
    ]


def select_video_bags(
    bags: list[VideoBag], seed: int, shuffle: bool, limit: int | None = None
) -> list[VideoBag]:
    selected = list(bags)
    randomizer = random.Random(seed)
    if limit is not None:
        pools = [[bag for bag in selected if bag.label == label] for label in (0, 1)]
        if limit < 2 or not all(pools):
            raise ValueError(
                "Video readiness requires at least two complete videos and both video classes"
            )
        for pool in pools:
            randomizer.shuffle(pool)
        selected = []
        for index in range(min(limit, len(bags))):
            label = index % 2
            if not pools[label]:
                label = 1 - label
            selected.append(pools[label].pop())
    elif shuffle:
        randomizer.shuffle(selected)
    return selected


def video_loader(
    dataset: Dataset[Any], bags: list[VideoBag], workers: int, device: torch.device, seed: int
) -> DataLoader[Any]:
    indices = [index for bag in bags for index in bag.indices]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def scored_video_bags(
    model: nn.Module,
    loader: DataLoader[Any],
    bags: list[VideoBag],
    device: torch.device,
    forward: Callable[..., torch.Tensor],
    progress: ProgressCallback | None = None,
    phase: str = "score complete videos",
) -> Iterator[dict[str, Any]]:
    """Retain only the current winning graph; never update inside a video bag."""
    batches = iter(loader)
    for bag_index, bag in enumerate(bags):
        winner = None
        largest_margin = float("-inf")
        labels, probabilities = [], []
        first_batch = None
        for event_index in range(len(bag.indices)):
            try:
                batch = next(batches)
            except StopIteration as error:
                raise ValueError("Incomplete video bag: loader ended before all events") from error
            if (
                list(batch["video_id"]) != [bag.video_id]
                or batch["video_label"].tolist() != [bag.label]
                or list(batch["event_id"]) != [bag.event_ids[event_index]]
            ):
                raise ValueError("Video loader mixed video IDs, labels or event membership")
            if progress is not None:
                progress(
                    phase,
                    bag_index,
                    len(bags),
                    f"{bag.video_id} | event {event_index + 1}/{len(bag.indices)}",
                )
            if first_batch is None:
                first_batch = batch
            videos = batch["video"].to(device, non_blocking=True)
            logits = forward(model, videos, batch, device)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all().item():
                raise FloatingPointError("Video objective requires finite binary event logits")
            margin_tensor = (logits[0, 1] - logits[0, 0]).detach()
            if not torch.isfinite(margin_tensor).item():
                raise FloatingPointError("Video event logit margin is not finite")
            margin = float(margin_tensor.item())
            if margin > largest_margin:
                winner, largest_margin = logits, margin
            labels.extend(batch["label"].tolist())
            probabilities.extend(F.softmax(logits.detach().float(), dim=1)[:, 1].cpu().tolist())
            del logits, videos, batch
        yield {
            "logits": winner,
            "label": torch.tensor([bag.label], device=device),
            "event_labels": labels,
            "event_probabilities": probabilities,
            "first_batch": first_batch,
        }
    if next(batches, None) is not None:
        raise ValueError("Video loader contains events outside the complete-bag plan")
    if progress is not None:
        progress(phase, len(bags), len(bags), "")


def train_video_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    bags: list[VideoBag],
    device: torch.device,
    forward: Callable[..., torch.Tensor],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    accumulation: int,
    parameters: list[nn.Parameter],
    progress: ProgressCallback | None = None,
    phase: str = "train complete videos",
) -> dict[str, Any]:
    loss_sum = 0.0
    labels, probabilities, video_probabilities = [], [], []
    first_batch = None
    updates = 0
    for index, scored in enumerate(
        scored_video_bags(model, loader, bags, device, forward, progress, phase)
    ):
        if first_batch is None:
            first_batch = scored["first_batch"]
        loss = loss_function(scored["logits"], scored["label"]).mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Video training loss is not finite")
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, len(bags) - group_start)
        (loss / group_size).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(bags):
            nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        loss_sum += float(loss.item())
        labels.extend(scored["event_labels"])
        probabilities.extend(scored["event_probabilities"])
        video_probabilities.append(max(scored["event_probabilities"]))
    return {
        "loss": loss_sum / len(bags),
        "loss_unit": "video",
        "batches": len(bags),
        "events_processed": len(labels),
        "videos_processed": len(bags),
        "complete_video_bags": True,
        "video_bags": [
            {"video_id": bag.video_id, "label": bag.label, "events": len(bag.indices)}
            for bag in bags
        ],
        "events": binary_classification_metrics(labels, probabilities),
        "videos": binary_classification_metrics([bag.label for bag in bags], video_probabilities),
        "optimizer_steps": updates,
        "first_batch": first_batch,
    }
