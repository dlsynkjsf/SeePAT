"""Masked visual-exposure labels affect training loss, never model inputs."""

from collections import defaultdict
from fractions import Fraction
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from seepat.training.metrics import binary_classification_metrics
from seepat.training.video_objective import event_diagnostic_metrics, scored_video_bags


def balance_targets(records):
    grouped = defaultdict(list)
    for row in records:
        target, mask = row["target"], row["target_mask"]
        if type(target) is not int or target not in {-1, 0, 1} or type(mask) is not bool or mask != (target != -1):
            raise ValueError("Auxiliary target/mask contract differs")
        if mask:
            grouped[row["video_id"]].append(target)
    masses = [sum((Fraction(values.count(c), len(values)) for values in grouped.values()), Fraction()) for c in (0, 1)]
    if not all(m > 0 for m in masses):
        raise ValueError("Auxiliary Train targets require both known classes")
    return {"known_videos": len(grouped), "class_mass": [str(m) for m in masses],
            "class_weights": [float(len(grouped) / (2 * m)) for m in masses]}


def masked_auxiliary_loss(logits, targets, mask, weights):
    if (logits.ndim != 2 or logits.shape[1] != 2 or targets.shape != logits.shape[:1]
            or mask.shape != targets.shape or mask.dtype != torch.bool or targets.dtype != torch.long
            or not torch.equal(mask, targets != -1) or not ((targets >= -1) & (targets <= 1)).all()
            or len(weights) != 2 or any(not isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("Auxiliary target/mask/weight contract differs")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Auxiliary logits are not finite")
    if not mask.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[mask], targets[mask], weight=logits.new_tensor(weights), reduction="sum") / mask.sum()


def train_auxiliary_epoch(model, loader, bags, device, forward, loss_function, optimizer,
                          accumulation, parameters, progress=None, phase="train auxiliary videos", *, weights):
    """Keep one update unit per complete video; retain supervised clip graphs."""
    losses, labels, probabilities, maxima = [], [], [], []
    auxiliary_terms, video_losses, auxiliary_losses = [], [], []
    first_batch, updates, known_total = None, 0, 0

    def dual_forward(model, video, batch, device):
        logits, auxiliary = model.forward_with_auxiliary(
            video, frame_mask=batch["frame_mask"].to(device),
            features=batch["evidence_features"].to(device),
            feature_mask=batch["evidence_feature_mask"].to(device),
        )
        target, mask = batch["auxiliary_target"].to(device), batch["auxiliary_mask"].to(device)
        term = masked_auxiliary_loss(auxiliary, target, mask, weights)
        if mask.any():
            auxiliary_terms.append(term)
        return logits

    for index, scored in enumerate(scored_video_bags(model, loader, bags, device, dual_forward, progress, phase)):
        if first_batch is None:
            first_batch = scored["first_batch"]
        video_loss = loss_function(scored["logits"], scored["label"]).mean()
        auxiliary_loss = torch.stack(auxiliary_terms).mean() if auxiliary_terms else video_loss * 0
        loss = video_loss + 0.25 * auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Auxiliary video training loss is not finite")
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, len(bags) - group_start)
        (loss / group_size).backward()
        known_total += len(auxiliary_terms)
        auxiliary_terms.clear()
        if (index + 1) % accumulation == 0 or index + 1 == len(bags):
            nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        losses.append(float(loss.detach()))
        video_losses.append(float(video_loss.detach()))
        auxiliary_losses.append(float(auxiliary_loss.detach()))
        labels.extend(scored["event_labels"])
        probabilities.extend(scored["event_probabilities"])
        maxima.append(max(scored["event_probabilities"]))
    return {"loss": sum(losses) / len(bags), "video_loss": sum(video_losses) / len(bags),
            "auxiliary_loss": sum(auxiliary_losses) / len(bags), "auxiliary_known_clips": known_total,
            "loss_unit": "video", "batches": len(bags), "events_processed": len(labels),
            "videos_processed": len(bags), "complete_video_bags": True,
            "video_bags": [{"video_id": b.video_id, "label": b.label, "events": len(b.indices)} for b in bags],
            "events": event_diagnostic_metrics(labels, probabilities),
            **({"context_clips_processed": labels.count(-1)} if -1 in labels else {}),
            "videos": binary_classification_metrics([b.label for b in bags], maxima),
            "optimizer_steps": updates, "first_batch": first_batch}
