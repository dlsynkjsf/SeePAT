from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence


def binary_classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, int | float]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    confusion = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in {0, 1}:
            raise ValueError("binary labels must be 0 or 1")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be finite values between 0 and 1")
        prediction = int(probability >= threshold)
        if label == 1 and prediction == 1:
            confusion["true_positive"] += 1
        elif label == 0 and prediction == 0:
            confusion["true_negative"] += 1
        elif label == 0:
            confusion["false_positive"] += 1
        else:
            confusion["false_negative"] += 1

    true_positive = confusion["true_positive"]
    true_negative = confusion["true_negative"]
    false_positive = confusion["false_positive"]
    false_negative = confusion["false_negative"]
    total = len(labels)

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    return {
        "samples": total,
        **confusion,
        "accuracy": divide(true_positive + true_negative, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": divide(2 * precision * recall, precision + recall),
        "false_positive_rate": divide(false_positive, false_positive + true_negative),
        "threshold": threshold,
    }


def aggregate_video_probabilities(
    video_ids: Sequence[str],
    video_labels: Sequence[int],
    event_probabilities: Sequence[float],
) -> tuple[list[str], list[int], list[float]]:
    if not (
        len(video_ids) == len(video_labels) == len(event_probabilities)
        and video_ids
    ):
        raise ValueError("video ids, labels, and probabilities must have equal non-zero length")

    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for video_id, label, probability in zip(
        video_ids,
        video_labels,
        event_probabilities,
        strict=True,
    ):
        if not video_id:
            raise ValueError("video ids must not be empty")
        if label not in {0, 1}:
            raise ValueError("video labels must be 0 or 1")
        previous_label = grouped_labels.setdefault(video_id, label)
        if previous_label != label:
            raise ValueError(f"Video {video_id!r} has conflicting labels")
        grouped_probabilities[video_id].append(probability)

    ordered_ids = sorted(grouped_probabilities)
    return (
        ordered_ids,
        [grouped_labels[video_id] for video_id in ordered_ids],
        [max(grouped_probabilities[video_id]) for video_id in ordered_ids],
    )
