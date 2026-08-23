from __future__ import annotations

import pytest

from seepat.training.metrics import (
    aggregate_video_probabilities,
    binary_classification_metrics,
)


def test_binary_metrics_report_each_confusion_case() -> None:
    metrics = binary_classification_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.7, 0.2],
    )

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["f1"] == 0.5


def test_video_aggregation_uses_maximum_event_probability() -> None:
    video_ids, labels, probabilities = aggregate_video_probabilities(
        video_ids=["video-b", "video-a", "video-b", "video-a"],
        video_labels=[1, 0, 1, 0],
        event_probabilities=[0.2, 0.1, 0.9, 0.4],
    )

    assert video_ids == ["video-a", "video-b"]
    assert labels == [0, 1]
    assert probabilities == [0.4, 0.9]


def test_video_aggregation_rejects_conflicting_labels() -> None:
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_video_probabilities(
            video_ids=["video-a", "video-a"],
            video_labels=[0, 1],
            event_probabilities=[0.1, 0.9],
        )
