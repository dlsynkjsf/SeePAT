from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
train_module = pytest.importorskip("seepat.training.train")
nn = torch.nn
Dataset = torch.utils.data.Dataset
TrainingOptions = train_module.TrainingOptions
source_group_overlap = train_module.source_group_overlap
train_model = train_module.train_model


class TinyEventDataset(Dataset):
    def __init__(self, prefix: str) -> None:
        self.rows = [
            {"source_group": f"{prefix}-real-a", "class_id": "0"},
            {"source_group": f"{prefix}-real-b", "class_id": "0"},
            {"source_group": f"{prefix}-fake-a", "class_id": "1"},
            {"source_group": f"{prefix}-fake-b", "class_id": "1"},
        ]
        self.samples = []
        for index, row in enumerate(self.rows):
            label = int(row["class_id"])
            self.samples.append(
                {
                    "video": torch.full((3, 1, 2, 2), float(label)),
                    "label": torch.tensor(label, dtype=torch.long),
                    "video_label": torch.tensor(label, dtype=torch.long),
                    "video_id": f"{prefix}-video-{index}",
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


class TinyVideoClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.classifier(video.mean(dim=(2, 3, 4)))


class ConstantVideoClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(video.shape[0], 2)


def _options(epochs: int) -> TrainingOptions:
    return TrainingOptions(
        epochs=epochs,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        sequence_length=1,
        image_size=2,
        amp=True,
    )


def test_source_group_overlap_finds_leakage() -> None:
    overlap = source_group_overlap(
        [{"source_group": "source-a"}, {"source_group": "source-b"}],
        [{"source_group": "source-b"}, {"source_group": "source-c"}],
    )

    assert overlap == {"source-b"}


def test_preflight_batch_limits_must_be_paired_and_positive() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        TrainingOptions(max_train_batches=1).validate()

    with pytest.raises(ValueError, match="must be positive"):
        TrainingOptions(max_train_batches=0, max_validation_batches=1).validate()


def test_training_writes_metrics_and_resumes_at_next_epoch(tmp_path: Path) -> None:
    output_dir = tmp_path / "training"
    train_dataset = TinyEventDataset("train")
    validation_dataset = TinyEventDataset("val")
    contract = {"model": "tiny", "train_manifest": "a", "val_manifest": "b"}

    first_report = train_model(
        model=TinyVideoClassifier(),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        options=_options(epochs=1),
        device=torch.device("cpu"),
        resume_contract=contract,
    )

    assert first_report["status"] == "complete"
    assert first_report["amp_enabled"] is False
    assert (output_dir / "checkpoint_last.pt").is_file()
    assert (output_dir / "checkpoint_best.pt").is_file()
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["validation"]["events"]["samples"] == 4
    assert history[0]["validation"]["videos"]["samples"] == 4

    resumed_report = train_model(
        model=TinyVideoClassifier(),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        options=_options(epochs=2),
        device=torch.device("cpu"),
        resume_contract=contract,
        resume_from=output_dir / "checkpoint_last.pt",
    )

    resumed_history = json.loads(
        (output_dir / "history.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output_dir / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_report["status"] == "complete"
    assert resumed_report["completed_epochs"] == 2
    assert [row["epoch"] for row in resumed_history] == [1, 2]
    assert checkpoint["completed_epoch"] == 2
    assert checkpoint["global_step"] == 4


def test_preflight_limits_batches_and_resumes(tmp_path: Path) -> None:
    output_dir = tmp_path / "preflight"
    train_dataset = TinyEventDataset("train")
    validation_dataset = TinyEventDataset("val")
    contract = {"model": "tiny", "train_manifest": "a", "val_manifest": "b"}

    first_options = TrainingOptions(
        epochs=1,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        sequence_length=1,
        image_size=2,
        gradient_accumulation_steps=2,
        max_train_batches=1,
        max_validation_batches=1,
    )
    first_report = train_model(
        model=TinyVideoClassifier(),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        options=first_options,
        device=torch.device("cpu"),
        resume_contract=contract,
    )

    first_history = json.loads(
        (output_dir / "history.json").read_text(encoding="utf-8")
    )
    assert first_report["run_type"] == "engineering_preflight"
    assert first_report["processed_train_events"] == 2
    assert first_report["processed_validation_events"] == 2
    assert first_history[0]["train"]["batches"] == 1
    assert first_history[0]["train"]["events"]["samples"] == 2
    assert first_history[0]["validation"]["batches"] == 1
    assert first_history[0]["validation"]["events"]["samples"] == 2
    resumed_options = TrainingOptions(
        epochs=2,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        sequence_length=1,
        image_size=2,
        gradient_accumulation_steps=2,
        max_train_batches=1,
        max_validation_batches=1,
    )
    resumed_report = train_model(
        model=TinyVideoClassifier(),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_dir=output_dir,
        options=resumed_options,
        device=torch.device("cpu"),
        resume_contract=contract,
        resume_from=output_dir / "checkpoint_last.pt",
    )
    checkpoint = torch.load(
        output_dir / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_report["completed_epochs"] == 2
    assert resumed_report["processed_train_events"] == 2
    assert resumed_report["processed_validation_events"] == 2
    assert checkpoint["global_step"] == 2
    assert checkpoint["resume_contract"]["batch_limits"] == {
        "train": 1,
        "validation": 1,
    }


def test_training_stops_after_validation_metric_stalls(tmp_path: Path) -> None:
    output_dir = tmp_path / "early-stop"
    options = TrainingOptions(
        epochs=5,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        sequence_length=1,
        image_size=2,
        early_stopping_patience=1,
    )

    report = train_model(
        model=ConstantVideoClassifier(),
        train_dataset=TinyEventDataset("train"),
        validation_dataset=TinyEventDataset("val"),
        output_dir=output_dir,
        options=options,
        device=torch.device("cpu"),
        resume_contract={"model": "constant"},
    )

    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert report["stopped_early"] is True
    assert report["completed_epochs"] == 2
    assert len(history) == 2
