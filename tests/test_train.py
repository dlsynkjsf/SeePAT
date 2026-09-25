from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
train_module = pytest.importorskip("seepat.training.train")
nn = torch.nn
Dataset = torch.utils.data.Dataset
TrainingOptions = train_module.TrainingOptions
CNN_TEMPORAL_MODEL = train_module.CNN_TEMPORAL_MODEL
SWIN_BASE_MODEL = train_module.SWIN_BASE_MODEL
model_contract_name = train_module.model_contract_name
source_group_overlap = train_module.source_group_overlap
train_model = train_module.train_model
training_version_for_model = train_module.training_version_for_model


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
                    "frame_mask": torch.tensor([True], dtype=torch.bool),
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


class MaskAwareVideoClassifier(TinyVideoClassifier):
    uses_frame_mask = True

    def __init__(self) -> None:
        super().__init__()
        self.mask_calls = 0

    def forward(
        self,
        video: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_mask is None or frame_mask.dtype is not torch.bool:
            raise TypeError("Expected a boolean frame mask")
        if frame_mask.shape != (video.shape[0], video.shape[2]):
            raise ValueError("Unexpected frame mask shape")
        self.mask_calls += 1
        return super().forward(video)


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


def _imbalanced_dataset():
    dataset = TinyEventDataset("imbalanced-train")
    dataset.rows[2]["class_id"] = "0"
    dataset.samples[2]["label"] = torch.tensor(0)
    dataset.samples[2]["video_label"] = torch.tensor(0)
    dataset.samples[2]["video"].zero_()
    return dataset


def test_readiness_sampling_is_bounded_deterministic_and_contains_both_classes():
    dataset = _imbalanced_dataset()
    samples = []
    for _ in range(2):
        loader = train_module._loader(dataset, 1, 0, torch.device("cpu"), False, 31, balanced_limit=2)
        samples.append([batch["label"].item() for batch in loader])
    assert samples[0] == samples[1] == [0, 1]
    ordinary = train_module._loader(dataset, 1, 0, torch.device("cpu"), False, 31)
    assert [batch["label"].item() for batch in ordinary] == [0, 0, 0, 1]


@pytest.mark.parametrize("weighting", ["balanced", "balanced_global"])
def test_batch_one_weighting_retains_minority_contribution(tmp_path, weighting):
    model = TinyVideoClassifier()
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    options = TrainingOptions(epochs=1, batch_size=1, gradient_accumulation_steps=4,
                              class_weighting=weighting, amp=False, weight_decay=0)
    train_model(model, _imbalanced_dataset(), TinyEventDataset("val"), tmp_path,
                options, torch.device("cpu"), {"model": "tiny", "class_weighting": weighting})
    checkpoint = torch.load(tmp_path / "checkpoint_last.pt", weights_only=False)
    bias_moment = checkpoint["optimizer_state"]["state"][1]["exp_avg"]
    # At equal logits, globally balanced real/fake contributions cancel the bias
    # gradient. Per-batch normalization instead treats the 3:1 inventory as unweighted.
    expected = 0.0 if weighting == "balanced_global" else 0.025
    assert float(bias_moment.abs().max()) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_global_weighting_accumulation_matches_full_batch_with_short_tail(tmp_path, batch_size):
    dataset = _imbalanced_dataset()
    dataset.rows.pop(0)
    dataset.samples.pop(0)
    states = []
    for index, (size, accumulation) in enumerate(((3, 1), (batch_size, 4))):
        torch.manual_seed(19)
        model = TinyVideoClassifier()
        directory = tmp_path / str(index)
        options = TrainingOptions(epochs=1, batch_size=size, gradient_accumulation_steps=accumulation,
                                  class_weighting="balanced_global", amp=False, weight_decay=0)
        train_model(model, dataset, TinyEventDataset("val"), directory,
                    options, torch.device("cpu"), {"model": "tiny"})
        states.append(torch.load(directory / "checkpoint_last.pt", weights_only=False)["optimizer_state"]["state"])
    for key in states[0]:
        torch.testing.assert_close(states[0][key]["exp_avg"], states[1][key]["exp_avg"], atol=1e-7, rtol=1e-6)


def test_bounded_fusion_tuning_weight_and_selection_contract(tmp_path, capsys):
    training = _imbalanced_dataset()
    weights = train_module._balanced_class_weights(
        training.rows,
        torch.device("cpu"),
        positive_ratio=10,
    )
    assert float(weights[1] / weights[0]) == pytest.approx(10)
    assert float((3 * weights[0] + weights[1]) / 4) == pytest.approx(1)

    options = TrainingOptions(
        epochs=1,
        batch_size=2,
        learning_rate=0.1,
        weight_decay=0,
        amp=False,
        class_weighting="balanced_global",
        positive_class_weight_ratio=10,
        selection_metric=train_module.VIDEO_BALANCED_ACCURACY_SELECTION,
    )
    report = train_model(
        TinyVideoClassifier(),
        training,
        TinyEventDataset("val"),
        tmp_path,
        options,
        torch.device("cpu"),
        {"model": "tiny"},
    )
    history = json.loads((tmp_path / "history.json").read_text())
    checkpoint = torch.load(tmp_path / "checkpoint_best.pt", weights_only=False)
    selection = history[0]["validation"]["selection"]
    selected_metrics = history[0]["validation"]["videos_at_selection_threshold"]
    assert selection["metric"] == train_module.VIDEO_BALANCED_ACCURACY_SELECTION
    assert selection["value"] == pytest.approx(
        (selected_metrics["recall"] + selected_metrics["specificity"]) / 2
    )
    assert checkpoint["selection_metric"] == train_module.VIDEO_BALANCED_ACCURACY_SELECTION
    assert report["best_selection_value"] == selection["value"]
    terminal = capsys.readouterr().out
    assert "balanced accuracy" in terminal and "TRAINING COMPLETE" in terminal


def test_model_contracts_distinguish_swin_and_cnn_temporal() -> None:
    assert model_contract_name(SWIN_BASE_MODEL) == "torchvision.swin3d_b"
    assert model_contract_name(CNN_TEMPORAL_MODEL).endswith("+tempcnn")
    assert training_version_for_model(SWIN_BASE_MODEL) != training_version_for_model(
        CNN_TEMPORAL_MODEL
    )

    with pytest.raises(ValueError, match="Unsupported training model"):
        model_contract_name("unknown")


def test_preflight_batch_limits_must_be_paired_and_positive() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        TrainingOptions(max_train_batches=1).validate()

    with pytest.raises(ValueError, match="must be positive"):
        TrainingOptions(max_train_batches=0, max_validation_batches=1).validate()

    with pytest.raises(ValueError, match="requires balanced_global"):
        TrainingOptions(positive_class_weight_ratio=10).validate()


@pytest.mark.parametrize("overflow_batches", [1, 2])
def test_amp_overflow_counts_only_updates_and_rejects_empty_training(
    tmp_path: Path, monkeypatch, overflow_batches: int,
) -> None:
    # Exercise the real scaler's skipped-step behavior on CPU without a GPU job.
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: scaler)
    model = TinyVideoClassifier()
    original_weight = model.classifier.weight.detach().clone()
    backward_calls = 0

    def overflow(gradient):
        nonlocal backward_calls
        backward_calls += 1
        return torch.full_like(gradient, float("inf")) if backward_calls <= overflow_batches else gradient

    model.classifier.weight.register_hook(overflow)
    arguments = {
        "model": model,
        "train_dataset": TinyEventDataset("train"),
        "validation_dataset": TinyEventDataset("val"),
        "output_dir": tmp_path,
        "options": TrainingOptions(
            epochs=1, batch_size=2, max_train_batches=2, max_validation_batches=1,
        ),
        "device": torch.device("cpu"),
        "resume_contract": {"model": "tiny"},
    }
    if overflow_batches == 2:
        with pytest.raises(RuntimeError, match="No optimizer updates"):
            train_model(**arguments)
        report = json.loads((tmp_path / "run.json").read_text())
        assert report["status"] == "failed"
        assert report["global_step"] == 0
        assert torch.equal(model.classifier.weight, original_weight)
        assert not (tmp_path / "checkpoint_last.pt").exists()
    else:
        report = train_model(**arguments)
        assert report["status"] == "complete"
        assert report["global_step"] == 1
        assert not torch.equal(model.classifier.weight, original_weight)
        checkpoint = torch.load(tmp_path / "checkpoint_last.pt", weights_only=False)
        assert {int(s["step"]) for s in checkpoint["optimizer_state"]["state"].values()} == {1}
        assert checkpoint["history"][0]["skipped_optimizer_steps"] == 1
    assert report["optimizer_steps"] == 2 - overflow_batches
    assert report["skipped_optimizer_steps"] == overflow_batches


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


def test_training_passes_frame_masks_to_mask_aware_models(tmp_path: Path) -> None:
    model = MaskAwareVideoClassifier()

    report = train_model(
        model=model,
        train_dataset=TinyEventDataset("train"),
        validation_dataset=TinyEventDataset("val"),
        output_dir=tmp_path / "mask-aware",
        options=_options(epochs=1),
        device=torch.device("cpu"),
        resume_contract={"model": "mask-aware"},
    )

    assert report["status"] == "complete"
    assert model.mask_calls == 4


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
