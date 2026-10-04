from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.training.train import (
    TrainingOptions,
    _balanced_class_weights,
    _forward_logits,
    train_model,
)
from seepat.training.video_objective import (
    VIDEO_SUPERVISION_CONTRACT,
    scored_video_bags,
    select_video_bags,
    train_video_epoch,
    video_bags,
    video_loader,
)


class BagDataset:
    def __init__(self, prefix):
        self.rows, self.samples = [], []
        # A manipulated video can legitimately have no retained fake event label.
        for identifier, label, values in (
            ("real", 0, [-1.0, -0.2]),
            ("context", 1, [0.1, 0.9, 0.3]),
            ("positive", 1, [-0.5]),
        ):
            for index, value in enumerate(values):
                event_id = f"{prefix}-{identifier}-{index}"
                event_label = int(identifier == "positive")
                video_id = f"{prefix}-{identifier}"
                self.rows.append(
                    {
                        "event_id": event_id,
                        "video_id": video_id,
                        "source_group": video_id,
                        "class_id": str(event_label),
                        "manipulation_modality": "real" if label == 0 else "both_modified",
                    }
                )
                self.samples.append(
                    {
                        "event_id": event_id,
                        "video_id": video_id,
                        "video": torch.full((3, 1, 2, 2), value),
                        "label": torch.tensor(event_label),
                        "video_label": torch.tensor(label),
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.samples[index]


class BagClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, video):
        return self.classifier(video.mean(dim=(1, 2, 3, 4))[:, None])


def options(**changes):
    return replace(
        TrainingOptions(
            amp=False,
            freeze_backbone=True,
            supervision="video_max",
            class_weighting="balanced_global",
            selection_metric="validation_video_balanced_accuracy",
            gradient_accumulation_steps=2,
            weight_decay=0,
        ),
        **changes,
    )


def test_complete_bag_sampling_weights_and_legacy_targets():
    dataset = BagDataset("train")
    before = [row.copy() for row in dataset.rows]
    bags = video_bags(dataset.rows)
    assert [len(bag.indices) for bag in bags] == [2, 3, 1]
    selected = select_video_bags(bags, 19, True, 2)
    assert selected == select_video_bags(bags, 19, True, 2)
    assert {bag.label for bag in selected} == {0, 1}
    loader = video_loader(dataset, selected, 0, torch.device("cpu"), 19)
    assert len(loader) == sum(len(bag.indices) for bag in selected)
    weights = _balanced_class_weights(
        [{"class_id": str(bag.label)} for bag in bags], torch.device("cpu")
    )
    torch.testing.assert_close(weights, torch.tensor([1.5, 0.75]))
    assert dataset.rows == before


@pytest.mark.parametrize("problem", ["duplicate", "source", "label", "missing_video"])
def test_video_bags_reject_inconsistent_membership(problem):
    rows = BagDataset("train").rows
    if problem == "duplicate":
        rows[1]["event_id"] = rows[0]["event_id"]
    elif problem == "source":
        rows[1]["source_group"] = "other"
    elif problem == "label":
        rows[1]["manipulation_modality"] = "both_modified"
    else:
        rows[0]["video_id"] = ""
    with pytest.raises(ValueError):
        video_bags(rows)


def test_streamed_maximum_matches_complete_bag_loss_and_gradients():
    dataset = BagDataset("train")
    bags = video_bags(dataset.rows)
    models = [BagClassifier(), BagClassifier()]
    models[1].load_state_dict(models[0].state_dict())
    weights = torch.tensor([1.5, 0.75])
    streamed = list(
        scored_video_bags(
            models[0],
            video_loader(dataset, bags, 0, torch.device("cpu"), 1),
            bags,
            torch.device("cpu"),
            _forward_logits,
        )
    )
    losses = []
    for model_index, model in enumerate(models):
        loss = torch.tensor(0.0)
        for index, bag in enumerate(bags):
            if model_index == 0:
                logits = streamed[index]["logits"]
            else:
                inputs = torch.stack([dataset[position]["video"] for position in bag.indices])
                all_logits = model(inputs)
                winner = (all_logits[:, 1] - all_logits[:, 0]).argmax()
                logits = all_logits[winner : winner + 1]
            loss = loss + nn.functional.cross_entropy(
                logits, torch.tensor([bag.label]), weight=weights, reduction="sum"
            ) / len(bags)
        losses.append(loss)
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1])
    for first, second in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)


@pytest.mark.parametrize("second_logits,expected", [([20.0, 20.5], 0), ([30.0, 31.0], 0)])
def test_margin_not_raw_positive_logit_and_ties_keep_first(second_logits, expected):
    dataset = BagDataset("train")
    bag = video_bags(dataset.rows)[0]

    class Lookup(nn.Module):
        def __init__(self):
            super().__init__()
            self.values = nn.Parameter(torch.tensor([[4.0, 5.0], second_logits]))
            self.calls = 0

        def forward(self, _video):
            value = self.values[self.calls : self.calls + 1]
            self.calls += 1
            return value

    model = Lookup()
    result = next(
        scored_video_bags(
            model,
            video_loader(dataset, [bag], 0, torch.device("cpu"), 1),
            [bag],
            torch.device("cpu"),
            _forward_logits,
        )
    )
    torch.testing.assert_close(result["logits"], model.values[expected : expected + 1])
    result["logits"].sum().backward()
    assert model.values.grad[1].abs().sum().item() == 0


def test_partial_and_wrong_event_bags_fail():
    dataset = BagDataset("train")
    bag = video_bags(dataset.rows)[0]
    for indices, message in (([0], "Incomplete"), ([0, 0], "membership")):
        with pytest.raises(ValueError, match=message):
            list(
                scored_video_bags(
                    BagClassifier(),
                    DataLoader(Subset(dataset, indices), batch_size=1),
                    [bag],
                    torch.device("cpu"),
                    _forward_logits,
                )
            )


def test_video_training_short_accumulation_tail_resume_and_supervision_guard(tmp_path):
    common = {
        "train_dataset": BagDataset("train"),
        "validation_dataset": BagDataset("val"),
        "device": torch.device("cpu"),
        "resume_contract": {"model": "tiny"},
    }
    states = []
    for split in (False, True):
        directory = tmp_path / str(split)
        torch.manual_seed(29)
        model = BagClassifier()
        if split:
            train_model(model, output_dir=directory, options=options(epochs=1), **common)
            resume = directory / "checkpoint_last.pt"
        else:
            resume = None
        run = train_model(
            model, output_dir=directory, options=options(epochs=2), resume_from=resume, **common
        )
        checkpoint = torch.load(directory / "checkpoint_last.pt", weights_only=False)
        states.append(checkpoint["model_state"])
        assert run["global_step"] == 4 and run["skipped_optimizer_steps"] == 0
        assert run["class_weights"] == [1.5, 0.75]
        assert checkpoint["resume_contract"]["supervision"] == VIDEO_SUPERVISION_CONTRACT
        for epoch in checkpoint["history"]:
            assert epoch["optimizer_steps"] == 2
            for section in ("train", "validation"):
                assert epoch[section]["loss_unit"] == "video"
                assert epoch[section]["videos_processed"] == 3
                assert epoch[section]["events_processed"] == 6
                assert epoch[section]["complete_video_bags"] is True
        with pytest.raises(ValueError, match="does not match"):
            train_model(
                BagClassifier(),
                output_dir=directory,
                options=replace(options(epochs=3), supervision="event"),
                resume_from=directory / "checkpoint_last.pt",
                **common,
            )
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "change",
    [
        {"amp": True},
        {"batch_size": 2},
        {"freeze_backbone": False},
        {"positive_class_weight_ratio": 10},
        {"loss_function": "focal"},
        {"max_train_batches": 1, "max_validation_batches": 1},
    ],
)
def test_unverified_video_combinations_are_rejected(change):
    with pytest.raises(ValueError):
        options(**change).validate()


def test_real_fusion_video_readiness_resume_skip_and_complete_bag_gate(tmp_path, monkeypatch):
    from test_fusion_model import _tiny_model
    from test_fusion_training import calibrated_inputs

    from seepat.training.readiness import audit_inputs, check_readiness
    from seepat.training.train import FUSION_MODEL
    from seepat.workflow import (
        ModelTrainingJob,
        model_training_outputs_are_current,
        run_model_training_job,
    )

    paths = calibrated_inputs(tmp_path)
    monkeypatch.setattr(
        "seepat.training.train.build_event_classifier",
        lambda **kwargs: _tiny_model(freeze_backbone=True),
    )
    settings = options(
        epochs=1, sequence_length=4, image_size=8, max_train_batches=2, max_validation_batches=2
    )
    job = ModelTrainingJob(
        "video",
        paths["train"],
        paths["val"],
        tmp_path / "model",
        tmp_path,
        "cpu",
        False,
        asdict(settings),
        FUSION_MODEL,
    )
    assert audit_inputs(job)["inputs"]["train"]["video_class_counts"] == {"0": 1, "1": 1}
    assert run_model_training_job(job)["action"] == "ran"
    assert check_readiness(job)["status"] == "pending"
    job = replace(job, options=asdict(replace(settings, epochs=2)))
    assert run_model_training_job(job)["action"] == "resumed"
    assert check_readiness(job)["status"] == "passed"
    assert run_model_training_job(job)["action"] == "skipped"
    assert model_training_outputs_are_current(job)
    checkpoint = job.output_dir / "checkpoint_last.pt"
    checkpoint_hash = file_sha256(checkpoint)
    history_path = job.output_dir / "history.json"
    history = json.loads(history_path.read_text())
    history[0]["train"]["video_bags"][0]["events"] += 1
    atomic_write_json(history_path, history)
    rejected = check_readiness(job)
    assert rejected["status"] == "pending" and "complete bags" in rejected["reason"]
    assert file_sha256(checkpoint) == checkpoint_hash
    history[0]["train"]["video_bags"][0]["events"] -= 1
    atomic_write_json(history_path, history)
    best_path = job.output_dir / "checkpoint_best.pt"
    selected = torch.load(best_path, weights_only=False)
    selected["resume_contract"]["supervision"]["loss_unit"] = "event"
    torch.save(selected, best_path)
    rejected = check_readiness(job)
    assert rejected["status"] == "pending" and "selected checkpoint" in rejected["reason"]


def test_phase3_profiles_are_one_fit_and_matching_complete_video_readiness(monkeypatch):
    from pathlib import Path

    from seepat.workflow import load_workflow_settings

    base = Path("configs")
    ready = load_workflow_settings(base / "fusion_phase3_readiness.yaml")
    full = load_workflow_settings(base / "fusion_phase3.yaml")
    assert len(ready.model_training_jobs) == len(full.model_training_jobs) == 1
    first, second = ready.model_training_jobs[0], full.model_training_jobs[0]
    assert first.initialize_from == second.initialize_from
    assert first.output_dir == second.readiness_dir != second.output_dir
    left, right = (
        asdict(TrainingOptions(**first.options)),
        asdict(TrainingOptions(**second.options)),
    )
    assert left["max_train_batches"] == 16 and left["max_validation_batches"] == 4
    assert right["epochs"] == 10 and right["max_train_batches"] is None
    for key in ("epochs", "max_train_batches", "max_validation_batches"):
        left.pop(key)
        right.pop(key)
    assert left == right and left["supervision"] == "video_max"
    assert left["positive_class_weight_ratio"] is None and not left["amp"]
    outer = load_workflow_settings(base / "fusion_phase3_outer_evaluation.yaml")
    assert len(outer.decision_jobs) == 2
    assert outer.decision_jobs[0].validation_manifest == outer.decision_jobs[1].validation_manifest
    assert all(job.threshold_artifact for job in outer.decision_jobs)
    cohort = load_workflow_settings(base / "phase3_cohort.yaml")
    assert not cohort.model_training_jobs and not cohort.decision_jobs
    from seepat import cli

    purposes = []

    def sample(**kwargs):
        purposes.append(kwargs["purpose"])
        return [], {"purpose": kwargs["purpose"]}

    monkeypatch.setattr(cli, "sample_training_canary", sample)
    arguments = [
        "seepat-data",
        "sample-disjoint-cohort",
        "--database",
        "synthetic.sqlite",
        "--output",
        "synthetic.csv",
        "--summary",
        "synthetic.json",
        "--exclude-manifest",
        "excluded.csv",
    ]
    for extra in ([], ["--purpose", "phase3_outer_development_cohort"]):
        monkeypatch.setattr("sys.argv", arguments + extra)
        cli.main()
    assert purposes == ["phase2_outer_development_cohort", "phase3_outer_development_cohort"]


def test_video_decisions_bind_supervision_and_explanations_do_not_claim_localization(
    tmp_path, monkeypatch
):
    from test_verdict import _event_row, _patch_builder, _seal_manifest, _write_checkpoint

    from seepat.artifacts import atomic_write_csv
    from seepat.decision import DecisionJob, decision_stage_status, run_decision_job
    from seepat.xai import build_evidence_bundle, build_prompts, render_evidence_summary

    _patch_builder(monkeypatch)
    manifest = tmp_path / "events.csv"
    atomic_write_csv(
        manifest,
        [
            _event_row(tmp_path, "val/a", "a", class_id="0", modality="real"),
            _event_row(tmp_path, "val/b", "b", class_id="0", modality="both_modified"),
        ],
    )
    _seal_manifest(manifest)
    checkpoint = tmp_path / "checkpoint_best.pt"
    _write_checkpoint(checkpoint, manifest)
    model = torch.load(checkpoint, weights_only=False)
    model["options"].update(supervision="video_max", seed=19)
    model["resume_contract"].update(supervision=dict(VIDEO_SUPERVISION_CONTRACT), seed=19)
    torch.save(model, checkpoint)
    run_path = checkpoint.parent / "run.json"
    run = json.loads(run_path.read_text())
    run["resume_contract"] = model["resume_contract"]
    atomic_write_json(run_path, run)
    job = DecisionJob(
        "video-aware",
        manifest,
        tmp_path / "decision",
        checkpoint,
        project_root=tmp_path,
        device="cpu",
    )
    run_decision_job(job)
    assert set(decision_stage_status(job).values()) == {"current"}
    evaluation = json.loads((job.output_dir / "verdict/evaluation.json").read_text())
    assert evaluation["supervision"] == VIDEO_SUPERVISION_CONTRACT
    assert evaluation["provenance"]["supervision"] == VIDEO_SUPERVISION_CONTRACT
    assert "diagnostic" in evaluation["event_metrics_scope"]
    bundle = build_evidence_bundle(job.output_dir / "verdict")
    text = render_evidence_summary(bundle)
    assert "not a verified manipulated interval" in text and "video evidence score=" in text
    assert "p(manipulated)=" not in text
    assert "Do not claim temporal localization" in build_prompts(bundle)[0]
    model["resume_contract"].pop("supervision")
    torch.save(model, checkpoint)
    with pytest.raises(ValueError, match="supervision"):
        run_decision_job(job)


def test_interrupted_video_epoch_replays_from_last_completed_checkpoint(tmp_path):
    common = {
        "train_dataset": BagDataset("train"),
        "validation_dataset": BagDataset("val"),
        "output_dir": tmp_path,
        "device": torch.device("cpu"),
        "resume_contract": {"model": "tiny"},
    }
    train_model(BagClassifier(), options=options(epochs=1), **common)
    last = tmp_path / "checkpoint_last.pt"
    saved_hash = file_sha256(last)

    class Interrupted(BagClassifier):
        def forward(self, video):
            raise RuntimeError("synthetic interruption inside video bag")

    with pytest.raises(RuntimeError, match="interruption"):
        train_model(Interrupted(), options=options(epochs=2), resume_from=last, **common)
    assert file_sha256(last) == saved_hash
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "failed"
    result = train_model(BagClassifier(), options=options(epochs=2), resume_from=last, **common)
    assert result["global_step"] == 4 and result["completed_epochs"] == 2
    with pytest.raises(ValueError, match="does not match"):
        train_model(BagClassifier(), options=options(epochs=3, seed=73), resume_from=last, **common)


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_video_accumulation_matches_group_means_including_short_tail(accumulation):
    dataset = BagDataset("train")
    bags = video_bags(dataset.rows)
    device = torch.device("cpu")
    models = [BagClassifier(), BagClassifier()]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in models]
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 0.75]), reduction="none")
    report = train_video_epoch(
        models[0],
        video_loader(dataset, bags, 0, device, 19),
        bags,
        device,
        _forward_logits,
        loss_function,
        optimizers[0],
        accumulation,
        list(models[0].parameters()),
    )
    for start in range(0, len(bags), accumulation):
        group = bags[start : start + accumulation]
        losses = []
        for bag in group:
            clips = torch.stack([dataset[index]["video"] for index in bag.indices])
            logits = models[1](clips)
            winner = (logits[:, 1] - logits[:, 0]).argmax()
            losses.append(
                loss_function(logits[winner : winner + 1], torch.tensor([bag.label])).mean()
            )
        torch.stack(losses).mean().backward()
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key])
    assert report["optimizer_steps"] == (len(bags) + accumulation - 1) // accumulation


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_video_logits_fail_before_update(value):
    dataset = BagDataset("train")
    bag = video_bags(dataset.rows)[0]
    with pytest.raises(FloatingPointError, match="finite binary event logits"):
        list(
            scored_video_bags(
                BagClassifier(),
                video_loader(dataset, [bag], 0, torch.device("cpu"), 1),
                [bag],
                torch.device("cpu"),
                lambda *_args: torch.tensor([[0.0, value]]),
            )
        )
