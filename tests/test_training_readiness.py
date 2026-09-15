from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_train import TinyEventDataset, TinyVideoClassifier

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256
from seepat.live_progress import WorkflowProgress
from seepat.progress import read_live_workflow_progress
from seepat.training.readiness import audit_inputs, check_readiness, require_readiness
from seepat.training.train import (
    SWIN_BASE_MODEL,
    TrainingOptions,
    model_contract_name,
    train_model,
    training_version_for_model,
)
from seepat.workflow import ModelTrainingJob, load_workflow_settings, run_model_training_job

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_share_experiment_settings_and_preserve_historical_outputs():
    ready = load_workflow_settings(ROOT / "configs/training_readiness.yaml")
    experiments = load_workflow_settings(ROOT / "configs/training_experiments.yaml")
    historical = load_workflow_settings(ROOT / "configs/workflow_scaled.yaml")
    historical_paths = {job.output_dir for job in historical.model_training_jobs}
    assert historical.model_training_config is None
    assert len(ready.model_training_jobs) == len(experiments.model_training_jobs) == 3
    assert len({job.model for job in experiments.model_training_jobs}) == 3
    for preflight, experiment in zip(ready.model_training_jobs, experiments.model_training_jobs, strict=True):
        left, right = TrainingOptions(**preflight.options), TrainingOptions(**experiment.options)
        left.validate()
        right.validate()
        assert left.max_train_batches == 16 and left.max_validation_batches == 8
        assert right.max_train_batches is right.max_validation_batches is None
        assert right.epochs == 10 and left.epochs == 1
        assert replace(left, epochs=right.epochs, max_train_batches=None, max_validation_batches=None) == right
        assert preflight.model == experiment.model
        assert preflight.pretrained is experiment.pretrained is True
        assert left.class_weighting == "balanced_global" and left.freeze_backbone and not left.amp
        assert preflight.train_manifest == experiment.train_manifest
        assert preflight.validation_manifest == experiment.validation_manifest
        assert experiment.readiness_dir == preflight.output_dir
        assert preflight.output_dir != experiment.output_dir
        assert not ({preflight.output_dir, experiment.output_dir} & historical_paths)


def test_profile_selection_and_live_tracker_notice_profile_changes(tmp_path):
    profile = tmp_path / "training.yaml"
    profile.write_text((ROOT / "configs/training_readiness.yaml").read_text(), encoding="utf-8")
    config = tmp_path / "workflow.yaml"
    report = tmp_path / "report.json"
    config.write_text(f"model_training_config: training.yaml\nreport: {report.as_posix()}\n", encoding="utf-8")
    settings = load_workflow_settings(config)
    assert settings.model_training_config == profile
    progress = WorkflowProgress(config, report, 3, model_training_config=profile)
    progress.start_job(1, "readiness-swin")
    assert read_live_workflow_progress(config)["status"] == "running"
    profile.write_text(profile.read_text().replace("epochs: 1", "epochs: 2"), encoding="utf-8")
    assert read_live_workflow_progress(config)["status"] == "unavailable"
    assert load_workflow_settings(config).model_training_jobs[0].options["epochs"] == 2


@pytest.mark.parametrize("profile_text", ["model_training: []", "model_training_config: other.yaml"])
def test_invalid_profiles_fail_before_execution(tmp_path, profile_text):
    (tmp_path / "training.yaml").write_text(profile_text, encoding="utf-8")
    path = tmp_path / "workflow.yaml"
    path.write_text("model_training_config: training.yaml", encoding="utf-8")
    with pytest.raises(ValueError):
        load_workflow_settings(path)


def _job_and_inputs(tmp_path):
    paths = {}
    for split in ("train", "val"):
        rows = []
        for label in (0, 1):
            clip = tmp_path / f"{split}-{label}.mp4"
            clip.write_bytes(b"not a real video: input audit must not decode")
            rows.append({"event_id": f"{split}-{label}", "video_id": f"{split}-{label}",
                         "dataset_split": split, "source_group": f"{split}-{label}",
                         "class_id": str(label), "phoneme": "m", "mouth_clip_path": str(clip),
                         "manipulation_modality": "real" if label == 0 else "visual_modified"})
        paths[split] = tmp_path / f"{split}.csv"
        atomic_write_csv(paths[split], rows)
    options = TrainingOptions(epochs=2, batch_size=1, sequence_length=1, image_size=2,
                              amp=False, class_weighting="balanced_global", learning_rate=0.01,
                              max_train_batches=4, max_validation_batches=4)
    job = ModelTrainingJob("readiness", paths["train"], paths["val"], tmp_path / "ready",
                           tmp_path, "cpu", True, asdict(options))
    return job, options


def test_input_audit_is_read_only_and_rejects_foreign_rows(tmp_path):
    job, _ = _job_and_inputs(tmp_path)
    before = file_sha256(job.train_manifest)
    assert audit_inputs(job)["inputs"]["train"]["events"] == 2
    assert not job.output_dir.exists() and file_sha256(job.train_manifest) == before
    text = job.train_manifest.read_text().replace("train,", "test,")
    job.train_manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="exclusively AV"):
        audit_inputs(job)


@pytest.fixture
def completed_readiness(tmp_path):
    job, options = _job_and_inputs(tmp_path)
    contract = {"training_version": training_version_for_model(SWIN_BASE_MODEL),
                "model": model_contract_name(SWIN_BASE_MODEL), "model_name": SWIN_BASE_MODEL,
                "pretrained": True, "train_manifest_sha256": file_sha256(job.train_manifest),
                "validation_manifest_sha256": file_sha256(job.validation_manifest)}
    common = {"train_dataset": TinyEventDataset("train"), "validation_dataset": TinyEventDataset("val"),
              "output_dir": job.output_dir, "device": torch.device("cpu"), "resume_contract": contract}
    train_model(model=TinyVideoClassifier(), options=replace(options, epochs=1), **common)
    assert check_readiness(job)["status"] == "pending"
    train_model(model=TinyVideoClassifier(), options=options,
                resume_from=job.output_dir / "checkpoint_last.pt", **common)
    experiment = replace(job, name="experiment", output_dir=tmp_path / "experiment",
                         readiness_dir=job.output_dir,
                         options=asdict(replace(options, epochs=10, max_train_batches=None, max_validation_batches=None)))
    return experiment


def test_resumed_readiness_checks_saved_state_and_leaves_it_unchanged(completed_readiness):
    job = completed_readiness
    checkpoint = job.readiness_dir / "checkpoint_last.pt"
    before = file_sha256(checkpoint)
    report = check_readiness(job)
    assert report["status"] == "passed", report
    assert report["optimizer_updates"] == 8
    require_readiness(job)
    assert file_sha256(checkpoint) == before
    assert not job.output_dir.exists()


@pytest.mark.parametrize("change", ["empty_optimizer", "nan_weight", "skipped_step", "changed_input", "changed_options"])
def test_readiness_rejects_invalid_or_mismatched_runs(completed_readiness, change):
    job = completed_readiness
    checkpoint_path = job.readiness_dir / "checkpoint_last.pt"
    if change in {"empty_optimizer", "nan_weight"}:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if change == "empty_optimizer":
            checkpoint["optimizer_state"]["state"] = {}
        else:
            next(iter(checkpoint["model_state"].values())).fill_(float("nan"))
        torch.save(checkpoint, checkpoint_path)
    elif change == "skipped_step":
        path = job.readiness_dir / "history.json"
        history = json.loads(path.read_text())
        history[0]["skipped_optimizer_steps"] = 1
        atomic_write_json(path, history)
    elif change == "changed_input":
        job.train_manifest.write_text(job.train_manifest.read_text() + "\n", encoding="utf-8")
    else:
        job = replace(job, options={**job.options, "learning_rate": 0.05})
    assert check_readiness(job)["status"] == "pending"
    with pytest.raises(RuntimeError, match="not accepted"):
        require_readiness(job)


def test_full_run_cannot_initialize_model_without_readiness(tmp_path, monkeypatch):
    job, options = _job_and_inputs(tmp_path)
    job = replace(job, readiness_dir=tmp_path / "missing", options=asdict(replace(
        options, max_train_batches=None, max_validation_batches=None)))
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before downloads or model initialization")
    monkeypatch.setattr("seepat.workflow._run_model_training", forbidden)
    with pytest.raises(RuntimeError, match="not accepted"):
        run_model_training_job(job)
    assert not job.output_dir.exists()


def test_recorded_gpu_change_requires_new_readiness(completed_readiness):
    path = completed_readiness.readiness_dir / "run.json"
    run = json.loads(path.read_text())
    run["environment"]["gpu"] = "another GPU"
    atomic_write_json(path, run)
    with pytest.raises(RuntimeError, match="selected training hardware"):
        require_readiness(completed_readiness)
