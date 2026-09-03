from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_json, stable_id
from seepat.config import load_pipeline_settings
from seepat.pipeline import PIPELINE_VERSION
from seepat.workflow import (
    ModelTrainingJob,
    WorkflowJob,
    WorkflowSettings,
    _sha256,
    load_workflow_settings,
    model_training_outputs_are_current,
    preprocessing_outputs_are_current,
    run_model_training_job,
    run_workflow,
    run_workflow_job,
    training_outputs_are_current,
)


def _write_pipeline_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_manifest = tmp_path / "source.csv"
    extracted_root = tmp_path / "extracted"
    output_dir = tmp_path / "outputs"
    atomic_write_csv(source_manifest, [{"file": "video.mp4"}])
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        f"""
dataset:
  split: train
  pilot_manifest: {source_manifest.as_posix()}
  extracted_root: {extracted_root.as_posix()}
preprocessing:
  output_dir: {output_dir.as_posix()}
  ffmpeg_path: null
  ffprobe_path: null
  whisper_model: base.en
  whisper_device: cpu
  whisper_compute_type: int8
  mfa_docker_image: mfa:test
  mfa_dictionary: dictionary
  mfa_acoustic_model: acoustic
  mfa_cache_dir: {(tmp_path / 'mfa').as_posix()}
  event_window_before_s: 0.2
  event_window_after_s: 0.2
  manipulation_boundary_tolerance_s: 0.04
  min_valid_landmark_ratio: 0.8
  min_bilabial_events: 1
  max_debug_overlays_per_video: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path, source_manifest, output_dir


def test_load_workflow_settings_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """
jobs:
  - name: duplicate
    pipeline_config: first.yaml
    manifest_output_dir: first
  - name: duplicate
    pipeline_config: second.yaml
    manifest_output_dir: second
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        load_workflow_settings(path)
    except ValueError as error:
        assert "Duplicate workflow job name" in str(error)
    else:
        raise AssertionError("Duplicate job names must be rejected")


def test_load_workflow_settings_accepts_multiple_model_jobs(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(
        """
jobs:
  - name: preparation
    pipeline_config: pipeline.yaml
    manifest_output_dir: manifests
model_training:
  - name: swin
    model: swin3d_b
    train_manifest: train.csv
    validation_manifest: val.csv
    output_dir: swin-output
  - name: cnn
    model: efficientnet_v2_s_tempcnn
    train_manifest: train.csv
    validation_manifest: val.csv
    output_dir: cnn-output
    pretrained: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = load_workflow_settings(path)

    assert [job.name for job in settings.model_training_jobs] == ["swin", "cnn"]
    assert [job.model for job in settings.model_training_jobs] == [
        "swin3d_b",
        "efficientnet_v2_s_tempcnn",
    ]
    assert settings.model_training_jobs[1].pretrained is False


def test_preprocessing_current_check_detects_manifest_change(tmp_path: Path) -> None:
    config_path, source_manifest, output_dir = _write_pipeline_config(tmp_path)
    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)

    atomic_write_csv(
        output_dir / "video_manifest.csv",
        [{"file": "video.mp4", "video_id": stable_id("video.mp4")}],
    )
    atomic_write_csv(output_dir / "bilabial_events.csv", [{"event_id": "event-1"}])
    atomic_write_csv(output_dir / "eligibility_report.csv", [{"video_id": "video-1"}])
    atomic_write_json(
        output_dir / "run_summary.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "cache_signature": settings.cache_signature,
            "videos_requested": 1,
        },
    )

    assert preprocessing_outputs_are_current(settings)
    atomic_write_csv(source_manifest, [{"file": "different.mp4"}])
    assert not preprocessing_outputs_are_current(settings)


def test_training_current_check_validates_hashes(tmp_path: Path) -> None:
    config_path, source_manifest, preprocessing_dir = _write_pipeline_config(tmp_path)
    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)
    video_manifest = preprocessing_dir / "video_manifest.csv"
    event_manifest = preprocessing_dir / "bilabial_events.csv"
    atomic_write_csv(video_manifest, [{"video_id": "video-1"}])
    atomic_write_csv(event_manifest, [{"event_id": "event-1"}])

    output_dir = tmp_path / "training"
    combined = output_dir / "events.csv"
    split = output_dir / "events_train.csv"
    atomic_write_csv(combined, [{"event_id": "event-1"}])
    atomic_write_csv(split, [{"event_id": "event-1"}])
    atomic_write_json(
        output_dir / "summary.json",
        {
            "input_artifacts": {
                "source_manifest": {"sha256": _sha256(source_manifest)},
                "video_manifest": {"sha256": _sha256(video_manifest)},
                "event_manifest": {"sha256": _sha256(event_manifest)},
            },
            "combined_manifest_sha256": _sha256(combined),
            "split_manifests": {"train": split.as_posix()},
            "split_manifest_sha256": {"train": _sha256(split)},
        },
    )

    assert training_outputs_are_current(settings, output_dir)
    atomic_write_csv(event_manifest, [{"event_id": "event-2"}])
    assert not training_outputs_are_current(settings, output_dir)


def test_workflow_job_runs_only_missing_stages(tmp_path: Path, monkeypatch) -> None:
    config_path, _, output_dir = _write_pipeline_config(tmp_path)
    training_dir = tmp_path / "training"
    calls: list[str] = []

    monkeypatch.setattr(
        "seepat.workflow.preprocessing_outputs_are_current",
        lambda settings: False,
    )
    monkeypatch.setattr(
        "seepat.workflow.training_outputs_are_current",
        lambda settings, path: False,
    )

    def fake_pipeline(path: Path, retry_failed: bool = False) -> dict[str, object]:
        calls.append("preprocessing")
        return {"videos_requested": 1}

    def fake_manifest(**kwargs) -> dict[str, object]:
        calls.append("manifest")
        return {"written_events": 1}

    monkeypatch.setattr("seepat.workflow.run_pipeline", fake_pipeline)
    monkeypatch.setattr("seepat.workflow.prepare_training_manifests", fake_manifest)

    report = run_workflow_job(
        WorkflowJob("test", config_path, training_dir),
    )

    assert calls == ["preprocessing", "manifest"]
    assert report["preprocessing"]["action"] == "ran"
    assert report["training_manifest"]["action"] == "built"
    assert output_dir == load_pipeline_settings(config_path, PIPELINE_VERSION).preprocessing.output_dir


def test_workflow_job_skips_current_stages(tmp_path: Path, monkeypatch) -> None:
    config_path, source_manifest, preprocessing_dir = _write_pipeline_config(tmp_path)
    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)
    video_manifest = preprocessing_dir / "video_manifest.csv"
    event_manifest = preprocessing_dir / "bilabial_events.csv"
    atomic_write_csv(
        video_manifest,
        [{"file": "video.mp4", "video_id": stable_id("video.mp4")}],
    )
    atomic_write_csv(event_manifest, [{"event_id": "event-1"}])
    atomic_write_csv(
        preprocessing_dir / "eligibility_report.csv",
        [{"video_id": stable_id("video.mp4")}],
    )
    pipeline_summary = {
        "pipeline_version": PIPELINE_VERSION,
        "cache_signature": settings.cache_signature,
        "videos_requested": 1,
    }
    atomic_write_json(preprocessing_dir / "run_summary.json", pipeline_summary)

    training_dir = tmp_path / "training"
    combined = training_dir / "events.csv"
    split = training_dir / "events_train.csv"
    atomic_write_csv(combined, [{"event_id": "event-1"}])
    atomic_write_csv(split, [{"event_id": "event-1"}])
    atomic_write_json(
        training_dir / "summary.json",
        {
            "input_artifacts": {
                "source_manifest": {"sha256": _sha256(source_manifest)},
                "video_manifest": {"sha256": _sha256(video_manifest)},
                "event_manifest": {"sha256": _sha256(event_manifest)},
            },
            "combined_manifest_sha256": _sha256(combined),
            "split_manifests": {"train": split.as_posix()},
            "split_manifest_sha256": {"train": _sha256(split)},
        },
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("A current stage must be skipped")

    monkeypatch.setattr("seepat.workflow.run_pipeline", unexpected_call)
    monkeypatch.setattr("seepat.workflow.prepare_training_manifests", unexpected_call)

    report = run_workflow_job(WorkflowJob("test", config_path, training_dir))

    assert report["preprocessing"]["action"] == "skipped"
    assert report["training_manifest"]["action"] == "skipped"


def _model_training_job(tmp_path: Path) -> ModelTrainingJob:
    train_manifest = tmp_path / "events_train.csv"
    validation_manifest = tmp_path / "events_val.csv"
    atomic_write_csv(train_manifest, [{"event_id": "train-event"}])
    atomic_write_csv(validation_manifest, [{"event_id": "val-event"}])
    return ModelTrainingJob(
        name="local-preflight",
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        output_dir=tmp_path / "model-output",
        project_root=tmp_path,
        device="cuda",
        pretrained=True,
        options={
            "epochs": 1,
            "max_train_batches": 1,
            "max_validation_batches": 1,
        },
    )


def test_model_training_current_check_validates_manifests(tmp_path: Path) -> None:
    train_module = pytest.importorskip("seepat.training.train")
    TrainingOptions = train_module.TrainingOptions
    training_version = train_module.TRAINING_VERSION
    job = _model_training_job(tmp_path)
    options = TrainingOptions(**job.options)
    job.output_dir.mkdir(parents=True)
    atomic_write_json(job.output_dir / "history.json", [{"epoch": 1}])
    (job.output_dir / "checkpoint_last.pt").write_bytes(b"checkpoint")
    (job.output_dir / "checkpoint_best.pt").write_bytes(b"checkpoint")
    atomic_write_json(
        job.output_dir / "run.json",
        {
            "status": "complete",
            "run_type": "engineering_preflight",
            "device": "cuda",
            "completed_epochs": 1,
            "options": asdict(options),
            "resume_contract": {
                "training_version": training_version,
                "model": "torchvision.swin3d_b",
                "pretrained": True,
                "train_manifest_sha256": _sha256(job.train_manifest),
                "validation_manifest_sha256": _sha256(job.validation_manifest),
            },
        },
    )

    assert model_training_outputs_are_current(job)
    atomic_write_csv(job.train_manifest, [{"event_id": "changed"}])
    assert not model_training_outputs_are_current(job)


def test_model_training_job_runs_or_skips_as_needed(tmp_path: Path, monkeypatch) -> None:
    job = _model_training_job(tmp_path)
    monkeypatch.setattr(
        "seepat.workflow.model_training_outputs_are_current",
        lambda current_job: False,
    )
    monkeypatch.setattr(
        "seepat.workflow._run_model_training",
        lambda current_job, resume_from: {"status": "complete"},
    )

    report = run_model_training_job(job)

    assert report["action"] == "ran"
    assert report["summary"] == {"status": "complete"}

    job.output_dir.mkdir(parents=True)
    atomic_write_json(job.output_dir / "run.json", {"status": "complete"})
    monkeypatch.setattr(
        "seepat.workflow.model_training_outputs_are_current",
        lambda current_job: True,
    )

    report = run_model_training_job(job)

    assert report["action"] == "skipped"


def test_model_training_job_resumes_when_epoch_target_increases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_module = pytest.importorskip("seepat.training.train")
    TrainingOptions = train_module.TrainingOptions
    training_version = train_module.TRAINING_VERSION
    first_job = _model_training_job(tmp_path)
    job = replace(first_job, options={**first_job.options, "epochs": 2})
    first_options = TrainingOptions(**first_job.options)
    job.output_dir.mkdir(parents=True)
    checkpoint_path = job.output_dir / "checkpoint_last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    atomic_write_json(
        job.output_dir / "run.json",
        {
            "status": "complete",
            "run_type": "engineering_preflight",
            "device": "cuda",
            "completed_epochs": 1,
            "options": asdict(first_options),
            "resume_contract": {
                "training_version": training_version,
                "model": "torchvision.swin3d_b",
                "pretrained": True,
                "train_manifest_sha256": _sha256(job.train_manifest),
                "validation_manifest_sha256": _sha256(job.validation_manifest),
            },
        },
    )
    received_resume_path = None

    def fake_training(current_job, resume_from):
        nonlocal received_resume_path
        received_resume_path = resume_from
        return {"status": "complete"}

    monkeypatch.setattr("seepat.workflow._run_model_training", fake_training)

    report = run_model_training_job(job)

    assert report["action"] == "resumed"
    assert received_resume_path == checkpoint_path


def test_workflow_runs_model_training_after_preparation(tmp_path: Path, monkeypatch) -> None:
    pipeline_config, _, _ = _write_pipeline_config(tmp_path)
    preparation_job = WorkflowJob("prepare", pipeline_config, tmp_path / "manifests")
    model_job = _model_training_job(tmp_path)
    cnn_job = replace(
        model_job,
        name="local-cnn-preflight",
        model="efficientnet_v2_s_tempcnn",
        output_dir=tmp_path / "cnn-output",
        pretrained=False,
    )
    settings = WorkflowSettings(
        jobs=(preparation_job,),
        model_training_jobs=(model_job, cnn_job),
        report_path=tmp_path / "workflow-report.json",
    )
    calls: list[str] = []

    monkeypatch.setattr("seepat.workflow.load_workflow_settings", lambda path: settings)
    monkeypatch.setattr(
        "seepat.workflow.run_workflow_job",
        lambda job, retry_failed=False: calls.append("preparation") or {"name": job.name},
    )
    monkeypatch.setattr(
        "seepat.workflow.run_model_training_job",
        lambda job: calls.append(f"model:{job.name}") or {"name": job.name},
    )

    report = run_workflow(tmp_path / "workflow.yaml")

    assert calls == [
        "preparation",
        "model:local-preflight",
        "model:local-cnn-preflight",
    ]
    assert report["model_training"] == [
        {"name": "local-preflight"},
        {"name": "local-cnn-preflight"},
    ]
