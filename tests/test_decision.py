from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from test_verdict import _event_row, _patch_builder, _seal_manifest, _write_checkpoint

from seepat import verdict
from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.decision import DecisionJob, decision_stage_status, run_decision_job
from seepat.workflow import load_workflow_settings, run_workflow
from seepat.xai import build_evidence_bundle, generate_forensic_trace


@pytest.fixture
def job(tmp_path, monkeypatch):
    _patch_builder(monkeypatch)
    rows = [
        _event_row(tmp_path, "val/a.mp4", "a1", class_id="0", modality="real"),
        _event_row(tmp_path, "val/a.mp4", "a2", class_id="0", modality="real"),
        _event_row(tmp_path, "val/b.mp4", "b1", class_id="1", modality="both_modified"),
    ]
    # Two missing fields on one event must count as one affected event, not two.
    rows[0]["vild_regression_residual_px"] = ""
    rows[0]["phoneme_viseme_residual_z"] = ""
    manifest = tmp_path / "events_val.csv"
    atomic_write_csv(manifest, rows)
    _seal_manifest(manifest)
    checkpoint = tmp_path / "checkpoint_best.pt"
    _write_checkpoint(checkpoint, manifest)
    source = tmp_path / "source.csv"
    atomic_write_csv(source, [{"file": f"val/{name}.mp4"} for name in ("a", "b", "no-events")])
    return DecisionJob("validation", manifest, tmp_path / "decisions", checkpoint, source, tmp_path, "cpu")


@pytest.mark.parametrize("split,source", [("test", verdict.VALIDATION_SOURCE), ("external-test", verdict.VALIDATION_SOURCE), ("train", verdict.VALIDATION_SOURCE), ("val", "Deepfake-Eval-2024")])
def test_threshold_rejects_foreign_scope_before_reading_data(tmp_path, split, source):
    with pytest.raises(ValueError, match="Validation"):
        verdict.select_threshold(tmp_path / "absent.csv", tmp_path / "absent.pt", tmp_path / "out.json", split=split, dataset_source=source)
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("field,value", [("dataset_split", "test"), ("dataset_source", "Deepfake-Eval-2024"), ("event_id", "a2")])
def test_declared_validation_does_not_hide_wrong_rows(job, field, value):
    rows = read_csv_rows(job.validation_manifest)
    rows[0][field] = value
    atomic_write_csv(job.validation_manifest, rows)
    with pytest.raises(ValueError):
        verdict.select_threshold(job.validation_manifest, job.checkpoint, job.output_dir / "threshold.json")


def test_preflight_and_incomplete_experiments_cannot_fit_thresholds(job):
    checkpoint = torch.load(job.checkpoint, weights_only=False)
    checkpoint["options"]["max_train_batches"] = 2
    torch.save(checkpoint, job.checkpoint)
    with pytest.raises(ValueError, match="preflight checkpoints"):
        run_decision_job(job)
    assert not job.output_dir.exists()
    checkpoint["options"]["max_train_batches"] = None
    torch.save(checkpoint, job.checkpoint)
    record = json.loads((job.checkpoint.parent / "run.json").read_text())
    record["status"] = "running"
    atomic_write_json(job.checkpoint.parent / "run.json", record)
    with pytest.raises(ValueError, match="completed model experiment"):
        run_decision_job(job)


def test_threshold_ties_and_full_precision_values():
    assert verdict.choose_threshold([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])["threshold"] == 0.2
    assert verdict.choose_threshold([0, 1], [1.0, 1.0])["threshold"] == 0.0
    score = 0.123456789123
    assert verdict.choose_threshold([0, 1], [0.1, score])["threshold"] == score
    for probability in (float("nan"), float("inf"), -0.01, 1.01):
        with pytest.raises(ValueError, match="finite"):
            verdict.choose_threshold([0, 1], [0.1, probability])


@pytest.mark.parametrize("field", ["checkpoint_sha256", "validation_manifest_sha256", "evidence_version", "fusion_inputs", "aggregation", "split", "selection_metric", "tie_break"])
def test_foreign_threshold_provenance_is_rejected(job, field):
    run_decision_job(job)
    path = job.output_dir / "threshold.json"
    artifact = json.loads(path.read_text())
    target = artifact["provenance"] if field in artifact["provenance"] else artifact
    target[field] = "wrong"
    atomic_write_json(path, artifact)
    with pytest.raises(ValueError, match="provenance"):
        verdict.run_verdict(job.validation_manifest, job.checkpoint, job.output_dir / "wrong", threshold_artifact=path, split="val", device_name="cpu")
    assert not (job.output_dir / "wrong").exists()


def test_decisions_count_events_preserve_missing_masks_and_resume(job, monkeypatch):
    updates = []
    first = run_decision_job(job, progress=lambda *args: updates.append(args))
    assert first["stages"] == dict.fromkeys(("threshold", "verdict", "explanation"), "ran")
    assert updates
    assert set(decision_stage_status(job).values()) == {"current"}
    events = read_csv_rows(job.output_dir / "verdict/event_predictions.csv")
    videos = read_csv_rows(job.output_dir / "verdict/video_verdicts.csv")
    assert [row["evidence_available"] for row in events] == ["5", "7", "7"]
    assert events[0]["vild_regression_residual_px_available"] == "False"
    for video in videos:
        scored = [row for row in events if row["video_id"] == video["video_id"]]
        assert int(video["event_count"]) == len(scored)
        if scored:
            assert float(video["maximum_probability"]) == max(float(row["manipulated_probability"]) for row in scored)
            assert video["verdict"] == ("manipulated" if float(video["maximum_probability"]) >= float(video["threshold"]) else "authentic")
        else:
            assert video["verdict"] == "not_evaluated" and video["maximum_probability"] == ""
    bundle = build_evidence_bundle(job.output_dir / "verdict")
    assert (bundle["videos_scored"], bundle["videos_not_evaluated"]) == (2, 1)
    assert (bundle["events_with_missing_evidence_values"], bundle["missing_evidence_values"]) == (1, 2)
    monkeypatch.setattr(verdict, "load_evaluation_model", lambda *args: pytest.fail("Current decisions must not run inference"))
    assert run_decision_job(job)["action"] == "skipped"
    (job.output_dir / "explanation.json").write_text('{}')
    result = run_decision_job(job)
    assert result["stages"] == {"threshold": "skipped", "verdict": "skipped", "explanation": "ran"}


def test_decision_restart_reuses_threshold_after_verdict_failure(job, monkeypatch):
    real_verdict = verdict.run_verdict
    def fail(**kwargs):
        raise RuntimeError("interrupted verdict")
    monkeypatch.setattr(verdict, "run_verdict", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_decision_job(job)
    threshold_hash = file_sha256(job.output_dir / "threshold.json")
    monkeypatch.setattr(verdict, "run_verdict", real_verdict)
    assert run_decision_job(job)["stages"]["threshold"] == "skipped"
    assert threshold_hash == file_sha256(job.output_dir / "threshold.json")


def test_corrupt_verdict_rebuilds_downstream_only(job, monkeypatch):
    run_decision_job(job)
    path = job.output_dir / "verdict/event_predictions.csv"
    path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(verdict, "load_evaluation_model", lambda *args: pytest.fail("Reuse verified threshold predictions"))
    assert run_decision_job(job)["stages"] == {"threshold": "skipped", "verdict": "ran", "explanation": "ran"}


def test_explanation_cannot_write_predictions_or_apply_client_decisions(job):
    run_decision_job(job)
    directory = job.output_dir / "verdict"
    paths = [directory / name for name in ("evaluation.json", "event_predictions.csv", "video_verdicts.csv")]
    paths += [job.checkpoint, job.validation_manifest, job.source_manifest, job.output_dir / "threshold.json"]
    paths += [job.output_dir / "threshold_events.csv"]
    hashes = {path: file_sha256(path) for path in paths}
    for path in paths:
        with pytest.raises(ValueError, match="overwrite"):
            generate_forensic_trace(directory, path)
    class AdversarialClient:
        label = "test-only"
        def generate(self, system_prompt, user_prompt):
            return '{"verdict":"authentic","threshold":0,"probability":0,"eligible":true}'
    result = generate_forensic_trace(directory, job.output_dir / "advisory.json", client=AdversarialClient())
    assert result["mode"] == "reasoning-llm"
    assert {path: file_sha256(path) for path in paths} == hashes
    assert "explanatory only" in result["evidence_summary"]


def test_entirely_unscorable_manifest_reports_requested_videos_without_model(job, monkeypatch):
    atomic_write_csv(job.validation_manifest, [])
    _seal_manifest(job.validation_manifest)
    _write_checkpoint(job.checkpoint, job.validation_manifest)
    monkeypatch.setattr(verdict, "load_evaluation_model", lambda *args: pytest.fail("No events to score"))
    result = verdict.run_verdict(job.validation_manifest, job.checkpoint, job.output_dir, threshold=0.5, engineering_preflight=True, split="val", source_manifest=job.source_manifest, device_name="cpu")
    assert (result["event_count"], result["not_evaluated_videos"]) == (0, 3)
    assert result["metrics"] is None


def test_workflow_defers_without_a_checkpoint_and_rejects_test_scope(tmp_path):
    config = {"jobs": [], "report": str(tmp_path / "report.json"), "decisions": [{
        "name": "decisions", "validation_manifest": "missing.csv", "checkpoint": None,
        "output_dir": str(tmp_path / "unused"),
    }]}
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run_workflow(path)["decisions"][0]["action"] == "deferred"
    assert not (tmp_path / "unused").exists()
    config["decisions"][0]["split"] = "test"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="external/test"):
        load_workflow_settings(path)
    scaled = load_workflow_settings(Path("configs/workflow_scaled.yaml"))
    assert scaled.decision_jobs[0].checkpoint is None
    assert set(decision_stage_status(scaled.decision_jobs[0]).values()) == {"awaiting_trained_checkpoint"}


def test_workflow_runs_production_decisions_after_model_jobs(job, tmp_path):
    config = {"jobs": [], "report": str(tmp_path / "workflow-report.json"), "decisions": [{
        "name": job.name, "validation_manifest": str(job.validation_manifest),
        "checkpoint": str(job.checkpoint), "source_manifest": str(job.source_manifest),
        "output_dir": str(job.output_dir), "device": "cpu",
    }]}
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run_workflow(path)["decisions"][0]["action"] == "ran"
    assert run_workflow(path)["decisions"][0]["action"] == "skipped"


def test_explicit_threshold_requires_engineering_mode(job):
    with pytest.raises(ValueError, match="engineering preflight"):
        verdict.run_verdict(job.validation_manifest, job.checkpoint, job.output_dir, threshold=0.5, split="val", device_name="cpu")


def test_changed_checkpoint_invalidates_threshold_and_downstream(job):
    run_decision_job(job)
    checkpoint = torch.load(job.checkpoint, weights_only=False)
    checkpoint["model_state"]["classifier.bias"] += 0.01
    torch.save(checkpoint, job.checkpoint)
    assert set(decision_stage_status(job).values()) == {"pending_or_stale"}
    assert set(run_decision_job(job)["stages"].values()) == {"ran"}


@pytest.mark.parametrize("split", ["val", "train"])
def test_external_adapter_cannot_relabel_test_as_development(tmp_path, split):
    from seepat.data.deepfake_eval import build_deepfake_eval_manifest

    with pytest.raises(ValueError, match="locked test split"):
        build_deepfake_eval_manifest(tmp_path / "missing.json", tmp_path / "out.csv", split=split)


def test_calibration_mismatch_is_rejected_before_inference(job, monkeypatch):
    checkpoint = torch.load(job.checkpoint, weights_only=False)
    checkpoint["resume_contract"]["fusion_inputs"]["calibration_sha256"] = "different"
    torch.save(checkpoint, job.checkpoint)
    run = json.loads((job.checkpoint.parent / "run.json").read_text())
    run["resume_contract"] = checkpoint["resume_contract"]
    atomic_write_json(job.checkpoint.parent / "run.json", run)
    monkeypatch.setattr(verdict, "load_evaluation_model", lambda *args: pytest.fail("Do not score mismatched calibration"))
    with pytest.raises(ValueError, match="Evaluation calibration"):
        run_decision_job(job)


def test_explanation_detects_source_changes_during_generation(job):
    run_decision_job(job)
    directory = job.output_dir / "verdict"
    class ChangingClient:
        label = "test-only"
        def generate(self, system_prompt, user_prompt):
            path = directory / "event_predictions.csv"
            path.write_text(path.read_text() + '\n')
            return "advisory"
    path = job.output_dir / "changed-trace.json"
    with pytest.raises(ValueError, match="changed during"):
        generate_forensic_trace(directory, path, client=ChangingClient())
    assert not path.exists()
