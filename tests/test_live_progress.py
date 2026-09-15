from __future__ import annotations

import json
from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.live_progress import WorkflowProgress, format_live_progress, live_progress_path
from seepat.progress import read_live_workflow_progress


def config_file(tmp_path):
    path = tmp_path / "workflow.yaml"
    report = tmp_path / "workflow_summary.json"
    path.write_text(
        f"report: {report.as_posix()}\njobs: []\nmodel_training:\n"
        "  name: model\n  train_manifest: train.csv\n"
        "  validation_manifest: val.csv\n  output_dir: model\n",
        encoding="utf-8",
    )
    return path, report


def test_live_progress_throttles_updates_and_resets_eta_per_phase(tmp_path, monkeypatch):
    config, report = config_file(tmp_path)
    now = [0.0]
    monkeypatch.setattr("seepat.live_progress.time.monotonic", lambda: now[0])
    progress = WorkflowProgress(config, report, 2)
    progress.start_job(1, "train")
    progress("audit preprocessing contract", 0, 100, "a")
    now[0] = 2.0
    progress("audit preprocessing contract", 10, 100, "b")
    assert json.loads(progress.path.read_text())["finished"] == 0  # Throttled disk writes.
    now[0] = 5.0
    progress("audit preprocessing contract", 25, 100, "c")
    record = read_live_workflow_progress(config)
    assert record["elapsed_seconds"] == 5
    assert record["eta_seconds"] == 15
    assert record["current_item"] == "c"
    text = format_live_progress(record)
    assert "25/100 (25.0%)" in text and "ETA 00:00:15" in text
    now[0] = 6.0
    progress("augment visual traces", 0, 20, "d")
    assert read_live_workflow_progress(config)["eta_seconds"] is None
    assert read_live_workflow_progress(config)["elapsed_seconds"] == 0
    progress("augment visual traces", 20, 20, "")  # Always publishes completion.
    assert read_live_workflow_progress(config)["finished"] == 20


def test_live_tracker_does_not_hash_dataset_or_import_training(tmp_path, monkeypatch):
    config, report = config_file(tmp_path)
    record = {
        "config_sha256": file_sha256(config),
        "status": "running",
        "job_index": 1,
        "jobs_total": 1,
        "phase": "audit preprocessing contract",
    }
    atomic_write_json(live_progress_path(report), record)

    def forbidden(*args, **kwargs):
        raise AssertionError("Live tracking must not run integrity/model checks")

    for name in (
        "preprocessing_outputs_are_current",
        "training_outputs_are_current",
        "model_training_outputs_are_current",
        "numerical_calibration_outputs_are_current",
    ):
        monkeypatch.setattr("seepat.workflow." + name, forbidden)
    monkeypatch.setattr(
        "seepat.preprocessing.augmentation.augmentation_outputs_are_current", forbidden
    )
    real_hash = file_sha256

    def config_hash_only(path):
        assert Path(path) == config
        return real_hash(path)

    monkeypatch.setattr("seepat.progress.file_sha256", config_hash_only)
    assert read_live_workflow_progress(config) == record


def test_live_tracker_reports_missing_or_mismatched_report(tmp_path):
    config, report = config_file(tmp_path)
    assert read_live_workflow_progress(config)["status"] == "unavailable"
    atomic_write_json(live_progress_path(report), {"config_sha256": "old", "status": "complete"})
    record = read_live_workflow_progress(config)
    assert record["status"] == "unavailable"
    assert "restarted" in format_live_progress(record)


@pytest.mark.parametrize(
    "error,status", [(RuntimeError("audit failed"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_workflow_records_failure_and_interruption_without_running_other_jobs(
    tmp_path, monkeypatch, error, status
):
    from seepat.workflow import run_workflow

    config, report = config_file(tmp_path)

    def fail(job, **kwargs):
        raise error

    monkeypatch.setattr("seepat.workflow.run_model_training_job", fail)
    with pytest.raises(type(error)):
        run_workflow(config)
    record = read_live_workflow_progress(config)
    assert record["status"] == status
    assert type(error).__name__ in record["error"]
    assert not report.exists()


@pytest.mark.parametrize("status", ["complete", "failed", "interrupted"])
def test_watch_exits_on_runner_terminal_state(tmp_path, monkeypatch, capsys, status):
    from seepat.progress import main

    config, report = config_file(tmp_path)
    atomic_write_json(
        live_progress_path(report),
        {
            "config_sha256": file_sha256(config),
            "status": status,
            "jobs_total": 1,
            "job_index": 1,
            "job": "model",
            "phase": "test",
        },
    )
    monkeypatch.setattr(
        "sys.argv", ["progress", "--workflow-config", str(config), "--watch-seconds", "5"]
    )
    monkeypatch.setattr(
        "seepat.progress.sleep", lambda seconds: pytest.fail("terminal state must exit")
    )
    main()
    assert f"Workflow {status}" in capsys.readouterr().out
