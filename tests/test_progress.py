from __future__ import annotations

import json
from pathlib import Path

from seepat.artifacts import stable_id
from seepat.progress import (
    format_progress,
    format_workflow_progress,
    read_workflow_progress,
    summarize_progress,
)
from seepat.workflow import ModelTrainingJob, WorkflowJob, WorkflowSettings


def _write_result(
    cache_dir: Path,
    file: str,
    signature: str,
    pipeline_status: str,
    eligibility_status: str,
) -> None:
    result_path = cache_dir / stable_id(file) / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "cache_signature": signature,
                "video_report": {
                    "pipeline_status": pipeline_status,
                    "eligibility_status": eligibility_status,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summarize_progress_counts_only_current_results(tmp_path: Path) -> None:
    rows = [{"file": "a.mp4"}, {"file": "b.mp4"}, {"file": "c.mp4"}]
    _write_result(tmp_path, "a.mp4", "current", "complete", "eligible")
    _write_result(tmp_path, "b.mp4", "current", "failed", "ineligible")
    _write_result(tmp_path, "c.mp4", "stale", "complete", "eligible")

    progress = summarize_progress(rows, tmp_path, "current")

    assert progress["videos_requested"] == 3
    assert progress["videos_finished"] == 2
    assert progress["videos_completed"] == 1
    assert progress["videos_failed"] == 1
    assert progress["videos_eligible"] == 1
    assert progress["videos_pending"] == 1
    assert progress["percent_finished"] == 66.7
    assert progress["latest_result_at_utc"] is not None


def test_format_progress_is_compact() -> None:
    text = format_progress(
        {
            "videos_finished": 4,
            "videos_requested": 20,
            "percent_finished": 20.0,
            "videos_completed": 3,
            "videos_failed": 1,
            "videos_eligible": 2,
            "videos_pending": 16,
        }
    )

    assert "finished=4/20 (20.0%)" in text
    assert "failed=1" in text
    assert "pending=16" in text


def test_read_workflow_progress_includes_model_jobs(tmp_path: Path, monkeypatch) -> None:
    preparation = WorkflowJob("prepare", tmp_path / "pipeline.yaml", tmp_path / "manifest")
    swin = ModelTrainingJob(
        name="swin",
        train_manifest=tmp_path / "train.csv",
        validation_manifest=tmp_path / "val.csv",
        output_dir=tmp_path / "swin",
        project_root=tmp_path,
        device="cpu",
        pretrained=False,
        options={"epochs": 1},
    )
    cnn = ModelTrainingJob(
        name="cnn",
        train_manifest=tmp_path / "train.csv",
        validation_manifest=tmp_path / "val.csv",
        output_dir=tmp_path / "cnn",
        project_root=tmp_path,
        device="cpu",
        pretrained=False,
        options={"epochs": 2},
        model="efficientnet_v2_s_tempcnn",
    )
    settings = WorkflowSettings(
        jobs=(preparation,),
        model_training_jobs=(swin, cnn),
        report_path=tmp_path / "summary.json",
    )
    cnn.output_dir.mkdir()
    (cnn.output_dir / "run.json").write_text(
        json.dumps({"status": "running", "options": {"epochs": 2}}),
        encoding="utf-8",
    )
    (cnn.output_dir / "history.json").write_text(
        json.dumps([{"epoch": 1}]),
        encoding="utf-8",
    )

    monkeypatch.setattr("seepat.workflow.load_workflow_settings", lambda path: settings)
    monkeypatch.setattr("seepat.progress.load_pipeline_settings", lambda *args: object())
    monkeypatch.setattr(
        "seepat.workflow.preprocessing_outputs_are_current", lambda settings: True
    )
    monkeypatch.setattr(
        "seepat.workflow.training_outputs_are_current", lambda settings, output: True
    )
    monkeypatch.setattr(
        "seepat.workflow.model_training_outputs_are_current",
        lambda job: job.name == "swin",
    )

    progress = read_workflow_progress(tmp_path / "workflow.yaml")

    assert progress["stages_current"] == 3
    assert progress["stages_total"] == 4
    assert progress["all_current"] is False
    assert progress["model_training"][0]["status"] == "current"
    assert progress["model_training"][1] == {
        "name": "cnn",
        "model": "efficientnet_v2_s_tempcnn",
        "status": "running",
        "completed_epochs": 1,
        "requested_epochs": 2,
    }


def test_format_workflow_progress_is_compact() -> None:
    text = format_workflow_progress(
        {
            "stages_current": 5,
            "stages_total": 6,
            "model_training": [
                {
                    "name": "swin",
                    "status": "current",
                    "completed_epochs": 1,
                    "requested_epochs": 1,
                },
                {
                    "name": "cnn",
                    "status": "running",
                    "completed_epochs": 0,
                    "requested_epochs": 1,
                },
            ],
        }
    )

    assert "workflow stages=5/6 current" in text
    assert "swin=current 1/1 epochs" in text
    assert "cnn=running 0/1 epochs" in text
