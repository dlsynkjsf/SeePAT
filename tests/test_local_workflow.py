"""Normal workflow interface; schema checks never load models or input media."""

import pytest

from seepat.artifacts import atomic_write_json
from seepat.workflow import load_workflow_settings


def test_local_profile_loads_without_reading_freeze_or_media(tmp_path):
    config = tmp_path / "workflow.json"
    output = tmp_path / "not-started"
    atomic_write_json(config, {"report": str(tmp_path / "summary.json"), "local_inference": [{
        "name": "local-demo", "freeze": str(tmp_path / "missing-freeze.json"),
        "raw_inputs": str(tmp_path / "missing-inputs.json"), "output_dir": str(output),
    }]})

    settings = load_workflow_settings(config)

    job, = settings.local_inference_jobs
    assert job.name == "local-demo" and job.device == "cpu" and job.batch_size == 1
    assert job.output_dir == output and not output.exists()


@pytest.mark.parametrize("collision", ["job_report", "freeze", "raw_inputs", "progress_input"])
def test_workflow_report_cannot_overwrite_local_evidence_or_inputs(tmp_path, collision):
    config = tmp_path / "workflow.json"
    output = tmp_path / "local-job"
    freeze, raw_inputs = tmp_path / "freeze.json", tmp_path / "inputs.json"
    report = {"job_report": output / "result/report.md", "freeze": freeze,
              "raw_inputs": raw_inputs, "progress_input": tmp_path / "other.json"}[collision]
    if collision == "progress_input":
        raw_inputs = tmp_path / "other.progress.json"
    atomic_write_json(config, {"report": str(report), "local_inference": [{
        "name": "local-demo", "freeze": str(freeze), "raw_inputs": str(raw_inputs),
        "output_dir": str(output),
    }]})

    with pytest.raises(ValueError, match="report.*local|local.*report"):
        load_workflow_settings(config)

    assert not output.exists() and not freeze.exists() and not raw_inputs.exists()
