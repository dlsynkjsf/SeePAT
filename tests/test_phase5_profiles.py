"""Prepared profile roles only; no real data, checkpoints or CUDA jobs."""

from dataclasses import replace
from pathlib import Path

from seepat.workflow import _model_training_options, load_workflow_settings

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_phase5_resume_profile_changes_only_epoch_target_and_report():
    first = load_workflow_settings(CONFIGS / "fusion_phase5_readiness.yaml")
    resume = load_workflow_settings(CONFIGS / "fusion_phase5_readiness_resume.yaml")
    assert first.report_path != resume.report_path
    assert len(first.model_training_jobs) == len(resume.model_training_jobs) == 2
    for old, new in zip(first.model_training_jobs, resume.model_training_jobs, strict=True):
        assert _model_training_options(old).epochs == 1
        assert _model_training_options(new).epochs == 2
        assert replace(new, options={**new.options, "epochs": 1}) == old


def test_phase5_outer_profiles_bind_same_export_and_own_inner_thresholds():
    full = load_workflow_settings(CONFIGS / "fusion_phase5.yaml")
    inner = load_workflow_settings(CONFIGS / "fusion_phase5_validation.yaml")
    outer = load_workflow_settings(CONFIGS / "fusion_phase5_outer_evaluation.yaml")
    assert not outer.model_training_jobs and len(outer.decision_jobs) == 2
    assert len({job.output_dir for job in inner.decision_jobs + outer.decision_jobs}) == 4
    for arm, fit, development, comparison in zip(
        ("control", "candidate"), full.model_training_jobs, inner.decision_jobs,
        outer.decision_jobs, strict=True,
    ):
        assert development.checkpoint == comparison.checkpoint == fit.output_dir / "checkpoint_best.pt"
        assert development.threshold_artifact is None
        assert comparison.threshold_artifact == development.output_dir / "threshold.json"
        assert comparison.source_manifest == Path("data/manifests/phase5_val_outer1000.csv")
        assert comparison.validation_manifest == Path(f"outputs/phase5_outer_context_inputs_v1/{arm}_val.csv")
        assert comparison.input_profile == {
            **fit.input_profile,
            "summary_sha256": "350cb7b230fc92844ca9f7dc9ea3ce67fda3fc4af5470d944553d14bc76d6c26",
        }
        assert comparison.input_profile["summary_sha256"] != fit.input_profile["summary_sha256"]
