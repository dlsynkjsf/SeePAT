import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_calibration import make_row

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.training import calibration
from seepat.training.study import (
    StudyJob,
    load_study,
    model_job,
    prepare_study,
    run_study_job,
    source_group_folds,
    study_phase_is_current,
)
from seepat.training.study_evaluation import (
    calibration_comparison,
    matched_geometry,
    paired_statistics,
    write_comparison,
)
from seepat.workflow import load_workflow_settings, run_workflow


def study_fixture(tmp_path, folds=2):
    rows = []
    for split, count in (("train", 12), ("val", 4)):
        for i in range(count):
            label = str(i % 2)
            row = make_row(tmp_path, f"{split}-{i}", split=split, label=label, baseline=1 + i / 2)
            row.update(manipulation_modality="real" if label == "0" else "both_modified",
                       mouth_clip_path=f"{split}-{i}.mp4", normalized_minimum_closure="0.2",
                       closure_time_s="2.2", video_phone_start_s="2.0", closure_duration_s=".1",
                       phone_duration_s=".1")
            (tmp_path / row["mouth_clip_path"]).write_bytes(b"clip-presence-only")
            rows.append(row)
    train, val = tmp_path / "train.csv", tmp_path / "val.csv"
    atomic_write_csv(train, [r for r in rows if r["dataset_split"] == "train"])
    atomic_write_csv(val, [r for r in rows if r["dataset_split"] == "val"])
    return StudyJob("test-study", train, val, tmp_path / "study", folds=folds, device="cpu", project_root=tmp_path)


def test_source_groups_stay_together_with_complete_deterministic_holdouts(tmp_path):
    job = study_fixture(tmp_path)
    train, val = read_csv_rows(job.train_manifest), read_csv_rows(job.validation_manifest)
    # Multiple events with different labels may belong to the same source group.
    train += [{**r, "event_id": r["event_id"] + "extra", "class_id": str(1 - int(r["class_id"]))} for r in train]
    folds = source_group_folds(train, val, 2, 91)
    assert folds == source_group_folds(train[::-1], val, 2, 91)
    assert sorted(r["event_id"] for _, held in folds for r in held) == sorted(r["event_id"] for r in train)
    for fitting, held in folds:
        assert not {r["source_group"] for r in fitting} & {r["source_group"] for r in held}
        assert all(r["dataset_split"] == "train" for r in held)
    with pytest.raises(ValueError, match="overlap"):
        source_group_folds(train, [{**r, "source_group": train[0]["source_group"]} for r in val], 2, 91)
    with pytest.raises(ValueError, match="external"):
        source_group_folds([{**r, "dataset_source": "Deepfake-Eval-2024"} for r in train], val, 2, 91)


def test_prepare_refits_population_per_fold_resumes_and_detects_tampering(tmp_path, monkeypatch):
    job = study_fixture(tmp_path)
    populations = []
    original = calibration._fit_population

    def observe(rows, *args):
        populations.append({r["event_id"] for r in rows})
        return original(rows, *args)

    monkeypatch.setattr(calibration, "_fit_population", observe)
    prepare_study(job)
    record = load_study(job)
    assert len(populations) == 2
    for fitting, fold in zip(populations, record["folds"], strict=True):
        assert fitting == {r["event_id"] for r in read_csv_rows(Path(fold["inputs"]["train"]))}
        assert not fitting & {r["event_id"] for r in read_csv_rows(Path(fold["inputs"]["heldout"]))}
    prepare_study(job)
    assert len(populations) == 2  # Reuse every current fold calibration.
    assert study_phase_is_current(job)
    assert not study_phase_is_current(replace(job, phase="train"))
    run_study_job(replace(job, phase="audit"))
    result = run_study_job(replace(job, phase="calibration"))
    assert len(result["results"]) == 4
    assert study_phase_is_current(replace(job, phase="calibration"))
    # A transferred export is sufficient; original traces/inputs are not reread.
    job.train_manifest.unlink()
    job.validation_manifest.unlink()
    assert load_study(job) == record
    path = Path(record["folds"][0]["manifests"]["heldout"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_study(job)


def test_fold_population_must_bind_its_training_subset(tmp_path):
    job = study_fixture(tmp_path)
    prepare_study(job)
    record_path = job.output_dir / "study.json"
    record = json.loads(record_path.read_text())
    fold = record["folds"][0]
    population_path = Path(fold["manifests"]["train"]).parent / "calibration.json"
    population = json.loads(population_path.read_text())
    population["train_manifest_sha256"] = file_sha256(job.train_manifest)
    atomic_write_json(population_path, population)
    fold["hashes"][population_path.as_posix()] = file_sha256(population_path)
    atomic_write_json(record_path, record)
    with pytest.raises(ValueError, match="not fitted"):
        load_study(job)


def test_workflow_profile_and_progress_include_the_study(tmp_path, monkeypatch):
    from seepat import workflow
    from seepat.progress import read_workflow_progress

    job = study_fixture(tmp_path)
    config = tmp_path / "study.yaml"
    config.write_text(f'''report: {tmp_path.as_posix()}/workflow.json
ablation_study:
  name: paper
  train_manifest: {job.train_manifest.as_posix()}
  validation_manifest: {job.validation_manifest.as_posix()}
  output_dir: {job.output_dir.as_posix()}
  folds: 2
''')
    wrapper = tmp_path / "workflow.yaml"
    wrapper.write_text(f"model_training_config: study.yaml\nreport: {tmp_path.as_posix()}/report.json\n")
    settings = load_workflow_settings(wrapper)
    assert settings.study_job.folds == 2 and not settings.model_training_jobs
    monkeypatch.setattr(workflow, "run_study_job", lambda job, progress: {"name": job.name, "action": "prepared"})
    assert run_workflow(wrapper)["ablation_study"][0]["action"] == "prepared"
    status = read_workflow_progress(wrapper)
    assert status["stages_total"] == 1 and not status["all_current"]
    assert status["ablation_study"]["phase"] == "prepare"
    for model in ("efficientnet_v2_s_only", "tempcnn_gray16", "swin3d_b_visual_fusion"):
        fold = {"fold": 1, "manifests": {"train": "train.csv", "val": "val.csv"}}
        ready = model_job(job, fold, model, readiness=True)
        full = model_job(job, fold, model, readiness=False)
        assert full.readiness_dir == ready.output_dir and ready.options["max_train_batches"] == 16
        assert full.options["max_train_batches"] is None
        assert ready.pretrained == (model != "tempcnn_gray16")


def geometry_rows():
    return [{"event_id": f"event-{i}", "video_id": f"video-{i}", "class_id": str(i % 2),
             "manipulation_modality": "real" if i % 2 == 0 else "both_modified",
             "normalized_minimum_closure": str(.1 + (i % 2) * .4),
             "isolation_forest_anomaly_score": str(.2 + (i % 2) * .6),
             "isolation_forest_available": "true"} for i in range(6)]


def test_geometry_uses_identical_label_independent_cohort_and_validation_thresholds():
    val = geometry_rows()
    held = geometry_rows()
    held[-1]["isolation_forest_available"] = "false"
    records, coverage = matched_geometry(held)
    assert coverage["paired_events"] == 5
    assert [r["event_id"] for r in records] == [r["event_id"] for r in matched_geometry(
        [{**r, "class_id": str(1 - int(r["class_id"]))} for r in held])[0]]
    results = calibration_comparison(val, held)
    changed_labels = [{**r, "class_id": str(1 - int(r["class_id"])),
                       "manipulation_modality": "real" if r["class_id"] == "1" else "both_modified"} for r in held]
    changed = calibration_comparison(val, changed_labels)
    assert [r["selection"] for r in results] == [r["selection"] for r in changed]
    assert results[0]["video_metrics"]["samples"] == results[1]["video_metrics"]["samples"] == 5
    assert results[0]["static_threshold_normalized_vild"] == pytest.approx(.5)
    assert results[0]["coverage"] == results[1]["coverage"]
    assert all("false_positive_rate" in r["video_metrics"] and "recall" in r["video_metrics"] for r in results)


def test_paired_report_keeps_five_fold_significance_limitation_and_ties(tmp_path):
    report = paired_statistics([1, 2, 3, 4, 5], [0] * 5)
    assert report["exact_two_sided_signed_rank_p"] == .0625
    assert paired_statistics([1] * 5, [0] * 5)["exact_two_sided_signed_rank_p"] == .0625
    assert paired_statistics([0] * 5, [0] * 5)["exact_two_sided_signed_rank_p"] == 1
    results = [{"model": model, "fold": fold, "video_metrics": {"f1": .5}}
               for model in ("efficientnet_v2_s_only", "efficientnet_v2_s_tempcnn") for fold in (1, 2)]
    write_comparison(tmp_path / "comparison", results)
    assert json.loads((tmp_path / "comparison.json").read_text())["paired_comparisons"][0]["pairs"] == 2
    with pytest.raises(ValueError, match="identical fold"):
        write_comparison(tmp_path / "mismatch", results[:-1])


def test_fold_model_real_resume_training_threshold_scoring_and_cache(tmp_path, monkeypatch):
    import torch
    from test_cnn_temporal import TinyFrameBackbone
    from test_verdict import _write_clip

    from seepat.models import cnn_temporal
    from seepat.training.readiness import check_readiness
    from seepat.training.study_evaluation import evaluate_model
    from seepat.workflow import run_model_training_job

    job = study_fixture(tmp_path)
    for manifest in (job.train_manifest, job.validation_manifest):
        rows = read_csv_rows(manifest)
        for row in rows:
            clip = tmp_path / (row["video_id"] + ".avi")
            _write_clip(clip)
            row["mouth_clip_path"] = clip.as_posix()
        atomic_write_csv(manifest, rows)
    prepare_study(job)
    fold = load_study(job)["folds"][0]
    monkeypatch.setattr(cnn_temporal, "efficientnet_v2_s", lambda **_: TinyFrameBackbone())
    ready = model_job(job, fold, "efficientnet_v2_s_only", readiness=True)
    ready = replace(ready, pretrained=False, options={**ready.options, "sequence_length": 4,
                    "image_size": 8, "max_train_batches": 4, "max_validation_batches": 4})
    assert run_model_training_job(ready)["action"] == "ran"
    ready = replace(ready, options={**ready.options, "epochs": 2})
    assert run_model_training_job(ready)["action"] == "resumed"
    assert check_readiness(ready)["status"] == "passed"
    trained = replace(ready, output_dir=job.output_dir / "full", readiness_dir=ready.output_dir,
                      options={**ready.options, "epochs": 1, "max_train_batches": None,
                               "max_validation_batches": None})
    assert run_model_training_job(trained)["action"] == "ran"
    assert run_model_training_job(trained)["action"] == "skipped"
    result = evaluate_model(job, fold, trained)
    assert result["video_metrics"]["samples"] == len(read_csv_rows(Path(fold["manifests"]["heldout"])))
    assert result["provenance"]["validation_manifest_sha256"] == file_sha256(trained.validation_manifest)

    def forbidden(**kwargs):
        raise AssertionError("Current predictions must be reused before model construction")

    monkeypatch.setattr(cnn_temporal, "efficientnet_v2_s", forbidden)
    assert evaluate_model(job, fold, trained) == result
    metrics_path = trained.output_dir / "fold_evaluation" / "metrics.json"
    corrupted = json.loads(metrics_path.read_text())
    corrupted["video_metrics"]["f1"] = .123456
    atomic_write_json(metrics_path, corrupted)
    with pytest.raises(AssertionError, match="Current predictions"):
        evaluate_model(job, fold, trained)  # Altered metrics must trigger recomputation.
    atomic_write_json(metrics_path, result)
    threshold_path = trained.output_dir / "fold_evaluation" / "threshold.json"
    threshold = json.loads(threshold_path.read_text())
    threshold["threshold"] = .123456
    atomic_write_json(threshold_path, threshold)
    with pytest.raises(AssertionError, match="Current predictions"):
        evaluate_model(job, fold, trained)  # Altered thresholds cannot be reused either.
    selected = Path(json.loads((trained.output_dir / "run.json").read_text())["best_checkpoint"])
    checkpoint = torch.load(selected, weights_only=False)
    checkpoint["options"]["max_train_batches"] = 1
    torch.save(checkpoint, selected)
    with pytest.raises(ValueError, match="preflight"):
        evaluate_model(job, fold, trained)
