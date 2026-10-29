"""End-to-end matched-input contracts on synthetic pixels and tiny models only."""

import json
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest
from test_fusion_model import _tiny_model
from test_fusion_training import calibrated_inputs

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.context_inputs import INPUT_VERSION, paired_indices
from seepat.training.matched_inputs import PROFILE_VERSION, MatchedClipDataset
from seepat.training.readiness import audit_inputs, check_readiness
from seepat.training.train import FUSION_MODEL, TrainingOptions, train_from_manifests
from seepat.training.video_objective import VIDEO_MAX, event_diagnostic_metrics, video_bags
from seepat.verdict import run_verdict, select_threshold
from seepat.workflow import (
    ModelTrainingJob,
    _model_training_options,
    load_workflow_settings,
    model_training_outputs_are_current,
    run_model_training_job,
)
from seepat.xai import build_evidence_bundle, render_evidence_summary


def synthetic_export(root):
    source = root / "source"
    source.mkdir()
    paths = calibrated_inputs(source)
    for split, path in paths.items():
        old = read_csv_rows(path)
        rows = [{**row, "event_id": f"{row['video_id']}-event-{j}", "video_phone_start_s": str(j)}
                for row in old for j in range(4)]
        atomic_write_csv(path, rows)
    source_summary = json.loads((source / "summary.json").read_text())
    source_summary["scored_manifest_sha256"] = {split: file_sha256(path) for split, path in paths.items()}
    atomic_write_json(source / "summary.json", source_summary)
    directory = root / "export"
    directory.mkdir()
    contract = {"input_sha256": {path.relative_to(root).as_posix(): file_sha256(path) for path in paths.values()}}
    atomic_write_json(directory / "contract.json", contract)
    indexes, hashes = {}, {}
    for split, path in paths.items():
        rows = read_csv_rows(path)
        indexes[f"control_{split}.csv"], indexes[f"candidate_{split}.csv"] = [], []
        for identifier in dict.fromkeys(row["video_id"] for row in rows):
            members = [row for row in rows if row["video_id"] == identifier]
            record_dir = directory / "records" / identifier
            record_dir.mkdir(parents=True)
            windows = []
            for index in range(2):
                context_id = f"{identifier}-context-{index:03d}"
                clip = record_dir / f"{context_id}.mp4"
                writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 25, (112, 112))
                assert writer.isOpened()
                for _ in range(16):
                    writer.write(np.full((112, 112, 3), 170, np.uint8))
                writer.release()
                windows.append({"context_id": context_id, "video_id": identifier, "dataset_split": split,
                                "source_group": members[0]["source_group"], "technical_ok": True,
                                "start_frame": index * 32, "frame_mask": [True] * 15 + [False],
                                "evidence_available": False, "phoneme": "", "event_class_id": "",
                                "clip_file": clip.name, "clip_sha256": file_sha256(clip)})
            record = {"video_id": identifier, "split": split, "source_group": members[0]["source_group"],
                      "windows": windows}
            atomic_write_json(record_dir / "record.json", record)
            hashes[f"records/{identifier}/record.json"] = file_sha256(record_dir / "record.json")
            left, right = paired_indices(members, windows)
            indexes[f"control_{split}.csv"].extend(left)
            indexes[f"candidate_{split}.csv"].extend(right)
    for name, rows in indexes.items():
        atomic_write_csv(directory / name, rows)
    summary = {"status": "complete", "input_version": INPUT_VERSION, "contract": contract,
               "engineering_gate": "review_for_training_wiring", "record_sha256": hashes,
               "output_sha256": {name: file_sha256(directory / name) for name in indexes}}
    atomic_write_json(directory / "summary.json", summary)
    profile = {"version": PROFILE_VERSION, "summary_sha256": file_sha256(directory / "summary.json"),
               "initial_checkpoint_sha256": "0" * 64}
    return paths, directory, profile


@pytest.fixture
def exported(tmp_path):
    return synthetic_export(tmp_path)


def test_context_slots_masks_unavailable_targets_and_original_bilabials(exported, tmp_path):
    paths, directory, profile = exported
    dataset = MatchedClipDataset(directory / "candidate_train.csv", tmp_path, "train", profile, image_size=8)
    assert dataset.audit_clips() == {"bilabial": 4, "context": 4}
    context_index = next(i for i, row in enumerate(dataset.rows) if row["kind"] == "context")
    sample = dataset[context_index]
    assert sample["video"].shape == (3, 16, 8, 8)
    assert sample["label"].item() == -1 and sample["phoneme"] == ""
    assert sample["frame_mask"].tolist() == [True] * 15 + [False]
    assert sample["video"][:, -1].count_nonzero().item() == 0
    assert sample["video"][:, 0].count_nonzero().item() > 0
    assert not sample["evidence_feature_mask"].any() and not sample["feature_mask"].any()
    assert not sample["evidence_features"].any() and not sample["features"].any()
    assert dataset[0]["label"].item() in {0, 1} and dataset[0]["phoneme"] == "p"
    assert all(row["class_id"] in {"0", "1"} for row in read_csv_rows(paths["train"]))
    with pytest.raises(ValueError, match="Invalid event label"):
        video_bags(dataset.rows)
    assert len(video_bags(dataset.rows, allow_context=True)) == 2


def test_contexts_are_excluded_only_from_event_metrics():
    metrics = event_diagnostic_metrics([0, -1, 1], [.1, .99, .8])
    assert metrics["samples"] == 2 and metrics["accuracy"] == 1
    assert metrics["unlabelled_context_clips_excluded"] == 1


def test_matched_pixel_cache_and_metadata_api_preserve_both_kinds(exported, tmp_path, monkeypatch):
    import torch

    from seepat.training import clip_cache

    _, directory, profile = exported
    original = MatchedClipDataset(directory / "candidate_train.csv", tmp_path, "train", profile,
                                  image_size=8)
    cached = MatchedClipDataset(directory / "candidate_train.csv", tmp_path, "train", profile,
                                image_size=8, clip_cache_dir=tmp_path / "pixel-cache")
    expected = [original[i] for i in range(len(original))]
    for index, reference in enumerate(expected):
        sample = cached[index]
        assert sample.keys() == reference.keys()
        for key, value in reference.items():
            assert torch.equal(value, sample[key]) if torch.is_tensor(value) else value == sample[key]
        path, frozen_hash, frame_mask = cached.clip_source(index)
        assert path.is_file()
        assert (frozen_hash is not None) == (sample["kind"] == "context")
        assert (frame_mask is not None) == (sample["kind"] == "context")

    def no_decode(*_args):
        raise AssertionError("A warmed cache or metadata lookup decoded pixels")

    monkeypatch.setattr(clip_cache, "_read_selected", no_decode)
    for index, reference in enumerate(expected):
        assert torch.equal(cached[index]["video"], reference["video"])
        metadata = cached.sample_metadata(index)
        assert metadata.keys() == reference.keys() - {"video", "frame_mask", "decoded_frame_count"}
        for key, value in metadata.items():
            assert torch.equal(value, reference[key]) if torch.is_tensor(value) else value == reference[key]


def test_tampered_record_or_profile_fails_before_decode(exported, tmp_path):
    _, directory, profile = exported
    bad = {**profile, "summary_sha256": "1" * 64}
    with pytest.raises(ValueError, match="summary hash"):
        MatchedClipDataset(directory / "candidate_train.csv", tmp_path, "train", bad)
    record = next((directory / "records").glob("*/record.json"))
    record.write_text("{}")
    with pytest.raises(ValueError, match="record hash"):
        MatchedClipDataset(directory / "candidate_train.csv", tmp_path, "train", profile)


def options(epochs=1, limited=False):
    return TrainingOptions(epochs=epochs, sequence_length=16, image_size=8, batch_size=1,
                           amp=False, freeze_backbone=True, supervision=VIDEO_MAX,
                           class_weighting="balanced_global", gradient_accumulation_steps=4,
                           selection_metric="validation_video_balanced_accuracy", early_stopping_patience=0,
                           max_train_batches=2 if limited else None,
                           max_validation_batches=2 if limited else None)


@pytest.mark.parametrize('cached_runtime', [False, True])
def test_matched_warm_start_resume_readiness_inner_threshold_and_context_trace(
    exported, tmp_path, monkeypatch, cached_runtime,
):
    paths, directory, profile = exported
    monkeypatch.setattr("seepat.training.train.build_event_classifier", lambda **kw: _tiny_model(freeze_backbone=kw["freeze_backbone"]))
    monkeypatch.setattr("seepat.verdict.build_event_classifier", lambda **kw: _tiny_model(freeze_backbone=kw["freeze_backbone"]))
    source_run = tmp_path / "reference"
    train_from_manifests(paths["train"], paths["val"], source_run, tmp_path, options(), "cpu", False,
                         model_name=FUSION_MODEL)
    initial = source_run / "checkpoint_best.pt"
    profile = {**profile, "initial_checkpoint_sha256": file_sha256(initial)}
    job = ModelTrainingJob("matched-ready", directory / "candidate_train.csv", directory / "candidate_val.csv",
                           tmp_path / "readiness", tmp_path, "cpu", False, asdict(options(limited=True)),
                           FUSION_MODEL, initialize_from=initial, input_profile=profile)
    if cached_runtime:
        job = replace(job, runtime={'feature_cache_dir': str(tmp_path / 'features'),
                                     'persistent_train_workers': True})
    assert audit_inputs(job)["status"] == "passed"
    first = run_model_training_job(job)["summary"]
    assert first["global_step"] == 1  # Reset optimizer; do not resume the source experiment.
    epoch = json.loads((job.output_dir / "history.json").read_text())[0]
    assert epoch["train"]["context_clips_processed"] == 4
    assert epoch["validation"]["events"]["samples"] == 4
    job = replace(job, options=asdict(options(epochs=2, limited=True)))
    resumed = run_model_training_job(job)
    assert resumed["action"] == "resumed" and resumed["summary"]["global_step"] == 2
    assert check_readiness(job)["status"] == "passed"
    if cached_runtime:
        wrong_runtime = replace(job, runtime={**job.runtime, 'persistent_train_workers': False})
        assert check_readiness(wrong_runtime)['status'] == 'pending'
        assert 'execution controls differ' in check_readiness(wrong_runtime)['reason']
    assert model_training_outputs_are_current(job)
    wrong_arm = replace(job, train_manifest=directory / "control_train.csv",
                        validation_manifest=directory / "control_val.csv")
    assert not model_training_outputs_are_current(wrong_arm)
    with pytest.raises(RuntimeError, match="does not match"):
        run_model_training_job(wrong_arm)
    full = replace(job, name="matched-full", output_dir=tmp_path / "fit", readiness_dir=job.output_dir,
                   options=asdict(options(epochs=1)))
    fit = run_model_training_job(full)
    assert fit["summary"]["global_step"] == 1
    selected = full.output_dir / "checkpoint_best.pt"
    threshold = tmp_path / "threshold.json"
    select_threshold(full.validation_manifest, selected, threshold, project_root=tmp_path, device_name="cpu")
    verdict = tmp_path / "verdict"
    result = run_verdict(full.validation_manifest, selected, verdict, threshold_artifact=threshold,
                         split="val", project_root=tmp_path, device_name="cpu")
    assert result["metrics"]["events"]["samples"] == 4
    predictions = read_csv_rows(verdict / "event_predictions.csv")
    contexts = [row for row in predictions if row["kind"] == "context"]
    assert len(contexts) == 4 and all(row["label"] == "" and row["phoneme"] == "" for row in contexts)
    bundle = build_evidence_bundle(verdict)
    assert bundle["context_clips_scored"] == 4
    assert "Visual-only context clip" in render_evidence_summary(bundle)
    with pytest.raises(ValueError, match="manifest differs"):
        select_threshold(directory / "control_val.csv", selected, tmp_path / "bad-threshold.json",
                         project_root=tmp_path, device_name="cpu")


def test_changed_inputs_cannot_use_legacy_warm_start_guard(exported, tmp_path, monkeypatch):
    paths, directory, profile = exported
    monkeypatch.setattr("seepat.training.train.build_event_classifier", lambda **kw: _tiny_model(freeze_backbone=True))
    reference = tmp_path / "reference"
    train_from_manifests(paths["train"], paths["val"], reference, tmp_path, options(), "cpu", False, model_name=FUSION_MODEL)
    initial = reference / "checkpoint_best.pt"
    with pytest.raises(ValueError, match="pinned original"):
        train_from_manifests(directory / "candidate_train.csv", directory / "candidate_val.csv",
                             tmp_path / "bad", tmp_path, options(), "cpu", False,
                             model_name=FUSION_MODEL, initialize_from=initial, input_profile=profile)


def test_input_audit_rejects_crossed_arms_and_missing_initializer(exported, tmp_path):
    _, directory, profile = exported
    job = ModelTrainingJob("bad-arm", directory / "candidate_train.csv", directory / "control_val.csv",
                           tmp_path / "readiness", tmp_path, "cpu", False, asdict(options(limited=True)),
                           FUSION_MODEL, input_profile=profile)
    with pytest.raises(ValueError, match="same arm"):
        audit_inputs(job)
    with pytest.raises(ValueError, match="pinned Phase-3"):
        audit_inputs(replace(job, validation_manifest=directory / "candidate_val.csv"))


def test_prepared_configs_keep_two_matched_fits_and_separate_fresh_cohort():
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    full = load_workflow_settings(configs / "fusion_phase5.yaml")
    ready = load_workflow_settings(configs / "fusion_phase5_readiness.yaml")
    assert len(full.model_training_jobs) == len(ready.model_training_jobs) == 2
    assert len({job.output_dir for job in full.model_training_jobs + ready.model_training_jobs}) == 4
    for fit, preflight in zip(full.model_training_jobs, ready.model_training_jobs, strict=True):
        actual, bounded = _model_training_options(fit), _model_training_options(preflight)
        assert actual.epochs == 10 and bounded.epochs == 1
        assert actual.max_train_batches is None and actual.max_validation_batches is None
        assert bounded.max_train_batches == 16 and bounded.max_validation_batches == 4
        assert fit.readiness_dir == preflight.output_dir
        assert fit.input_profile == preflight.input_profile
        assert fit.initialize_from == preflight.initialize_from
        assert actual.supervision == VIDEO_MAX and actual.freeze_backbone and not actual.amp
        assert actual.gradient_accumulation_steps == 4
        assert not actual.unfreeze_final_backbone_stages
        with pytest.raises(ValueError, match="Matched inputs"):
            _model_training_options(replace(fit, options={**fit.options, "sequence_length": 8}))
    decisions = load_workflow_settings(configs / "fusion_phase5_validation.yaml").decision_jobs
    assert len(decisions) == 2 and all(job.threshold_artifact is None for job in decisions)
    cohort = load_workflow_settings(configs / "phase5_cohort.yaml")
    assert not cohort.model_training_jobs and not cohort.decision_jobs and cohort.study_job is None
    assert len(cohort.jobs) == 1 and len(cohort.trace_augmentation_jobs) == 1
    assert cohort.numerical_calibration_jobs[0].calibration is not None
