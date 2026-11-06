from __future__ import annotations

import json
from dataclasses import replace

import pytest
from media_fixtures import calibrated_inputs
from model_fixtures import tiny_fusion_model as _tiny_model

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.evidence import CALIBRATED_INPUT_VERSION, EVIDENCE_VERSION, FUSION_EVIDENCE_FIELDS
from seepat.training.dataset import MouthEventDataset
from seepat.training.train import FUSION_MODEL, TrainingOptions, train_from_manifests
from seepat.workflow import (
    ModelTrainingJob,
    model_training_outputs_are_current,
    run_model_training_job,
)

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_calibrated_fusion_trains_resumes_skips_and_rejects_old_contract(tmp_path, monkeypatch):
    paths = calibrated_inputs(tmp_path)
    built = []

    def build(**kwargs):
        assert kwargs["model_name"] == FUSION_MODEL and not kwargs["pretrained"]
        model = _tiny_model(freeze_backbone=kwargs["freeze_backbone"])
        built.append((model, model.classifier.weight.detach().clone()))
        return model

    monkeypatch.setattr("seepat.training.train.build_event_classifier", build)
    job = ModelTrainingJob(
        "fusion",
        paths["train"],
        paths["val"],
        tmp_path / "model",
        tmp_path,
        "cpu",
        False,
        {
            "epochs": 1,
            "sequence_length": 4,
            "image_size": 8,
            "freeze_backbone": True,
            "gradient_accumulation_steps": 2,
            "max_train_batches": 2,
            "max_validation_batches": 1,
        },
        FUSION_MODEL,
    )
    updates = []
    first = run_model_training_job(job, progress=lambda *args: updates.append(args))
    assert first["action"] == "ran"
    assert not torch.equal(built[0][0].classifier.weight, built[0][1])
    report = first["summary"]
    contract = report["resume_contract"]
    assert contract["training_version"] == "hybrid-fusion-v3"
    assert report["optimizer_steps"] == 1
    assert report["skipped_optimizer_steps"] == 0
    assert contract["fusion_inputs"]["evidence_version"] == EVIDENCE_VERSION
    assert contract["fusion_inputs"]["evidence_fields"] == list(FUSION_EVIDENCE_FIELDS)
    assert report["data"]["first_train_batch"]["video_shape"] == [1, 3, 4, 8, 8]
    assert report["data"]["first_train_batch"]["evidence_shape"] == [1, 7]
    assert report["data"]["evidence_coverage"]["train"]["isolation_forest_anomaly_score"] == 0.5
    assert any(
        phase == "train epoch 1/1" and done == total == 2 for phase, done, total, _ in updates
    )
    assert any(
        phase == "validate epoch 1/1" and done == total == 1 for phase, done, total, _ in updates
    )
    assert model_training_outputs_are_current(job)

    resumed_job = replace(job, options={**job.options, "epochs": 2})
    resumed = run_model_training_job(resumed_job)
    assert resumed["action"] == "resumed"
    assert resumed["summary"]["completed_epochs"] == 2
    checkpoint = torch.load(job.output_dir / "checkpoint_last.pt", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert resumed["summary"]["optimizer_steps"] == 1
    assert checkpoint["optimizer_state"]["state"]
    assert {int(s["step"]) for s in checkpoint["optimizer_state"]["state"].values()} == {2}
    assert [r["epoch"] for r in checkpoint["history"]] == [1, 2]
    assert (
        checkpoint["history"][1]["validation"]["video_aggregation"]
        == "maximum event fake probability"
    )
    assert all("sync_gap" not in key for key in checkpoint["model_state"])
    assert run_model_training_job(resumed_job)["action"] == "skipped"
    assert len(built) == 2

    run_path = job.output_dir / "run.json"
    record = json.loads(run_path.read_text())
    record["resume_contract"]["fusion_inputs"]["evidence_version"] = "fusion-evidence-v1"
    atomic_write_json(run_path, record)
    assert not model_training_outputs_are_current(resumed_job)
    with pytest.raises(RuntimeError, match="does not match"):
        run_model_training_job(resumed_job)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("uncalibrated", "calibrated manifest"),
        ("version", "calibration version"),
        ("mask", "availability disagree"),
        ("hash", "manifest hash"),
        ("population", "artifact hash/version"),
        ("split", "only the requested split"),
    ],
)
def test_fusion_rejects_invalid_inputs_before_decoding(tmp_path, problem, match):
    paths = calibrated_inputs(tmp_path)
    train = paths["train"]
    rows = read_csv_rows(train)
    if problem == "uncalibrated":
        for row in rows:
            del row["isolation_forest_available"]
    elif problem == "version":
        rows[0]["calibration_version"] = "old"
    elif problem == "mask":
        rows[0]["isolation_forest_available"] = "False"
    elif problem == "population":
        atomic_write_json(tmp_path / "calibration.json", {"calibration_version": "old"})
    elif problem == "split":
        rows[0]["dataset_split"] = "val"
    else:
        rows[0]["phone_duration_s"] = "0.5"
    atomic_write_csv(train, rows)
    if problem == "split":
        summary = json.loads((tmp_path / "summary.json").read_text())
        summary["scored_manifest_sha256"]["train"] = file_sha256(train)
        atomic_write_json(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match=match):
        MouthEventDataset(train, dataset_split="train", require_calibration=True)


def test_fusion_rejects_different_train_and_validation_populations(tmp_path, monkeypatch):
    train_dir, val_dir = tmp_path / "a", tmp_path / "b"
    train_dir.mkdir()
    val_dir.mkdir()
    train, val = calibrated_inputs(train_dir), calibrated_inputs(val_dir)
    population = val_dir / "calibration.json"
    atomic_write_json(
        population, {"calibration_version": CALIBRATED_INPUT_VERSION, "different": True}
    )
    summary = json.loads((val_dir / "summary.json").read_text())
    summary["calibration_sha256"] = file_sha256(population)
    atomic_write_json(val_dir / "summary.json", summary)
    monkeypatch.setattr(
        "seepat.training.train.build_event_classifier",
        lambda **kwargs: pytest.fail("Do not build a model for incompatible inputs"),
    )
    with pytest.raises(ValueError, match="same frozen calibration"):
        train_from_manifests(
            train["train"],
            val["val"],
            tmp_path / "model",
            tmp_path,
            TrainingOptions(),
            "cpu",
            False,
            FUSION_MODEL,
        )


def test_phase2_initializes_from_selected_checkpoint_and_uses_two_learning_rates(
    tmp_path,
    monkeypatch,
):
    paths = calibrated_inputs(tmp_path)

    def build(**kwargs):
        return _tiny_model(
            freeze_backbone=kwargs["freeze_backbone"],
            unfreeze_final_backbone_stages=kwargs.get(
                "unfreeze_final_backbone_stages",
                False,
            ),
        )

    monkeypatch.setattr("seepat.training.train.build_event_classifier", build)
    common = {
        "sequence_length": 4,
        "image_size": 8,
        "amp": False,
        "freeze_backbone": True,
        "class_weighting": "balanced_global",
        "positive_class_weight_ratio": 10.0,
        "selection_metric": "validation_video_balanced_accuracy",
    }
    source = ModelTrainingJob(
        "source",
        paths["train"],
        paths["val"],
        tmp_path / "source",
        tmp_path,
        "cpu",
        False,
        {"epochs": 1, **common},
        FUSION_MODEL,
    )
    run_model_training_job(source)
    initial = source.output_dir / "checkpoint_best.pt"
    phase2 = ModelTrainingJob(
        "phase2",
        paths["train"],
        paths["val"],
        tmp_path / "phase2",
        tmp_path,
        "cpu",
        False,
        {
            "epochs": 1,
            "learning_rate": 0.01,
            "unfreeze_final_backbone_stages": True,
            "backbone_learning_rate": 0.001,
            **common,
        },
        FUSION_MODEL,
        initialize_from=initial,
    )

    report = run_model_training_job(phase2)["summary"]
    checkpoint = torch.load(
        phase2.output_dir / "checkpoint_last.pt",
        weights_only=False,
    )

    assert report["resume_contract"]["initial_checkpoint"]["sha256"] == file_sha256(
        initial
    )
    assert [group["lr"] for group in checkpoint["optimizer_state"]["param_groups"]] == [
        0.01,
        0.001,
    ]
    assert checkpoint["history"][0]["learning_rates"] == [0.01, 0.001]
