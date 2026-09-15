from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_fusion_model import _tiny_model

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.evidence import CALIBRATED_INPUT_VERSION, EVIDENCE_VERSION, FUSION_EVIDENCE_FIELDS
from seepat.training.dataset import MouthEventDataset
from seepat.training.train import FUSION_MODEL, TrainingOptions, train_from_manifests
from seepat.workflow import (
    ModelTrainingJob,
    load_workflow_settings,
    model_training_outputs_are_current,
    run_model_training_job,
)

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def calibrated_inputs(tmp_path):
    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (8, 8))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write a synthetic MJPG clip")
    for index in range(4):
        writer.write(np.full((8, 8, 3), 30 + index * 20, dtype=np.uint8))
    writer.release()
    population = tmp_path / "calibration.json"
    atomic_write_json(population, {"calibration_version": CALIBRATED_INPUT_VERSION})
    paths = {}
    for split in ("train", "val"):
        rows = []
        for index in range(2):
            rows.append(
                {
                    **{field: "0" for field in FUSION_EVIDENCE_FIELDS},
                    "event_id": f"{split}-{index}",
                    "video_id": f"{split}-{index}",
                    "source_group": f"{split}-{index}",
                    "dataset_split": split,
                    "class_id": str(index),
                    "phoneme": "p",
                    "mouth_clip_path": clip.as_posix(),
                    "manipulation_modality": "real" if index == 0 else "both_modified",
                    "closure_time_s": "0.2",
                    "video_phone_start_s": "0.21",
                    "calibration_version": CALIBRATED_INPUT_VERSION,
                    "isolation_forest_available": "True" if index == 0 else "False",
                    "isolation_forest_anomaly_score": "0" if index == 0 else "",
                    "isolation_forest_scope": "input_video" if index == 0 else "unavailable",
                    "isolation_forest_unavailable_reason": (
                        "" if index == 0 else "insufficient_video_reference_frames"
                    ),
                }
            )
        paths[split] = tmp_path / f"events_{split}_calibrated.csv"
        atomic_write_csv(paths[split], rows)
    atomic_write_json(
        tmp_path / "summary.json",
        {
            "scored_manifests": {k: p.as_posix() for k, p in paths.items()},
            "scored_manifest_sha256": {k: file_sha256(p) for k, p in paths.items()},
            "calibration": population.as_posix(),
            "calibration_sha256": file_sha256(population),
        },
    )
    return paths


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


def test_fusion_preflight_config_resumes_into_normal_workflow_without_changing_baselines():
    scaled = load_workflow_settings(Path("configs/workflow_scaled.yaml"))
    setup = load_workflow_settings(Path("configs/workflow_fusion_preflight.yaml"))
    swin, cnn, fusion = scaled.model_training_jobs
    assert (swin.model, cnn.model) == ("swin3d_b", "efficientnet_v2_s_tempcnn")
    assert str(swin.output_dir).endswith("local_swin_preflight_guarded")
    assert str(cnn.output_dir).endswith("local_cnn_temporal_preflight_guarded_fp32")
    assert not cnn.options["amp"] and not fusion.options["amp"]
    assert len({job.output_dir for job in scaled.model_training_jobs}) == 3
    assert (
        not setup.jobs
        and not setup.trace_augmentation_jobs
        and not setup.numerical_calibration_jobs
    )
    assert setup.model_training_jobs == (replace(fusion, options={**fusion.options, "epochs": 1}),)
    assert fusion.options["epochs"] == 2
    assert not fusion.pretrained and fusion.options["freeze_backbone"]
