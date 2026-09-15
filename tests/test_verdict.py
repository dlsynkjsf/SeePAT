from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
verdict_module = pytest.importorskip("seepat.verdict")
from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.evidence import (
    CALIBRATED_INPUT_VERSION,
    FUSION_EVIDENCE_FIELDS,
    calibrated_manifest_contract,
)

nn = torch.nn
choose_threshold = verdict_module.choose_threshold
run_verdict = verdict_module.run_verdict
THRESHOLD_VERSION = verdict_module.THRESHOLD_VERSION
VERDICT_NOT_EVALUATED = verdict_module.VERDICT_NOT_EVALUATED


class TinySwinBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.head = nn.Linear(6, 10)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.projection(video.mean(dim=(2, 3, 4)))


class TinyFrameBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.classifier = nn.Sequential(nn.Linear(6, 10))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projection(frames.mean(dim=(2, 3))))


def _tiny_fusion_model() -> torch.nn.Module:
    from seepat.models.fusion import HybridFusionEventClassifier

    return HybridFusionEventClassifier(
        pretrained=False,
        swin_backbone=TinySwinBackbone(),
        frame_backbone=TinyFrameBackbone(),
        evidence_hidden=8,
        temporal_channels=8,
        embedding_features=7,
        fusion_dimension=8,
        fusion_heads=2,
        dropout=0.0,
    )


def _write_clip(path: Path, frames: int = 4, size: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (size, size),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write MJPG clips in this environment")
    try:
        for index in range(frames):
            value = int(255 * index / max(frames - 1, 1))
            frame = np.full((size, size, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _event_row(
    tmp_path: Path,
    file_name: str,
    event_id: str,
    *,
    class_id: str,
    modality: str,
    phoneme: str = "p",
) -> dict[str, object]:
    clip_path = tmp_path / f"{event_id}.avi"
    _write_clip(clip_path)
    return {
        "event_id": event_id,
        "video_id": stable_id(file_name),
        "file": file_name,
        "dataset_split": "val",
        "source_group": file_name,
        "subject_id": "subject-a",
        "manipulation_modality": modality,
        "phoneme": phoneme,
        "class_id": class_id,
        "mouth_clip_path": clip_path.as_posix(),
        "video_phone_start_s": "1.0",
        "closure_time_s": "1.2",
        "vild_regression_residual_px": "1.5",
        "phoneme_viseme_residual_z": "-0.8",
        "isolation_forest_anomaly_score": "0.55",
        "normalized_minimum_closure": "0.002",
        "closure_duration_s": "0.2",
        "phone_duration_s": "0.08",
        "calibration_version": CALIBRATED_INPUT_VERSION,
        "isolation_forest_available": "True",
        "isolation_forest_scope": "input_video",
        "isolation_forest_unavailable_reason": "",
    }


def _seal_manifest(manifest: Path) -> None:
    population = manifest.parent / "calibration.json"
    atomic_write_json(population, {"calibration_version": CALIBRATED_INPUT_VERSION})
    atomic_write_json(manifest.parent / "summary.json", {
        "scored_manifests": {"val": manifest.as_posix()},
        "scored_manifest_sha256": {"val": file_sha256(manifest)},
        "calibration": population.as_posix(), "calibration_sha256": file_sha256(population),
    })


def _write_checkpoint(path: Path, manifest: Path) -> None:
    model = _tiny_fusion_model()
    contract = {
        "training_version": "hybrid-fusion-v3",
        "model": "seepat.hybrid_fusion.swin3d_b_tempcnn_evidence",
        "model_name": "swin3d_b_vild_fusion", "sequence_length": 4, "image_size": 8,
        "validation_manifest_sha256": file_sha256(manifest),
        "fusion_inputs": calibrated_manifest_contract(manifest, allow_empty=True),
    }
    torch.save(
        {
            "checkpoint_type": "seepat_evaluation_model",
            "completed_epoch": 1,
            "model_state": model.state_dict(),
            "resume_contract": contract,
            "options": {"max_train_batches": None, "max_validation_batches": None},
        },
        path,
    )
    atomic_write_json(path.parent / "run.json", {
        "status": "complete", "run_type": "training_experiment", "global_step": 1,
        "resume_contract": contract, "best_checkpoint": path.as_posix(),
    })


def _patch_builder(monkeypatch) -> None:
    monkeypatch.setattr(
        verdict_module,
        "build_event_classifier",
        lambda model_name, pretrained, freeze_backbone: _tiny_fusion_model(),
    )


def test_choose_threshold_maximizes_balanced_accuracy() -> None:
    selection = choose_threshold(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.2, 0.8, 0.9],
    )

    assert selection["threshold"] == 0.8
    assert selection["balanced_accuracy"] == 1.0
    assert selection["positives"] == 2
    assert selection["negatives"] == 2


def test_choose_threshold_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        choose_threshold(labels=[1, 1], probabilities=[0.9, 0.8])


def test_verdict_requires_a_frozen_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen threshold is required"):
        run_verdict(
            manifest_path=tmp_path / "missing.csv",
            checkpoint_path=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            device_name="cpu",
        )


def test_run_verdict_aggregates_maximum_event_probability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_builder(monkeypatch)
    fake_file = "eval/fake-01.mp4"
    real_file = "eval/real-01.mp4"
    missing_file = "eval/no-events.mp4"
    rows = [
        _event_row(
            tmp_path,
            fake_file,
            "fake-01-a",
            class_id="1",
            modality="both_modified",
            phoneme="b",
        ),
        _event_row(
            tmp_path,
            fake_file,
            "fake-01-b",
            class_id="1",
            modality="both_modified",
        ),
        _event_row(tmp_path, real_file, "real-01-a", class_id="0", modality="real"),
    ]
    manifest_path = tmp_path / "events_test.csv"
    atomic_write_csv(manifest_path, rows)
    source_path = tmp_path / "source.csv"
    atomic_write_csv(
        source_path,
        [{"file": fake_file}, {"file": real_file}, {"file": missing_file}],
    )
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    _seal_manifest(manifest_path)
    _write_checkpoint(checkpoint_path, manifest_path)
    threshold_path = tmp_path / "threshold.json"
    verdict_module.select_threshold(manifest_path, checkpoint_path, threshold_path, device_name="cpu")

    summary = run_verdict(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "out",
        threshold_artifact=threshold_path,
        split="val",
        source_manifest=source_path,
        project_root=tmp_path,
        device_name="cpu",
        batch_size=2,
        sequence_length=4,
        image_size=8,
    )

    assert summary["verdict_version"] == "verdict-v2"
    assert summary["threshold"]["source"] == "artifact"
    assert summary["event_count"] == 3
    assert summary["video_count"] == 3
    assert summary["not_evaluated_videos"] == 1
    assert summary["sync_gap_scores"] is False
    assert summary["metrics"] is not None
    assert summary["metrics"]["video_aggregation"] == "maximum event probability"

    verdicts = {
        row["video_id"]: row
        for row in read_csv_rows(tmp_path / "out" / "video_verdicts.csv")
    }
    assert verdicts[stable_id(missing_file)]["verdict"] == VERDICT_NOT_EVALUATED
    for file_name in (fake_file, real_file):
        row = verdicts[stable_id(file_name)]
        assert row["verdict"] in {"manipulated", "authentic"}
        assert row["event_count"] == ("2" if file_name == fake_file else "1")
        assert float(row["maximum_probability"]) >= 0.0

    events = read_csv_rows(tmp_path / "out" / "event_predictions.csv")
    assert len(events) == 3
    for event in events:
        assert event["sync_gap_score"] == ""
        assert event["evidence_available"] == "7"
        for field in FUSION_EVIDENCE_FIELDS:
            assert field in event
    evaluation = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert evaluation["manifest"]["sha256"]
    assert evaluation["checkpoint"]["sha256"]
