"""Approved public inference/CLI seams; tiny synthetic artifacts only."""

import json
from pathlib import Path

import pytest
import torch
from media_fixtures import write_clip

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256
from seepat.evidence import CALIBRATED_INPUT_VERSION, calibrated_manifest_contract
from seepat.inference.frozen_inference import validate_inputs
from seepat.training.video_objective import VIDEO_SUPERVISION_CONTRACT
from seepat.verdict import AGGREGATION, THRESHOLD_VERSION


def binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


@pytest.fixture
def inputs(tmp_path):
    model_dir = tmp_path / "model"
    prepared = tmp_path / "prepared"
    model_dir.mkdir()
    prepared.mkdir()
    calibration = model_dir / "calibration.json"
    preprocess = model_dir / "preprocessing.json"
    atomic_write_json(calibration, {"calibration_version": CALIBRATED_INPUT_VERSION})
    atomic_write_json(preprocess, {"version": "synthetic-preprocessing-only"})
    clip = prepared / "clip.avi"
    clip.write_bytes(b"synthetic-placeholder-never-decoded-in-contract-checks")
    manifest = prepared / "events.csv"
    atomic_write_csv(manifest, [{
        "event_id": "event-a", "video_id": "video-a", "source_group": "group-a",
        "dataset_split": "local", "phoneme": "m", "kind": "bilabial",
        "mouth_clip_path": str(clip), "mouth_clip_sha256": file_sha256(clip),
        "video_phone_start_s": "1.52", "closure_time_s": "1.60",
        "vild_regression_residual_px": "", "phoneme_viseme_residual_z": "",
        "isolation_forest_anomaly_score": "", "isolation_forest_available": "False",
        "isolation_forest_scope": "input_video", "isolation_forest_unavailable_reason": "insufficient_reference",
        "normalized_minimum_closure": "0.02", "closure_duration_s": "0.04", "phone_duration_s": "0.06",
        "calibration_version": CALIBRATED_INPUT_VERSION,
    }])
    calibration_summary = prepared / "summary.json"
    atomic_write_json(calibration_summary, {
        "mode": "score_only", "calibration": str(calibration), "calibration_sha256": file_sha256(calibration),
        "scored_manifests": {"local": str(manifest)},
        "scored_manifest_sha256": {"local": file_sha256(manifest)},
    })
    contract = {
        "model_name": "swin3d_b_vild_fusion", "training_version": "hybrid-fusion-v3",
        "sequence_length": 16, "image_size": 8, "seed": 17,
        "validation_manifest_sha256": "a" * 64,
        "fusion_inputs": calibrated_manifest_contract(manifest),
        "supervision": VIDEO_SUPERVISION_CONTRACT,
    }
    checkpoint = model_dir / "checkpoint_best.pt"
    torch.save({"checkpoint_type": "seepat_evaluation_model", "model_state": {}, "resume_contract": contract,
                "options": {"max_train_batches": None, "max_validation_batches": None,
                            "supervision": "video_max", "seed": 17}}, checkpoint)
    run = model_dir / "run.json"
    atomic_write_json(run, {"status": "complete", "run_type": "training_experiment", "global_step": 3,
                            "best_checkpoint": str(checkpoint), "resume_contract": contract})
    selection_events = model_dir / "threshold_events.csv"
    selection_events.write_text("synthetic saved inner selection; not reused for new inputs\n")
    threshold = model_dir / "threshold.json"
    provenance = {
        "checkpoint_sha256": file_sha256(checkpoint), "validation_manifest_sha256": "a" * 64,
        "model_name": contract["model_name"], "training_version": contract["training_version"],
        "sequence_length": 16, "image_size": 8, "evidence_version": "fusion-evidence-v2",
        "fusion_inputs": contract["fusion_inputs"], "supervision": VIDEO_SUPERVISION_CONTRACT,
    }
    atomic_write_json(threshold, {
        "threshold_version": THRESHOLD_VERSION, "threshold": 0.6, "selection": {"threshold": 0.6},
        "split": "val", "dataset_source": "AV-Deepfake1M++", "aggregation": AGGREGATION,
        "selection_metric": "balanced_accuracy", "tie_break": "lowest threshold among equal balanced accuracies",
        "provenance": provenance, "predictions": binding(selection_events),
    })
    freeze = tmp_path / "freeze.json"
    atomic_write_json(freeze, {
        "version": "frozen-inference-v2", "status": "preparation",
        "representation": "legacy-bilabial-v1", "checkpoint": binding(checkpoint), "run": binding(run),
        "threshold": binding(threshold), "calibration": binding(calibration), "preprocessing": binding(preprocess),
    })
    spec = tmp_path / "inputs.json"
    atomic_write_json(spec, {
        "version": "prepared-inference-v1", "dataset_source": "local-unlabelled", "split": "local",
        "representation": "legacy-bilabial-v1", "preprocessing_sha256": file_sha256(preprocess),
        "manifest": binding(manifest), "calibration_summary": binding(calibration_summary),
        "requested_videos": [{"video_id": "video-a", "source_group": "group-a", "status": "ready"},
                             {"video_id": "video-b", "source_group": "group-b", "status": "alignment_failed"}],
    })
    return freeze, spec


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda *a, **k: pytest.fail("No network in this preparation"))


def test_validate_frozen_bindings_without_inventing_labels_or_loading_a_real_model(inputs):
    report = validate_inputs(*inputs)
    assert report["status"] == "passed"
    assert report["threshold"]["value"] == 0.6
    assert report["events"] == 1 and report["requested"] == 2
    assert report["execution_ready"] is False
    assert "class_id" not in report["rows"][0] and "manipulation_modality" not in report["rows"][0]


def test_legacy_event_supervision_retains_its_own_cutoff_without_contexts(inputs):
    """Phase-2-style contracts remain supported; do not substitute a later clip policy."""
    freeze = json.loads(inputs[0].read_text())
    checkpoint = Path(freeze["checkpoint"]["path"])
    saved = torch.load(checkpoint, weights_only=False)
    saved["options"]["supervision"] = "event"
    saved["resume_contract"].pop("supervision")
    torch.save(saved, checkpoint)
    run_path = Path(freeze["run"]["path"])
    run = json.loads(run_path.read_text())
    run["resume_contract"] = saved["resume_contract"]
    atomic_write_json(run_path, run)
    cutoff_path = Path(freeze["threshold"]["path"])
    cutoff = json.loads(cutoff_path.read_text())
    cutoff["provenance"].pop("supervision")
    cutoff["provenance"]["checkpoint_sha256"] = file_sha256(checkpoint)
    atomic_write_json(cutoff_path, cutoff)
    for name in ("checkpoint", "run", "threshold"):
        freeze[name] = binding(Path(freeze[name]["path"]))
    atomic_write_json(inputs[0], freeze)
    checked = validate_inputs(*inputs)
    assert checked["threshold"]["value"] == .6 and checked["matched_coverage"] is None
    assert checked["contract"].get("supervision") is None
    assert checked["rows"][0]["kind"] == "bilabial"


class TinyInputs:
    """Concrete tensor dataset: synthetic logits encoded in video pixels, no media."""

    def __init__(self):
        self.rows = [{"event_id": "a1", "video_id": "a", "phoneme": "m"},
                     {"event_id": "a2", "video_id": "a", "phoneme": "b"},
                     {"event_id": "c1", "video_id": "c", "phoneme": "p"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {**self.rows[index], "video": torch.full((3, 16, 8, 8), 2.0 if index == 1 else -2.0),
                "frame_mask": torch.ones(16, dtype=torch.bool),
                "evidence_features": torch.zeros(7), "evidence_feature_mask": torch.zeros(7, dtype=torch.bool),
                "label": torch.tensor(-1), "video_label": torch.tensor(-1)}


class TinyModel(torch.nn.Module):
    uses_frame_mask = True
    uses_evidence_features = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, video, *, frame_mask, features, feature_mask):
        assert frame_mask.dtype == torch.bool and feature_mask.dtype == torch.bool
        assert not features[~feature_mask].any()
        margin = video.mean(dim=(1, 2, 3, 4)) * self.scale
        return torch.stack((torch.zeros_like(margin), margin), dim=1)


def test_synthetic_scoring_uses_video_max_and_preserves_unknown_labels_and_failures():
    from seepat.inference.frozen_inference import score_prepared

    model = TinyModel()
    result = score_prepared(model, TinyInputs(), [
        {"video_id": "a", "status": "ready"}, {"video_id": "c", "status": "ready"},
        {"video_id": "b", "status": "alignment_failed"}], 0.6)
    a, c, b = result["videos"]
    assert a["maximum_probability"] == pytest.approx(0.8807970285)
    assert a["verdict"] == "manipulated" and a["event_count"] == 2
    assert c["verdict"] == "authentic" and c["maximum_probability"] == pytest.approx(0.1192029119)
    assert b["verdict"] == "not_evaluated" and b["reason"] == "alignment_failed"
    assert b["maximum_probability"] == "" and b["event_count"] == 0
    assert all("label" not in row and "video_label" not in row for row in result["events"])
    assert result["metrics"] is None and result["labels_available"] is False
    assert model.scale.item() == 1.0 and model.scale.grad is None


def test_cli_defaults_to_verification_and_synthetic_export_is_explicit_and_preserved(inputs, tmp_path, capsys):
    from seepat.inference.frozen_inference import export_result, main, score_prepared

    result = main(["--freeze", str(inputs[0]), "--inputs", str(inputs[1])])
    assert result["status"] == "passed" and result["execution_ready"] is False
    assert "PREPARATION ONLY" in capsys.readouterr().out
    validated = validate_inputs(*inputs)
    dataset = TinyInputs()
    dataset.rows = [{"event_id": "event-a", "video_id": "video-a", "phoneme": "m"}]
    scored = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    output = tmp_path / "result"
    before = {str(p): file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    summary = export_result(validated, scored, output, engineering=True)
    assert summary["status"] == "complete" and summary["scope"] == "synthetic_engineering"
    assert "ENGINEERING CHECK" in (output / "report.md").read_text()
    assert "not_evaluated" in (output / "video_verdicts.csv").read_text()
    for name, expected in summary["output_sha256"].items():
        assert file_sha256(output / name) == expected
    for path, expected in before.items():
        assert file_sha256(Path(path)) == expected
    with pytest.raises(FileExistsError):
        export_result(validated, scored, output, engineering=True)


def test_final_status_alone_does_not_authorize_execution(inputs):
    freeze = json.loads(inputs[0].read_text())
    freeze["status"] = "final_frozen"
    atomic_write_json(inputs[0], freeze)
    assert validate_inputs(*inputs)["execution_ready"] is False


def test_cli_run_stays_locked_for_a_preparation_record(inputs, tmp_path, capsys):
    from seepat.inference.frozen_inference import main

    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        main(["--freeze", str(inputs[0]), "--inputs", str(inputs[1]), "--run", "--output-dir", str(output)])
    assert error.value.code == 1
    assert "not_final_frozen" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("annotation", ["0", "1"])
def test_prepared_clip_decoding_and_scoring_do_not_depend_on_training_annotations(inputs, annotation):
    from seepat.artifacts import read_csv_rows
    from seepat.inference.frozen_inference import PreparedDataset, score_prepared

    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    spec = json.loads(inputs[1].read_text())
    manifest = Path(spec["manifest"]["path"])
    rows = read_csv_rows(manifest)
    clip = Path(rows[0]["mouth_clip_path"])
    write_clip(clip, (np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(3)),
               codec="MJPG", fps=5, skip_unavailable=True)
    rows[0].update(mouth_clip_sha256=file_sha256(clip), class_id=annotation,
                   manipulation_modality="real" if annotation == "0" else "both_modified")
    atomic_write_csv(manifest, rows)
    summary_path = Path(spec["calibration_summary"]["path"])
    summary = json.loads(summary_path.read_text())
    summary["scored_manifest_sha256"]["local"] = file_sha256(manifest)
    atomic_write_json(summary_path, summary)
    spec.update(manifest=binding(manifest), calibration_summary=binding(summary_path))
    atomic_write_json(inputs[1], spec)
    validated = validate_inputs(*inputs)
    dataset = PreparedDataset(validated)
    scored = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    assert scored["videos"][0]["maximum_probability"] == pytest.approx(0.7310585786)
    assert scored["videos"][0]["verdict"] == "manipulated"
    assert dataset[0]["label"].item() == -1 and dataset[0]["video_label"].item() == -1
    assert dataset[0]["frame_mask"].tolist() == [True] * 3 + [False] * 13
    assert "label" not in scored["events"][0]


def test_cli_handles_an_all_failed_inventory_and_skips_a_verified_completed_restart(inputs, tmp_path, capsys):
    from seepat.inference.frozen_inference import main

    spec = json.loads(inputs[1].read_text())
    manifest = Path(spec["manifest"]["path"])
    atomic_write_csv(manifest, [])
    summary_path = Path(spec["calibration_summary"]["path"])
    summary = json.loads(summary_path.read_text())
    summary["scored_manifest_sha256"]["local"] = file_sha256(manifest)
    atomic_write_json(summary_path, summary)
    spec.update(manifest=binding(manifest), calibration_summary=binding(summary_path))
    atomic_write_json(inputs[1], spec)
    freeze = json.loads(inputs[0].read_text())
    from seepat.inference.frozen_inference import _runtime_hashes
    freeze.update(status="final_frozen", approved_scope="local_unlabelled_inference",
                  runtime_sha256=_runtime_hashes())
    atomic_write_json(inputs[0], freeze)
    output = tmp_path / "empty-result"
    args = ["--freeze", str(inputs[0]), "--inputs", str(inputs[1]), "--run", "--output-dir", str(output)]
    result = main(args)
    assert result["status"] == "complete" and result["scope"] == "local_unlabelled_inference"
    text = (output / "report.md").read_text()
    assert text.count("not_evaluated") == 2 and "manipulated;" not in text
    before = {path.name: file_sha256(path) for path in output.iterdir()}
    repeated = main(args)
    assert repeated["action"] == "skipped"
    assert {path.name: file_sha256(path) for path in output.iterdir()} == before
    assert "VERIFIED COMPLETE; SKIPPED" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["external-spec", "test-split", "external-row", "duplicate-video",
                                  "empty-status", "unsafe-id", "wrong-group", "duplicate-event", "context",
                                  "clip-bytes", "calibration-refit", "wrong-preprocessing", "matched-profile"])
def test_incompatible_inputs_stop_before_scoring(inputs, fault):
    from seepat.artifacts import read_csv_rows

    freeze, spec = json.loads(inputs[0].read_text()), json.loads(inputs[1].read_text())
    manifest = Path(spec["manifest"]["path"])
    rows = read_csv_rows(manifest)
    if fault == "external-spec":
        spec["dataset_source"] = "Deepfake-Eval-2024"
    elif fault == "test-split":
        spec["split"] = "test"
    elif fault == "external-row":
        rows[0]["dataset_source"] = "Deepfake-Eval-2024"
    elif fault == "duplicate-video":
        spec["requested_videos"].append(spec["requested_videos"][0])
    elif fault == "empty-status":
        spec["requested_videos"][1]["status"] = ""
    elif fault == "unsafe-id":
        spec["requested_videos"][1]["video_id"] = "<script>bad</script>"
    elif fault == "wrong-group":
        rows[0]["source_group"] = "foreign"
    elif fault == "duplicate-event":
        rows.append(rows[0])
    elif fault == "context":
        rows[0]["kind"] = "context"
    elif fault == "clip-bytes":
        Path(rows[0]["mouth_clip_path"]).write_bytes(b"changed")
    elif fault == "wrong-preprocessing":
        spec["preprocessing_sha256"] = "0" * 64
    elif fault == "matched-profile":
        freeze["representation"] = spec["representation"] = "matched-candidate-v1"
    summary_path = Path(spec["calibration_summary"]["path"])
    summary = json.loads(summary_path.read_text())
    if fault == "calibration-refit":
        summary["mode"] = "fit_and_score"
    atomic_write_csv(manifest, rows)
    summary["scored_manifest_sha256"]["local"] = file_sha256(manifest)
    atomic_write_json(summary_path, summary)
    spec.update(manifest=binding(manifest), calibration_summary=binding(summary_path))
    atomic_write_json(inputs[1], spec)
    atomic_write_json(inputs[0], freeze)
    with pytest.raises(ValueError):
        validate_inputs(*inputs)


@pytest.mark.parametrize("fault", ["wrong-video", "wrong-score", "wrong-verdict", "wrong-count", "wrong-reason",
                                  "nan", "video-label", "phantom-clip"])
def test_export_rejects_inconsistent_scored_decisions(inputs, tmp_path, fault):
    from seepat.inference.frozen_inference import export_result, score_prepared

    validated = validate_inputs(*inputs)
    dataset = TinyInputs()
    dataset.rows = [{"event_id": "event-a", "video_id": "video-a", "phoneme": "m"}]
    scored = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    if fault == "wrong-video":
        scored["events"][0]["video_id"] = "video-b"
    elif fault == "wrong-score":
        scored["videos"][0]["maximum_probability"] = 0.99
    elif fault == "wrong-verdict":
        scored["videos"][0]["verdict"] = "manipulated"
    elif fault == "wrong-count":
        scored["videos"][0]["event_count"] = 67
    elif fault == "wrong-reason":
        scored["videos"][1]["reason"] = ""
    elif fault == "nan":
        scored["events"][0]["manipulated_probability"] = float("nan")
    elif fault == "video-label":
        scored["videos"][0]["label"] = 0
    else:
        scored["events"][0]["event_id"] = "invented"
    output = tmp_path / "invalid-output"
    with pytest.raises(ValueError):
        export_result(validated, scored, output, engineering=True)
    assert not output.exists()


def test_zero_probability_is_not_reported_as_unavailable(inputs, tmp_path):
    from seepat.inference.frozen_inference import export_result, score_prepared

    class ZeroInputs(TinyInputs):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["video"].fill_(-1000)
            return sample

    validated = validate_inputs(*inputs)
    dataset = ZeroInputs()
    dataset.rows = [{"event_id": "event-a", "video_id": "video-a", "phoneme": "m"}]
    result = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    output = tmp_path / "zero"
    export_result(validated, result, output, engineering=True)
    assert "maximum score 0.0" in (output / "report.md").read_text()


@pytest.mark.parametrize("artifact", ["checkpoint", "run", "threshold", "calibration", "preprocessing"])
def test_modified_frozen_artifacts_are_rejected(inputs, artifact):
    freeze = json.loads(inputs[0].read_text())
    with Path(freeze[artifact]["path"]).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        validate_inputs(*inputs)


@pytest.mark.parametrize("fault", ["zero-updates", "preflight", "resumable-only", "supervision-seed", "matched-checkpoint",
                                  "threshold-selection", "missing-evidence-flag"])
def test_resealed_but_scientifically_incompatible_artifacts_still_fail(inputs, fault):
    freeze = json.loads(inputs[0].read_text())
    if fault == "zero-updates":
        path = Path(freeze["run"]["path"])
        run = json.loads(path.read_text())
        run["global_step"] = 0
        atomic_write_json(path, run)
        freeze["run"] = binding(path)
    elif fault in {"preflight", "resumable-only", "supervision-seed", "matched-checkpoint"}:
        path = Path(freeze["checkpoint"]["path"])
        checkpoint = torch.load(path, weights_only=False)
        if fault == "preflight":
            checkpoint["options"]["max_train_batches"] = 2
        elif fault == "resumable-only":
            checkpoint["checkpoint_type"] = "seepat_resumable_training"
        elif fault == "supervision-seed":
            checkpoint["options"]["seed"] = 999
        else:
            checkpoint["resume_contract"]["clip_input_contract"] = {"version": "matched-clips-v1"}
        torch.save(checkpoint, path)
        freeze["checkpoint"] = binding(path)
    elif fault == "threshold-selection":
        path = Path(freeze["threshold"]["path"])
        threshold = json.loads(path.read_text())
        threshold["selection_metric"] = "accuracy"
        atomic_write_json(path, threshold)
        freeze["threshold"] = binding(path)
    else:
        from seepat.artifacts import read_csv_rows

        spec = json.loads(inputs[1].read_text())
        manifest = Path(spec["manifest"]["path"])
        rows = read_csv_rows(manifest)
        rows[0]["isolation_forest_available"] = "True"
        atomic_write_csv(manifest, rows)
        summary_path = Path(spec["calibration_summary"]["path"])
        summary = json.loads(summary_path.read_text())
        summary["scored_manifest_sha256"]["local"] = file_sha256(manifest)
        atomic_write_json(summary_path, summary)
        spec.update(manifest=binding(manifest), calibration_summary=binding(summary_path))
        atomic_write_json(inputs[1], spec)
    atomic_write_json(inputs[0], freeze)
    with pytest.raises(ValueError):
        validate_inputs(*inputs)


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1, True])
def test_scoring_rejects_invalid_thresholds(threshold):
    from seepat.inference.frozen_inference import score_prepared

    with pytest.raises(ValueError):
        score_prepared(TinyModel(), TinyInputs(), [{"video_id": "a", "status": "ready"}], threshold)


@pytest.mark.parametrize("bad_logits", ["nan", "wrong-shape"])
def test_scoring_rejects_invalid_neural_outputs(bad_logits):
    from seepat.inference.frozen_inference import score_prepared

    class Invalid(torch.nn.Module):
        def forward(self, video):
            return torch.full((video.shape[0], 3 if bad_logits == "wrong-shape" else 2), float("nan"))

    with pytest.raises(ValueError):
        score_prepared(Invalid(), TinyInputs(), [{"video_id": "a", "status": "ready"},
                                                {"video_id": "c", "status": "ready"}], 0.6)


@pytest.mark.parametrize("where", ["model", "prepared", "existing"])
def test_export_preserves_protected_directories(inputs, tmp_path, where):
    from seepat.inference.frozen_inference import export_result, score_prepared

    validated = validate_inputs(*inputs)
    dataset = TinyInputs()
    dataset.rows = [{"event_id": "event-a", "video_id": "video-a", "phoneme": "m"}]
    scored = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    output = tmp_path / where / "nested"
    if where == "existing":
        output.mkdir(parents=True)
        (output / "keep.txt").write_text("preserve")
    before = {str(path): file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        export_result(validated, scored, output, engineering=True)
    assert {str(path): file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()} == before


def test_source_mutation_after_scoring_stops_export(inputs, tmp_path):
    from seepat.inference.frozen_inference import export_result, score_prepared

    validated = validate_inputs(*inputs)
    dataset = TinyInputs()
    dataset.rows = [{"event_id": "event-a", "video_id": "video-a", "phoneme": "m"}]
    scored = score_prepared(TinyModel(), dataset, validated["requested_videos"], 0.6)
    with inputs[1].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError):
        export_result(validated, scored, tmp_path / "changed", engineering=True)
    assert not (tmp_path / "changed").exists()


def test_cli_failure_is_redacted_and_has_no_output(inputs, tmp_path, capsys):
    from seepat.inference.frozen_inference import main

    inputs[0].write_text("SYNTHETIC_SECRET_DO_NOT_ECHO")
    with pytest.raises(SystemExit) as error:
        main(["--freeze", str(inputs[0]), "--inputs", str(inputs[1])])
    assert error.value.code == 1
    printed = capsys.readouterr()
    assert "input_contract_failed" in printed.err
    assert "SYNTHETIC_SECRET_DO_NOT_ECHO" not in printed.out + printed.err


@pytest.mark.parametrize("flags", [["--run"], ["--output-dir", "never"], ["--threshold", "0.1"], ["--fit"]])
def test_cli_refuses_ambiguous_execution_and_refitting_flags(inputs, flags):
    from seepat.inference.frozen_inference import main

    with pytest.raises(SystemExit) as error:
        main(["--freeze", str(inputs[0]), "--inputs", str(inputs[1]), *flags])
    assert error.value.code == 2


def test_cli_invalid_checkpoint_failure_is_redacted(inputs, capsys):
    from seepat.inference.frozen_inference import main

    freeze = json.loads(inputs[0].read_text())
    checkpoint = Path(freeze["checkpoint"]["path"])
    checkpoint.write_bytes(b"SYNTHETIC_SECRET_INVALID_CHECKPOINT")
    freeze["checkpoint"] = binding(checkpoint)
    atomic_write_json(inputs[0], freeze)
    with pytest.raises(SystemExit) as error:
        main(["--freeze", str(inputs[0]), "--inputs", str(inputs[1])])
    assert error.value.code == 1
    printed = capsys.readouterr()
    assert "SYNTHETIC_SECRET" not in printed.out + printed.err
