"""Frozen inner threshold transfer across distinct synthetic matched export profiles."""

import json
import shutil
from dataclasses import replace

import pytest
from test_fusion_model import _tiny_model
from test_matched_training import options, synthetic_export

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.decision import DecisionJob, decision_stage_status, run_decision_job
from seepat.training.train import FUSION_MODEL, train_from_manifests
from seepat.verdict import select_threshold


def outer_export(root, paths, directory, profile):
    output = root / "outer_export"
    shutil.copytree(directory, output)
    source = root / "outer_source"
    source.mkdir()
    original_rows = read_csv_rows(paths["val"])
    original_ids = list(dict.fromkeys(row["video_id"] for row in original_rows))
    manifest = source / "events_outer_calibrated.csv"
    atomic_write_csv(manifest, [{**row, **{key: "outer-" + row[key] for key in
                                         ("event_id", "video_id", "source_group")}} for row in original_rows])
    calibration = json.loads((paths["val"].parent / "summary.json").read_text())
    atomic_write_json(source / "summary.json", {
        "scored_manifests": {"outer": manifest.as_posix()},
        "scored_manifest_sha256": {"outer": file_sha256(manifest)},
        "calibration": calibration["calibration"], "calibration_sha256": calibration["calibration_sha256"]})
    contract = {"input_sha256": {path.relative_to(root).as_posix(): file_sha256(path)
                                 for path in (paths["train"], manifest)}}
    atomic_write_json(output / "contract.json", contract)
    summary = json.loads((output / "summary.json").read_text())
    summary["contract"] = contract
    for arm in ("control", "candidate"):
        path = output / f"{arm}_val.csv"
        rows = read_csv_rows(path)
        for row in rows:
            for key in ("clip_id", "video_id", "source_group", "source_event_id", "context_record_id"):
                if row.get(key):
                    row[key] = "outer-" + row[key]
        atomic_write_csv(path, rows)
        summary["output_sha256"][path.name] = file_sha256(path)
    for identifier in original_ids:
        directory = output / "records" / identifier
        renamed = directory.with_name("outer-" + identifier)
        directory.rename(renamed)
        record = json.loads((renamed / "record.json").read_text())
        record["video_id"] = "outer-" + record["video_id"]
        record["source_group"] = "outer-" + record["source_group"]
        for window in record["windows"]:
            for key in ("context_id", "video_id", "source_group"):
                window[key] = "outer-" + window[key]
            old_clip = renamed / window["clip_file"]
            window["clip_file"] = "outer-" + window["clip_file"]
            old_clip.rename(renamed / window["clip_file"])
        atomic_write_json(renamed / "record.json", record)
        atomic_write_json(renamed / "record_hash.json", {"sha256": file_sha256(renamed / "record.json")})
        summary["record_sha256"].pop(f"records/{identifier}/record.json")
        summary["record_sha256"][f"records/outer-{identifier}/record.json"] = file_sha256(renamed / "record.json")
    atomic_write_json(output / "summary.json", summary)
    return output, {**profile, "summary_sha256": file_sha256(output / "summary.json")}


def test_outer_decision_uses_frozen_inner_threshold_and_correct_new_profile(tmp_path, monkeypatch):
    paths, directory, profile = synthetic_export(tmp_path)
    monkeypatch.setattr("seepat.training.train.build_event_classifier", lambda **kw: _tiny_model(freeze_backbone=True))
    monkeypatch.setattr("seepat.verdict.build_event_classifier", lambda **kw: _tiny_model(freeze_backbone=True))
    reference = tmp_path / "reference"
    train_from_manifests(paths["train"], paths["val"], reference, tmp_path, options(), "cpu", False, model_name=FUSION_MODEL)
    initial = reference / "checkpoint_best.pt"
    profile = {**profile, "initial_checkpoint_sha256": file_sha256(initial)}
    fit = tmp_path / "fit"
    train_from_manifests(directory / "candidate_train.csv", directory / "candidate_val.csv", fit,
                         tmp_path, options(), "cpu", False, model_name=FUSION_MODEL,
                         initialize_from=initial, input_profile=profile)
    selected = fit / "checkpoint_best.pt"
    threshold = tmp_path / "inner_threshold.json"
    select_threshold(directory / "candidate_val.csv", selected, threshold, project_root=tmp_path, device_name="cpu")
    original_threshold_hash = file_sha256(threshold)
    output, outer_profile = outer_export(tmp_path, paths, directory, profile)
    job = DecisionJob(name="outer", validation_manifest=output / "candidate_val.csv", source_manifest=None,
                      checkpoint=selected, output_dir=tmp_path / "outer_decisions", project_root=tmp_path,
                      device="cpu", threshold_artifact=threshold, input_profile=outer_profile)
    assert run_decision_job(job)["action"] == "ran"
    assert set(decision_stage_status(job).values()) == {"current"}
    assert run_decision_job(job)["action"] == "skipped"
    assert file_sha256(threshold) == original_threshold_hash
    evaluation = json.loads((job.output_dir / "verdict" / "evaluation.json").read_text())
    assert evaluation["mode"] == "outer_validation"
    assert evaluation["provenance"]["clip_input_contract"]["profile"] == outer_profile
    assert evaluation["metrics"]["events"]["unlabelled_context_clips_excluded"] == 4
    rows = read_csv_rows(job.output_dir / "verdict" / "video_verdicts.csv")
    assert all(row["video_id"].startswith("outer-") for row in rows)
    with pytest.raises(ValueError, match="arm or export differs"):
        run_decision_job(replace(job, validation_manifest=output / "control_val.csv"))
    with pytest.raises(ValueError, match="pinned initializer"):
        run_decision_job(replace(job, input_profile={**outer_profile, "initial_checkpoint_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="manifest differs"):
        select_threshold(output / "candidate_val.csv", selected, tmp_path / "bad_threshold.json",
                         project_root=tmp_path, device_name="cpu")
