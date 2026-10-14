from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from seepat.artifacts import atomic_write_csv, file_sha256, read_csv_rows, stable_id
from seepat.coverage import (
    CoveragePilotJob,
    context_windows,
    coverage_outputs_are_current,
    frames_overlap,
    run_coverage_pilot,
    select_pilot_sources,
)
from seepat.progress import read_live_workflow_progress, read_workflow_progress
from seepat.workflow import format_workflow_summary, run_workflow


def make_job(tmp_path: Path, count: int = 4) -> CoveragePilotJob:
    sources, videos, events = [], [], []
    for index in range(count):
        file = f"video-{index}.mp4"
        identifier = stable_id(file)
        category = ("real", "audio_modified", "visual_modified", "both_modified")[index % 4]
        segment = "[[5.5, 5.8]]"
        sources.append({
            "file": file, "original": f"original-{index}.mp4", "split": "train",
            "modify_type": category, "video_frames": 250,
            "fake_segments_json": "[]" if category == "real" else segment,
            "audio_fake_segments_json": segment if category in {
                "audio_modified", "both_modified"} else "[]",
            "visual_fake_segments_json": segment if category in {
                "visual_modified", "both_modified"} else "[]",
        })
        videos.append({
            "file": file, "video_id": identifier, "split": "train", "fps": 25,
            "video_frame_count": 250, "pipeline_status": "complete",
            "eligibility_status": "eligible",
        })
        events.append({
            "file": file, "video_id": identifier, "event_id": f"{identifier}-000",
            "source_group": f"original-{index}.mp4", "dataset_split": "train",
            "mouth_crop_frame_indices_json": json.dumps(list(range(16))),
        })
    paths = {"source_manifest": tmp_path / "source.csv",
             "video_manifest": tmp_path / "videos.csv",
             "event_manifest": tmp_path / "events.csv"}
    for key, rows in zip(paths, (sources, videos, events)):
        atomic_write_csv(paths[key], rows)
    return CoveragePilotJob("fixture", **paths, output_dir=tmp_path / "pilot", max_videos=count,
                            expected_sha256={key: file_sha256(p) for key, p in paths.items()})


def rebind(job: CoveragePilotJob) -> CoveragePilotJob:
    return replace(job, expected_sha256={key: file_sha256(getattr(job, key))
                                        for key in job.expected_sha256})


def test_fixed_schedule_is_bounded_and_short_padding_is_not_new_evidence():
    windows = context_windows(250, 25)
    assert len(windows) == 8
    assert windows[0]["start_frame"] == 8
    assert windows[-1]["stop_frame_exclusive"] == 242
    assert all(len(w["source_frames"]) == len(w["frame_mask"]) == 16 for w in windows)
    short = context_windows(4, 25)
    assert len(short) == 1
    assert short[0]["source_frames"][:4] == [0, 1, 2, 3]
    assert sum(short[0]["frame_mask"]) == 4
    for fps in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            context_windows(250, fps)


def test_frame_proxy_uses_half_open_intervals():
    assert frames_overlap({10}, 10, [(1.0, 1.01)])
    assert not frames_overlap({10}, 10, [(1.1, 1.2)])


def test_group_selection_is_order_and_label_independent():
    rows = [{"file": f"v-{i}", "original": f"group-{i}", "modify_type": "real"}
            for i in range(10)]
    expected = [r["file"] for r in select_pilot_sources(rows, 4, 42)]
    changed = [dict(r, modify_type="both_modified", fake_segments_json="[[0, 1]]")
               for r in reversed(rows)]
    assert [r["file"] for r in select_pilot_sources(changed, 4, 42)] == expected
    with pytest.raises(ValueError, match="duplicate source groups"):
        select_pilot_sources(rows + [rows[0]], 4, 42)


def test_synthetic_recovery_of_missed_visual_span_does_not_export_training_rows(tmp_path):
    job = make_job(tmp_path, 128)
    result = run_coverage_pilot(job)
    summary = result["summary"]
    assert summary["selected_videos"] == summary["videos_audited"] == 128
    assert summary["visual_manipulated_videos_audited"] == 64
    assert summary["baseline_visual_covered"] == 0
    assert summary["combined_visual_covered"] == 64
    assert summary["gate"]["proceed_to_media_qa"] is True
    plan = read_csv_rows(job.output_dir / "context_plan.csv")
    assert len(plan) == 1024
    assert not ({"class_id", "phoneme", "mouth_clip_path", "category", "event_id"} & set(plan[0]))
    assert coverage_outputs_are_current(job)
    assert run_coverage_pilot(job)["action"] == "skipped"


def test_changing_annotation_and_category_cannot_change_context_plan(tmp_path):
    job = make_job(tmp_path)
    run_coverage_pilot(job)
    before = (job.output_dir / "context_plan.csv").read_bytes()
    rows = read_csv_rows(job.source_manifest)
    for row in rows:
        row.update(modify_type="real", fake_segments_json="[]",
                   audio_fake_segments_json="[]", visual_fake_segments_json="[]")
    atomic_write_csv(job.source_manifest, rows)
    run_coverage_pilot(replace(rebind(job), output_dir=tmp_path / "other"))
    assert (tmp_path / "other/context_plan.csv").read_bytes() == before


def test_existing_failure_and_bad_map_are_retained_without_replacement(tmp_path):
    job = make_job(tmp_path)
    rows = read_csv_rows(job.video_manifest)
    rows[0]["pipeline_status"] = "failed"
    atomic_write_csv(job.video_manifest, rows)
    events = read_csv_rows(job.event_manifest)
    events[1]["mouth_crop_frame_indices_json"] = "[999]"
    atomic_write_csv(job.event_manifest, events)
    result = run_coverage_pilot(rebind(job))["summary"]
    assert result["selected_videos"] == 4
    assert result["status_counts"] == {"unavailable": 2, "audited": 2}
    assert result["gate"]["proceed_to_media_qa"] is False
    assert len(read_csv_rows(job.output_dir / "video_coverage.csv")) == 4


@pytest.mark.parametrize("bad_field,bad_value", [("split", "val"),
                                                ("dataset_source", "Deepfake-Eval-2024")])
def test_non_train_or_external_tables_are_rejected_before_output(tmp_path, bad_field, bad_value):
    job = make_job(tmp_path)
    rows = read_csv_rows(job.source_manifest)
    rows[0][bad_field] = bad_value
    atomic_write_csv(job.source_manifest, rows)
    with pytest.raises(ValueError, match=r"original AV\+\+ Train"):
        run_coverage_pilot(rebind(job))
    assert not job.output_dir.exists()


def test_input_drift_and_completed_output_corruption_are_not_overwritten(tmp_path):
    job = make_job(tmp_path)
    run_coverage_pilot(job)
    path = job.output_dir / "context_plan.csv"
    path.write_text("corrupted\n")
    assert not coverage_outputs_are_current(job)
    with pytest.raises(FileExistsError):
        run_coverage_pilot(job)
    assert path.read_text() == "corrupted\n"
    job.source_manifest.write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen"):
        run_coverage_pilot(job)


def test_incomplete_owned_output_resumes_with_same_contract(tmp_path):
    job = make_job(tmp_path)
    run_coverage_pilot(job)
    (job.output_dir / "summary.json").unlink()
    assert run_coverage_pilot(job)["action"] == "ran"
    assert coverage_outputs_are_current(job)


def test_invalid_annotation_is_reported_without_changing_planned_windows(tmp_path):
    job = make_job(tmp_path)
    rows = read_csv_rows(job.source_manifest)
    rows[0]["fake_segments_json"] = "[[NaN, 1]]"
    atomic_write_csv(job.source_manifest, rows)
    summary = run_coverage_pilot(rebind(job))["summary"]
    assert summary["status_counts"]["annotation_unavailable"] == 1
    assert summary["proposed_context_windows"] == 32


def test_source_group_provenance_and_output_input_overlap_are_rejected(tmp_path):
    job = make_job(tmp_path)
    with pytest.raises(ValueError, match="contain an input"):
        replace(job, output_dir=tmp_path).validate()
    rows = read_csv_rows(job.event_manifest)
    rows[0]["source_group"] = "wrong"
    atomic_write_csv(job.event_manifest, rows)
    with pytest.raises(ValueError, match="provenance"):
        run_coverage_pilot(rebind(job))


def test_workflow_tracker_verification_and_negative_gate_are_independent(tmp_path):
    job = make_job(tmp_path)
    raw_job = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(job).items()}
    config = tmp_path / "workflow.yaml"
    report_path = tmp_path / "workflow.json"
    config.write_text(yaml.safe_dump({"report": str(report_path), "coverage_pilot": raw_job}))
    report = run_workflow(config)
    assert report["coverage_pilot"][0]["summary"]["gate"]["proceed_to_media_qa"] is False
    assert read_live_workflow_progress(config)["status"] == "complete"
    verification = read_workflow_progress(config)
    assert verification["all_current"] and verification["stages_total"] == 1
    assert verification["coverage_pilots"][0]["gate"]["decision"] == "stop_and_review"
    assert "not a classifier result" in format_workflow_summary(report, report_path)
    assert run_workflow(config)["coverage_pilot"][0]["action"] == "skipped"
