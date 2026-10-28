"""Synthetic public boundaries for the Train-only auxiliary-target audit."""

import importlib.util
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_gzip_json,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.live_progress import live_progress_path

SPEC = importlib.util.spec_from_file_location(
    "clip_targets", Path(__file__).parents[1] / "scripts/audit_clip_targets.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def annotation(category="visual_modified", visual=((1., 2.),)):
    audio = ((3., 4.),) if category in {"audio_modified", "both_modified"} else ()
    return {"modify_type": category, "visual_fake_segments_json": json.dumps(visual),
            "audio_fake_segments_json": json.dumps(audio),
            "fake_segments_json": json.dumps([*visual, *audio])}


def slots(times, *, padded=False):
    result = [{"source_frame": i, "timestamp_s": t, "rgb_valid": True, "clock_valid": True}
              for i, t in enumerate(times)]
    if padded:
        result.extend({**result[-1], "rgb_valid": False, "clock_valid": False}
                      for _ in range(16 - len(result)))
    return result


def test_targets_mean_visual_edit_exposure_not_video_fake_or_lip_anomaly():
    result = audit.visual_target(annotation(), slots([.2, 1.3, 2.5], padded=True), fps=25, duration=5)
    assert result["aux_visual_target"] == 1 and result["aux_target_mask"] is True
    assert result["inside_unique_frames"] == 1 and result["rgb_unique_frames"] == 3
    audio_only = annotation("audio_modified", ())
    negative = audit.visual_target(audio_only, slots([3.2, 3.5], padded=True), fps=25, duration=5)
    assert negative["aux_visual_target"] == 0 and negative["aux_target_mask"] is True
    assert audio_only["modify_type"] == "audio_modified"
    edge = audit.visual_target(annotation(), slots([1., 2.], padded=True), fps=25, duration=5)
    assert edge["aux_visual_target"] == -1 and edge["aux_target_mask"] is False


@pytest.mark.parametrize("change", [
    lambda a: a.pop("visual_fake_segments_json"),
    lambda a: a.update(visual_fake_segments_json="[]"),
    lambda a: a.update(visual_fake_segments_json="[[NaN,2]]"),
    lambda a: a.update(visual_fake_segments_json="[[1,1]]"),
    lambda a: a.update(visual_fake_segments_json="[[1,200]]"),
    lambda a: a.update(modify_type="audio_modified"),
])
def test_missing_invalid_or_contradictory_annotations_stay_unknown(change):
    source = annotation()
    change(source)
    result = audit.visual_target(source, slots([1.5], padded=True), fps=25, duration=5)
    assert result["aux_visual_target"] == -1 and not result["aux_target_mask"]
    assert result["reason"].startswith("annotation_unavailable")


def test_unknown_rgb_timing_cannot_become_negative_and_padding_cannot_become_positive():
    support = slots([.2, 1.5], padded=True)
    support[1]["clock_valid"] = False
    result = audit.visual_target(annotation(), support, fps=25, duration=5)
    assert result["aux_visual_target"] == -1
    support[1]["rgb_valid"] = False
    result = audit.visual_target(annotation(), support, fps=25, duration=5)
    assert result["aux_visual_target"] == 0 and result["rgb_unique_frames"] == 1
    unknown = audit.visual_target(annotation(), slots([.2]), fps=25, duration=5, clock_ok=False)
    assert unknown["aux_visual_target"] == -1
    outside = audit.visual_target(annotation("real", ()), slots([50.]), fps=25, duration=5)
    assert outside["aux_visual_target"] == -1


def joined_inputs():
    source = {**annotation(), "file": "visual/train.mp4", "split": "train", "source_group": "source/train"}
    identifier = stable_id(source["file"])
    inventory = [{"video_id": identifier, "source_group": source["source_group"], "dataset_split": "train",
                  "category": "visual_modified", "status": "paired", "reason": "", "expanded_clips": "2"},
                 {"video_id": stable_id("real/missing.mp4"), "source_group": "source/missing", "dataset_split": "train",
                  "category": "real", "status": "unevaluated", "reason": "existing_missing", "expanded_clips": "0"}]
    missing = {**annotation("real", ()), "file": "real/missing.mp4", "split": "train", "source_group": "source/missing"}
    inner = [{"video_id": "inner-id", "source_group": "source/inner", "dataset_split": "val"}]
    common = {"video_id": identifier, "source_group": source["source_group"], "dataset_split": "train"}
    index = [{**common, "clip_id": f"{identifier}-0", "source_event_id": f"{identifier}-0", "kind": "bilabial"},
             {**common, "clip_id": f"{identifier}-context-0", "source_event_id": "", "kind": "context"}]
    events = [{**common, "event_id": index[0]["clip_id"], "file": source["file"],
               "manipulation_modality": source["modify_type"], "class_id": "0", "phoneme": "p"}]
    timing = {r["clip_id"]: {**common, "kind": r["kind"], "decoded_frame_count": 3 if r["kind"] == "bilabial" else 16,
                           "clip_sha256": "b" * 64, "trace_sha256": ""} for r in index}
    return [source, missing], inventory, inner, index, events, timing


def test_fixed_train_join_retains_unavailable_and_is_independent_of_interval_contents():
    inputs = joined_inputs()
    cases = audit.bind_train(*inputs)
    assert len(cases) == 2 and cases[1]["clips"] == []
    ids = [r["clip_id"] for r in cases[0]["clips"]]
    changed = deepcopy(inputs)
    changed[0][0]["visual_fake_segments_json"] = "malformed annotations"
    assert [r["clip_id"] for r in audit.bind_train(*changed)[0]["clips"]] == ids
    leaked = deepcopy(inputs)
    leaked[2][0]["source_group"] = leaked[1][1]["source_group"]
    with pytest.raises(ValueError, match="separation"):
        audit.bind_train(*leaked)
    foreign = deepcopy(inputs)
    foreign[0][0]["split"] = "val"
    with pytest.raises(ValueError, match="Train"):
        audit.bind_train(*foreign)


def case_fixture():
    from test_sequence_evidence import example

    data = joined_inputs()
    row, trace = example()
    source, event = data[0][0], data[4][0]
    event.update({k: v for k, v in row.items() if k not in {"event_id", "video_id", "file", "source_group", "dataset_split"}})
    trace.update(video_id=event["video_id"], source_file=source["file"], source_group=source["source_group"],
                 summary={"fps_fallback_timestamp_frames": 0})
    trace["timing"].update(duration_s=5., video_start_time_s=0.)
    trace["bilabial_event_windows"][0]["event_id"] = event["event_id"]
    data[5][event["event_id"]]["timing_frame_indices"] = [10, 12, 15] + [15] * 13
    case = audit.bind_train(*data)[0]
    context_id = data[3][1]["clip_id"]
    ids = [10, 12, 15] + [15] * 13
    mask = [True] * 3 + [False] * 13
    points = [1., 1.13, 1.41] + [1.41] * 13
    window = {"context_id": context_id, "video_id": event["video_id"], "dataset_split": "train",
              "source_group": source["source_group"], "technical_ok": True, "phoneme": "",
              "event_class_id": "", "evidence_available": False, "clip_sha256": "b" * 64,
              "source_frames": ids, "frame_mask": mask, "planned_mask": mask,
              "slots": [{"source_frame": f, "decoder_pts_s": t, "trace_pts_s": t,
                         "planned_valid": keep, "crop_valid": keep, "timing_ok": True, "face_count": 1}
                        for f, t, keep in zip(ids, points, mask)]}
    record = {"video_id": event["video_id"], "split": "train", "file": source["file"],
              "source_group": source["source_group"], "windows": [window]}
    return case, trace, record


def test_support_uses_recorded_video_times_without_reapplying_audio_offset_or_changing_labels():
    case, trace, record = case_fixture()
    result = audit.audit_case(case, trace, record)
    assert [c["clip_id"] for c in result["clips"]] == [c["clip_id"] for c in case["clips"]]
    first, context = result["clips"]
    assert first["support"][1]["timestamp_s"] == 1.13
    assert first["original_event_class_id"] == "0" and first["aux_visual_target"] == 1
    assert context["original_event_class_id"] == "-1" and context["original_phoneme"] == ""
    assert context["aux_visual_target"] == 1
    assert sum(s["rgb_valid"] for s in context["support"]) == 3
    trace["timing"]["video_start_time_s"] = .5
    unknown = audit.audit_case(case, trace, record)
    assert all(not c["aux_target_mask"] for c in unknown["clips"])


def test_audit_writes_only_a_target_sidecar_preserves_requests_and_reuses_sealed_completion(tmp_path):
    case, trace, record = case_fixture()
    missing = audit.bind_train(*joined_inputs())[1]
    output = tmp_path / "audit"
    result = audit.run_audit([case, missing], output, lambda _: (trace, record), {"fixed": "yes"})
    assert result["requested"] == 2 and result["original_usable"] == 1 and result["original_unavailable"] == 1
    assert result["clips"] == 2 and result["targets"] == {"-1": 0, "0": 0, "1": 2}
    assert result["gate"] == "review_auxiliary_supervision_design" and result["training_ready"] is False
    rows = read_csv_rows(output / "clips.csv")
    assert rows[1]["original_event_class_id"] == "-1" and rows[1]["original_phoneme"] == ""
    first = file_sha256(output / "seal.json")
    assert audit.run_audit([], output, lambda _: pytest.fail("No inputs should reopen"), {"fixed": "yes"}) == result
    assert audit.verify_saved(output, {"fixed": "yes"}) == result
    assert file_sha256(output / "seal.json") == first
    with pytest.raises(ValueError, match="binding|settings"):
        audit.run_audit([], output, lambda _: (None, None), {"fixed": "changed"})
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.run_audit([], partial, lambda _: pytest.fail("No partial overwrite"), {})


def test_cli_help_and_terminal_watch_do_not_open_inputs_or_launch_a_worker(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        audit.main(["--help"])
    assert stopped.value.code == 0
    status = live_progress_path(tmp_path / audit.STATUS)
    atomic_write_json(status, {"status": "complete", "jobs_total": 1, "job_index": 1,
                               "phase": "complete", "finished": 128, "total": 128})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    audit.main(["watch", "--project", str(tmp_path)])
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert "complete" in capsys.readouterr().out.lower()


def test_protocol_drift_is_rejected_and_verify_does_not_open_original_metadata(tmp_path):
    old = json.loads((Path(__file__).parents[1] / "configs/frozen_feature_probe_v1.json").read_text())
    paths = {p for p in old["code_bindings"] if p.startswith("src/")} | {"scripts/audit_clip_targets.py"}
    data = b"synthetic source binding"
    definition = {"design": deepcopy(audit.DESIGN), "metadata_bindings": dict.fromkeys(audit.PATHS.values(), "a" * 64),
                  "code_bindings": dict.fromkeys(paths, sha256(data).hexdigest())}
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    config = tmp_path / "configs/clip_target_audit_v1.json"
    atomic_write_json(config, definition)
    assert audit.load_protocol(tmp_path, config)[0] == definition
    for change in (lambda p: p["design"].update(training_ready=True),
                   lambda p: p["design"]["inputs"].update(source="data/deepfake_eval/source.csv"),
                   lambda p: p["design"].update(boundary_guard="tuned")):
        changed = deepcopy(definition)
        change(changed)
        atomic_write_json(config, changed)
        with pytest.raises(ValueError, match="design"):
            audit.load_protocol(tmp_path, config)
    atomic_write_json(config, definition)
    with pytest.raises(FileNotFoundError):
        audit.main(["verify", "--project", str(tmp_path)])
    assert not (tmp_path / audit.OUTPUT).exists()
    (tmp_path / "scripts/audit_clip_targets.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        audit.load_protocol(tmp_path, config)


def test_saved_train_metadata_loader_needs_no_media_and_rejects_foreign_trace_paths(tmp_path):
    case, trace, record = case_fixture()
    identifier = case["inventory"]["video_id"]
    relative = f"outputs/train-subset5000/vild_traces/{identifier}.json.gz"
    atomic_write_gzip_json(tmp_path / relative, trace)
    digest = file_sha256(tmp_path / relative)
    event = case["clips"][0]["event"]
    event.update(vild_trace_path=relative, vild_trace_sha256=digest)
    case["clips"][0]["timing"]["trace_sha256"] = digest
    record.update(trace_path=relative, trace_sha256=digest)
    name = f"records/{identifier}/record.json"
    path = tmp_path / "outputs/phase5_context_inputs_v1" / name
    atomic_write_json(path, record)
    summary = {"record_sha256": {name: file_sha256(path)}}
    loaded_trace, loaded_context = audit.load_support(tmp_path, case, summary)
    assert loaded_trace == trace and loaded_context == record
    event["vild_trace_path"] = "data/deepfake_eval/trace.json.gz"
    with pytest.raises(ValueError, match="Train"):
        audit.load_support(tmp_path, case, summary)


def test_clock_and_context_contract_changes_are_not_silently_relabeled():
    case, trace, record = case_fixture()
    trace["summary"]["fps_fallback_timestamp_frames"] = False
    assert all(c["aux_visual_target"] == -1 for c in audit.audit_case(case, trace, record)["clips"])
    trace["summary"]["fps_fallback_timestamp_frames"] = 1
    unknown = audit.audit_case(case, trace, record)
    assert all(c["aux_visual_target"] == -1 for c in unknown["clips"])
    summary = audit.summarize([case["inventory"]], [{"category": case["inventory"]["category"], **c} for c in unknown["clips"]])
    assert summary["support_unavailable_reasons"]["fallback_timestamp_provenance_unavailable"] == 2
    case, trace, record = case_fixture()
    trace["timing"]["audio_video_start_offset_s"] = .2
    with pytest.raises(ValueError, match="clocks"):
        audit.audit_case(case, trace, record)
    case, trace, record = case_fixture()
    record["windows"][0]["frame_mask"][0] = False
    with pytest.raises(ValueError, match="mask"):
        audit.audit_case(case, trace, record)


def test_complete_cli_preflight_accepts_bound_train_metadata_without_reading_traces(tmp_path):
    sources, inventory, index, events, timing = [], [], [], [], {}
    paired = 0
    for category in sorted(audit.CATEGORIES):
        for i in range(32):
            file = f"{category}/train-{i}.mp4"
            identifier = stable_id(file)
            source = {**annotation(category, ((1., 2.),) if category in {"visual_modified", "both_modified"} else ()),
                      "file": file, "split": "train", "source_group": file}
            sources.append(source)
            missing = category == "real" and i == 0 or category == "visual_modified" and i < 2
            count = 0 if missing else 11 if paired < 90 else 10
            row = {"video_id": identifier, "dataset_split": "train", "source_group": file,
                   "category": category, "status": "unevaluated" if missing else "paired",
                   "reason": "fixture_missing" if missing else "", "expanded_clips": str(count)}
            inventory.append(row)
            if missing:
                continue
            paired += 1
            common = {k: row[k] for k in ("video_id", "dataset_split", "source_group")}
            for j in range(count):
                kind = "bilabial" if j < count - 8 else "context"
                clip = f"{identifier}-{j}"
                index.append({**common, "clip_id": clip, "kind": kind,
                              "source_event_id": clip if kind == "bilabial" else ""})
                if kind == "bilabial":
                    events.append({**common, "file": file, "event_id": clip, "class_id": "0",
                                   "manipulation_modality": category, "phoneme": "p"})
                timing[clip] = {**common, "kind": kind, "decoded_frame_count": 3 if kind == "bilabial" else 16,
                                "clip_sha256": "b" * 64, "trace_sha256": ""}
    excluded = [{"video_id": f"inner-{i}", "dataset_split": "val", "source_group": f"inner/{i}"} for i in range(64)]
    csvs = {"source": sources, "inventory": inventory, "excluded": excluded, "index": index, "events": events}
    for key, rows in csvs.items():
        atomic_write_csv(tmp_path / audit.PATHS[key], rows)
    atomic_write_json(tmp_path / audit.PATHS["timing"], timing)
    atomic_write_json(tmp_path / audit.PATHS["timing_job"], {"stage": "inputs", "timing_version": "fusion-vild-sequence-v1"})
    atomic_write_json(tmp_path / audit.PATHS["timing_summary"], {"status": "complete", "roles": {"train": {"clips": 1340}}})
    atomic_write_json(tmp_path / audit.PATHS["timing_seal"], {
        Path(audit.PATHS[k]).name: {"sha256": file_sha256(tmp_path / audit.PATHS[k])}
        for k in ("timing", "timing_job", "timing_summary")})
    atomic_write_json(tmp_path / audit.PATHS["selection_seal"], {
        Path(audit.PATHS[k]).name: {"sha256": file_sha256(tmp_path / audit.PATHS[k])}
        for k in ("inventory", "excluded", "index")})
    atomic_write_json(tmp_path / audit.PATHS["contexts"], {"status": "complete", "record_sha256": {},
        "contract": {"input_sha256": {audit.PATHS[k]: file_sha256(tmp_path / audit.PATHS[k]) for k in ("source", "events")}}})
    old = json.loads((Path(__file__).parents[1] / "configs/frozen_feature_probe_v1.json").read_text())
    code_paths = {p for p in old["code_bindings"] if p.startswith("src/")} | {"scripts/audit_clip_targets.py"}
    data = b"synthetic bound source"
    for name in code_paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    protocol = {"design": deepcopy(audit.DESIGN), "code_bindings": dict.fromkeys(code_paths, sha256(data).hexdigest()),
                "metadata_bindings": {p: file_sha256(tmp_path / p) for p in audit.PATHS.values()}}
    atomic_write_json(tmp_path / "configs/clip_target_audit_v1.json", protocol)
    assert audit.main(["preflight", "--project", str(tmp_path)]) == {"requested": 128, "usable": 125, "clips": 1340}
    assert not (tmp_path / audit.OUTPUT).exists()
    assert not list(tmp_path.rglob("*.json.gz"))
