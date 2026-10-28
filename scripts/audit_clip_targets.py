"""Train-only visual-edit target audit; no media decoding, fitting or model scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.data.sampling import canonical_source_group
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.models.sequence_evidence import event_timing_sequence
from seepat.preprocessing.vild import load_vild_trace
from seepat.training import experiment as runtime
from seepat.training.dataset import sequence_frame_indices
from seepat.training.matched_inputs import inside

EPS = 1e-6
CATEGORIES = {"real", "audio_modified", "visual_modified", "both_modified"}
GATE = "review_auxiliary_supervision_design"
OUTPUT = "outputs/engineering/clip_target_audit_v1"
STATUS = "outputs/engineering/clip_target_audit_v1_status.json"
SELECTION = "outputs/engineering/smooth_loss_pilot_v1/selection"
SAVED = "outputs/engineering/sequence_evidence_pilot_v2/inputs"
PATHS = {
    "source": "data/manifests/train_subset5000.csv",
    "events": "outputs/numerical_calibration/guarded_per_video_v3/events_train_calibrated.csv",
    "inventory": f"{SELECTION}/inventory_train.csv", "excluded": f"{SELECTION}/inventory_inner.csv",
    "index": f"{SELECTION}/index_train.csv", "selection_seal": f"{SELECTION}/seal.json",
    "timing": f"{SAVED}/train.json", "timing_job": f"{SAVED}/job.json",
    "timing_summary": f"{SAVED}/summary.json", "timing_seal": f"{SAVED}/seal.json",
    "contexts": "outputs/phase5_context_inputs_v1/summary.json",
}
DESIGN = {"version": "clip-target-audit-v1", "output": OUTPUT, "inputs": PATHS,
          "requests": 128, "usable": 125, "clips": 1340, "excluded_inner_requests": 64,
          "selection": "unchanged saved expanded Train bags; no annotation-based selection",
          "clock": "zero video origin; recorded compatible timestamps; no FPS fallback",
          "boundary_guard": "one source-frame period plus 1e-6 seconds; audit convention, not a proven error bound",
          "target": "visual-edit exposure in usable sampled RGB; not authenticity or mouth-artifact truth",
          "positive": "at least one unique usable sample strictly inside a guarded visual interval",
          "negative": "all valid stored RGB frames have safe known timestamps outside visual intervals",
          "otherwise": "unknown -1, target mask false; original clip remains in the bag",
          "gate": GATE, "training_ready": False}


def intervals(source, duration, guard):
    """Validate explicit modality annotations; missing fields are never negatives."""
    parsed = {}
    for kind, field in (("any", "fake_segments_json"), ("audio", "audio_fake_segments_json"),
                        ("visual", "visual_fake_segments_json")):
        values = json.loads(source[field])
        if not isinstance(values, list):
            raise TypeError("Annotations must be lists")
        spans = []
        for pair in values:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                    or not 0 <= pair[0] < pair[1] <= duration + guard + EPS):
                raise ValueError("Annotation interval/unit is invalid or outside video duration")
            spans.append(tuple(pair))
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        parsed[kind] = merged
    category = source["modify_type"]
    expected = {"any": category != "real", "audio": category in {"audio_modified", "both_modified"},
                "visual": category in {"visual_modified", "both_modified"}}
    if category not in CATEGORIES or any(bool(parsed[k]) != required for k, required in expected.items()):
        raise ValueError("Modality annotations conflict with the video category")
    return parsed


def visual_target(source, slots, *, fps, duration, clock_ok=True):
    """Classify observed visual-edit exposure, not authenticity or lip-sync error."""
    result = {"aux_visual_target": -1, "aux_target_mask": False, "reason": "",
              "rgb_unique_frames": 0, "clock_unique_frames": 0,
              "inside_unique_frames": 0, "boundary_unique_frames": 0}
    rgb, timed = set(), {}
    for slot in slots:
        if any(type(slot[k]) is not bool for k in ("rgb_valid", "clock_valid")):
            raise ValueError("Saved support masks must be explicit booleans")
        if slot["clock_valid"] and not slot["rgb_valid"]:
            raise ValueError("Clock support cannot invent an RGB frame")
        if not slot["rgb_valid"]:
            continue
        frame, time = slot["source_frame"], slot["timestamp_s"]
        if type(frame) is not int or frame < 0:
            raise ValueError("Valid RGB support needs a recorded source frame")
        rgb.add(frame)
        if slot["clock_valid"]:
            if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
                raise ValueError("Valid clock support needs a finite recorded timestamp")
            if frame in timed and abs(timed[frame] - time) > EPS:
                raise ValueError("Duplicate frame has conflicting recorded timestamps")
            timed[frame] = time
    result.update(rgb_unique_frames=len(rgb), clock_unique_frames=len(timed))
    if (not clock_ok or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0):
        return {**result, "reason": "clock_unavailable"}
    guard = 1 / fps
    if any(t > duration + guard + EPS for t in timed.values()):
        return {**result, "reason": "clock_outside_video"}
    try:
        spans = intervals(source, duration, guard)["visual"]
    except (ValueError, TypeError, KeyError) as error:
        return {**result, "reason": f"annotation_unavailable: {error}"}
    if not timed:
        return {**result, "reason": "no_usable_rgb_timing"}
    inside = {f for f, t in timed.items() if any(a + guard + EPS < t < b - guard - EPS for a, b in spans)}
    boundary = {f for f, t in timed.items() if any(abs(t - edge) <= guard + EPS for span in spans for edge in span)}
    result.update(inside_unique_frames=len(inside), boundary_unique_frames=len(boundary))
    if inside:
        return {**result, "aux_visual_target": 1, "aux_target_mask": True,
                "reason": "safely_sampled_visual_edit"}
    if boundary or set(timed) != rgb:
        return {**result, "reason": "boundary_or_unknown_rgb_timing"}
    return {**result, "aux_visual_target": 0, "aux_target_mask": True,
            "reason": "safely_outside_recorded_visual_edits"}


def bind_train(sources, inventory, inner_requests, index, events, timing):
    """Join fixed IDs before inspecting intervals; inner data supply exclusion IDs only."""
    by_source, by_event = {}, {}
    excluded_ids = {r["video_id"] for r in inner_requests}
    excluded_groups = {canonical_source_group(r["source_group"], None) for r in inner_requests}
    if any(r["dataset_split"] != "val" for r in inner_requests):
        raise ValueError("Saved inner exclusion requests have the wrong split")
    for row in sources:
        if row["split"] != "train" or row.get("dataset_source", "AV-Deepfake1M++") != "AV-Deepfake1M++":
            raise ValueError("Only original AV++ Train annotations are allowed")
        identifier = stable_id(row["file"])
        if identifier in by_source:
            raise ValueError("Duplicate source video identity")
        by_source[identifier] = row
    for row in events:
        if row["dataset_split"] != "train" or row["event_id"] in by_event:
            raise ValueError("Train event identity/split differs")
        by_event[row["event_id"]] = row
    lookup = {r["video_id"]: r for r in inventory}
    if len(lookup) != len(inventory):
        raise ValueError("Repeated fixed Train request")
    grouped, clip_ids = defaultdict(list), set()
    for row in index:
        identifier, clip = row["video_id"], row["clip_id"]
        original = lookup[identifier]
        if (row["dataset_split"] != "train" or clip in clip_ids
                or canonical_source_group(row["source_group"], None) != canonical_source_group(original["source_group"], None)
                or row["kind"] not in {"bilabial", "context"}):
            raise ValueError("Frozen Train clip identity/source/kind differs")
        saved = timing[clip]
        if any(saved[k] != row[k] for k in ("video_id", "dataset_split", "kind", "source_group")):
            raise ValueError("Saved decoder provenance differs from the fixed clip index")
        event = None
        if row["kind"] == "bilabial":
            event = by_event[row["source_event_id"]]
            source = by_source[identifier]
            if (event["event_id"] != clip or event["video_id"] != identifier
                    or event["file"] != source["file"] or event["manipulation_modality"] != source["modify_type"]
                    or canonical_source_group(event["source_group"], None) != canonical_source_group(row["source_group"], None)
                    or event["class_id"] not in {"0", "1"} or event["phoneme"] not in {"m", "b", "p"}):
                raise ValueError("Original bilabial identity/target/provenance differs")
        clip_ids.add(clip)
        grouped[identifier].append({**row, "event": event, "timing": saved})
    if set(timing) != clip_ids:
        raise ValueError("Saved Train timing membership differs from fixed clips")
    cases, seen_groups = [], set()
    for row in inventory:
        identifier = row["video_id"]
        source = by_source[identifier]
        group = canonical_source_group(source.get("source_group") or source.get("original"), source["file"])
        if (identifier in excluded_ids or group in excluded_groups or group in seen_groups
                or row["dataset_split"] != "train" or group != canonical_source_group(row["source_group"], None)
                or row["category"] != source["modify_type"] or source["modify_type"] not in CATEGORIES):
            raise ValueError("Fixed Train selection/source separation differs, including unavailable requests")
        seen_groups.add(group)
        clips = grouped[identifier]
        if (len(clips) != int(row["expanded_clips"]) or row["status"] not in {"paired", "unevaluated"}
                or bool(clips) != (row["status"] == "paired") or len(clips) > 16
                or clips and (not any(c["kind"] == "bilabial" for c in clips)
                              or sum(c["kind"] == "context" for c in clips) > 8)
                or not clips and not row["reason"]):
            raise ValueError("Original usable/unavailable bag membership differs")
        cases.append({"inventory": row, "source": source, "clips": clips})
    return cases


def video_clock(trace):
    if trace is None:
        return 0., 0., "missing_verified_trace"
    clock = trace.get("timing", {})
    fps, duration, origin = (clock.get(k) for k in ("fps", "duration_s", "video_start_time_s"))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (fps, duration, origin)) or fps <= 0 or duration <= 0:
        return 0., 0., "incomplete_video_clock"
    if abs(origin) > EPS:
        return fps, duration, "nonzero_video_clock_origin"
    fallbacks = trace.get("summary", {}).get("fps_fallback_timestamp_frames")
    if type(fallbacks) is not int or fallbacks != 0:
        return fps, duration, "fallback_timestamp_provenance_unavailable"
    return fps, duration, ""


def clip_support(clip, trace, windows, *, fps, duration):
    """Reconcile saved masks and actual frame maps; never use edit intervals here."""
    frames = {f["frame_index"]: f for f in trace["frames"]}
    record = clip["timing"]
    if clip["kind"] == "bilabial":
        data = event_timing_sequence(clip["event"], trace, decoded_frame_count=record["decoded_frame_count"])
        ids = list(data["timing_frame_indices"])
        if ids != record["timing_frame_indices"]:
            raise ValueError("Saved decoded RGB frame mapping differs")
        _, mask = sequence_frame_indices(record["decoded_frame_count"], 16)
        times = [frames[f]["timestamp_s"] for f in ids]
        flags = [True] * 16
    else:
        window = windows[clip["clip_id"]]
        if (window["video_id"] != clip["video_id"] or window["dataset_split"] != "train"
                or canonical_source_group(window["source_group"], None) != canonical_source_group(clip["source_group"], None)
                or window["technical_ok"] is not True or window["phoneme"] != ""
                or window["event_class_id"] != "" or window["evidence_available"] is not False
                or window["clip_sha256"] != record["clip_sha256"] or record["decoded_frame_count"] != 16
                or any(len(window[k]) != 16 for k in ("source_frames", "frame_mask", "planned_mask", "slots"))):
            raise ValueError("Original context identity/mask/target contract differs")
        ids, mask, times, flags = window["source_frames"], window["frame_mask"], [], []
        for f, keep, planned, slot in zip(ids, mask, window["planned_mask"], window["slots"], strict=True):
            if (type(keep) is not bool or type(planned) is not bool or keep and not planned
                    or slot["source_frame"] != f or slot["crop_valid"] is not keep
                    or slot["planned_valid"] is not planned):
                raise ValueError("Original context slot/frame masks differ")
            t, reference = slot["decoder_pts_s"], frames[f]["timestamp_s"]
            valid_time = (type(t) in (int, float) and math.isfinite(t)
                          and type(slot["trace_pts_s"]) in (int, float) and math.isfinite(slot["trace_pts_s"])
                          and abs(slot["trace_pts_s"] - reference) <= EPS
                          and abs(t - reference) <= 1 / fps + EPS)
            times.append(t if type(t) in (int, float) and math.isfinite(t) else None)
            flags.append(valid_time and slot["timing_ok"] is True and slot["face_count"] == 1)
    support = []
    for f, keep, time, flag in zip(ids, mask, times, flags, strict=True):
        if type(f) is not int or f < 0 or type(keep) is not bool:
            raise ValueError("Saved source frames/crop masks must be explicit")
        compatible = (type(time) in (int, float) and math.isfinite(time)
                      and 0 <= time <= duration + 1 / fps + EPS and abs(time - f / fps) <= 1 / fps + EPS)
        support.append({"source_frame": f, "timestamp_s": time, "rgb_valid": keep,
                        "clock_valid": bool(keep and flag and compatible)})
    return support


def audit_case(case, trace, context_record):
    original, source = case["inventory"], case["source"]
    result = {"inventory": original, "clips": []}
    if original["status"] == "unevaluated":
        return {**result, "clock_reason": "original_unavailable"}
    if trace is not None and (
        trace.get("video_id") != original["video_id"] or trace.get("dataset_split") != "train"
        or trace.get("source_file") != source["file"]
        or canonical_source_group(trace.get("source_group"), None) != canonical_source_group(original["source_group"], None)
    ):
        raise ValueError("Trace does not belong to the selected Train source")
    windows = {}
    if context_record is not None:
        if (context_record["video_id"] != original["video_id"] or context_record["split"] != "train"
                or context_record["file"] != source["file"]
                or canonical_source_group(context_record["source_group"], None) != canonical_source_group(original["source_group"], None)):
            raise ValueError("Context record does not belong to the selected Train source")
        windows = {w["context_id"]: w for w in context_record["windows"]}
        if len(windows) != len(context_record["windows"]):
            raise ValueError("Repeated saved context identity")
    fps, duration, reason = video_clock(trace)
    for clip in case["clips"]:
        missing_context = clip["kind"] == "context" and clip["clip_id"] not in windows
        support = [] if reason or missing_context else clip_support(clip, trace, windows, fps=fps, duration=duration)
        target = visual_target(source, support, fps=fps, duration=duration, clock_ok=not reason and not missing_context)
        event = clip["event"]
        result["clips"].append({"clip_id": clip["clip_id"], "kind": clip["kind"],
                                "declared_clip_sha256": clip["timing"]["clip_sha256"],
                                "declared_trace_sha256": clip["timing"]["trace_sha256"],
                                "original_event_class_id": event["class_id"] if event else "-1",
                                "original_phoneme": event["phoneme"] if event else "",
                                "support": support, "support_reason": reason or ("missing_context_record" if missing_context else ""),
                                **target})
    return {**result, "clock_reason": reason, "fps": fps, "duration_s": duration,
            "source_annotations": {k: source.get(k) for k in ("fake_segments_json", "audio_fake_segments_json", "visual_fake_segments_json")}}


def summarize(inventory, clips):
    counts = Counter(str(r["aux_visual_target"]) for r in clips)
    return {"status": "complete", "requested": len(inventory),
            "original_usable": sum(r["status"] == "paired" for r in inventory),
            "original_unavailable": sum(r["status"] == "unevaluated" for r in inventory),
            "clips": len(clips), "targets": {k: counts[k] for k in ("-1", "0", "1")},
            "by_category": {c: dict(Counter(str(r["aux_visual_target"]) for r in clips if r["category"] == c))
                            for c in sorted({r["category"] for r in inventory})},
            "by_kind": {k: dict(Counter(str(r["aux_visual_target"]) for r in clips if r["kind"] == k))
                        for k in ("bilabial", "context")},
            "unknown_reasons": dict(Counter(r["reason"] for r in clips if str(r["aux_visual_target"]) == "-1")),
            "support_unavailable_reasons": dict(Counter(r["support_reason"] for r in clips
                                                       if str(r["aux_visual_target"]) == "-1" and r["support_reason"])),
            "gate": GATE, "training_ready": False, "model_calls": 0, "optimizer_updates": 0,
            "scope": "Train metadata target-design audit; not accuracy, confirmed mouth artifacts or training approval"}


def verify_saved(output, binding):
    job = {"stage": "clip-target-audit-v1", "binding": binding}
    summary = runtime.verify_output(output, job)
    inventory, clips = read_csv_rows(output / "inventory.csv"), read_csv_rows(output / "clips.csv")
    if len({r["clip_id"] for r in clips}) != len(clips) or len({r["video_id"] for r in inventory}) != len(inventory):
        raise ValueError("Repeated output identities")
    for row in clips:
        if (row["aux_visual_target"] not in {"-1", "0", "1"}
                or row["aux_target_mask"] != str(row["aux_visual_target"] != "-1")
                or row["kind"] == "context" and (row["original_event_class_id"] != "-1" or row["original_phoneme"] != "")):
            raise ValueError("Sidecar target/mask/original context fields differ")
    if summarize(inventory, clips) != summary:
        raise ValueError("Saved target counts differ")
    return summary


def run_audit(cases, output, support_loader, binding, *, progress=None):
    output = Path(output)
    if output.exists():
        return verify_saved(output, binding)
    job = {"stage": "clip-target-audit-v1", "binding": binding}
    inventory, clips = [], []
    with runtime.output_stage(output, job, progress) as complete:
        for number, case in enumerate(cases, 1):
            original = case["inventory"]
            if progress:
                progress("audit saved Train clip targets", number - 1, len(cases), original["video_id"])
            trace, context = support_loader(case) if original["status"] == "paired" else (None, None)
            result = audit_case(case, trace, context)
            atomic_write_json(output / "cases" / f"{original['video_id']}.json", result)
            inventory.append(original)
            clips.extend({"video_id": original["video_id"], "source_group": original["source_group"],
                          "category": original["category"], **{k: v for k, v in row.items() if k != "support"}}
                         for row in result["clips"])
            if progress:
                progress("audit saved Train clip targets", number, len(cases), original["video_id"])
        atomic_write_csv(output / "inventory.csv", inventory)
        atomic_write_csv(output / "clips.csv", clips)
        return complete(summarize(inventory, clips))


def load_protocol(project, config):
    protocol = runtime.read_json(config)
    if (set(protocol) != {"design", "metadata_bindings", "code_bindings"}
            or protocol["design"] != DESIGN or set(protocol["metadata_bindings"]) != set(PATHS.values())):
        raise ValueError("Unsupported Train target-audit design; preserve the frozen version")
    sources = {p.relative_to(project).as_posix() for p in (project / "src/seepat").rglob("*.py")}
    if set(protocol["code_bindings"]) != sources | {"scripts/audit_clip_targets.py"}:
        raise ValueError("Frozen source membership changed")
    for relative, expected in protocol["code_bindings"].items():
        raw = inside(project, relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    return protocol, {"protocol_sha256": hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def load_cases(project, protocol):
    """Niko-run metadata preflight: fixed Train tables and inner exclusion IDs only."""
    for relative, expected in protocol["metadata_bindings"].items():
        if file_sha256(inside(project, relative)) != expected:
            raise ValueError(f"Frozen metadata changed: {relative}")
    tables = {name: (read_csv_rows(path) if path.suffix == ".csv" else runtime.read_json(path))
              for name, relative in PATHS.items() for path in (inside(project, relative),)}
    seal = tables["timing_seal"]
    if (tables["timing_job"]["stage"] != "inputs" or tables["timing_job"]["timing_version"] != "fusion-vild-sequence-v1"
            or tables["timing_summary"]["status"] != "complete" or tables["contexts"]["status"] != "complete"
            or tables["timing_summary"]["roles"]["train"]["clips"] != DESIGN["clips"]
            or any(seal[name]["sha256"] != protocol["metadata_bindings"][f"{SAVED}/{name}"]
                   for name in ("train.json", "summary.json", "job.json"))
            or any(tables["selection_seal"][name]["sha256"] != protocol["metadata_bindings"][f"{SELECTION}/{name}"]
                   for name in ("inventory_train.csv", "inventory_inner.csv", "index_train.csv"))
            or any(tables["contexts"]["contract"]["input_sha256"][PATHS[name]] != protocol["metadata_bindings"][PATHS[name]]
                   for name in ("source", "events"))):
        raise ValueError("Completed upstream metadata bindings differ")
    if (Counter(r["category"] for r in tables["inventory"]) != {c: 32 for c in CATEGORIES}
            or len(tables["excluded"]) != 64
            or sum(r["status"] == "paired" for r in tables["inventory"]) != 125
            or len(tables["index"]) != 1340):
        raise ValueError("Frozen Train request/usable/clip counts differ")
    return bind_train(tables["source"], tables["inventory"], tables["excluded"], tables["index"],
                      tables["events"], tables["timing"]), tables["contexts"]


def load_support(project, case, context_summary):
    """Read only selected saved trace/context JSON; never follow a media path."""
    identifier = case["inventory"]["video_id"]
    events = [c["event"] for c in case["clips"] if c["event"] is not None]
    paths = {(r.get("vild_trace_path", "").replace("\\", "/"), r.get("vild_trace_sha256", "")) for r in events}
    if len(paths) != 1:
        raise ValueError("Selected bilabial trace lineage differs")
    relative, expected = paths.pop()
    trace = None
    if relative and expected:
        if relative != f"outputs/train-subset5000/vild_traces/{identifier}.json.gz":
            raise ValueError("Trace path is outside the selected Train metadata family")
        if any(c["timing"]["trace_sha256"] != expected for c in case["clips"] if c["kind"] == "bilabial"):
            raise ValueError("Saved decoder/trace hashes differ")
        path = inside(project, relative)
        if path.is_file():
            trace = load_vild_trace(path, expected)
    name = f"records/{identifier}/record.json"
    path = inside(project, f"outputs/phase5_context_inputs_v1/{name}")
    record = None
    digest = context_summary["record_sha256"].get(name)
    if digest and path.is_file():
        if file_sha256(path) != digest:
            raise ValueError("Saved context record hash differs")
        record = runtime.read_json(path)
        if any(c["kind"] == "context" for c in case["clips"]) and (
            record.get("trace_path", "").replace("\\", "/") != relative or record.get("trace_sha256") != expected
        ):
            raise ValueError("Context and bilabial records use different source traces")
    return trace, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify", "watch"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/clip_target_audit_v1.json"))
    args = parser.parse_args(argv)
    project = args.project.resolve()
    output = inside(project, OUTPUT)
    if args.command == "watch":
        return runtime.watch_progress(live_progress_path(inside(project, STATUS)), "CLIP TARGET AUDIT")
    config = inside(project, str(args.config))
    protocol, binding = load_protocol(project, config)
    if args.command == "verify" or args.command == "run" and output.exists():
        result = verify_saved(output, binding)
        print(f"CLIP TARGET AUDIT VERIFIED | {result['requested']} requests | saved sidecar only", flush=True)
        return result
    cases, context_summary = load_cases(project, protocol)
    if args.command == "preflight":
        print("CLIP TARGET INPUTS READY | 128 requests | 125 original usable | 1340 clips | no trace/media/model load", flush=True)
        return {"requested": len(cases), "usable": 125, "clips": 1340}
    progress = WorkflowProgress(config, inside(project, STATUS), 1)
    progress.start_job(1, "saved-train-auxiliary-target-audit")
    try:
        result = run_audit(cases, output, lambda case: load_support(project, case, context_summary),
                           binding, progress=progress)
        progress.finish()
        print(f"CLIP TARGET AUDIT COMPLETE | {result['requested']}/128 requests | {result['clips']} clips | targets {result['targets']}", flush=True)
        print(f"Gate: {GATE} | NOT A TRAINING MANIFEST OR ACCURACY RESULT", flush=True)
        return result
    except BaseException as error:
        progress.finish(error)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        raise SystemExit(f"CLIP TARGET AUDIT STOPPED | {error}") from error
