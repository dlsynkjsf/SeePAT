"""Validation and selection of matched bilabial and context inference inputs."""

import json
import math
from collections import defaultdict
from itertools import pairwise

from seepat.context_inputs import INPUT_VERSION, paired_indices
from seepat.coverage import context_windows
from seepat.evidence import FUSION_EVIDENCE_FIELDS
from seepat.training.dataset import FEATURE_FIELDS
from seepat.training.matched_inputs import inside, validate_profile

from .expanded_context import REPRESENTATION as EXPANDED_REPRESENTATION
from .expanded_context import expanded_indices

REPRESENTATION = "matched-clips-v1"
CONTEXT_PROTOCOL_SHA256 = "bfb0b300cf6e1f23b6ee4bf6157ff6d8f48c946ef39ddb3b1bffab2b2be43bef"
POLICY = "cap16; >=half bilabial; same per-video counts; context event target unavailable"
MASKS = "16 saved crop-valid/padding flags; invalid decoded pixels zeroed"
EVIDENCE = "all values zero placeholders, all availability flags false"
UNAVAILABLE_CONTEXT_FIELDS = set(FUSION_EVIDENCE_FIELDS) | set(FEATURE_FIELDS) | {"video_phone_start_s"}


def validate_training_context(freeze, contract, bindings, bound):
    matched = contract["clip_input_contract"]
    if freeze["representation"] == EXPANDED_REPRESENTATION:
        bound(freeze["context_protocol"], bindings)
        if freeze["context_protocol"]["sha256"] != CONTEXT_PROTOCOL_SHA256:
            raise ValueError("Context preprocessing differs from the accepted protocol")
        return
    if (matched.get("version") != REPRESENTATION or matched.get("arm") not in {"control", "candidate"}
            or matched.get("policy") != POLICY or matched.get("context_masks") != MASKS
            or matched.get("context_evidence") != EVIDENCE or contract["sequence_length"] != 16):
        raise ValueError("Unsupported matched checkpoint policy or masks")
    profile = validate_profile(matched["profile"])
    bound(freeze["context_protocol"], bindings)
    if freeze["context_protocol"]["sha256"] != CONTEXT_PROTOCOL_SHA256:
        raise ValueError("Context preprocessing differs from the accepted Phase-5 protocol")
    path = bound(freeze["matched_training_summary"], bindings)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (freeze["matched_training_summary"]["sha256"] != profile["summary_sha256"]
            or summary.get("status") != "complete" or summary.get("input_version") != INPUT_VERSION
            or summary.get("engineering_gate") != "review_for_training_wiring"
            or summary["contract"]["protocol_sha256"] != CONTEXT_PROTOCOL_SHA256
            or summary["output_sha256"] != matched["index_sha256"]
            or contract["validation_manifest_sha256"] != matched["index_sha256"][f"{matched['arm']}_val.csv"]):
        raise ValueError("Matched training-summary/profile/threshold input binding differs")


def _validate_window(window, plan, source, fps):
    expected_id = f"{source['video_id']}-context-{plan['slot']:03d}"
    slots = window["slots"]
    if (window.get("context_id") != expected_id or window.get("video_id") != source["video_id"]
            or window.get("source_group") != source["source_group"] or window.get("dataset_split") != "local"
            or window.get("kind") != "context" or window.get("evidence_available") is not False
            or window.get("phoneme") or window.get("event_class_id")
            or any(window.get(field) not in (None, "") for field in UNAVAILABLE_CONTEXT_FIELDS)
            or any(window.get(f"{field}_available") not in (None, False) for field in UNAVAILABLE_CONTEXT_FIELDS)
            or window["start_frame"] != plan["start_frame"]
            or window["stop_frame_exclusive"] != plan["stop_frame_exclusive"]
            or window["source_frames"] != plan["source_frames"] or window["planned_mask"] != plan["frame_mask"]
            or len(slots) != 16 or len(window["frame_mask"]) != 16
            or any(type(flag) is not bool for flag in window["frame_mask"])
            or any(type(flag) is not bool for flag in window["planned_mask"])
            or type(window["technical_ok"]) is not bool):
        raise ValueError("Context identity, plan, masks or unavailable evidence differs")
    for index, slot in enumerate(slots):
        if (slot["source_frame"] != plan["source_frames"][index]
                or type(slot["planned_valid"]) is not bool or type(slot["crop_valid"]) is not bool
                or slot["planned_valid"] != plan["frame_mask"][index]
                or slot["crop_valid"] != window["frame_mask"][index]
                or type(slot["face_count"]) is not int or slot["face_count"] < 0
                or slot["crop_valid"] and (not slot["planned_valid"] or slot["face_count"] != 1)):
            raise ValueError("Context slot masks or exactly-one-face guard differs")
        times = [slot.get("decoder_pts_s"), slot.get("trace_pts_s")]
        timing_ok = (all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times)
                     and abs(times[0] - slot["source_frame"] / fps) <= 1 / fps + 1e-6
                     and abs(times[0] - times[1]) <= 1 / fps + 1e-6)
        if type(slot["timing_ok"]) is not bool or slot["timing_ok"] != timing_ok:
            raise ValueError("Context recorded timing does not match source timestamps")
    active = [slot for slot in slots if slot["planned_valid"]]
    fraction = sum(slot["crop_valid"] for slot in active) / len(active)
    multiple = sum(slot["face_count"] > 1 for slot in active)
    monotonic = all(a["decoder_pts_s"] is not None and b["decoder_pts_s"] is not None
                    and b["decoder_pts_s"] > a["decoder_pts_s"] for a, b in pairwise(active))
    usable = fraction >= .8 and multiple == 0 and monotonic and all(slot["timing_ok"] for slot in active)
    if (window["technical_ok"] != usable or window["usable_frame_fraction"] != fraction
            or window["multiple_face_slots"] != multiple
            or not usable and (window.get("clip_file") or window.get("clip_sha256"))):
        raise ValueError("Context technical gate or failed-window handling differs")


def select_local_rows(freeze, spec, contract, bilabials, requested, bindings, bound):
    path = bound(spec["matched_inputs"], bindings)
    packet = json.loads(path.read_text(encoding="utf-8"))
    arm = contract["clip_input_contract"]["arm"]
    expanded = freeze["representation"] == EXPANDED_REPRESENTATION
    if (packet.get("version") != ("expanded-local-inputs-v1" if expanded else "matched-local-inputs-v1") or packet.get("arm") != arm
            or packet.get("context_protocol_sha256") != freeze["context_protocol"]["sha256"]
            or packet.get("bilabial_manifest_sha256") != spec["manifest"]["sha256"]
            or set(packet["indexes"]) != ({"expanded"} if expanded else {"control", "candidate"})):
        raise ValueError("Prepared matched arm, policy or bilabial binding differs")
    records = packet["records"]
    if (len(records) != len(requested) or len({row["video_id"] for row in records}) != len(records)
            or {row["video_id"] for row in records} != {row["video_id"] for row in requested}):
        raise ValueError("Matched records must retain every requested video, including failures")
    by_video = {row["video_id"]: row for row in records}
    grouped = defaultdict(list)
    for row in bilabials:
        grouped[row["video_id"]].append(row)
    expected = {"expanded": []} if expanded else {"control": [], "candidate": []}
    context_rows, coverage = {}, []
    for source in requested:
        record = by_video[source["video_id"]]
        if (record.get("source_group") != source["source_group"] or record.get("dataset_split") != "local"
                or record.get("dataset_source", "local-unlabelled") != "local-unlabelled"):
            raise ValueError("Matched record identity or local-only scope differs")
        base, windows = grouped[source["video_id"]], record["windows"]
        if len(base) >= 2:
            if type(record["video_frame_count"]) is not int or type(record["fps"]) not in (int, float):
                raise ValueError("Context frame metadata must be numeric")
            plans = context_windows(record["video_frame_count"], record["fps"])
            if len(windows) != len(plans):
                raise ValueError("All planned context windows, including failures, must remain")
            for window, plan in zip(windows, plans, strict=True):
                _validate_window(window, plan, source, record["fps"])
                if window["technical_ok"]:
                    clip = inside(path.parent, window["clip_file"])
                    if clip.suffix.lower() != ".mp4":
                        raise ValueError("Context storage must retain the Phase-5 MP4 family")
                    bound({"path": str(clip), "sha256": window["clip_sha256"]}, bindings)
                    context_rows[window["context_id"]] = {
                        "event_id": window["context_id"], "video_id": source["video_id"],
                        "source_group": source["source_group"], "dataset_split": "local", "kind": "context",
                        "phoneme": "", "mouth_clip_path": str(clip), "mouth_clip_sha256": window["clip_sha256"],
                        "_frame_mask": window["frame_mask"],
                    }
        elif windows:
            raise ValueError("No context slots exist without at least two eligible bilabials")
        control, candidate = paired_indices(base, windows)
        if expanded:
            candidate = expanded_indices(base, windows)
            expected["expanded"].extend(candidate)
        else:
            expected["control"].extend(control)
            expected["candidate"].extend(candidate)
        contexts = sum(row["kind"] == "context" for row in candidate)
        fallback = (source["status"] if source["status"] != "ready" else "no_eligible_bilabials"
                    if not base else "single_bilabial" if len(base) == 1 else "no_usable_context"
                    if not contexts else "")
        coverage.append({"video_id": source["video_id"], "bilabials_available": len(base),
                         "control_clips": len(control), "candidate_clips": len(candidate),
                         "candidate_context_clips": contexts, "fallback_reason": fallback})
    if packet["indexes"] != expected:
        raise ValueError("Matched indexes differ from frozen count-matched selection")
    originals = {row["event_id"]: row for row in bilabials}
    if originals.keys() & context_rows.keys():
        raise ValueError("Context and bilabial identities must not collide")
    selected = [({**originals[item["source_event_id"]], "kind": "bilabial"}
                 if item["kind"] == "bilabial" else context_rows[item["clip_id"]]) for item in expected[arm]]
    return selected, str(path.parent), coverage
