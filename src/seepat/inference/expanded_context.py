"""Expanded-context inference and original-development threshold bindings."""

import json
import math

from seepat.artifacts import file_sha256
from seepat.context_inputs import paired_indices
from seepat.data.sampling import canonical_source_group
from seepat.training.train import FUSION_MODEL
from seepat.training.video_objective import VIDEO_MAX, VIDEO_SUPERVISION_CONTRACT

REPRESENTATION = "context-budget-training-v1"
POLICY = "same selected bilabials plus all usable existing contexts; <=8 contexts/16 total"
MASKS = "unchanged original decoder; 16 saved flags; invalid pixels zeroed"
EVIDENCE = "unavailable; zero values and false masks; target -1 is not a class"


def expanded_indices(events, windows):
    """Retain the matched candidate's bilabials, then every usable context."""
    if (any(row.get("dataset_split") != "local" for row in events + windows)
            or len(windows) > 8
            or any(type(w.get("start_frame")) is not int or w["start_frame"] < 0 for w in windows)
            or len(events) < 2 and windows):
        raise ValueError("Expanded local inputs require local provenance and at most eight planned contexts")
    _, matched = paired_indices(events, windows)
    expanded = [row for row in matched if row["kind"] == "bilabial"]
    for window in sorted((w for w in windows if w["technical_ok"]),
                         key=lambda w: (w["start_frame"], w["context_id"])):
        expanded.append({"clip_id": window["context_id"], "kind": "context", "source_event_id": "",
                         "video_id": window["video_id"], "dataset_split": "local",
                         "source_group": canonical_source_group(window["source_group"], None),
                         "context_record_id": window["context_id"]})
    if len(expanded) > 16 or len({r["clip_id"] for r in expanded}) != len(expanded):
        raise ValueError("Expanded bags require at most sixteen unique clips")
    return expanded


def validate_supervision(checkpoint, contract):
    """Validate this representation separately from the legacy matched reader."""
    options = checkpoint.get("options", {})
    clip = contract.get("clip_input_contract", {})
    if (options.get("supervision") != VIDEO_MAX or contract.get("supervision") != VIDEO_SUPERVISION_CONTRACT
            or contract.get("model_name") != FUSION_MODEL or contract.get("seed") != options.get("seed")
            or contract.get("context_budget_stage") != "training"
            or clip.get("version") != REPRESENTATION or clip.get("arm") != "expanded"
            or clip.get("policy") != POLICY or clip.get("context_masks") != MASKS
            or clip.get("context_evidence") != EVIDENCE or contract.get("sequence_length") != 16
            or set(clip.get("index_sha256", {})) != {"train", "inner", "outer"}
            or contract.get("validation_manifest_sha256") != clip["index_sha256"]["inner"]
            or contract.get("train_manifest_sha256") != clip["index_sha256"]["train"]):
        raise ValueError("Unsupported Phase-6 expanded checkpoint, policy or supervision")


def saved_threshold(freeze, contract, run, bindings, bound):
    """Bind the existing cutoff to the completed study; never choose a cutoff."""
    definitions = freeze["phase6_study"]
    required = {"profile", "inputs_summary", "inputs_seal", "training_summary", "training_seal",
                "evaluation_job", "evaluation_summary", "evaluation_seal", "inner_predictions"}
    if set(definitions) != required:
        raise ValueError("Phase-6 freeze requires the complete saved study bindings")
    paths = {name: bound(definitions[name], bindings) for name in required}
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
               if name != "inner_predictions"}
    cutoff_path = bound(freeze["threshold"], bindings)
    cutoff = json.loads(cutoff_path.read_text(encoding="utf-8"))
    selected = cutoff.get("threshold")
    binding = contract["context_budget_binding"]
    clip = contract["clip_input_contract"]
    if (type(selected) not in (int, float) or not math.isfinite(selected) or not 0 <= selected <= 1
            or cutoff.get("selection") != "original_inner_validation_only" or cutoff.get("arm") != "expanded"
            or cutoff.get("binding") != records["evaluation_job"].get("binding")
            or any(cutoff["binding"].get(k) != v for k, v in binding.items())
            or cutoff["binding"]["checkpoints"]["expanded"] != freeze["checkpoint"]["sha256"]
            or definitions["profile"]["sha256"] != binding["profile_sha256"]
            or definitions["inputs_summary"]["sha256"] != binding["input_summary_sha256"]
            or clip["summary_sha256"] != binding["input_summary_sha256"]
            or definitions["inputs_seal"]["sha256"] != binding["input_seal_sha256"]
            or definitions["training_seal"]["sha256"] != cutoff["binding"]["training_seal_sha256"]):
        raise ValueError("Phase-6 cutoff crosses model, input or training bindings")
    for seal_name, entries in {
        "inputs_seal": {"summary.json": definitions["inputs_summary"]},
        "training_seal": {"summary.json": definitions["training_summary"], "expanded/run.json": freeze["run"],
                          "expanded/checkpoint_best.pt": freeze["checkpoint"]},
        "evaluation_seal": {"job.json": definitions["evaluation_job"], "summary.json": definitions["evaluation_summary"],
                            "expanded_threshold.json": freeze["threshold"],
                            "expanded_inner_videos.csv": definitions["inner_predictions"]},
    }.items():
        if any(records[seal_name].get(name, {}).get("sha256") != entry["sha256"] for name, entry in entries.items()):
            raise ValueError("Phase-6 sealed artifact linkage differs")
    profile, inputs, training, evaluation = (records[k] for k in
                                            ("profile", "inputs_summary", "training_summary", "evaluation_summary"))
    arm = training.get("arms", {}).get("expanded", {})
    if (profile.get("version") != REPRESENTATION or inputs.get("status") != "complete"
            or training.get("status") != "complete" or training.get("stage") != "training"
            or training.get("through_epoch") != 4 or arm.get("completed_epochs") != 4
            or arm.get("optimizer_updates") != run.get("global_step") or arm.get("skipped_updates") != 0
            or arm.get("best_epoch") != run.get("best_epoch")
            or evaluation.get("status") != "complete" or evaluation.get("optimizer_updates") != 0
            or evaluation["comparison"]["development_gate"] != "review_expanded_candidate"
            or evaluation["comparison"]["thresholds"]["expanded"]["threshold"] != selected
            or any(profile["training"][k] != contract[k] for k in ("sequence_length", "image_size", "seed"))):
        raise ValueError("Phase-6 completed fit, selected checkpoint or saved cutoff differs")
    return {"value": selected, "source": "saved_phase6_original_inner", "artifact": str(cutoff_path),
            "artifact_sha256": file_sha256(cutoff_path), "predictions": definitions["inner_predictions"],
            "provenance": cutoff["binding"]}
