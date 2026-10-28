"""Bounded sequence-evidence comparison; real execution is explicit."""

import argparse
import hashlib
import os
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import torch

from seepat.artifacts import atomic_write_csv, file_sha256, read_csv_rows
from seepat.data.sampling import canonical_source_group
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.models.sequence_evidence import FRAME_WINDOW_POLICY, MODEL_VERSION, SequenceAwareFusion
from seepat.training import experiment as runtime
from seepat.training.context_dataset import CONTEXT_INPUTS, ExpandedContextDataset
from seepat.training.dataset import MouthEventDataset
from seepat.training.matched_inputs import inside
from seepat.training.metrics import binary_classification_metrics
from seepat.training.sequence_inputs import SequenceInputDataset, export_timing_inputs
from seepat.training.train import (
    FUSION_MODEL,
    TrainingOptions,
    _forward_logits,
    build_event_classifier,
    train_model,
)
from seepat.training.video_objective import scored_video_bags, video_bags, video_loader

ARMS = ("control", "candidate")
VERSION = "sequence-evidence-pilot-v2"
OUTPUT = "outputs/engineering/sequence_evidence_pilot_v2"
SELECTION = "outputs/engineering/smooth_loss_pilot_v1/selection"
ROLES = {"train": "train", "inner": "val"}
CATEGORIES = {"real", "audio_modified", "visual_modified", "both_modified"}
REQUIRED_SOURCE = {f"src/seepat/{name}.py" for name in (
    "models/sequence_evidence", "models/fusion", "training/sequence_pilot", "training/sequence_inputs",
    "training/train", "training/experiment", "training/dataset", "training/matched_inputs",
    "training/context_dataset", "training/video_objective", "evidence", "artifacts", "preprocessing/face",
)}


def fixed_options():
    return TrainingOptions(epochs=4, seed=20261025, amp=False, freeze_backbone=True,
                           supervision="video_max", class_weighting="balanced_global",
                           gradient_accumulation_steps=4, early_stopping_patience=0)


def validate_protocol(protocol):
    expected_roles = {
        role: {"events": f"outputs/numerical_calibration/guarded_per_video_v3/events_{split}_calibrated.csv",
               "export": "outputs/phase5_context_inputs_v1"} for role, split in ROLES.items()
    }
    required_metadata = {
        *(d["events"] for d in expected_roles.values()),
        "outputs/numerical_calibration/guarded_per_video_v3/calibration.json",
        "outputs/numerical_calibration/guarded_per_video_v3/summary.json",
        "outputs/training/fusion_phase5_v1/experiments/candidate/run.json",
        *(f"{CONTEXT_INPUTS}/{name}" for name in (
            "inventory_train.csv", "inventory_inner.csv", "expanded_train.csv", "expanded_inner.csv",
            "records_train.json", "records_inner.json", "summary.json", "seal.json")),
        *(f"{SELECTION}/{name}" for name in (
            "inventory_train.csv", "inventory_inner.csv", "index_train.csv", "index_inner.csv", "seal.json")),
    }
    if (protocol["version"] != VERSION or protocol["output"] != OUTPUT
            or protocol.get("frame_window_policy") != FRAME_WINDOW_POLICY
            or protocol["selection"] != SELECTION or protocol["roles"] != expected_roles
            or protocol["training"] != asdict(fixed_options())
            or protocol["initial_checkpoint"]["path"] != "outputs/training/fusion_phase5_v1/experiments/candidate/checkpoint_best.pt"
            or protocol["initial_checkpoint"]["sha256"] != "46fb132230fb2292ecadab63ac84aca34882174d6dbd4680ade5adab82dc4b07"
            or protocol["gate"] != "strict_auc_and_ba_improvement_with_no_extra_false_positives"
            or protocol["coverage"] != "at_least_80_percent_of_usable_videos_in_each_role"
            or not required_metadata <= protocol["metadata_bindings"].keys()
            or not REQUIRED_SOURCE <= protocol["code_bindings"].keys()):
        raise ValueError("Unsupported sequence pilot design; preserve this frozen version")


def load_protocol(project, config):
    protocol = runtime.read_json(config)
    validate_protocol(protocol)
    actual_source = {p.relative_to(project).as_posix() for p in (project / "src/seepat").rglob("*.py")}
    if set(protocol["code_bindings"]) != actual_source:
        raise ValueError("Frozen sequence source membership changed")
    for relative, expected in protocol["code_bindings"].items():
        raw = inside(project, relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen sequence source changed: {relative}")
    for relative, expected in protocol["metadata_bindings"].items():
        if file_sha256(inside(project, relative)) != expected:
            raise ValueError(f"Frozen sequence metadata changed: {relative}")
    return protocol, {"protocol_sha256": hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def selected_inputs(project):
    selection = inside(project, SELECTION)
    runtime.verify_output(selection)
    inventories, indexes, identities, groups = {}, {}, set(), set()
    for role, split in ROLES.items():
        inventories[role] = read_csv_rows(selection / f"inventory_{role}.csv")
        indexes[role] = read_csv_rows(selection / f"index_{role}.csv")
        count = 32 if role == "train" else 16
        if Counter(r["category"] for r in inventories[role]) != {c: count for c in CATEGORIES}:
            raise ValueError("Original fixed category/request counts changed")
        if sum(r["status"] == "paired" for r in inventories[role]) != (125 if role == "train" else 62):
            raise ValueError("Original usable/unavailable counts changed")
        lookup, by_video = {}, defaultdict(list)
        for row in inventories[role]:
            group = canonical_source_group(row["source_group"], None)
            if (row["video_id"] in identities or group in groups or row["dataset_split"] != split
                    or row["status"] not in {"paired", "unevaluated"}):
                raise ValueError("Original source separation/availability changed, including unavailable cases")
            identities.add(row["video_id"])
            groups.add(group)
            lookup[row["video_id"]] = row
        clip_ids = set()
        for row in indexes[role]:
            original = lookup[row["video_id"]]
            if (row["clip_id"] in clip_ids or row["dataset_split"] != split or row["kind"] not in {"bilabial", "context"}
                    or canonical_source_group(row["source_group"], None) != canonical_source_group(original["source_group"], None)):
                raise ValueError("Original clip identity/source changed")
            clip_ids.add(row["clip_id"])
            by_video[row["video_id"]].append(row)
        for identifier, row in lookup.items():
            clips = by_video[identifier]
            if (len(clips) != int(row["expanded_clips"]) or bool(clips) != (row["status"] == "paired")
                    or clips and (len(clips) > 16 or not any(c["kind"] == "bilabial" for c in clips)
                                  or sum(c["kind"] == "context" for c in clips) > 8)):
                raise ValueError("Original complete bag or availability changed")
    return inventories, indexes


def base_dataset(project, protocol, role, index):
    definition = protocol["roles"][role]
    original = MouthEventDataset(inside(project, definition["events"]), project,
                                 dataset_split=ROLES[role], require_calibration=True)
    records = runtime.read_json(inside(project, f"{CONTEXT_INPUTS}/records_{role}.json"))
    return ExpandedContextDataset(original, index, records, project, inside(project, definition["export"]))


def fit_pair(output, options, inputs_loader, model_loader, binding, *, stage,
             resume=False, device="cpu", progress=None):
    options.validate()
    job = {"stage": stage, "study": VERSION, "binding": binding, "options": asdict(options),
           "environment": {"device": device, "torch": torch.__version__, "python": platform.python_version()}}

    def train_arm(arm, model, train, validation, initial, directory):
        contract = {"pilot": VERSION, "model": MODEL_VERSION, "arm": arm, "stage": stage,
                    "binding": binding, "initial_state_sha256": initial,
                    "clip_input_contract": train.input_contract, "fusion_inputs": train.calibration_contract,
                    "timing_enabled": arm == "candidate", "checkpoint": "fixed last epoch, never best"}
        run = train_model(model, train, validation, directory, options, torch.device(device), contract,
                          resume_from=directory / "checkpoint_last.pt" if resume else None, progress=progress)
        history = runtime.read_json(directory / "history.json")
        runtime.check_history(run, history, train, validation, options)
        if any(e["validation"]["selection"]["threshold"] != 0.5 for e in history):
            raise ValueError("Sequence pilot must keep its fixed diagnostic cutoff")
        return {"completed_epochs": run["completed_epochs"], "optimizer_updates": run["global_step"],
                "skipped_updates": run["skipped_optimizer_steps"], "initial_state_sha256": initial,
                "final_validation": history[-1]["validation"], "elapsed_seconds": run["elapsed_seconds"],
                "peak_cuda_memory_bytes": run.get("peak_cuda_memory_bytes"),
                "timing_projection_l1": float(model.timing_encoder.output.weight.detach().float().abs().sum()),
                "checkpoint": f"{arm}/checkpoint_last.pt"}

    return runtime.run_training_pair(output, job, inputs_loader, model_loader, train_arm,
                                     arms=ARMS, resume=resume, progress=progress,
                                     summary_fields={"scope": "small reused-inner development; no promotion"})


def evaluate_pair(output, dataset_loader, model_loader, selected, binding, *, device="cpu", progress=None):
    job = {"stage": "evaluation", "study": VERSION, "binding": binding,
           "threshold": 0.5, "optimizer_updates": 0, "device": device, "checkpoint": "fixed last epoch4"}
    if output.exists():
        return runtime.verify_output(output, job)
    results, signatures = {}, []
    with runtime.output_stage(output, job, progress) as complete:
        for number, arm in enumerate(ARMS, 1):
            if progress:
                progress.start_job(number, arm)
            dataset, model = dataset_loader(arm), model_loader(arm)
            model.to(device).eval().requires_grad_(False)
            bags = video_bags(dataset.rows, True)
            loader = video_loader(dataset, bags, 0, torch.device(device), 20261025)
            signatures.append([(r["event_id"], r["video_id"], r["source_group"], r["kind"], r["class_id"])
                               for r in dataset.rows])
            scores, clips = {}, []
            with torch.inference_mode():
                for bag, scored in zip(bags, scored_video_bags(
                    model, loader, bags, torch.device(device), _forward_logits, progress,
                    "score fixed maximum and cutoff",
                ), strict=True):
                    scores[bag.video_id] = {"probability": max(scored["event_probabilities"]), "label": bag.label}
                    clips.extend({"video_id": bag.video_id, "clip_id": identifier, "probability": probability}
                                 for identifier, probability in zip(bag.event_ids, scored["event_probabilities"], strict=True))
            if set(scores) != {r["video_id"] for r in selected if r["status"] == "paired"}:
                raise ValueError("Evaluation changed original usable membership")
            if any(scores[r["video_id"]]["label"] != int(r["category"] != "real") for r in selected if r["video_id"] in scores):
                raise ValueError("Evaluation changed original labels")
            rows = [{"video_id": r["video_id"], "source_group": r["source_group"], "category": r["category"],
                     "status": "evaluated" if r["video_id"] in scores else "not_evaluated", "reason": r["reason"],
                     **scores.get(r["video_id"], {"probability": "", "label": ""})} for r in selected]
            available = [r for r in rows if r["status"] == "evaluated"]
            labels, probabilities = [r["label"] for r in available], [r["probability"] for r in available]
            metrics = binary_classification_metrics(labels, probabilities, 0.5)
            positive = [p for y, p in zip(labels, probabilities, strict=True) if y]
            negative = [p for y, p in zip(labels, probabilities, strict=True) if not y]
            wins = sum(2 * (p > n) + (p == n) for p in positive for n in negative)
            pairs = len(positive) * len(negative)
            if not pairs:
                raise ValueError("Paired evaluation requires both classes")
            results[arm] = {"metrics": metrics, "auc": wins / (2 * pairs), "auc_twice_pair_wins": wins,
                            "auc_pair_count": pairs, "balanced_accuracy": (metrics["recall"] + metrics["specificity"]) / 2,
                            "requested": len(rows), "evaluated": len(available), "unavailable": len(rows) - len(available)}
            atomic_write_csv(output / f"{arm}_videos.csv", rows)
            atomic_write_csv(output / f"{arm}_clips.csv", clips)
            del model, dataset
            if device == "cuda":
                torch.cuda.empty_cache()
        if signatures[0] != signatures[1]:
            raise ValueError("Evaluation arms changed original clips, identities or targets")
        def balanced_numerator(result):
            m = result["metrics"]
            return m["true_positive"] * (m["true_negative"] + m["false_positive"]) + m["true_negative"] * (m["true_positive"] + m["false_negative"])
        a, b = (results[arm] for arm in ARMS)
        passes = (b["auc_twice_pair_wins"] > a["auc_twice_pair_wins"]
                  and balanced_numerator(b) > balanced_numerator(a)
                  and b["metrics"]["false_positive"] <= a["metrics"]["false_positive"])
        return complete({"status": "complete", "arms": results, "optimizer_updates": 0,
                         "gate": "review_larger_design_only" if passes else "stop_and_review",
                         "interpretation": "small reused-inner diagnostic; not H1, model promotion or an automatic full fit"})


def stage_path(project, stage):
    return inside(project, f"{OUTPUT}/{'readiness' if stage.startswith('readiness') else stage}")


def verify_stage(project, stage, binding):
    directory = stage_path(project, stage)
    result = runtime.verify_output(directory)
    job = runtime.read_json(directory / "job.json")
    if job["binding"] != binding or stage != "inputs" and job.get("study") != VERSION:
        raise ValueError("Stage belongs to different sequence inputs/protocol")
    if stage == "inputs":
        if result["gate"] != "ready_for_readiness":
            raise ValueError("Timing coverage gate failed; preserve and review inputs")
    elif stage != "evaluation":
        epoch = int(stage[-1]) if stage.startswith("readiness") else 4
        updates = epoch * (4 if stage.startswith("readiness") else 32)
        if (result["through_epoch"] != epoch or set(result["arms"]) != set(ARMS)
                or any(a["optimizer_updates"] != updates or a["skipped_updates"] for a in result["arms"].values())):
            raise ValueError("Sequence epoch/update accounting changed")
        if epoch >= 2 and (result["arms"]["control"]["timing_projection_l1"] != 0
                           or result["arms"]["candidate"]["timing_projection_l1"] <= 0):
            raise ValueError("Timing branch did not follow its enabled/disabled learning contract; preserve for review")
    elif result["optimizer_updates"] != 0:
        raise ValueError("Evaluation must not fit parameters")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="One fixed sequence-evidence development pilot; no automatic promotion.")
    parser.add_argument("command", choices=("export", "verify", "readiness1", "readiness2", "train", "evaluate", "watch"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/sequence_evidence_pilot_v2.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("inputs", "readiness1", "readiness2", "training", "evaluation"), default="inputs")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    os.chdir(project)
    if args.command == "watch":
        return runtime.watch_progress(live_progress_path(inside(project, f"{OUTPUT}/{args.stage}_status.json")), "SEQUENCE PILOT")
    config = args.config.resolve()
    protocol, binding = load_protocol(project, config)
    inventories, indexes = selected_inputs(project)
    if args.command == "export":
        if (stage_path(project, "inputs") / "worker.lock").exists():
            raise ValueError("Inputs are active or interrupted; preserve and inspect before recovery")
        progress = WorkflowProgress(config, inside(project, f"{OUTPUT}/inputs_status.json"), 2)
        try:
            result = export_timing_inputs(
                {role: base_dataset(project, protocol, role, indexes[role]) for role in ROLES},
                inventories, stage_path(project, "inputs"), binding, progress=progress,
            )
            if result["gate"] != "ready_for_readiness":
                raise ValueError("Fewer than80% of usable videos have aligned timing; stop and review")
            progress.finish()
            print("SEQUENCE INPUTS COMPLETE | existing clips/traces only | no training or pixel export")
            return result
        except BaseException as error:
            progress.finish(error)
            raise
    verify_stage(project, "inputs", binding)
    bound = {**binding, "inputs_seal_sha256": file_sha256(stage_path(project, "inputs") / "seal.json")}
    evaluation_binding = None
    if args.command == "evaluate" or args.command == "verify" and args.stage == "evaluation":
        verify_stage(project, "training", bound)
        evaluation_binding = {**bound, "training_seal_sha256": file_sha256(stage_path(project, "training") / "seal.json")}
    if args.command == "verify":
        expected = binding if args.stage == "inputs" else evaluation_binding if args.stage == "evaluation" else bound
        result = verify_stage(project, args.stage, expected)
        print(f"SEQUENCE PILOT VERIFIED | {args.stage} | checkpoint bytes not rehashed")
        return result

    def dataset(role, arm):
        return SequenceInputDataset(base_dataset(project, protocol, role, indexes[role]),
                                    stage_path(project, "inputs"), role, arm == "candidate")

    def fresh_model(resuming):
        if resuming:
            base = build_event_classifier(FUSION_MODEL, pretrained=False, freeze_backbone=True)
        else:
            original = protocol["initial_checkpoint"]
            run = runtime.read_json(inside(project, original["run"]))
            base = runtime.warm_start(inside(project, original["path"]), run["resume_contract"], original["sha256"])
        torch.manual_seed(protocol["training"]["seed"])
        return SequenceAwareFusion(base)

    stage = "training" if args.command == "train" else "evaluation" if args.command == "evaluate" else args.command
    if (stage_path(project, stage) / "worker.lock").exists():
        raise ValueError("Stage is active or interrupted; preserve and inspect before recovery")
    progress = WorkflowProgress(config, inside(project, f"{OUTPUT}/{stage}_status.json"), 2)
    try:
        if stage != "evaluation":
            options = TrainingOptions(**protocol["training"])
            resume = False
            if stage.startswith("readiness"):
                options = replace(options, epochs=int(stage[-1]), max_train_batches=16, max_validation_batches=4)
                if stage == "readiness2":
                    previous = runtime.verify_output(stage_path(project, stage))
                    if previous["through_epoch"] not in {1, 2}:
                        raise ValueError("Only sealed readiness1->2 resume is supported")
                    resume = previous["through_epoch"] == 1
            else:
                verify_stage(project, "readiness2", bound)
            result = fit_pair(stage_path(project, stage), options,
                              lambda arm: tuple(dataset(role, arm) for role in ROLES),
                              lambda _arm, resuming: fresh_model(resuming), bound,
                              stage="training" if stage == "training" else "readiness",
                              resume=resume, device=args.device, progress=progress)
        else:
            directory = stage_path(project, "training")
            seal = runtime.read_json(directory / "seal.json")
            def evaluated_model(arm):
                path = directory / arm / "checkpoint_last.pt"
                if file_sha256(path) != seal[f"{arm}/checkpoint_last.pt"]["sha256"]:
                    raise ValueError("Pilot checkpoint bytes changed")
                saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                declared = runtime.read_json(directory / arm / "run.json")["resume_contract"]
                if (saved["checkpoint_type"] != "seepat_resumable_training" or saved["completed_epoch"] != 4
                        or saved["resume_contract"] != declared or declared["binding"] != bound
                        or declared["arm"] != arm or declared["model"] != MODEL_VERSION):
                    raise ValueError("Pilot checkpoint/arm/input contract changed")
                model = fresh_model(True)
                model.load_state_dict(saved["model_state"], strict=True)
                return model
            result = evaluate_pair(stage_path(project, stage), lambda arm: dataset("inner", arm),
                                   evaluated_model, inventories["inner"], evaluation_binding,
                                   device=args.device, progress=progress)
        progress.finish()
        print(f"SEQUENCE PILOT {stage.upper()} COMPLETE | {result.get('gate', 'both arms complete')} | diagnostic only")
        return result
    except BaseException as error:
        progress.finish(error)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        raise SystemExit(f"SEQUENCE PILOT STOPPED | {error}") from error
