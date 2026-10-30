"""One deadline-bound auxiliary comparison; real execution is explicitly user-run."""

import argparse
import hashlib
import math
import os
import platform
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from seepat.artifacts import atomic_write_csv, file_sha256, read_csv_rows
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.models.visual_auxiliary import MODEL_VERSION, AuxiliaryVisualFusion
from seepat.training import experiment as runtime
from seepat.training.auxiliary_comparison_inputs import (
    ANNOTATIONS,
    CONTEXT_INPUTS,
    MONITOR,
    OUTPUT,
    ROLES,
    VERSION,
    ComparisonDataset,
    base_dataset,
    prepare_inputs,
    read_selection,
)
from seepat.training.auxiliary_visual import train_auxiliary_epoch
from seepat.training.matched_inputs import inside
from seepat.training.metrics import binary_classification_metrics
from seepat.training.sequence_pilot import fixed_options
from seepat.training.train import (
    FUSION_MODEL,
    VideoObjective,
    _forward_logits,
    build_event_classifier,
    train_model,
)
from seepat.training.video_objective import (
    VIDEO_SUPERVISION_CONTRACT,
    scored_video_bags,
    train_video_epoch,
    video_bags,
    video_loader,
)

ARMS = ("control", "candidate")
MODELS = (*ARMS, "anchor")
INITIAL = "outputs/training/fusion_phase5_v1/experiments/candidate/checkpoint_best.pt"
INITIAL_SHA = "46fb132230fb2292ecadab63ac84aca34882174d6dbd4680ade5adab82dc4b07"
INITIAL_RUN = "outputs/training/fusion_phase5_v1/experiments/candidate/run.json"
CONFIG = "configs/larger_auxiliary_comparison_v1.json"
DESIGN = {"version": VERSION, "output": OUTPUT, "train_requests": 1000, "train_per_category": 250,
          "monitor_requests": 64, "comparison_requests": 1000, "comparison_usable": 965,
          "comparison_clips": 10404, "auxiliary_coefficient": .25, "threshold": .5,
          "training": asdict(fixed_options()), "initial_checkpoint": INITIAL, "initial_sha256": INITIAL_SHA,
          "checkpoint": "fixed last epoch4", "seed_count": 1,
          "gate": "strict_AUC_BA_no_extra_FP_no_lower_total_TP_with_anchor_guard",
          "scope": "reused AV++ Validation development; no significance/H1 or automatic promotion"}
METADATA = {
    ANNOTATIONS, INITIAL_RUN,
    *(definition[1] for definition in ROLES.values()),
    "outputs/numerical_calibration/guarded_per_video_v3/calibration.json",
    *(f"{directory}/summary.json" for directory in (
        "outputs/numerical_calibration/guarded_per_video_v3", "outputs/numerical_calibration/phase5_outer_v1")),
    *(f"{CONTEXT_INPUTS}/{name}" for name in (
        "inventory_train.csv", "inventory_inner.csv", "inventory_outer.csv", "expanded_train.csv",
        "expanded_inner.csv", "expanded_outer.csv", "records_train.json", "records_inner.json", "records_outer.json",
        "summary.json", "seal.json")),
    *(f"{MONITOR}/{name}" for name in ("inventory_inner.csv", "index_inner.csv", "seal.json")),
    "outputs/phase5_context_inputs_v1/summary.json", "outputs/phase5_outer_context_inputs_v1/summary.json",
}


def load_protocol(project, config, *, metadata=True):
    protocol = runtime.read_json(config)
    actual = {p.relative_to(project).as_posix() for p in (project / "src/seepat").rglob("*.py")}
    actual.add("scripts/audit_clip_targets.py")
    actual.add("scripts/transfer_auxiliary_comparison.py")
    if (set(protocol) != {"design", "code_bindings", "metadata_bindings"} or protocol["design"] != DESIGN
            or set(protocol["code_bindings"]) != actual or set(protocol["metadata_bindings"]) != METADATA):
        raise ValueError("Unsupported larger comparison protocol/source membership; preserve the frozen design")
    for name, expected in protocol["code_bindings"].items():
        if hashlib.sha256(inside(project, name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Frozen comparison code changed: {name}")
    if metadata:
        for name, expected in protocol["metadata_bindings"].items():
            if file_sha256(inside(project, name)) != expected:
                raise ValueError(f"Frozen comparison metadata changed: {name}")
    return protocol, {"protocol_sha256": hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def train_arm(directory, arm, model, train, validation, options, binding, initial, *,
              stage, device="cpu", resume_from=None, progress=None):
    coefficient = .25 if arm == "candidate" else 0.
    weights = train.auxiliary_balance["class_weights"]
    supervision = {**VIDEO_SUPERVISION_CONTRACT, "version": VERSION, "auxiliary_coefficient": coefficient,
                   "auxiliary_weights": weights, "auxiliary_reduction": "known-clip mean within video; all-unknown zero"}
    objective = VideoObjective("video maximum plus masked visual exposure", supervision,
                               partial(train_auxiliary_epoch, weights=weights) if coefficient else train_video_epoch)
    contract = {"study": VERSION, "model": MODEL_VERSION, "arm": arm, "stage": stage, "binding": binding,
                "initial_state_sha256": initial, "clip_input_contract": train.input_contract,
                "fusion_inputs": train.calibration_contract, "auxiliary_coefficient": coefficient,
                "checkpoint": "fixed last epoch4 for training; readiness weights never initialize full fits"}
    run = train_model(model, train, validation, directory, options, torch.device(device), contract,
                      resume_from=resume_from, progress=progress, video_objective=objective)
    history = runtime.read_json(directory / "history.json")
    runtime.check_history(run, history, train, validation, options)
    if any(e["validation"]["selection"]["threshold"] != .5 for e in history):
        raise ValueError("Comparison cannot select a new cutoff")
    head = torch.cat([p.detach().flatten() for p in model.auxiliary_head.parameters()])
    if not torch.isfinite(head).all() or (not coefficient and head.any()) or (coefficient and not head.any()):
        raise ValueError("Auxiliary head learning/zero control differs; preserve for review")
    return {"completed_epochs": run["completed_epochs"], "optimizer_updates": run["global_step"],
            "skipped_updates": run["skipped_optimizer_steps"], "initial_state_sha256": initial,
            "train_videos": len(video_bags(train.rows, True)), "train_clips": len(train),
            "monitor_videos": len(video_bags(validation.rows, True)), "monitor_clips": len(validation),
            "final_validation": history[-1]["validation"], "auxiliary_head_l1": float(head.abs().sum()),
            "invocation_elapsed_seconds": run["elapsed_seconds"], "peak_cuda_memory_bytes": run.get("peak_cuda_memory_bytes")}


def fit_epochs(output, options, inputs_loader, model_loader, binding, *, stage, device="cpu", progress=None):
    """Two continuing fits, immutable per-epoch seals; completed work is reused."""
    output = Path(output)
    marker = output / f"through-{options.epochs}"
    job = {"study": VERSION, "stage": stage, "binding": binding, "options": asdict(options),
           "environment": {"device": device, "torch": torch.__version__, "python": platform.python_version()}}
    if marker.exists():
        return verify_fit(output, options.epochs, binding, expected_job=job)
    initial, results = None, {}
    # Preflight every saved epoch before loading a model or starting a sibling arm.
    for arm in ARMS:
        for epoch in range(1, options.epochs + 1):
            directory = output / arm / f"epoch-{epoch}"
            if directory.exists():
                if epoch > 1 and not (output / arm / f"epoch-{epoch - 1}").is_dir():
                    raise ValueError("Saved epoch chain has a gap; preserve before recovery")
                saved = runtime.verify_output(directory)
                old = runtime.read_json(directory / "job.json")
                expected = {**job, "arm": arm, "options": asdict(replace(options, epochs=epoch))}
                if old != expected or saved["completed_epochs"] != epoch:
                    raise ValueError("Saved epoch binding/settings changed")
                if initial is not None and initial != saved["initial_state_sha256"]:
                    raise ValueError("Saved arms have different actual initial weights")
                initial = saved["initial_state_sha256"]
    for number, arm in enumerate(ARMS, 1):
        if progress:
            progress.start_job(number, f"{stage}-{arm}")
        train, validation, model = None, None, None
        for epoch in range(1, options.epochs + 1):
            directory = output / arm / f"epoch-{epoch}"
            if directory.exists():
                results[arm] = runtime.verify_output(directory)
                continue
            if train is None:
                train, validation = inputs_loader(arm)
                if (train.input_contract != validation.input_contract
                        or train.calibration_contract != validation.calibration_contract):
                    raise ValueError("Train/monitor input or calibration contracts differ")
                model = model_loader(arm, epoch > 1)
            previous = output / arm / f"epoch-{epoch - 1}"
            if epoch == 1:
                actual = runtime.state_digest(model)
                if initial is not None and initial != actual:
                    raise ValueError("Fits do not share identical initial weights/buffers")
                initial = actual
            else:
                runtime.verify_output(previous, full=True)
            epoch_job = {**job, "arm": arm, "options": asdict(replace(options, epochs=epoch))}
            with runtime.output_stage(directory, epoch_job) as complete:
                result = train_arm(directory, arm, model, train, validation, replace(options, epochs=epoch),
                                   binding, initial, stage=stage, device=device,
                                   resume_from=previous / "checkpoint_last.pt" if epoch > 1 else None, progress=progress)
                results[arm] = complete({"status": "complete", **result})
        del model, train, validation
        if device == "cuda":
            torch.cuda.empty_cache()
    with runtime.output_stage(marker, job) as complete:
        summary = complete({"status": "complete", "stage": stage, "through_epoch": options.epochs,
                            "initial_state_sha256": initial, "arms": results,
                            "epoch_seals": {f"{arm}/epoch-{e}": file_sha256(output / arm / f"epoch-{e}/seal.json")
                                            for arm in ARMS for e in range(1, options.epochs + 1)}})
    verify_fit(output, options.epochs, binding)
    return summary


def verify_fit(output, epoch, binding, *, expected_job=None, full=False):
    marker = output / f"through-{epoch}"
    summary = runtime.verify_output(marker, expected_job)
    job = runtime.read_json(marker / "job.json")
    if job["study"] != VERSION or job["binding"] != binding or job["options"]["epochs"] != epoch:
        raise ValueError("Fit binding/epoch differs")
    if job["options"] != asdict(replace(fixed_options(), epochs=epoch)):
        raise ValueError("Fit options differ from the frozen comparison")
    expected_seals = {f"{a}/epoch-{e}" for a in ARMS for e in range(1, epoch + 1)}
    if set(summary["epoch_seals"]) != expected_seals or set(summary["arms"]) != set(ARMS):
        raise ValueError("Fit epoch/arm membership differs")
    for name, digest in summary["epoch_seals"].items():
        directory = inside(output, name)
        entry = runtime.verify_output(directory, full=full)
        arm, number = name.split("/")
        count = int(number.removeprefix("epoch-"))
        saved_job = runtime.read_json(directory / "job.json")
        run, history = runtime.read_json(directory / "run.json"), runtime.read_json(directory / "history.json")
        contract = run["resume_contract"]
        if (file_sha256(directory / "seal.json") != digest or saved_job["binding"] != binding
                or saved_job["study"] != VERSION or saved_job["arm"] != arm
                or saved_job["options"] != {**job["options"], "epochs": count}
                or entry["completed_epochs"] != count or run["completed_epochs"] != count or len(history) != count
                or entry["initial_state_sha256"] != summary["initial_state_sha256"]
                or contract["binding"] != binding or contract["study"] != VERSION or contract["arm"] != arm
                or contract["initial_state_sha256"] != summary["initial_state_sha256"]
                or entry["optimizer_updates"] != count * math.ceil(entry["train_videos"] / 4)
                or run["global_step"] != entry["optimizer_updates"] or entry["skipped_updates"]
                or any(h["skipped_optimizer_steps"] or h["optimizer_steps"] != math.ceil(entry["train_videos"] / 4)
                       or h["validation"]["selection"]["threshold"] != .5 for h in history)
                or (arm == "control" and entry["auxiliary_head_l1"] != 0)
                or (arm == "candidate" and not entry["auxiliary_head_l1"] > 0)):
            raise ValueError("Fit history/update/equal-start/head accounting differs")
        if count == epoch and entry != summary["arms"][arm]:
            raise ValueError("Final arm summary differs")
    return summary


def result_metrics(labels, probabilities):
    metrics = binary_classification_metrics(labels, probabilities, .5)
    pos = [p for y, p in zip(labels, probabilities, strict=True) if y]
    neg = [p for y, p in zip(labels, probabilities, strict=True) if not y]
    pairs = len(pos) * len(neg)
    if not pairs:
        raise ValueError("Paired scoring requires both video classes")
    wins = sum(2 * (p > n) + (p == n) for p in pos for n in neg)
    ba_num = metrics["true_positive"] * len(neg) + metrics["true_negative"] * len(pos)
    return {"metrics": metrics, "balanced_accuracy": ba_num / (2 * pairs), "ba_numerator": ba_num,
            "auc": wins / (2 * pairs), "auc_twice_pair_wins": wins, "auc_pair_count": pairs}


def compare_predictions(labels, control, candidate, anchor):
    arms = {name: result_metrics(labels, scores) for name, scores in zip(MODELS, (control, candidate, anchor), strict=True)}
    a, b, c = (arms[k] for k in MODELS)
    checks = {"higher_auc": b["auc_twice_pair_wins"] > a["auc_twice_pair_wins"],
              "higher_balanced_accuracy": b["ba_numerator"] > a["ba_numerator"],
              "no_extra_total_false_alarms": b["metrics"]["false_positive"] <= a["metrics"]["false_positive"],
              "no_lost_total_detections": b["metrics"]["true_positive"] >= a["metrics"]["true_positive"]}
    anchor_ok = b["auc_twice_pair_wins"] >= c["auc_twice_pair_wins"] and b["ba_numerator"] >= c["ba_numerator"]
    return {"arms": arms, "checks": checks, "pair_passed": all(checks.values()), "anchor_guard_passed": anchor_ok,
            "gate": "review_inference_integration_only" if all(checks.values()) and anchor_ok else "stop_and_review",
            "scope": DESIGN["scope"], "automatic_promotion": False}


def evaluate_models(output, dataset_loader, model_loader, selected, binding, *, device="cpu", progress=None):
    """Each completed model is reusable; every case and clip remains in the report."""
    output = Path(output)
    job = {"study": VERSION, "stage": "evaluation", "binding": binding, "threshold": .5, "optimizer_updates": 0}
    final = output / "complete"
    if final.exists():
        result = runtime.verify_output(final, job)
        for name, digest in result["model_seals"].items():
            runtime.verify_output(output / name)
            if file_sha256(output / name / "seal.json") != digest:
                raise ValueError("Saved scored model seal differs")
        return result
    model_results, signatures = {}, []
    for number, name in enumerate(MODELS, 1):
        directory, model_job = output / name, {**job, "model": name}
        if progress:
            progress.start_job(number, f"score-{name}")
        if not directory.exists():
            started = perf_counter()
            dataset, model = dataset_loader(name), model_loader(name)
            model.to(device).eval().requires_grad_(False)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            signature = hashlib.sha256(repr([(r["event_id"], r["video_id"], r["source_group"], r["kind"], r["class_id"])
                                             for r in dataset.rows]).encode()).hexdigest()
            bags = video_bags(dataset.rows, True)
            if {bag.video_id: bag.label for bag in bags} != {
                    r["video_id"]: int(r["category"] != "real") for r in selected if r["status"] == "paired"}:
                raise ValueError("Scored video labels differ from the fixed inventory")
            loader = video_loader(dataset, bags, 0, torch.device(device), 20261025)
            scores, clips = {}, []
            with runtime.output_stage(directory, model_job) as complete:
                with torch.inference_mode():
                    for bag, scored in zip(bags, scored_video_bags(model, loader, bags, torch.device(device),
                            _forward_logits, progress, "score frozen original video classifier; maximum/.5"), strict=True):
                        scores[bag.video_id] = max(scored["event_probabilities"])
                        clips.extend({"video_id": bag.video_id, "clip_id": clip, "probability": p}
                                     for clip, p in zip(bag.event_ids, scored["event_probabilities"], strict=True))
                if set(scores) != {r["video_id"] for r in selected if r["status"] == "paired"}:
                    raise ValueError("Scoring changed fixed available video membership")
                rows = [{**{k: r[k] for k in ("video_id", "source_group", "category", "reason")},
                         "status": "evaluated" if r["video_id"] in scores else "not_evaluated",
                         "label": int(r["category"] != "real") if r["video_id"] in scores else "",
                         "probability": scores.get(r["video_id"], "")} for r in selected]
                available = [r for r in rows if r["status"] == "evaluated"]
                result_metrics([r["label"] for r in available], [r["probability"] for r in available])
                atomic_write_csv(directory / "videos.csv", rows)
                atomic_write_csv(directory / "clips.csv", clips)
                complete({"status": "complete", "requested": len(rows), "evaluated": len(available),
                          "unavailable": len(rows) - len(available), "clips": len(clips), "signature": signature,
                          "elapsed_seconds": perf_counter() - started,
                          "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
            del model, dataset
            if device == "cuda":
                torch.cuda.empty_cache()
        model_results[name] = runtime.verify_output(directory, model_job)
        signatures.append(model_results[name]["signature"])
    if len(set(signatures)) != 1:
        raise ValueError("Scored models used different clip identities/order/labels")
    tables = {n: read_csv_rows(output / n / "videos.csv") for n in MODELS}
    reference = [{k: r[k] for k in ("video_id", "source_group", "category", "reason", "status", "label")} for r in tables["control"]]
    for name in MODELS:
        if [{k: r[k] for k in reference[0]} for r in tables[name]] != reference:
            raise ValueError("Paired video identities/availability/labels changed")
    available = [r for r in tables["control"] if r["status"] == "evaluated"]
    labels = [int(r["label"]) for r in available]
    probabilities = {n: [float(r["probability"]) for r in tables[n] if r["status"] == "evaluated"] for n in MODELS}
    result = compare_predictions(labels, *(probabilities[n] for n in MODELS))
    changed, changes = [], Counter()
    for row, a, b in zip(available, probabilities["control"], probabilities["candidate"], strict=True):
        if (a >= .5) != (b >= .5):
            kind = ("detection_gained" if b >= .5 else "detection_lost") if int(row["label"]) else (
                "false_alarm_added" if b >= .5 else "false_alarm_corrected")
            changes[kind] += 1
            changed.append({"video_id": row["video_id"], "category": row["category"], "change": kind, "control": a, "candidate": b})
    for name in MODELS:
        result["arms"][name].update(model_results[name])
        result["arms"][name]["by_category"] = {
            c: {"requested": sum(r["category"] == c for r in tables[name]),
                "evaluated": sum(r["category"] == c and r["status"] == "evaluated" for r in tables[name]),
                "correct": sum(int(float(r["probability"]) >= .5) == int(r["label"]) for r in tables[name]
                               if r["category"] == c and r["status"] == "evaluated")}
            for c in sorted({r["category"] for r in selected})}
    with runtime.output_stage(final, job) as complete:
        if changed:
            atomic_write_csv(final / "paired_changes.csv", changed)
        return complete({"status": "complete", **result, "optimizer_updates": 0,
                         "paired_changes": {"changed": len(changed), **dict(changes), "videos": changed},
                         "model_seals": {n: file_sha256(output / n / "seal.json") for n in MODELS}})


def input_binding(project, binding):
    directory = inside(project, f"{OUTPUT}/inputs")
    result = runtime.verify_output(directory)
    selected = file_sha256(inside(project, f"{OUTPUT}/selection/seal.json"))
    if (runtime.read_json(directory / "job.json") != {"study": VERSION, "stage": "inputs",
            "binding": {**binding, "selection_seal_sha256": selected}} or result["gate"] != "ready_for_readiness"):
        raise ValueError("Prepared input protocol/selection differs")
    return result, {**binding, "inputs_seal_sha256": file_sha256(directory / "seal.json")}


def fresh_model(project, resuming):
    if resuming:
        base = build_event_classifier(FUSION_MODEL, pretrained=False, freeze_backbone=True)
    else:
        contract = runtime.read_json(inside(project, INITIAL_RUN))["resume_contract"]
        base = runtime.warm_start(inside(project, INITIAL), contract, INITIAL_SHA)
    torch.manual_seed(20261025)
    return AuxiliaryVisualFusion(base)


def verify_stage(project, stage, binding):
    inputs, bound = input_binding(project, binding)
    if stage == "inputs":
        cases = runtime.read_json(inside(project, f"{OUTPUT}/inputs/case_seals.json"))
        inventories, _ = read_selection(project)
        expected = {f"{OUTPUT}/preparation/{role}/{r['video_id']}" for role, rows in inventories.items() for r in rows}
        if set(cases) != expected:
            raise ValueError("Prepared case membership differs, including unavailable requests")
        for name, digest in cases.items():
            directory = inside(project, name)
            runtime.verify_output(directory)
            if file_sha256(directory / "seal.json") != digest:
                raise ValueError("Prepared case provenance seal changed")
        return inputs
    for name, count in (("readiness1", 1), ("readiness2", 2), ("training1", 1), ("training", 4)):
        if stage in {name, "all"}:
            kind = "readiness" if name.startswith("readiness") else "training"
            result = verify_fit(inside(project, f"{OUTPUT}/{kind}"), count, bound)
            expected = 16 if kind == "readiness" else inputs["roles"]["train"]["usable"]
            if any(a["train_videos"] != expected or a["monitor_videos"] != (4 if kind == "readiness" else 62)
                   for a in result["arms"].values()):
                raise ValueError("Fit readiness/full membership differs")
            if stage != "all":
                return result
    if stage in {"evaluation", "all"}:
        verify_fit(inside(project, f"{OUTPUT}/training"), 4, bound)
        evaluation_binding = {**bound, "training_seal_sha256": file_sha256(inside(project, f"{OUTPUT}/training/through-4/seal.json"))}
        inventories, _ = read_selection(project)
        if not inside(project, f"{OUTPUT}/evaluation/complete/seal.json").is_file():
            raise FileNotFoundError("Evaluation is not sealed; verify never starts or completes a worker")
        result = evaluate_models(inside(project, f"{OUTPUT}/evaluation"), None, None,
                                 inventories["outer"], evaluation_binding)
        if (result["optimizer_updates"] or set(result["arms"]) != set(MODELS)
                or any((v["requested"], v["evaluated"], v["unavailable"], v["clips"]) != (1000, 965, 35, 10404)
                       for v in result["arms"].values())):
            raise ValueError("Comparison coverage/fitting differs")
        return result
    raise ValueError("Unknown comparison verification stage")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "run", "readiness1", "readiness2", "train", "evaluate", "verify", "watch"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(CONFIG))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--through-epoch", type=int, choices=(1, 4), default=4)
    parser.add_argument("--stage", choices=("inputs", "run", "readiness1", "readiness2", "training1", "training", "evaluation", "all"), default="inputs")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if args.command == "watch":
        return runtime.watch_progress(live_progress_path(inside(project, f"{OUTPUT}/{args.stage}_status.json")), "LARGER AUXILIARY")
    os.chdir(project)
    config = inside(project, str(args.config))
    _, binding = load_protocol(project, config)
    if args.command == "preflight":
        print("LARGER AUXILIARY PREFLIGHT PASSED | pinned code/metadata only; no model/media load", flush=True)
        return binding
    if args.command == "verify":
        result = verify_stage(project, args.stage, binding)
        print(f"LARGER AUXILIARY VERIFIED | {args.stage} | checkpoint bytes not rehashed", flush=True)
        return result
    stage = "inputs" if args.command == "prepare" else "training1" if args.command == "train" and args.through_epoch == 1 else "training" if args.command == "train" else args.command
    root = inside(project, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "operation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} command={args.command}; preserve after hard interruption\n")
    progress = WorkflowProgress(config, inside(project, f"{OUTPUT}/{stage}_status.json"), 3)
    progress.start_job(1, stage)
    try:
        if args.command == "prepare":
            result = prepare_inputs(project, binding, progress=progress)
            progress.finish()
            print(f"LARGER AUXILIARY INPUTS COMPLETE | {result['roles']} | no training or new pixels", flush=True)
            return result
        inputs, bound = input_binding(project, binding)
        inventories, indexes = read_selection(project)
        def dataset(role, readiness=False):
            return ComparisonDataset(base_dataset(project, role, indexes[role]), inside(project, f"{OUTPUT}/inputs"), role,
                                     videos=set(inputs["readiness"][role]) if readiness else None)

        def fit(kind, epoch):
            return fit_epochs(inside(project, f"{OUTPUT}/{kind}"), replace(fixed_options(), epochs=epoch),
                              lambda _arm: (dataset("train", kind == "readiness"), dataset("inner", kind == "readiness")),
                              lambda _arm, resume: fresh_model(project, resume), bound, stage=kind,
                              device=args.device, progress=progress)

        if args.command in {"run", "readiness1", "readiness2"}:
            result = fit("readiness", 1)
            if args.command != "readiness1":
                result = fit("readiness", 2)
        if args.command in {"run", "train"}:
            verify_stage(project, "readiness2", binding)
            fit("training", 1)
            result = fit("training", args.through_epoch)
        if args.command in {"run", "evaluate"}:
            verify_stage(project, "training", binding)
            evaluation_binding = {**bound, "training_seal_sha256": file_sha256(inside(project, f"{OUTPUT}/training/through-4/seal.json"))}
            def model(name):
                if name == "anchor":
                    return fresh_model(project, False)
                directory = inside(project, f"{OUTPUT}/training/{name}/epoch-4")
                runtime.verify_output(directory, full=True)
                saved = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False, mmap=True)
                declared = runtime.read_json(directory / "run.json")["resume_contract"]
                if (saved["checkpoint_type"] != "seepat_resumable_training" or saved["completed_epoch"] != 4
                        or saved["resume_contract"] != declared or declared["binding"] != bound
                        or declared["study"] != VERSION or declared["arm"] != name):
                    raise ValueError("Scoring checkpoint/arm/protocol differs")
                restored = fresh_model(project, True)
                restored.load_state_dict(saved["model_state"], strict=True)
                return restored
            result = evaluate_models(inside(project, f"{OUTPUT}/evaluation"), lambda _name: dataset("outer"), model,
                                     inventories["outer"], evaluation_binding, device=args.device, progress=progress)
        progress.finish()
        print(f"LARGER AUXILIARY COMPLETE | {stage} | {result.get('gate', 'completed epoch boundary')}", flush=True)
        return result
    except BaseException as error:
        progress.finish(error)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as error:
        raise SystemExit(f"LARGER AUXILIARY STOPPED | {error}") from error
