"""Read-only input and optimizer-state checks before the first experiments."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from seepat.artifacts import read_csv_rows

if TYPE_CHECKING:
    from seepat.workflow import ModelTrainingJob


def audit_inputs(job: ModelTrainingJob) -> dict[str, object]:
    """Check metadata and clip presence without decoding clips or building a model."""
    from seepat.evidence import calibrated_manifest_contract
    from seepat.training.dataset import MouthEventDataset
    from seepat.training.train import FUSION_MODEL, TrainingOptions, source_group_overlap

    options = TrainingOptions(**job.options)
    options.validate()
    inputs, summaries, contracts = {}, {}, []
    for split, path in (("train", job.train_manifest), ("val", job.validation_manifest)):
        rows = read_csv_rows(path)
        if not rows or any(
            row.get("dataset_split") != split
            or row.get("dataset_source", "AV-Deepfake1M++") != "AV-Deepfake1M++"
            for row in rows
        ):
            raise ValueError("Experiments require exclusively AV++ Train and Validation inputs")
        ids = [row.get("event_id") for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{split} event IDs must be nonempty and unique")
        if {row["class_id"] for row in rows} != {"0", "1"}:
            raise ValueError(f"{split} must contain both binary event classes")
        dataset = MouthEventDataset(path, job.project_root, dataset_split=split)
        missing = [row["mouth_clip_path"] for row in rows
                   if not (job.project_root / row["mouth_clip_path"].replace("\\", "/")).is_file()]
        if missing:
            raise ValueError(f"{split} has {len(missing)} missing mouth clips; first: {missing[0]}")
        if job.model == FUSION_MODEL:
            contracts.append(calibrated_manifest_contract(path, rows))
        inputs[split] = rows
        summaries[split] = {
            "events": len(dataset), "class_counts": dict(Counter(row["class_id"] for row in rows)),
            "source_groups": len({row["source_group"] for row in rows}),
        }
    if source_group_overlap(inputs["train"], inputs["val"]):
        raise ValueError("Train and Validation source groups overlap")
    if {row["event_id"] for row in inputs["train"]} & {row["event_id"] for row in inputs["val"]}:
        raise ValueError("Train and Validation event IDs overlap")
    if contracts and contracts[0] != contracts[1]:
        raise ValueError("Fusion inputs must share one frozen Train calibration")
    return {"name": job.name, "status": "passed", "inputs": summaries,
            "device": job.device, "pretrained": job.pretrained, "options": asdict(options)}


def _finite(value: object) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_readiness(job: ModelTrainingJob) -> dict[str, object]:
    """Inspect a completed, separately resumed preflight; never launch training."""
    import torch

    from seepat.training.train import VIDEO_BALANCED_ACCURACY_SELECTION, TrainingOptions
    from seepat.workflow import _model_training_configuration_matches

    directory = job.readiness_dir or job.output_dir
    result: dict[str, object] = {"name": job.name, "directory": str(directory), "status": "pending"}
    if not (directory / "run.json").is_file():
        result["reason"] = "No readiness run saved yet; complete epoch 1 and resume to epoch 2"
        return result
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        options = TrainingOptions(**run["options"])
        options.validate()
        if (run["status"] != "complete" or options.max_train_batches is None
                or options.max_validation_batches is None or options.epochs < 2
                or int(run.get("completed_epochs", 0)) < 2 or not run.get("resumed_from")):
            raise ValueError("Complete epoch 1, then resume to epoch 2 before accepting readiness")
        actual, expected = asdict(options), asdict(TrainingOptions(**job.options))
        for key in ("epochs", "max_train_batches", "max_validation_batches"):
            actual.pop(key)
            expected.pop(key)
        if actual != expected:
            raise ValueError("Readiness settings differ from the requested experiment")
        ready_job = replace(job, output_dir=directory, options=asdict(options), readiness_dir=None)
        if not _model_training_configuration_matches(ready_job, run):
            raise ValueError("Readiness input hashes, calibration, model or settings are stale")
        if Path(run["resumed_from"]).resolve() != (directory / "checkpoint_last.pt").resolve():
            raise ValueError("Readiness must resume its own checkpoint")
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        if len(history) < 2 or any(
            row.get("optimizer_steps", 0) <= 0 or row.get("skipped_optimizer_steps", -1) != 0
            or not math.isfinite(row["train"]["loss"])
            or not math.isfinite(row["validation"]["loss"])
            for row in history
        ):
            raise ValueError("Every readiness epoch must have finite losses and completed, unskipped updates")
        for epoch in history:
            for split in ("train", "validation"):
                counts = epoch[split]["events"]
                if (counts["true_positive"] + counts["false_negative"] == 0
                        or counts["true_negative"] + counts["false_positive"] == 0):
                    raise ValueError("Readiness must exercise both event classes in each split and epoch")
            if options.selection_metric == VIDEO_BALANCED_ACCURACY_SELECTION:
                selection = epoch["validation"].get("selection", {})
                if (
                    selection.get("metric") != VIDEO_BALANCED_ACCURACY_SELECTION
                    or selection.get("positives", 0) < 1
                    or selection.get("negatives", 0) < 1
                ):
                    raise ValueError(
                        "Readiness must exercise both video classes for balanced-accuracy selection"
                    )
        updates = sum(row["optimizer_steps"] for row in history)
        if run.get("global_step") != updates or history[-1].get("global_step") != updates:
            raise ValueError("Readiness update counts disagree")
        checkpoint = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False, mmap=True)
        if (checkpoint.get("checkpoint_type") != "seepat_resumable_training"
                or checkpoint.get("resume_contract") != run["resume_contract"]
                or checkpoint.get("options") != run["options"]
                or checkpoint.get("completed_epoch") != run["completed_epochs"]
                or checkpoint.get("global_step") != updates or checkpoint.get("history") != history):
            raise ValueError("Readiness checkpoint and reports disagree")
        optimizer = checkpoint["optimizer_state"]
        states = optimizer["state"]
        parameter_ids = {key for group in optimizer["param_groups"] for key in group["params"]}
        if not states or set(states) != parameter_ids or any(float(row.get("step", 0)) <= 0 for row in states.values()):
            raise ValueError("Readiness optimizer has missing or uninitialized parameter state")
        if not checkpoint["model_state"] or not _finite(checkpoint["model_state"]) or not _finite(optimizer):
            raise ValueError("Readiness model or optimizer contains non-finite values")
        if not (directory / "checkpoint_best.pt").is_file():
            raise ValueError("Readiness did not save its selected checkpoint")
        selected = torch.load(
            directory / "checkpoint_best.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if selected.get("selection_metric") != options.selection_metric:
            raise ValueError("Readiness selected checkpoint uses the wrong metric")
        result.update(status="passed", optimizer_updates=updates, epochs=len(history),
                      environment=run["environment"], peak_cuda_memory_bytes=run["peak_cuda_memory_bytes"],
                      last_invocation_seconds=run["elapsed_seconds"],
                      last_invocation_events_per_second=run["processed_events_per_second"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result["reason"] = str(error)
    return result


def require_readiness(job: ModelTrainingJob) -> None:
    import torch

    report = check_readiness(job)
    if report["status"] != "passed":
        raise RuntimeError(f"[{job.name}] training readiness is not accepted: {report.get('reason')}")
    environment = report["environment"]
    gpu = torch.cuda.get_device_name() if job.device == "cuda" and torch.cuda.is_available() else None
    if (environment.get("torch") != torch.__version__ or environment.get("gpu") != gpu
            or environment.get("cuda_runtime") != torch.version.cuda):
        raise RuntimeError("Repeat readiness on the selected training hardware/software before a full run")


def main() -> None:
    from seepat.workflow import load_workflow_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training_experiments.yaml"))
    parser.add_argument("--inputs-only", action="store_true", help="Check manifests and clip paths; no checkpoint loading")
    args = parser.parse_args()
    settings = load_workflow_settings(args.config)
    jobs = list(settings.model_training_jobs)
    if settings.study_job is not None:
        from seepat.training.study import load_study, model_job

        study = settings.study_job
        folds = load_study(study)["folds"]
        jobs.extend(model_job(study, fold, model, readiness=False)
                    for fold in folds for model in study.models)
    results = []
    for job in jobs:
        try:
            results.append(audit_inputs(job) if args.inputs_only else check_readiness(job))
        except (OSError, ValueError, KeyError, TypeError) as error:
            results.append({"name": job.name, "status": "failed", "reason": str(error)})
    print(json.dumps({"read_only": True, "jobs": results}, indent=2))
    if not jobs or any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
