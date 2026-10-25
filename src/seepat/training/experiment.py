"""Sealed experiment outputs and paired training, shared by the two studies."""

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.training.matched_inputs import inside


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def seal_output(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"seal.json", "worker.lock"} and not path.name.endswith(".progress.json"):
            name = path.relative_to(directory).as_posix()
            inside(directory, name)
            files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    atomic_write_json(directory / "seal.json", files)



def verify_output(directory, job=None, *, full=False):
    """Check small records by hash; checkpoint-byte hashing is explicit and Niko-run."""
    if (directory / "worker.lock").exists():
        raise ValueError("Output active/interrupted; preserve before recovery")
    saved = read_json(directory / "job.json")
    if job is not None and saved != job:
        raise ValueError("Output binding/settings changed; preserve existing files")
    summary, seal = read_json(directory / "summary.json"), read_json(directory / "seal.json")
    if summary["status"] != "complete" or not {"job.json", "summary.json"} <= set(seal):
        raise ValueError("Output partial or failed; no automatic replay")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
              and p.name not in {"seal.json", "worker.lock"} and not p.name.endswith(".progress.json")}
    if actual != set(seal):
        raise ValueError("Output membership changed; preserve before recovery")
    for name, item in seal.items():
        path = inside(directory, name)
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or (full or path.suffix != ".pt") and file_sha256(path) != item["sha256"]):
            raise ValueError(f"Output changed: {name}")
    return summary



def state_digest(model):
    """Compare actual initial weights/buffers, not just filenames or random seeds."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



def check_history(run, history, train, validation, options):
    from seepat.training.video_objective import select_video_bags, video_bags

    if (run["status"] != "complete" or run["completed_epochs"] != options.epochs
            or len(history) != options.epochs or run["global_step"] != sum(e["optimizer_steps"] for e in history)
            or not 1 <= run["best_epoch"] <= options.epochs):
        raise ValueError("Training epoch/update/checkpoint accounting differs")
    for epoch in history:
        for key, dataset, limit in (("train", train, options.max_train_batches),
                                    ("validation", validation, options.max_validation_batches)):
            is_train = key == "train"
            plan = select_video_bags(video_bags(dataset.rows, True), options.seed + epoch["epoch"] if is_train else options.seed,
                                     is_train, limit)
            section = epoch[key]
            expected = [{"video_id": bag.video_id, "label": bag.label, "events": len(bag.indices)} for bag in plan]
            contexts = sum(dataset.rows[i]["kind"] == "context" for bag in plan for i in bag.indices)
            if (section.get("complete_video_bags") is not True or section.get("video_bags") != expected
                    or section.get("videos_processed") != len(plan)
                    or section.get("events_processed") != sum(len(bag.indices) for bag in plan)
                    or not math.isfinite(section["loss"]) or {b.label for b in plan} != {0, 1}
                    or contexts < 1 or section.get("context_clips_processed") != contexts):
                raise ValueError("Every epoch must use complete bags, both classes and the planned context clips")
            if is_train and (epoch["optimizer_steps"] != math.ceil(len(plan) / options.gradient_accumulation_steps)
                             or epoch["skipped_optimizer_steps"] != 0):
                raise ValueError("Optimizer updates differ from the frozen complete-video budget")



def watch_progress(path):
    import time

    from seepat.live_progress import format_live_progress

    try:
        while True:
            if path.is_file():
                record = read_json(path)
                print(format_live_progress(record), flush=True)
                if record["status"] != "running":
                    if record["status"] != "complete":
                        raise ValueError("Worker stopped without completing; preserve outputs and logs")
                    return record
            else:
                print("CONTEXT COMPARISON | waiting for worker; this does not start it", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Watcher stopped; worker was not stopped.", flush=True)
        return None


@contextmanager
def output_stage(directory, job, progress=None, *, resume=False, partial_results=None):
    """Own one stage's lock, failure record, seal and completion verification."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=resume)
    lock = directory / "worker.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Stage active; preserve partial output after interruption.\n")
    committed = False

    def complete(summary):
        nonlocal committed
        if committed or summary.get("status") != "complete":
            raise ValueError("A stage must commit exactly one complete summary")
        atomic_write_json(directory / "summary.json", summary)
        seal_output(directory)
        committed = True
        return summary

    try:
        atomic_write_json(directory / "job.json", job)
        yield complete
        if not committed:
            raise RuntimeError("Stage ended without a completed summary")
    except BaseException as error:
        failure = {"status": "failed", "stage": job.get("stage"), "error": str(error)}
        if partial_results is not None:
            failure["arms"] = partial_results
        atomic_write_json(directory / "summary.json", failure)
        if progress:
            progress.finish(error)
        raise
    finally:
        lock.unlink()
    verify_output(directory, job)
    if progress:
        progress.finish()


def warm_start(path, contract, expected_sha256, *, factory=None):
    """Load exact evaluation weights while preserving the template's freeze mask."""
    import torch

    from seepat.training.train import FUSION_MODEL, build_event_classifier

    if file_sha256(path) != expected_sha256:
        raise ValueError("Original checkpoint hash differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("checkpoint_type") != "seepat_evaluation_model":
        raise ValueError("Original checkpoint type differs")
    if checkpoint.get("resume_contract") != contract:
        raise ValueError("Original checkpoint contract differs")
    model = factory() if factory else build_event_classifier(FUSION_MODEL, pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval()


def run_training_pair(directory, job, inputs_loader, model_loader, train_arm, *,
                      arms, resume=False, progress=None, summary_fields=None, verify=verify_output):
    """Keep scientific arm rules in the caller and lifecycle/identity checks here."""
    import torch

    directory = Path(directory)
    epoch = job["options"]["epochs"]
    previous = None
    if directory.exists():
        if not resume:
            return verify(directory, job)
        previous = verify(directory, full=True)
        previous_job = read_json(directory / "job.json")
        resumed_job = {**previous_job, "options": {**previous_job["options"], "epochs": epoch}}
        if (job["stage"] != "readiness" or epoch != 2 or previous["through_epoch"] != 1
                or resumed_job != job):
            raise ValueError("Only the sealed readiness epoch-one state may resume to epoch two")
    elif resume:
        raise ValueError("Missing completed readiness state")
    summaries = {}
    initial = previous["initial_state_sha256"] if previous else None
    with output_stage(directory, job, progress, resume=resume, partial_results=summaries) as complete:
        for number, arm in enumerate(arms, 1):
            if progress:
                progress.start_job(number, arm)
                progress("load fixed inputs and model", 0, 0, arm)
            train, validation = inputs_loader(arm)
            if (train.input_contract != validation.input_contract
                    or train.calibration_contract != validation.calibration_contract):
                raise ValueError("Train/inner input or calibration contracts differ")
            model = model_loader(arm, resume)
            if not resume:
                actual = state_digest(model)
                if initial is not None and initial != actual:
                    raise ValueError("Arms do not start from identical model weights/buffers")
                initial = actual
            summaries[arm] = train_arm(arm, model, train, validation, initial, directory / arm)
            del model, train, validation
            if job["environment"]["device"] == "cuda":
                torch.cuda.empty_cache()
        return complete({"status": "complete", "stage": job["stage"], "through_epoch": epoch,
                         "initial_state_sha256": initial, "arms": summaries, **(summary_fields or {})})
