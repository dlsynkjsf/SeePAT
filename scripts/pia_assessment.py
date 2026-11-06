"""Resumable PIA predictions and sealed metrics for the fixed 1,000-video selection."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import pia_baseline as author
from scripts.assessment_runtime import (
    archive_lock,
    assessment,
    empty_crop_failure,
    external_target,
    process_alive,
    read,
    run_cases,
    safe_path,
    verify_predictions,
)
from seepat.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    read_csv_rows,
    stable_id,
)
from seepat.live_progress import WorkflowProgress, format_live_progress, live_progress_path

CONFIG = ROOT / "configs/pia_assessment.optimized.json"
DECODE_RUNTIME = "scripts/pia_cpu_decode_runtime.py"
RUNTIME_HELPERS = [
    "scripts/pia_cpu_consumed_runtime.py",
    "scripts/pia_cpu_identity_runtime.py",
    "scripts/pia_cpu_spin_runtime.py",
]


SOURCE_FILES = (
    "scripts/pia_assessment.py", "scripts/assessment_runtime.py", "scripts/pia_baseline.py",
    "src/seepat/artifacts.py", "src/seepat/live_progress.py",
    ".tools/pia_feasibility_v1/source/main.py", ".tools/pia_feasibility_v1/source/src/model.py",
    ".tools/pia_feasibility_v1/source/src/utils.py", *RUNTIME_HELPERS,
)


def selected_requests(selection, source_root):
    names = selection["selected_files"]
    if len(names) != 1000 or len(set(names)) != 1000:
        raise ValueError("Exactly the original 1,000 unique requests are required")
    pool = {row["file"]: row for row in selection["pool"]}
    if len(pool) != len(selection["pool"]):
        raise ValueError("Duplicate selection inventory entries")
    rows = []
    for name in names:
        source = safe_path(source_root, name)
        if name not in pool:
            raise ValueError("Selected request is missing from the frozen inventory")
        item = pool[name]
        rows.append(
            {
                "video_id": "dfe-" + stable_id(name),
                "file": name,
                "path": str(source),
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
        )
    return rows


def checked_file(binding):
    path = safe_path(ROOT, binding["path"])
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("Bound file changed: " + binding["path"])
    return path


def load_profile(path):
    config = read(path)
    if config["version"] not in {"pia-assessment-1000-v1", "pia-assessment-1000-v2", "pia-assessment-1000-v3"} or config["requested_videos"] != 1000:
        raise ValueError("Unsupported PIA assessment profile")
    if (
        config["runtime_helpers"] != RUNTIME_HELPERS + ([DECODE_RUNTIME] if config["version"].endswith("v3") else [])
        or not {*RUNTIME_HELPERS, "scripts/pia_assessment.py"} <= config["source_sha256"].keys()
    ):
        raise ValueError("Runtime helpers require explicit source bindings")
    if type(config["case_timeout_seconds"]) is not int or not (
        0 < config["case_timeout_seconds"] <= 86400
    ):
        raise ValueError("Per-video timeout must be between 1 second and 24 hours")
    required = set(SOURCE_FILES) | ({DECODE_RUNTIME} if config["version"].endswith("v3") else set())
    if config["version"] != "pia-assessment-1000-v1" and not required <= config["source_sha256"].keys():
        raise ValueError("Assessment runtime requires explicit source bindings")
    for name, digest in config["source_sha256"].items():
        checked_file({"path": name, "sha256": digest})
    output = safe_path(ROOT, config["output"])
    if not output.is_relative_to(ROOT / "outputs") or output == ROOT / "outputs":
        raise ValueError("Assessment output must be a dedicated outputs subdirectory")
    prior = safe_path(ROOT, config["reuse"]["output"])
    if output == prior or output in prior.parents or prior in output.parents:
        raise ValueError("New assessment and historical output must be separate")
    requests = selected_requests(
        read(checked_file(config["selection"])), safe_path(ROOT, config["source_root"])
    )
    proof_path = checked_file(config["parity_proof"])
    proof = read(proof_path)
    if (
        file_sha256(proof_path) != read(proof_path.parent / "seal.json")["proof_sha256"]
        or proof["parity"]["passed"] is not True
        or proof["parity"]["absolute_probability_difference"] > 1e-6
    ):
        raise ValueError("Original consumed-identity parity proof is invalid")
    return config, output, requests


def freeze_runtime(config, config_path, output):
    setup = read(checked_file(config["setup"]))
    work = safe_path(ROOT, config["work_dir"])
    if (
        setup["status"] != "complete"
        or setup["image"] != config["image"]
        or setup["environment"]["device"] != "cpu"
        or setup["environment"]["strict_author_checkpoint_load"] is not True
        or author._image_id(config["image"]) != config["image"]
    ):
        raise ValueError("Previously captured PIA CPU setup is unavailable")
    stamps = {}
    for item in setup["assets"] + setup["cache_files"]:
        path = Path(item["path"])
        if not path.resolve().is_relative_to(work) or path.stat().st_size != item["bytes"]:
            raise ValueError("Captured PIA asset missing or changed")
        stamps[str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
    contract = {
        "config_sha256": file_sha256(config_path),
        "image": config["image"],
        "selection_sha256": config["selection"]["sha256"],
        "parity_proof_sha256": config["parity_proof"]["sha256"],
        "runtime_asset_stamps": stamps,
        "device": "cpu",
        "network": "none",
        "policy": "exact-author-first-five-identity-v1",
    }
    freeze = output / "freeze.json"
    if freeze.exists():
        if read(freeze) != contract:
            raise ValueError("Frozen runtime changed; preserve this run")
    else:
        checked_file(config["checkpoint"])
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(freeze, contract)
    return work, contract


def reused_prediction(config, row):
    case = safe_path(ROOT, config["reuse"]["output"] + "/cases/" + row["video_id"])
    summary = case / "summary.json"
    if not (case / "seal.json").exists():
        return None
    digest = file_sha256(summary)
    job, saved = read(case / "job.json"), read(summary)
    if (
        digest != read(case / "seal.json")["summary_sha256"]
        or job["source"] != row
        or saved["source_sha256"] != row["sha256"]
        or job["contract"]["image"] != config["image"]
        or job["contract"]["config_sha256"] != config["reuse"]["config_sha256"]
    ):
        raise ValueError("Historical prediction binding differs: " + row["video_id"])
    return {
        "eligible": saved["eligible"],
        "probability": saved["probability"],
        "reason": saved["reason"],
        "reused_from": str(summary),
        "reused_summary_sha256": digest,
        "original_seconds": saved["seconds"],
    }


def run_author(config, work, row, directory, progress):
    pointer = directory / "attempt.json"
    attempt = read(pointer) if pointer.exists() else None

    def heartbeat(phase, _done, _total, _current):
        marker = ""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            markers = [v for v in lines if v.startswith(("[CPU_PROGRESS]", "[STEP]"))]
            if markers:
                marker = " | " + markers[-1]
        progress(phase, row["video_id"] + marker)

    active = False
    if attempt:
        folder = safe_path(directory, attempt["folder"])
        log = folder / "author.log"
        active = (
            subprocess.run(
                ["docker", "inspect", attempt["container"], "--format", "{{.State.Running}}"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            == "true"
        )
    if active:
        author._command(
            ["docker", "logs", "--follow", attempt["container"]],
            folder / "reconnect.log",
            heartbeat,
            "reconnect PIA case",
            timeout=config["case_timeout_seconds"],
        )
    saved_ok = (
        attempt
        and (folder / "author/results.json").exists()
        and any(
            p.exists()
            and "[SAVED] /output/results.json" in p.read_text(encoding="utf-8", errors="replace")
            for p in [folder / "author.log", folder / "reconnect.log"]
        )
    )
    if not saved_ok:
        failure = empty_crop_failure(directory)
        if failure:
            return failure
        folder = directory / ("attempt-" + str(time.time_ns()))
        (folder / "author").mkdir(parents=True)
        command, image = author._container(config["image"], work)
        attempt = {"folder": folder.name, "container": command[command.index("--name") + 1]}
        atomic_write_json(pointer, attempt)
        for name in config["runtime_helpers"]:
            command += author._mount(safe_path(ROOT, name), "/helpers/" + Path(name).name)
        command += [
            *author._mount(Path(row["path"]), "/input/video.mp4"),
            *author._mount(folder / "author", "/output", False),
            image,
            "python",
            "/helpers/" + Path(DECODE_RUNTIME if config.get("version", "").endswith("v3") else RUNTIME_HELPERS[0]).name,
            "--video",
            "/input/video.mp4",
            "--outdir",
            "/output",
        ]
        log = folder / "author.log"
        try:
            author._command(
                command,
                log,
                heartbeat,
                "PIA author inference; CPU; offline",
                timeout=config["case_timeout_seconds"],
            )
        except RuntimeError:
            text = log.read_text(encoding="utf-8", errors="replace")
            if "Failed to load audio:" in text:
                return {
                    "eligible": False,
                    "probability": None,
                    "reason": "audio_decode_failed",
                    "attempt": attempt,
                }
            failure = empty_crop_failure(directory)
            if failure:
                return failure
            raise
    result = author.parse_prediction(folder / "author/results.json")
    try:
        crops = author._crop_records(folder, config["phonemes"])
    except ValueError as error:
        if "No usable author phoneme crops" not in str(error):
            raise
        return {
            "eligible": False,
            "probability": None,
            "reason": "no_supported_crops",
            "attempt": attempt,
        }
    receipt = folder / "author/consumed_identity.json"
    if not receipt.is_file():
        raise ValueError("Consumed identity receipt missing")
    atomic_write_json(folder / "selected_crops.json", crops)
    return {
        "eligible": True,
        "probability": result["fake_probability"],
        "reason": None,
        "attempt": attempt,
        "selected_crop_files": len(crops),
        "identity_selection_sha256": file_sha256(receipt),
        "author_results_sha256": file_sha256(folder / "author/results.json"),
    }


def verify(output):
    summary = verify_predictions(output)
    job, predictions = read(output / "job.json"), read(output / "predictions.json")
    if (
        job["contract"] != read(output / "freeze.json")
        or summary["contract"] != job["contract"]
        or len(predictions) != len(job["requests"])
    ):
        raise ValueError("Prediction count or frozen binding changed")
    for row, result in zip(job["requests"], predictions, strict=True):
        case = safe_path(output, "cases/" + row["video_id"])
        if (
            read(case / "job.json")["source"] != row
            or read(case / "summary.json") != result
            or file_sha256(case / "summary.json") != read(case / "seal.json")["summary_sha256"]
            or result["source_sha256"] != row["sha256"]
        ):
            raise ValueError("Saved case differs from sealed predictions")
        if (
            "reused_from" in result
            and file_sha256(Path(result["reused_from"])) != result["reused_summary_sha256"]
        ):
            raise ValueError("Reused historical prediction changed")
    return summary


def verify_metrics(config, output):
    summary = verify(output)
    result = read(output / "metrics.json")
    seal = read(output / "metrics.seal.json")
    if (
        result["binding"]
        != {
            "predictions_sha256": summary["predictions_sha256"],
            "metadata_sha256": config["metadata"]["sha256"],
        }
        or file_sha256(output / "metrics.json") != seal["metrics_sha256"]
        or file_sha256(output / "report.md") != seal["report_sha256"]
    ):
        raise ValueError("Saved assessment changed; preserve it")
    return result


def write_metrics(config, output):
    summary = verify(output)
    binding = {
        "predictions_sha256": summary["predictions_sha256"],
        "metadata_sha256": config["metadata"]["sha256"],
    }
    target = output / "metrics.json"
    if (output / "metrics.seal.json").exists():
        return verify_metrics(config, output)
    truth = {}
    for item in read_csv_rows(checked_file(config["metadata"])):
        name = Path(item["Filename"].replace("\\", "/")).name
        if name in truth:
            raise ValueError("Ambiguous metadata filename")
        truth[name] = item
    predictions = read(output / "predictions.json")
    labels = {"primary_either_modality": [], "secondary_visual_only": []}
    for row in predictions:
        item = truth[Path(row["file"]).name]
        labels["primary_either_modality"].append(
            external_target(item["Video Ground Truth"], item["Audio Ground Truth"])
        )
        labels["secondary_visual_only"].append(external_target(item["Video Ground Truth"], "real"))
    scores = [row["probability"] for row in predictions]
    result = {
        "status": "complete",
        "binding": binding,
        "requested": len(predictions),
        "metrics": {
            key: assessment(values, scores, math.nextafter(0.5, math.inf))
            for key, values in labels.items()
        },
        "decision_rule": "P(fake) > 0.5",
        "training_performed": False,
        "threshold_selection_performed": False,
        "five_fold_performed": False,
    }
    if target.exists() and read(target) != result:
        raise ValueError("Partial assessment differs; preserve it")
    lines = [
        "# PIA external assessment",
        "",
        f"Requested videos: {len(predictions)}.",
        "",
        "| Target | Known scored | Accuracy | Precision | Recall | F1 | Specificity | BA | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in result["metrics"].items():
        fields = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"]
        cells = ["undefined" if values[k] is None else f"{100 * values[k]:.2f}%" for k in fields]
        auc = "undefined" if values["roc_auc"] is None else f"{values['roc_auc']:.4f}"
        lines.append(
            f"| {key} | {values['scored_known_target']} | " + " | ".join(cells) + f" | {auc} |"
        )
    lines += ["", "Unknown targets and unavailable predictions (counts can overlap):", ""]
    for key, values in result["metrics"].items():
        lines.append(
            f"- {key}: {values['unknown_target']} unknown targets; "
            f"{values['unavailable_prediction']} unavailable predictions."
        )
    lines.append(
        "\n"
        "Frozen author cutoff; no fitting or threshold selection. "
        "Unknown labels and unavailable inputs remain in requested coverage."
    )
    atomic_write_text(output / "report.md", "\n".join(lines) + "\n")
    atomic_write_json(target, result)
    atomic_write_json(
        output / "metrics.seal.json",
        {"metrics_sha256": file_sha256(target), "report_sha256": file_sha256(output / "report.md")},
    )
    return result


def watch(output):
    try:
        while True:
            path = live_progress_path(output / "workflow.json")
            if path.exists():
                record = read(path)
                print(format_live_progress(record), flush=True)
                if record["status"] in {"complete", "failed", "interrupted"}:
                    return 0 if record["status"] == "complete" else 1
            else:
                print("PIA | waiting for worker", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Watcher stopped; worker is unchanged.", flush=True)
        return 0


def run(config, path, output, requests):
    prior = safe_path(ROOT, config["reuse"]["output"])
    lock = Path(str(prior) + ".worker.lock")
    if lock.exists() and process_alive(int(lock.read_text())):
        raise RuntimeError("Historical PIA worker is active; finish or stop it before this run")
    with archive_lock(output):
        progress = WorkflowProgress(path, output / "workflow.json", 2)
        try:
            progress.start_job(1, "PIA 1,000-video predictions")
            work, contract = freeze_runtime(config, path, output)
            positions = {r["video_id"]: i for i, r in enumerate(requests)}

            def worker(row, directory):
                saved = reused_prediction(config, row)
                if saved is not None:
                    return saved
                return run_author(
                    config,
                    work,
                    row,
                    directory,
                    lambda phase, text: progress(
                        phase, positions[row["video_id"]], len(requests), text
                    ),
                )

            run_cases(output, requests, contract, worker, progress)
            progress.start_job(2, "PIA sealed assessment")
            result = write_metrics(config, output)
            atomic_write_json(output / "workflow.json", result)
            progress.finish()
            print(f"PIA COMPLETE | {len(requests)} requests | {output / 'report.md'}", flush=True)
            return 0
        except BaseException as error:
            progress.finish(error)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run", "watch", "verify"], default="run")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    try:
        if args.command == "watch":
            return watch(safe_path(ROOT, read(args.config)["output"]))
        config, output, requests = load_profile(args.config)
        if args.command == "verify":
            job = read(output / "job.json")
            if (
                job["contract"]["config_sha256"] != file_sha256(args.config)
                or job["requests"] != requests
            ):
                raise ValueError("Saved assessment belongs to a different profile or selection")
            verify_metrics(config, output)
            print("PIA VERIFIED | sealed predictions and metrics; no media scan", flush=True)
            return 0
        return run(config, args.config, output, requests)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        print("PIA STOPPED | " + str(error), flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
