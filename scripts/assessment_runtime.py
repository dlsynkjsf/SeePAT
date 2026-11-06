"""Shared assessment paths, worker locks, sealed predictions, and metrics."""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from seepat.artifacts import atomic_write_json, file_sha256

POLICY = "author-empty-mouth-crop-unavailable-v1"


def safe_path(root, name):
    pure = PurePosixPath(name)
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not name
        or any(ord(c) < 32 or c in '<>"|?*' for c in name)
        or "\\" in name
        or ":" in name
        or pure.is_absolute()
        or name != pure.as_posix()
        or any(
            p in {".", ".."} or p.endswith((".", " ")) or p.split(".")[0].upper() in reserved
            for p in pure.parts
        )
    ):
        raise ValueError("Unsafe Windows handoff path")
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("Handoff path escapes its root")
    return path


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("Unrecognized worker PID; inspect before recovery")
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    # os.kill(pid, 0) is not a Windows liveness check. Query without signalling.
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in {87, 1168}:
            return False
        raise OSError(error, "Cannot establish worker liveness; preserve locks")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def reclaim_lock(path, *, alive=None):
    alive = alive or process_alive
    if path.exists():
        pid = int(path.read_text())
        if alive(pid):
            raise RuntimeError(f"H1 worker PID {pid} is still running; use the watcher")
        preserved = path.with_name(path.name + f".interrupted-{time.time_ns()}")
        path.rename(preserved)


@contextmanager
def archive_lock(archive):
    lock = Path(str(archive) + ".worker.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    reclaim_lock(lock)
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def assessment(labels, probabilities, cutoff):
    if len(labels) != len(probabilities) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Invalid assessment inputs/cutoff")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    unknown = unavailable = 0
    targets, scores = [], []
    for label, value in zip(labels, probabilities, strict=True):
        if label is not None and (type(label) is not int or label not in {0, 1}):
            raise ValueError("Invalid binary target")
        if value is not None and (
            type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError("Invalid probability")
        unknown += label is None
        unavailable += value is None
        if label is None or value is None:
            continue
        targets.append(label)
        scores.append(value)
        counts[("tp" if label else "fp") if value >= cutoff else ("fn" if label else "tn")] += 1
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))

    def ratio(a, b):
        return a / b if b else None

    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    auc = None
    if set(targets) == {0, 1}:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(targets, scores))
    return {
        "requested": len(labels),
        "scored_known_target": len(targets),
        "unknown_target": unknown,
        "unavailable_prediction": unavailable,
        "counts": counts,
        "accuracy": ratio(tp + tn, len(targets)),
        "precision": ratio(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "balanced_accuracy": (recall + specificity) / 2
        if None not in (recall, specificity)
        else None,
        "roc_auc": auc,
        "cutoff": cutoff,
    }


def external_target(video_label, audio_label):
    """Either-modality target: real requires two known real labels."""

    def normalize(value):
        if value is None or str(value).strip().lower() in {"", "unknown", "na", "n/a"}:
            return None
        text = str(value).strip().lower()
        if text in {"real", "authentic", "0"}:
            return 0
        if text in {"fake", "manipulated", "1"}:
            return 1
        raise ValueError("Unsupported external target value")

    labels = [normalize(video_label), normalize(audio_label)]
    return 1 if 1 in labels else 0 if labels == [0, 0] else None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_predictions(output):
    output = Path(output)
    saved = read(output / "summary.json")
    if (
        saved.get("status") != "complete"
        or file_sha256(output / "predictions.json") != saved["predictions_sha256"]
    ):
        raise ValueError("Saved PIA predictions changed or are incomplete")
    return saved


def run_cases(output, requests, contract, worker, progress=None):
    """Commit one prediction at a time; preserve unknowns, partial work and exact requests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    job = {"contract": contract, "requests": requests}
    if (output / "job.json").exists() and read(output / "job.json") != job:
        raise ValueError("PIA request/runtime binding changed; preserve this output")
    if (output / "summary.json").exists():
        verify_predictions(output)
    atomic_write_json(output / "job.json", job)
    predictions = []
    for i, row in enumerate(requests, 1):
        if shutil.disk_usage(output).free < 4 * 1024**3:
            raise OSError("Less than 4 GiB free; preserve cases and free space before resuming")
        source = Path(row["path"])
        stamp = source.stat()
        if stamp.st_size != row["bytes"]:
            raise ValueError("Selected source size changed; no replacement")
        directory = safe_path(output, "cases/" + row["video_id"])
        expected = {
            "contract": contract,
            "source": row,
            "source_stamp": [stamp.st_size, stamp.st_mtime_ns],
        }
        if directory.exists() and (
            not (directory / "job.json").exists() or read(directory / "job.json") != expected
        ):
            raise ValueError("PIA case/source binding changed; preserve partial work")
        if progress:
            progress("PIA CPU predictions", i - 1, len(requests), row["video_id"])
        if (directory / "seal.json").exists():
            if (
                file_sha256(directory / "summary.json")
                != read(directory / "seal.json")["summary_sha256"]
            ):
                raise ValueError("Completed PIA prediction changed")
            result = read(directory / "summary.json")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / "job.json", expected)
            if file_sha256(source) != row["sha256"]:
                raise ValueError("Selected source bytes changed; no replacement")
            started = time.monotonic()
            result = worker(row, directory)
            probability = result["probability"]
            if (
                result["eligible"]
                and (
                    probability is None
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                )
            ) or (not result["eligible"] and probability is not None):
                raise ValueError("Invalid PIA probability/eligibility result")
            result = {
                **result,
                "video_id": row["video_id"],
                "file": row["file"],
                "source_sha256": row["sha256"],
                "seconds": time.monotonic() - started,
                "verdict": None
                if probability is None
                else ("manipulated" if probability > 0.5 else "authentic"),
            }
            atomic_write_json(directory / "summary.json", result)
            atomic_write_json(
                directory / "seal.json", {"summary_sha256": file_sha256(directory / "summary.json")}
            )
        predictions.append(result)
        if progress:
            progress("PIA CPU predictions", i, len(requests), row["video_id"])
    atomic_write_json(output / "predictions.json", predictions)
    result = {
        "status": "complete",
        "requested": len(requests),
        "scored": sum(r["probability"] is not None for r in predictions),
        "unavailable": sum(r["probability"] is None for r in predictions),
        "predictions_sha256": file_sha256(output / "predictions.json"),
        "contract": contract,
        "labels_read": False,
        "training_performed": False,
        "paired_comparison_performed": False,
        "decision_rule": "P(fake) > 0.5",
        "scope": "PIA-only predictions; pair exact IDs with returned SeePAT scores later",
    }
    atomic_write_json(output / "summary.json", result)
    return result


def empty_crop_failure(directory):
    """Recognize only the terminal OpenCV error in the author's mouth-crop writer."""
    pointer = directory / "attempt.json"
    if not pointer.exists():
        return None
    attempt = read(pointer)
    folder = safe_path(directory, attempt["folder"])
    log = folder / "author.log"
    if not log.exists() or (folder / "author/results.json").exists():
        return None
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 8192))
        text = stream.read().decode("utf-8", errors="replace").strip()
    terminal = text.splitlines()[-1] if text else ""
    if not (
        terminal.startswith("cv2.error: OpenCV(")
        and "(-215:Assertion failed) !_img.empty() in function 'imwrite'" in terminal
        and 'File "/pia/src/utils.py"' in text
        and "in extract_mouth_states" in text
        and "cv2.imwrite(crop_path, crop_img)" in text
    ):
        return None
    return {
        "eligible": False,
        "probability": None,
        "reason": "author_empty_mouth_crop",
        "attempt": attempt,
        "failure_log": log.relative_to(directory).as_posix(),
        "failure_log_sha256": file_sha256(log),
        "failure_policy": POLICY,
    }
