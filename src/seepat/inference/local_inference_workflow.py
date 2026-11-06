"""Local inference job/report adapter; real execution requires reviewed bindings."""

import argparse
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.evidence import FUSION_EVIDENCE_FIELDS, evidence_feature_values
from seepat.live_progress import WorkflowProgress

from .frozen_inference import (
    PreparedDataset,
    _json,
    _unchanged,
    export_result,
    load_local_model,
    read_result,
    score_prepared,
    validate_inputs,
)
from .local_raw_inputs import _destination, export_prepared, validate_raw_request
from .xai_paragraphs import human_summary

VERSION = "local-inference-workflow-v1"


@dataclass(frozen=True)
class LocalInferenceJob:
    name: str
    freeze: Path
    raw_inputs: Path
    output_dir: Path
    device: str = "cpu"
    batch_size: int = 1

    def validate(self):
        if (not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.name)
                or self.device not in {"cpu", "cuda"} or type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("Invalid local inference job name/device/batch size")


def load_local_jobs(config):
    """Parse local-inference jobs without opening models or media."""
    raw = config if isinstance(config, dict) else yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("local_inference", []), list):
        raise TypeError("local_inference must be a workflow list")
    jobs = []
    allowed = {"name", "freeze", "raw_inputs", "output_dir", "device", "batch_size"}
    for item in raw.get("local_inference", []):
        if not isinstance(item, dict) or set(item) - allowed or not {"name", "freeze", "raw_inputs", "output_dir"} <= set(item):
            raise ValueError("Unsupported or missing local workflow fields")
        job = LocalInferenceJob(item["name"], Path(item["freeze"]), Path(item["raw_inputs"]), Path(item["output_dir"]),
                                item.get("device", "cpu"), item.get("batch_size", 1))
        job.validate()
        output = job.output_dir.resolve()
        if any(old.name == job.name or output.is_relative_to(old.output_dir.resolve())
               or old.output_dir.resolve().is_relative_to(output) for old in jobs):
            raise ValueError("Local workflow names/output trees must be distinct")
        jobs.append(job)
    return tuple(jobs)


def workflow_runtime_hashes():
    here = Path(__file__).resolve().parent
    return {name: file_sha256(here / name) for name in
            ("local_inference_workflow.py", "frozen_inference.py", "local_raw_inputs.py", "local_matched_inputs.py",
             "readable_xai.py", "xai_paragraphs.py")}


def _binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _clock(seconds):
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _timestamp(row, validated):
    if row.get("kind") == "context":
        spec = _json(Path(validated["prepared_spec"]))
        if not spec.get("raw_provenance"):
            return {"status": "unavailable", "reason": "no_bound_source_trace_clock"}
        packet = _json(Path(spec["matched_inputs"]["path"]))
        matches = [window for video in packet["records"] for window in video["windows"]
                   if window["context_id"] == row["event_id"] and window["video_id"] == row["video_id"]]
        if len(matches) != 1:
            raise ValueError("Context clock identity mismatch")
        active = [slot for slot in matches[0]["slots"] if slot["planned_valid"]]
        start = active[0]["decoder_pts_s"]
        return {"status": "verified", "kind": "sampled_context_source_frame", "start_s": start,
                "last_sample_s": active[-1]["decoder_pts_s"], "display": _clock(start),
                "source_frame": active[0]["source_frame"],
                "meaning": "First planned source frame's saved decoder PTS, joined to the original trace; not an edit location"}
    start, end = row.get("video_phone_start_s"), row.get("video_phone_end_s")
    if start in (None, "") or end in (None, ""):
        return {"status": "unavailable", "reason": "no_bound_video_phoneme_clock"}
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Invalid bound video phoneme interval")
    return {"status": "verified", "kind": "video_phoneme_interval", "start_s": start, "end_s": end,
            "display": _clock(start), "meaning": "Aligned phoneme onset on the original video clock; not padded clip start"}


def build_reports(validated, result_dir: Path):
    """Compact offline presentation from verified label-free numerical outputs."""
    scope = _json(result_dir / "summary.json")["scope"]
    saved = read_result(validated, result_dir, engineering=scope == "synthetic_engineering")
    by_event = {row["event_id"]: row for row in validated["rows"]}
    reports = []
    for video in saved["scored"]["videos"]:
        clips = sorted((row for row in saved["scored"]["events"] if row["video_id"] == video["video_id"]),
                       key=lambda row: (-row["manipulated_probability"], row["event_id"]))
        highlight = None
        if clips:
            best = clips[0]
            original = by_event[best["event_id"]]
            timestamp = _timestamp(original, validated)
            if original.get("kind") == "context":
                values, mask = [0.0] * len(FUSION_EVIDENCE_FIELDS), [False] * len(FUSION_EVIDENCE_FIELDS)
            else:
                values, mask = evidence_feature_values(original)
            highlight = {"event_id": best["event_id"], "kind": original.get("kind", "bilabial"),
                         "probability": best["manipulated_probability"], "timestamp": timestamp,
                         "evidence_available": sum(mask),
                         "evidence": {field: value if available else None for field, value, available in
                                      zip(FUSION_EVIDENCE_FIELDS, values, mask, strict=True)},
                         "ranking": "highest classifier probability, not Isolation Forest anomaly ranking"}
        video_report = {"video_id": video["video_id"], "verdict": video["verdict"], "threshold": video["threshold"],
                        "video_score": video["maximum_probability"], "clips_scored": len(clips),
                        "highlight": highlight, "reason": video["reason"]}
        video_report["summary_text"] = human_summary(video_report)
        reports.append(video_report)
    _unchanged(saved["bindings"])
    return {"version": VERSION, "scope": scope, "network_requests": 0, "rendering": "local_deterministic",
            "videos": reports, "input_bindings": saved["bindings"],
            "limitations": ["Unknown ground truth; no classification metrics",
                            "Scores are not calibrated confidence; descriptions are not causal feature attribution",
                            "Classifier ranking is not Isolation Forest anomaly ranking or manipulation localization"],
            "caution": "Review highlight, not a confirmed altered segment. The verdict is a model result, not proof."}


def _explanation_content(report, result_dir):
    lines = ["# SeePAT result", "", "ENGINEERING CHECK: synthetic inputs, not thesis performance."
             if report["scope"] == "synthetic_engineering" else "LOCAL UNLABELLED INFERENCE: not an external-test evaluation.", ""]
    for video in report["videos"]:
        lines.extend((f"## {video['video_id']}", "", video["summary_text"], ""))
    lines.extend((report["caution"], "", "[Full numerical details](technical_report.md) | [Structured evidence](report.json)", ""))
    return {"report.md": "\n".join(lines).encode("utf-8"),
            "report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            "technical_report.md": (result_dir / "report.md").read_bytes()}


def read_explanations(validated, result_dir, output_dir):
    """Rebuild deterministic presentation and verify its seal without rewriting."""
    report = build_reports(validated, result_dir)
    saved = _json(output_dir / "summary.json")
    content = _explanation_content(report, result_dir)
    if (saved.get("version") != "local-inference-explanation-v1" or saved.get("status") != "complete"
            or saved.get("scope") != report["scope"] or saved.get("network_requests") != 0
            or saved.get("input_bindings") != report["input_bindings"]
            or saved.get("implementation_sha256") != workflow_runtime_hashes()
            or set(saved.get("output_sha256", {})) != set(content)
            or any(file_sha256(output_dir / name) != saved["output_sha256"][name]
                   or (output_dir / name).read_bytes() != data for name, data in content.items())):
        raise ValueError("Explanation is partial, changed or unsupported by frozen evidence")
    _unchanged(report["input_bindings"])
    return saved


def export_explanations(validated, result_dir: Path, output_dir: Path):
    """Fresh local compact view; numerical outputs and earlier XAI stay untouched."""
    output = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Preserve the existing explanation")
    protected = [result_dir.resolve(), Path(validated["checkpoint"]).parent, Path(validated["manifest"]).parent]
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("Explanation must remain outside numerical/model/prepared directories")
    report = build_reports(validated, result_dir)
    if any(Path(item["path"]).is_relative_to(output) for item in report["input_bindings"]):
        raise ValueError("Explanation cannot contain protected inputs")
    content = _explanation_content(report, result_dir)
    _unchanged(report["input_bindings"])
    output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    _unchanged(report["input_bindings"])
    summary = {"version": "local-inference-explanation-v1", "status": "complete", "scope": report["scope"],
               "network_requests": 0, "input_bindings": report["input_bindings"],
               "implementation_sha256": workflow_runtime_hashes(),
               "output_sha256": {name: file_sha256(output / name) for name in content}}
    atomic_write_json(output / "summary.json", summary)
    return summary


def _job_contract(job, engineering):
    return {"version": VERSION, "name": job.name, "freeze": _binding(job.freeze), "raw_inputs": _binding(job.raw_inputs),
            "device": job.device, "batch_size": job.batch_size,
            "scope": "synthetic_engineering" if engineering else "local_unlabelled_inference",
            "implementation_sha256": workflow_runtime_hashes()}


def _read_complete(output, contract):
    if not (output / "summary.json").exists():
        return None
    saved = _json(output / "summary.json")
    names = {"preparation", "inference", "explanation"}
    if (saved.get("version") != VERSION or saved.get("status") != "complete"
            or saved.get("contract") != contract or saved.get("network_requests") != 0
            or saved.get("scope") != contract["scope"] or set(saved.get("stage_summaries", {})) != names
            or set(saved.get("stages", {})) != names or any(value not in {"ran", "skipped"} for value in saved["stages"].values())):
        raise ValueError("Completed workflow metadata differs")
    for name, directory in (("preparation", "prepared"), ("inference", "result"), ("explanation", "explanation")):
        if saved["stage_summaries"][name] != _binding(output / directory / "summary.json"):
            raise ValueError("Completed stage binding differs")
    return saved


def local_job_status(job, *, engineering=False):
    """Full read-only stage verification; never create a missing stage or model."""
    job.validate()
    validated = validate_raw_request(job.freeze, job.raw_inputs)
    output = _destination(validated, job.output_dir)
    stages = {name: "pending_or_stale" for name in ("preparation", "inference", "explanation")}
    contract = _job_contract(job, engineering)
    completed = None
    if output.exists():
        if job.output_dir.is_symlink() or _json(output / "job.json") != contract:
            raise ValueError("Foreign or changed workflow contract")
        completed = _read_complete(output, contract)
        if (output / "prepared/summary.json").is_file():
            # Existing directory + completion file guarantee this call only verifies/skips.
            export_prepared(validated, output / "prepared", engineering=engineering)
            prepared = validate_inputs(job.freeze, output / "prepared/inputs.json")
            stages["preparation"] = "current"
            if (output / "result/summary.json").is_file():
                read_result(prepared, output / "result", engineering=engineering)
                stages["inference"] = "current"
                if (output / "explanation/summary.json").is_file():
                    read_explanations(prepared, output / "result", output / "explanation")
                    stages["explanation"] = "current"
            if completed and completed.get("videos_requested") != prepared["requested"]:
                raise ValueError("Completed workflow coverage differs")
    _unchanged(validated["bindings"])
    if completed is None and all(value == "current" for value in stages.values()):
        stages["explanation"] = "pending_or_stale"  # Final workflow seal is part of the last gate.
    return {"name": job.name, "stages": stages, "completion_current": completed is not None,
            "execution_ready": validated["execution_ready"]
            and validated["freeze"].get("local_workflow_sha256") == workflow_runtime_hashes()}


@contextmanager
def _tracking(job, completed, progress):
    if completed or progress is not None:
        yield progress
        return
    tracker = WorkflowProgress(job.raw_inputs, job.output_dir / "summary.json", jobs_total=1)
    tracker.start_job(1, job.name)
    try:
        yield tracker
    except BaseException as error:
        tracker.finish(KeyboardInterrupt() if isinstance(error, KeyboardInterrupt) else
                       RuntimeError(f"{type(error).__name__}: local job aborted; preserve artifacts"))
        raise
    tracker.finish()


def run_local_job(job, *, progress=None, engineering=False, processor_factory=None, model=None):
    """Workflow-callable chain; custom media/model adapters are synthetic-only."""
    job.validate()
    validated = validate_raw_request(job.freeze, job.raw_inputs)
    if not engineering and (not validated["execution_ready"]
                            or validated["freeze"].get("local_workflow_sha256") != workflow_runtime_hashes()
                            or processor_factory is not None or model is not None):
        raise ValueError("Local workflow needs its reviewed final freeze; synthetic adapters cannot run as real")
    output = _destination(validated, job.output_dir)
    contract = _job_contract(job, engineering)
    if job.output_dir.is_symlink():
        raise FileExistsError("Workflow output cannot be a symlink")
    completed = None
    if output.exists():
        if _json(output / "job.json") != contract:
            raise FileExistsError("Workflow output belongs to another contract; preserve it")
        completed = _read_complete(output, contract)
    else:
        if engineering and processor_factory is None:
            raise ValueError("Synthetic workflows require a recorded media adapter")
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "job.json", contract)
    with _tracking(job, completed, progress) as callback:
        return _run_stages(job, validated, output, contract, completed, callback, engineering, processor_factory, model)


def _run_stages(job, validated, output, contract, completed, progress, engineering, processor_factory, model):
    stages = {}
    if progress and not completed:
        progress("local inputs", 0, 3, job.name)
    if engineering and not (output / "prepared").exists() and processor_factory is None:
        raise ValueError("Synthetic workflows require a recorded media adapter")
    preparation = export_prepared(validated, output / "prepared", engineering=engineering,
                                  processor_factory=processor_factory, progress=progress if not completed else None)
    stages["preparation"] = "skipped" if preparation.get("action") == "skipped" else "ran"
    prepared = validate_inputs(job.freeze, output / "prepared/inputs.json")
    if progress and not completed:
        progress("frozen prediction", 1, 3, job.name)
    if (output / "result").exists():
        read_result(prepared, output / "result", engineering=engineering)
        stages["inference"] = "skipped"
    else:
        dataset = PreparedDataset(prepared)
        if len(dataset) and model is None:
            if engineering:
                raise ValueError("Supply a concrete tiny synthetic model for engineering scoring")
            model, loaded_contract = load_local_model(Path(prepared["checkpoint"]), torch.device(job.device))
            if loaded_contract != prepared["contract"]:
                raise ValueError("Loaded model contract differs")
        if model is None:
            model = torch.nn.Identity()
        scored = score_prepared(model, dataset, prepared["requested_videos"], prepared["threshold"]["value"],
                                device=job.device, batch_size=job.batch_size, progress=progress)
        export_result(prepared, scored, output / "result", engineering=engineering)
        stages["inference"] = "ran"
    if progress and not completed:
        progress("compact explanation", 2, 3, job.name)
    if (output / "explanation").exists():
        read_explanations(prepared, output / "result", output / "explanation")
        stages["explanation"] = "skipped"
    else:
        export_explanations(prepared, output / "result", output / "explanation")
        stages["explanation"] = "ran"
    if completed:
        if completed.get("videos_requested") != prepared["requested"]:
            raise ValueError("Completed workflow coverage differs")
        return {"name": job.name, "action": "skipped", "stages": stages, "summary": completed}
    summary = {"version": VERSION, "status": "complete", "scope": "synthetic_engineering" if engineering else "local_unlabelled_inference",
               "network_requests": 0, "contract": contract, "stages": stages, "videos_requested": prepared["requested"],
               "stage_summaries": {name: _binding(output / directory / "summary.json") for name, directory in
                                   (("preparation", "prepared"), ("inference", "result"), ("explanation", "explanation"))}}
    _unchanged(validated["bindings"])
    atomic_write_json(output / "summary.json", summary)
    if progress:
        progress("local inference complete", 3, 3, job.name)
    return {"name": job.name, "action": "ran", "stages": stages, "summary": summary}


def main(argv=None, *, engineering=False):
    """Preparation preview/read-only tracker, deliberately not a second worker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--watch-output", type=Path)
    parser.add_argument("--watch-seconds", type=float, default=5)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.watch_output is not None:
        if args.verify:
            parser.error("--verify needs --config, not the lightweight watcher")
        from .local_raw_inputs import main as watch

        return watch(["--watch-output", str(args.watch_output), "--watch-seconds", str(args.watch_seconds)])
    try:
        jobs = load_local_jobs(args.config)
        if not jobs:
            raise ValueError("No local inference jobs in this preparation profile")
        if args.verify:
            statuses = [local_job_status(job, engineering=engineering) for job in jobs]
            current = sum(value == "current" for item in statuses for value in item["stages"].values())
            result = {"local_inference": statuses, "stages_current": current, "stages_total": 3 * len(jobs),
                      "all_current": current == 3 * len(jobs) and all(item["completion_current"] for item in statuses),
                      "entry_point": "preparation_companion", "execution": "not_started"}
            print(json.dumps(result, indent=2), flush=True)
            if not result["all_current"]:
                raise SystemExit(1)
            return result
    except (OSError, ValueError, KeyError, TypeError):
        print("LOCAL INFERENCE CHECK FAILED | config_gate; preserve inputs")
        raise SystemExit(1) from None
    print(f"PREPARATION ONLY | {len(jobs)} local inference job(s); schema preview; nothing started", flush=True)
    return {"execution": "not_started", "jobs": [job.name for job in jobs], "entry_point": "preparation_companion"}


if __name__ == "__main__":
    main()
