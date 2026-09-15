"""Restartable Validation decisions; external evaluation is deliberately locked."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from seepat.artifacts import file_sha256
from seepat.live_progress import ProgressCallback


@dataclass(frozen=True)
class DecisionJob:
    name: str
    validation_manifest: Path
    output_dir: Path
    checkpoint: Path | None = None
    source_manifest: Path | None = None
    project_root: Path = Path(".")
    device: str = "auto"
    batch_size: int = 1


def decision_stage_status(job: DecisionJob) -> dict[str, str]:
    names = ("threshold", "verdict", "explanation")
    if job.checkpoint is None:
        return dict.fromkeys(names, "awaiting_trained_checkpoint")
    from seepat.verdict import VERDICT_VERSION, _load_threshold, evaluation_inputs
    from seepat.xai import FORENSIC_TRACE_VERSION, build_evidence_bundle, render_evidence_summary

    status = dict.fromkeys(names, "pending_or_stale")
    try:
        _, provenance = evaluation_inputs(job.validation_manifest, job.checkpoint, "val")
        threshold_path = job.output_dir / "threshold.json"
        _load_threshold(None, threshold_path, provenance)
        status["threshold"] = "current"
        directory = job.output_dir / "verdict"
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        source_hash = file_sha256(job.source_manifest) if job.source_manifest else None
        if (
            evaluation.get("verdict_version") != VERDICT_VERSION
            or evaluation.get("status") != "complete" or evaluation.get("mode") != "validation"
            or evaluation.get("provenance") != provenance
            or evaluation.get("threshold", {}).get("artifact_sha256") != file_sha256(threshold_path)
            or (evaluation.get("source_manifest") or {}).get("sha256") != source_hash
        ):
            return status
        bundle = build_evidence_bundle(directory)
        status["verdict"] = "current"
        trace = json.loads((job.output_dir / "explanation.json").read_text(encoding="utf-8"))
        expected = render_evidence_summary(bundle)
        if (
            trace.get("trace_version") == FORENSIC_TRACE_VERSION
            and trace.get("mode") == "deterministic" and trace.get("sources") == bundle["sources"]
            and trace.get("trace") == expected and trace.get("evidence_summary") == expected
        ):
            status["explanation"] = "current"
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return status


def run_decision_job(job: DecisionJob, progress: ProgressCallback | None = None) -> dict[str, object]:
    if job.checkpoint is None:
        print(f"[{job.name}] deferred: select a completed experiment checkpoint before threshold fitting")
        return {"name": job.name, "action": "deferred", "stages": decision_stage_status(job)}
    from seepat.verdict import evaluation_inputs, run_verdict, select_threshold
    from seepat.xai import generate_forensic_trace

    # Fail before writing anything if a preflight, foreign split or stale calibration is supplied.
    evaluation_inputs(job.validation_manifest, job.checkpoint, "val")
    current = decision_stage_status(job)
    actions = {}
    for stage in ("threshold", "verdict", "explanation"):
        if current[stage] == "current":
            actions[stage] = "skipped"
            print(f"[{job.name}] {stage}: skipped (artifacts are current)")
            continue
        if progress:
            progress(f"{stage} Validation decisions", 0, 0, "")
        common = {
            "manifest_path": job.validation_manifest, "checkpoint_path": job.checkpoint,
            "project_root": job.project_root, "device_name": job.device,
            "batch_size": job.batch_size, "progress": progress,
        }
        if stage == "threshold":
            select_threshold(output_path=job.output_dir / "threshold.json", **common)
        elif stage == "verdict":
            run_verdict(
                output_dir=job.output_dir / "verdict", split="val",
                source_manifest=job.source_manifest,
                threshold_artifact=job.output_dir / "threshold.json", **common,
            )
        else:
            generate_forensic_trace(job.output_dir / "verdict", job.output_dir / "explanation.json")
        actions[stage] = "ran"
        print(f"[{job.name}] {stage}: ran")
    return {"name": job.name, "action": "skipped" if set(actions.values()) == {"skipped"} else "ran", "stages": actions}
