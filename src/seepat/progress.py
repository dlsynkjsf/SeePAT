from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import sleep

from seepat.artifacts import stable_id
from seepat.config import load_pipeline_settings
from seepat.pipeline import PIPELINE_VERSION, selected_manifest_rows


def summarize_progress(
    manifest_rows: list[dict[str, str]],
    cache_dir: Path,
    cache_signature: str,
) -> dict[str, object]:
    completed = 0
    failed = 0
    eligible = 0
    latest_result_at: datetime | None = None

    for row in manifest_rows:
        result_path = cache_dir / stable_id(row["file"]) / "result.json"
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if cached.get("cache_signature") != cache_signature:
            continue

        report = cached.get("video_report", {})
        if not isinstance(report, dict):
            continue
        if report.get("pipeline_status") == "complete":
            completed += 1
            if report.get("eligibility_status") == "eligible":
                eligible += 1
        else:
            failed += 1

        modified_at = datetime.fromtimestamp(result_path.stat().st_mtime, tz=UTC)
        if latest_result_at is None or modified_at > latest_result_at:
            latest_result_at = modified_at

    requested = len(manifest_rows)
    finished = completed + failed
    return {
        "videos_requested": requested,
        "videos_finished": finished,
        "videos_completed": completed,
        "videos_failed": failed,
        "videos_eligible": eligible,
        "videos_pending": requested - finished,
        "percent_finished": round(100 * finished / requested, 1) if requested else 100.0,
        "latest_result_at_utc": (
            latest_result_at.isoformat() if latest_result_at is not None else None
        ),
    }


def read_progress(config_path: Path, limit: int | None = None) -> dict[str, object]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)
    manifest_rows = selected_manifest_rows(settings, limit=limit)
    return summarize_progress(
        manifest_rows=manifest_rows,
        cache_dir=settings.preprocessing.output_dir / "cache",
        cache_signature=settings.cache_signature,
    )


def format_progress(progress: dict[str, object]) -> str:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    return (
        f"[{timestamp}] "
        f"finished={progress['videos_finished']}/{progress['videos_requested']} "
        f"({progress['percent_finished']}%) | "
        f"completed={progress['videos_completed']} | "
        f"failed={progress['videos_failed']} | "
        f"eligible={progress['videos_eligible']} | "
        f"pending={progress['videos_pending']}"
    )


def _read_run_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _completed_history_epochs(path: Path) -> int:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0
    if not isinstance(value, list):
        return 0
    epochs = [row.get("epoch", 0) for row in value if isinstance(row, dict)]
    return max((int(epoch) for epoch in epochs), default=0)


def read_workflow_progress(config_path: Path) -> dict[str, object]:
    from seepat.workflow import (
        load_workflow_settings,
        model_training_outputs_are_current,
        preprocessing_outputs_are_current,
        training_outputs_are_current,
    )

    settings = load_workflow_settings(config_path)
    preparation: list[dict[str, object]] = []
    current_stages = 0
    for job in settings.jobs:
        pipeline_settings = load_pipeline_settings(job.pipeline_config, PIPELINE_VERSION)
        preprocessing_current = preprocessing_outputs_are_current(pipeline_settings)
        manifest_current = training_outputs_are_current(
            pipeline_settings,
            job.manifest_output_dir,
        )
        current_stages += int(preprocessing_current) + int(manifest_current)
        preparation.append(
            {
                "name": job.name,
                "preprocessing": "current" if preprocessing_current else "pending_or_stale",
                "training_manifest": "current" if manifest_current else "pending_or_stale",
            }
        )

    model_training: list[dict[str, object]] = []
    for job in settings.model_training_jobs:
        current = model_training_outputs_are_current(job)
        run_record = _read_run_record(job.output_dir / "run.json")
        recorded_status = str(run_record.get("status", "pending"))
        if current:
            status = "current"
            current_stages += 1
        elif recorded_status == "running":
            status = "running"
        elif recorded_status == "failed":
            status = "failed"
        elif recorded_status == "complete":
            status = "stale"
        else:
            status = "pending"
        options = run_record.get("options", {})
        requested_epochs = (
            int(options.get("epochs", 0)) if isinstance(options, dict) else 0
        )
        completed_epochs = max(
            int(run_record.get("completed_epochs", 0)),
            _completed_history_epochs(job.output_dir / "history.json"),
        )
        model_training.append(
            {
                "name": job.name,
                "model": job.model,
                "status": status,
                "completed_epochs": completed_epochs,
                "requested_epochs": requested_epochs or int(job.options.get("epochs", 0)),
            }
        )

    total_stages = 2 * len(settings.jobs) + len(settings.model_training_jobs)
    return {
        "workflow_config": config_path.as_posix(),
        "stages_current": current_stages,
        "stages_total": total_stages,
        "all_current": current_stages == total_stages,
        "preparation": preparation,
        "model_training": model_training,
    }


def format_workflow_progress(progress: dict[str, object]) -> str:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    models = progress.get("model_training", [])
    model_text = ""
    if isinstance(models, list) and models:
        values = []
        for model in models:
            if not isinstance(model, dict):
                continue
            status = str(model.get("status", "pending"))
            epochs = ""
            requested = int(model.get("requested_epochs", 0))
            if requested:
                epochs = f" {int(model.get('completed_epochs', 0))}/{requested} epochs"
            values.append(f"{model.get('name')}={status}{epochs}")
        if values:
            model_text = " | " + ", ".join(values)
    return (
        f"[{timestamp}] workflow stages="
        f"{progress['stages_current']}/{progress['stages_total']} current"
        f"{model_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.progress")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--workflow-config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        help="Refresh until every requested video or workflow stage is current",
    )
    args = parser.parse_args()
    if args.watch_seconds is not None and args.watch_seconds <= 0:
        parser.error("--watch-seconds must be positive")

    while True:
        if args.workflow_config is not None:
            if args.limit is not None:
                parser.error("--limit can only be used with --config")
            progress = read_workflow_progress(args.workflow_config)
            finished = bool(progress["all_current"])
            formatted = format_workflow_progress(progress)
        else:
            progress = read_progress(args.config, limit=args.limit)
            finished = progress["videos_pending"] == 0
            formatted = format_progress(progress)
        if args.watch_seconds is None:
            print(json.dumps(progress, indent=2))
            return
        print(formatted, flush=True)
        if finished:
            return
        try:
            sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("Tracking stopped; the running workflow was not interrupted.")
            return


if __name__ == "__main__":
    main()
