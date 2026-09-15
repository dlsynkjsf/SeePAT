"""Small operational progress snapshots; integrity checks remain in the workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from seepat.artifacts import atomic_write_json, file_sha256

ProgressCallback = Callable[[str, int, int, str], None]


def live_progress_path(report: Path) -> Path:
    return report.with_name(f"{report.stem}.progress.json")


def duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_live_progress(record: dict[str, object]) -> str:
    if record.get("status") == "unavailable":
        return str(record["message"])
    done, total = int(record.get("finished", 0)), int(record.get("total", 0))
    count = f"{done}/{total} ({done / total:.1%})" if total else "count pending"
    age = max(0, time.time() - float(record.get("updated_at", time.time())))
    lines = [
        (
            f"Workflow {record['status']} | job {record.get('job_index', 0)}/{record['jobs_total']} "
            f"| {record.get('job', '')}"
        ),
        (
            f"  {record.get('phase', 'starting')} | {count} "
            f"| phase elapsed {duration_text(record.get('elapsed_seconds'))} "
            f"| phase ETA {duration_text(record.get('eta_seconds'))}"
        ),
    ]
    if record.get("current_item"):
        lines.append(f"  Current: {record['current_item']}")
    lines.append(f"  Last update {age:.0f}s ago | source: workflow report")
    if record.get("error"):
        lines.append(f"  {record['error']}")
    return "\n".join(lines)


class WorkflowProgress:
    def __init__(
        self, config: Path, report: Path, jobs_total: int,
        model_training_config: Path | None = None,
    ):
        self.path = live_progress_path(report)
        self.record: dict[str, object] = {
            "config_sha256": file_sha256(config),
            "status": "running",
            "jobs_total": jobs_total,
            "job_index": 0,
        }
        self.phase_started = time.monotonic()
        if model_training_config is not None:
            self.record["model_training_config_sha256"] = file_sha256(model_training_config)
        self.last_publish = -float("inf")

    def start_job(self, index: int, name: str) -> None:
        self.record.update(job_index=index, job=name, phase="")
        self("starting", 0, 0, "")

    def __call__(self, phase: str, finished: int = 0, total: int = 0, item: str = "") -> None:
        now = time.monotonic()
        changed = phase != self.record.get("phase")
        if changed:
            self.phase_started = now
        elapsed = now - self.phase_started
        self.record.update(
            phase=phase,
            finished=finished,
            total=total,
            current_item=item,
            elapsed_seconds=round(elapsed, 2),
            eta_seconds=(
                round(elapsed * (total - finished) / finished, 2)
                if finished > 0 and total >= finished
                else None
            ),
        )
        if changed or now - self.last_publish >= 5 or (total > 0 and finished == total):
            self.publish()

    def publish(self) -> None:
        self.record["updated_at"] = time.time()
        atomic_write_json(self.path, self.record)
        print(format_live_progress(self.record), flush=True)
        self.last_publish = time.monotonic()

    def finish(self, error: BaseException | None = None, *, deferred: bool = False) -> None:
        self.record["status"] = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "failed"
            if error
            else "complete"
        )
        if error is not None:
            self.record["error"] = f"{type(error).__name__}: {error}"
        else:
            self.record.update(
                phase=("available work completed; decisions await a trained checkpoint"
                       if deferred else "all configured jobs completed"),
                current_item="",
                eta_seconds=0,
                finished=self.record["jobs_total"],
                total=self.record["jobs_total"],
            )
        self.publish()
