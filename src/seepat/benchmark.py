from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from seepat.artifacts import atomic_write_json, read_csv_rows
from seepat.config import PipelineSettings, load_pipeline_settings
from seepat.pipeline import PIPELINE_VERSION, run_pipeline


def directory_size(path: Path, excluded: Path | None = None) -> int:
    excluded = excluded.resolve() if excluded is not None else None
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and (excluded is None or item.resolve() != excluded)
    )


def input_video_size(settings: PipelineSettings, limit: int | None = None) -> int:
    rows = read_csv_rows(settings.dataset.pilot_manifest)
    if limit is not None:
        rows = rows[:limit]
    return sum(
        path.stat().st_size
        for row in rows
        if (path := settings.dataset.extracted_root / Path(row["file"])).is_file()
    )


def build_benchmark_report(
    summary: dict[str, object],
    elapsed_seconds: float,
    input_bytes: int,
    output_bytes: int,
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, object]:
    if elapsed_seconds <= 0:
        raise ValueError("Benchmark duration must be positive")

    requested = int(summary["videos_requested"])
    completed = int(summary["videos_completed"])
    eligible = int(summary["videos_eligible"])
    return {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "videos_requested": requested,
        "videos_completed": completed,
        "videos_failed": requested - completed,
        "videos_eligible": eligible,
        "eligible_coverage": round(eligible / requested, 6) if requested else None,
        "seconds_per_completed_video": (
            round(elapsed_seconds / completed, 3) if completed else None
        ),
        "completed_videos_per_hour": (
            round(completed * 3600 / elapsed_seconds, 3) if completed else 0.0
        ),
        "input_video_bytes": input_bytes,
        "output_bytes": output_bytes,
        "output_mib_per_completed_video": (
            round(output_bytes / completed / 1024**2, 3) if completed else None
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "pipeline_summary": summary,
    }


def run_benchmark(
    config_path: Path,
    report_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> dict[str, object]:
    settings = load_pipeline_settings(config_path, PIPELINE_VERSION)
    report_path = report_path or settings.preprocessing.output_dir / "benchmark_report.json"

    started_at = datetime.now(UTC)
    started_counter = perf_counter()
    summary = run_pipeline(
        config_path,
        limit=limit,
        force=force,
        retry_failed=retry_failed,
    )
    elapsed_seconds = perf_counter() - started_counter
    completed_at = datetime.now(UTC)

    report = build_benchmark_report(
        summary=summary,
        elapsed_seconds=elapsed_seconds,
        input_bytes=input_video_size(settings, limit=limit),
        output_bytes=directory_size(settings.preprocessing.output_dir, report_path),
        started_at=started_at,
        completed_at=completed_at,
    )
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(
        config_path=args.config,
        report_path=args.report,
        limit=args.limit,
        force=args.force,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
