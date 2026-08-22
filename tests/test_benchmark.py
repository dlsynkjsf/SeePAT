from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from seepat.benchmark import build_benchmark_report, directory_size


def test_directory_size_can_exclude_report(tmp_path: Path) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"12345")
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_bytes(b"not counted")

    assert directory_size(tmp_path, excluded=report_path) == 5


def test_build_benchmark_report_calculates_rates() -> None:
    timestamp = datetime(2026, 8, 22, tzinfo=UTC)
    summary: dict[str, object] = {
        "videos_requested": 100,
        "videos_completed": 96,
        "videos_eligible": 90,
    }

    report = build_benchmark_report(
        summary=summary,
        elapsed_seconds=480.0,
        input_bytes=1_000,
        output_bytes=96 * 1024**2,
        started_at=timestamp,
        completed_at=timestamp,
    )

    assert report["videos_failed"] == 4
    assert report["eligible_coverage"] == 0.9
    assert report["seconds_per_completed_video"] == 5.0
    assert report["completed_videos_per_hour"] == 720.0
    assert report["output_mib_per_completed_video"] == 1.0


def test_build_benchmark_report_rejects_zero_duration() -> None:
    timestamp = datetime(2026, 8, 22, tzinfo=UTC)

    with pytest.raises(ValueError, match="duration must be positive"):
        build_benchmark_report({}, 0.0, 0, 0, timestamp, timestamp)
