from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seepat.artifacts import file_sha256, stable_id
from seepat.pipeline import _load_cached_result, selected_manifest_rows


def _write_result(path: Path, status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "cache_signature": "signature",
                "video_report": {"pipeline_status": status},
                "events": [],
            }
        ),
        encoding="utf-8",
    )


def test_retry_failed_reuses_only_complete_results(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_result(result_path, "failed")

    assert _load_cached_result(result_path, "signature") is not None
    assert _load_cached_result(result_path, "signature", retry_failed=True) is None

    _write_result(result_path, "complete")
    assert _load_cached_result(result_path, "signature", retry_failed=True) is not None


def test_cached_result_requires_its_recorded_vild_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace_path.write_bytes(b"trace")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "cache_signature": "signature",
                "video_report": {
                    "pipeline_status": "complete",
                    "bilabial_event_count": 1,
                    "vild_trace_path": str(trace_path),
                    "vild_trace_sha256": file_sha256(trace_path),
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    assert _load_cached_result(
        result_path, "signature", require_vild_trace=True
    ) is not None
    trace_path.write_bytes(b"changed")
    assert _load_cached_result(result_path, "signature", require_vild_trace=True) is None


def test_selected_manifest_rows_filters_by_stable_video_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"file": "a.mp4"}, {"file": "b.mp4"}, {"file": "c.mp4"}]
    settings = SimpleNamespace(
        dataset=SimpleNamespace(
            pilot_manifest=Path("pilot.csv"),
            include_video_ids=(stable_id("c.mp4"), stable_id("a.mp4")),
        )
    )
    monkeypatch.setattr("seepat.pipeline.read_csv_rows", lambda path: rows)

    assert selected_manifest_rows(settings) == [rows[0], rows[2]]


def test_selected_manifest_rows_rejects_unknown_video_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        dataset=SimpleNamespace(
            pilot_manifest=Path("pilot.csv"),
            include_video_ids=("missing",),
        )
    )
    monkeypatch.setattr(
        "seepat.pipeline.read_csv_rows", lambda path: [{"file": "a.mp4"}]
    )

    with pytest.raises(ValueError, match="absent from the manifest: missing"):
        selected_manifest_rows(settings)
