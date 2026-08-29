from __future__ import annotations

import json
from pathlib import Path

from seepat.pipeline import _load_cached_result


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
