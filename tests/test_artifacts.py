from __future__ import annotations

import json
from pathlib import Path

from seepat.artifacts import atomic_write_csv, atomic_write_json, read_csv_rows, stable_id


def test_atomic_artifact_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "summary.json"
    csv_path = tmp_path / "nested" / "events.csv"

    atomic_write_json(json_path, {"completed": 2})
    atomic_write_csv(csv_path, [{"video_id": "a", "status": "eligible"}])

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"completed": 2}
    assert read_csv_rows(csv_path) == [{"video_id": "a", "status": "eligible"}]
    assert not (json_path.parent / ".summary.json.tmp").exists()
    assert not (csv_path.parent / ".events.csv.tmp").exists()


def test_stable_id_is_deterministic_and_length_limited() -> None:
    assert stable_id("same video") == stable_id("same video")
    assert len(stable_id("same video", length=10)) == 10
    assert stable_id("same video") != stable_id("different video")
