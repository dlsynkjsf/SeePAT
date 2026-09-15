from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from seepat.artifacts import (
    atomic_write_csv,
    atomic_write_gzip_json,
    atomic_write_json,
    read_csv_rows,
    read_gzip_json,
    stable_id,
)


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


def test_atomic_gzip_json_is_deterministic_and_readable(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    value = {"frames": [{"timestamp_s": 0.0, "normalized_vild": 0.1}]}

    atomic_write_gzip_json(first, value)
    atomic_write_gzip_json(second, value)

    assert first.read_bytes() == second.read_bytes()
    assert read_gzip_json(first) == value


@pytest.mark.parametrize("writer,reader", [
    (atomic_write_json, lambda path: json.loads(path.read_text(encoding="utf-8"))),
    (atomic_write_gzip_json, read_gzip_json),
    (atomic_write_csv, read_csv_rows),
])
def test_atomic_writers_retry_transient_windows_replace_locks(tmp_path, monkeypatch, writer, reader):
    path = tmp_path / "artifact"
    old, new = [{"state": "old"}], [{"state": "new"}]
    writer(path, old)
    replace = Path.replace
    attempts, delays = [], []

    def locked_replace(temporary, target):
        attempts.append(target)
        assert reader(path) == old  # Never delete/truncate the previous artifact.
        if len(attempts) < 3:
            error = PermissionError("reader has the destination open")
            error.winerror = 5
            raise error
        return replace(temporary, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("seepat.artifacts.sleep", delays.append)
    writer(path, new)
    assert reader(path) == new
    assert len(attempts) == 3
    assert delays == [0.05, 0.1]
    assert not path.with_name(".artifact.tmp").exists()


@pytest.mark.parametrize("winerror,expected_attempts", [(5, 6), (32, 6), (33, 6), (None, 1)])
def test_atomic_write_preserves_old_file_and_raises_when_retry_cannot_help(
    tmp_path, monkeypatch, winerror, expected_attempts,
):
    path = tmp_path / "progress.json"
    atomic_write_json(path, {"finished": 1})
    error = PermissionError("persistent denial")
    if winerror is not None:
        error.winerror = winerror
    attempts, delays = [], []

    def denied(temporary, target):
        attempts.append(target)
        raise error

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr("seepat.artifacts.sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        atomic_write_json(path, {"finished": 2})
    assert raised.value is error
    assert len(attempts) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert sum(delays) <= 1.55
    assert json.loads(path.read_text()) == {"finished": 1}
    assert json.loads(path.with_name(".progress.json.tmp").read_text()) == {"finished": 2}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows denies rename over an open reader")
def test_atomic_write_recovers_from_real_windows_reader_lock(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    atomic_write_json(path, {"finished": 1})
    delays = []
    with path.open(encoding="utf-8") as reader:
        def release_reader(delay):
            delays.append(delay)
            assert json.loads(reader.read()) == {"finished": 1}
            reader.close()

        monkeypatch.setattr("seepat.artifacts.sleep", release_reader)
        atomic_write_json(path, {"finished": 2})
    assert delays == [0.05]  # A real replacement failed once before the reader closed.
    assert json.loads(path.read_text()) == {"finished": 2}
