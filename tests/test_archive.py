from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from seepat.data.archive import read_manifest_video_paths, run_visible_command


def _write_manifest(path: Path, rows: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file",))
        writer.writeheader()
        writer.writerows({"file": row} for row in rows)


def test_manifest_paths_are_normalized_and_deduplicated(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.csv"
    _write_manifest(manifest, ["a\\b.mp4", "a/b.mp4", "c/d.mp4"])

    assert read_manifest_video_paths(manifest) == ["a/b.mp4", "c/d.mp4"]


def test_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot.csv"
    _write_manifest(manifest, ["../outside.mp4"])

    with pytest.raises(ValueError, match="Unsafe archive path"):
        read_manifest_video_paths(manifest)


def test_long_command_output_is_visible_and_retained(monkeypatch, capsys) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = StringIO("starting\n50%\ncomplete\n")

        def wait(self, timeout=None) -> int:
            return 0

    def fake_popen(*args, **kwargs) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr("seepat.data.archive.subprocess.Popen", fake_popen)

    return_code, output_tail = run_visible_command(["fake-command"])

    assert return_code == 0
    assert output_tail == "starting\n50%\ncomplete\n"
    assert capsys.readouterr().out == output_tail
