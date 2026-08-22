from __future__ import annotations

import csv
from pathlib import Path

import pytest

from seepat.data.archive import read_manifest_video_paths


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
