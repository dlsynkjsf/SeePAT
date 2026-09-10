from __future__ import annotations

import json
from pathlib import Path

import pytest

from seepat.artifacts import read_csv_rows
from seepat.data.deepfake_eval import (
    DEEPFAKE_EVAL_SOURCE,
    MANIFEST_FIELDS,
    build_deepfake_eval_manifest,
)


def _write_index(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_build_manifest_maps_labels_and_modalities(tmp_path: Path) -> None:
    index_path = _write_index(
        tmp_path / "index.json",
        [
            {"file": "videos/fake-visual.mp4", "label": "fake", "modality": "visual", "duration_s": 12.5},
            {"file": "videos/fake-unknown.mp4", "label": "fake"},
            {"file": "videos/real.mp4", "label": "REAL"},
            {"file": "videos/fake-audio.mp4", "label": 1, "modality": "audio"},
        ],
    )
    output_path = tmp_path / "manifest.csv"
    summary_path = tmp_path / "summary.json"

    summary = build_deepfake_eval_manifest(
        index_path=index_path,
        output_manifest=output_path,
        summary_path=summary_path,
    )

    rows = read_csv_rows(output_path)
    assert len(rows) == 4
    assert list(rows[0]) == list(MANIFEST_FIELDS)
    by_file = {row["file"]: row for row in rows}
    assert by_file["videos/fake-visual.mp4"]["modify_type"] == "visual_modified"
    assert json.loads(by_file["videos/fake-visual.mp4"]["fake_segments_json"]) == [
        [0.0, 12.5]
    ]
    assert by_file["videos/fake-unknown.mp4"]["modify_type"] == "both_modified"
    assert by_file["videos/real.mp4"]["modify_type"] == "real"
    assert by_file["videos/real.mp4"]["fake_segments_json"] == "[]"
    assert by_file["videos/fake-audio.mp4"]["modify_type"] == "audio_modified"
    assert all(row["split"] == "test" for row in rows)

    assert summary["source"] == DEEPFAKE_EVAL_SOURCE
    assert summary["records"] == 4
    assert summary["label_counts"] == {"fake": 3, "real": 1}
    assert summary["modify_type_counts"]["both_modified"] == 1
    assert summary["index"]["sha256"]
    assert summary["manifest"]["sha256"]
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary == summary


def test_build_manifest_accepts_csv_index_and_custom_split(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "file,label,modality\nvideos/a.mp4,fake,visual\nvideos/b.mp4,real,\n",
        encoding="utf-8",
    )

    build_deepfake_eval_manifest(
        index_path=index_path,
        output_manifest=tmp_path / "manifest.csv",
        split="external-test",
    )

    rows = read_csv_rows(tmp_path / "manifest.csv")
    assert [row["split"] for row in rows] == ["external-test", "external-test"]
    assert rows[0]["modify_type"] == "visual_modified"
    assert rows[1]["modify_type"] == "real"


def test_build_manifest_rejects_invalid_records(tmp_path: Path) -> None:
    duplicate = _write_index(
        tmp_path / "duplicate.json",
        [
            {"file": "videos/a.mp4", "label": "real"},
            {"file": "videos/a.mp4", "label": "fake"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build_deepfake_eval_manifest(duplicate, tmp_path / "out.csv")

    bad_label = _write_index(
        tmp_path / "bad-label.json",
        [{"file": "videos/a.mp4", "label": "maybe"}],
    )
    with pytest.raises(ValueError, match="unsupported label"):
        build_deepfake_eval_manifest(bad_label, tmp_path / "out.csv")

    bad_modality = _write_index(
        tmp_path / "bad-modality.json",
        [{"file": "videos/a.mp4", "label": "fake", "modality": "subtitles"}],
    )
    with pytest.raises(ValueError, match="unsupported modality"):
        build_deepfake_eval_manifest(bad_modality, tmp_path / "out.csv")

    with pytest.raises(ValueError, match="non-empty list"):
        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        build_deepfake_eval_manifest(empty, tmp_path / "out.csv")
