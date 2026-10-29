from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_local_subset.py"
_SPEC = importlib.util.spec_from_file_location("prepare_local_subset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
subset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(subset)


def _row(file: str, group: str, split: str, order: int = 0) -> dict[str, str]:
    row = dict.fromkeys(subset.METADATA_COLUMNS, "")
    row.update(
        {
            "file": file,
            "original": group,
            "source_group": group,
            "split": split,
            "modify_type": "real",
            "fake_segments_json": "[]",
            "audio_fake_segments_json": "[]",
            "visual_fake_segments_json": "[]",
            "video_frames": "10",
            "audio_frames": "6400",
            "subject_id": "person1",
            "selection_order": str(order),
            "sampling_seed": "123",
        }
    )
    return row


def _write_dataset(root: Path, train: list[dict[str, str]], val: list[dict[str, str]]) -> None:
    root.mkdir()
    index = []
    for split, rows, name in (
        ("train", train, "train_subset5000.csv"),
        ("val", val, "val_subset1000.csv"),
    ):
        (root / name).write_bytes(subset._csv_bytes(list(rows[0]), rows))
        for row in rows:
            media = root / split / row["file"]
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"synthetic media placeholder")
            index.append(
                {
                    "split": split,
                    "file": row["file"],
                    "source_group": row["source_group"],
                    "category": row["modify_type"],
                    "relative_path": f"{split}/{row['file']}",
                    "binary_label": str(int(row["modify_type"] != "real")),
                }
            )
    (root / "videos.csv").write_bytes(subset._csv_bytes(subset.INDEX_COLUMNS, index))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_prefix_rule_is_narrow_and_preserves_path_case() -> None:
    assert subset.canonical_source_group("VOX_CELEB_2/voxceleb2/dev/IdCase/A.mp4") == (
        "VoxCeleb2/dev/IdCase/A.mp4"
    )
    for other in ("vox_celeb_2/id1/A.mp4", "lrs3/Case/A.mp4", "nested/vox_celeb_2/VoxCeleb2/A"):
        assert subset.canonical_source_group(other) == other
    assert subset.canonical_source_group("  vox_celeb_2\\VoxCeleb2//dev/IdCase/A.mp4  ") == (
        "VoxCeleb2/dev/IdCase/A.mp4"
    )


def test_cleaning_preserves_validation_metadata_and_is_idempotent(tmp_path: Path) -> None:
    train = [
        _row("shared-real.mp4", "VoxCeleb2/dev/shared.mp4", "train", 0),
        _row("later.mp4", "vox_celeb_2/VoxCeleb2/dev/duplicate.mp4", "train", 3),
        _row("first.mp4", "VoxCeleb2/dev/duplicate.mp4", "train", 2),
        _row("edited.mp4", "lrs3/source.mp4", "train", 4),
    ]
    train[-1].update(
        {
            "modify_type": "audio_modified",
            "fake_segments_json": "[[0.1, 0.20]]",
            "audio_fake_segments_json": "[[0.1, 0.20]]",
            "extra_column": "keep exact text",
        }
    )
    for row in train[:-1]:
        row["extra_column"] = "also preserved"
    val = [_row("val-shared.mp4", "vox_celeb_2/VoxCeleb2/dev/shared.mp4", "val")]
    root = tmp_path / "dataset"
    _write_dataset(root, train, val)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "run"
    audit = subset.prepare_local_subset(root, output)
    assert audit["output_rows"] == {"train": 2, "val": 1}
    assert audit["output_canonical_source_overlap"] == 0
    assert audit["exclusion_counts"] == {
        "duplicate_train_source_group": 1,
        "train_validation_source_overlap": 1,
    }
    assert _read(output / "train.csv") == [train[2], train[3]]
    assert _read(output / "val.csv") == [{**val[0], "source_group": "VoxCeleb2/dev/shared.mp4"}]
    excluded = _read(output / "exclusions.csv")
    assert excluded[0]["retained_file"] == "val-shared.mp4"
    assert excluded[1]["retained_file"] == "first.mp4"
    assert excluded[1]["original_source_group"].startswith("vox_celeb_2/")
    output_before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert subset.prepare_local_subset(root, output) == audit
    assert {p.name: p.read_bytes() for p in output.iterdir()} == output_before
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (
        json.loads((output / "audit.json").read_text())["media_content_decoded_or_hashed"] is False
    )


@pytest.mark.parametrize(
    "problem", ["missing", "zero", "escape", "json", "split", "header", "index"]
)
def test_invalid_dataset_is_rejected_without_outputs(tmp_path: Path, problem: str) -> None:
    train = [_row("train.mp4", "lrs3/train.mp4", "train")]
    val = [_row("val.mp4", "lrs3/val.mp4", "val")]
    root = tmp_path / "dataset"
    _write_dataset(root, train, val)
    if problem == "missing":
        (root / "train/train.mp4").unlink()
    elif problem == "zero":
        (root / "train/train.mp4").write_bytes(b"")
    elif problem == "escape":
        train[0]["file"] = "../escape.mp4"
        index = _read(root / "videos.csv")
        index[0]["file"] = "../escape.mp4"
        index[0]["relative_path"] = "train/../escape.mp4"
        (root / "videos.csv").write_bytes(subset._csv_bytes(subset.INDEX_COLUMNS, index))
    elif problem == "json":
        train[0]["fake_segments_json"] = "{}"
    elif problem == "split":
        train[0]["split"] = "val"
    elif problem == "header":
        del train[0]["fake_segments_json"]
    elif problem == "index":
        index = _read(root / "videos.csv")
        index[0]["category"] = "both_modified"
        (root / "videos.csv").write_bytes(subset._csv_bytes(subset.INDEX_COLUMNS, index))
    (root / "train_subset5000.csv").write_bytes(subset._csv_bytes(list(train[0]), train))
    with pytest.raises(ValueError):
        subset.prepare_local_subset(root, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_refuses_changed_outputs_and_dataset_destination(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root, [_row("t.mp4", "lrs3/t.mp4", "train")], [_row("v.mp4", "lrs3/v.mp4", "val")]
    )
    output = tmp_path / "run"
    subset.prepare_local_subset(root, output)
    (output / "train.csv").write_text("changed")
    with pytest.raises(FileExistsError, match="new output directory"):
        subset.prepare_local_subset(root, output)
    assert (output / "train.csv").read_text() == "changed"
    with pytest.raises(ValueError, match="outside"):
        subset.prepare_local_subset(root, root / "new-run")


@pytest.mark.parametrize(
    "path", ["../outside.mp4", "C:/outside.mp4", "/outside.mp4", "a/../../outside.mp4"]
)
def test_media_path_cannot_escape_split(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="Unsafe media path"):
        subset._media_path(tmp_path, "train", path)


def test_conflicting_source_identity_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train = [_row("t.mp4", "lrs3/t.mp4", "train")]
    train[0]["source_group"] = "lrs3/another-original.mp4"
    _write_dataset(root, train, [_row("v.mp4", "lrs3/v.mp4", "val")])
    with pytest.raises(ValueError, match="Conflicting original/source_group"):
        subset.prepare_local_subset(root, tmp_path / "run")


def test_recorded_group_accepts_existing_separator_equivalence_and_blank_original(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    train = [_row("t.mp4", "VoxCeleb2/dev/t.mp4", "train")]
    train[0]["original"] = "VoxCeleb2//dev/t.mp4"
    val = [_row("v.mp4", "lrs3/v.mp4", "val")]
    val[0]["original"] = ""
    _write_dataset(root, train, val)
    subset.prepare_local_subset(root, tmp_path / "run")
    assert _read(tmp_path / "run/train.csv")[0] == train[0]
    assert _read(tmp_path / "run/val.csv")[0] == val[0]


def test_source_identity_can_fall_back_to_original_or_file(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train = [_row("t.mp4", "lrs3/source.mp4", "train")]
    train[0]["source_group"] = ""
    val = [_row("v.mp4", "", "val")]
    _write_dataset(root, train, val)
    subset.prepare_local_subset(root, tmp_path / "run")
    assert _read(tmp_path / "run/train.csv")[0]["source_group"] == "lrs3/source.mp4"
    assert _read(tmp_path / "run/val.csv")[0]["source_group"] == "v.mp4"
