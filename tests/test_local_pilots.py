from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts import prepare_local_pilots as pilots
from scripts import prepare_local_subset as source


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_clean(cleaned: Path, rows: dict[str, list[dict[str, str]]]) -> None:
    hashes = {}
    for split, values in rows.items():
        data = source._csv_bytes(list(values[0]), values)
        (cleaned / f"{split}.csv").write_bytes(data)
        hashes[f"{split}.csv"] = hashlib.sha256(data).hexdigest()
    (cleaned / "audit.json").write_text(
        json.dumps({"version": "local-subset-v1", "output_sha256": hashes})
    )


@pytest.fixture
def inputs(tmp_path: Path):
    cleaned, media = tmp_path / "cleaned", tmp_path / "media"
    cleaned.mkdir()
    rows = {}
    for split, count in (("train", 20), ("val", 6)):
        rows[split] = []
        for category in pilots.CATEGORY_ORDER:
            for index in range(count):
                file = f"{category}/{index}.mp4"
                group = f"lrs3/{split}/{category}/{index}.mp4"
                row = dict.fromkeys(source.METADATA_COLUMNS, "")
                row.update(
                    file=file,
                    original=group,
                    source_group=group,
                    split=split,
                    modify_type=category,
                    subject_id="shared-subject-is-allowed",
                    selection_order=str(len(rows[split])),
                    sampling_seed="original-seed",
                    video_frames="25",
                    audio_frames="16000",
                    fake_segments_json="[]" if category == "real" else "[[0.10, 0.20]]",
                    audio_fake_segments_json="[]",
                    visual_fake_segments_json="[]",
                    extra_column=f"preserve {category} {index}",
                )
                rows[split].append(row)
                path = media / split / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"nonempty presence-check fixture; not a decoded video")
    _save_clean(cleaned, rows)
    return cleaned, media, rows


def _prepare(inputs, destination: Path, **kwargs):
    cleaned, media, _ = inputs
    return pilots.prepare_local_pilots(
        cleaned, media, destination, train_sizes=(8, 16), validation_size=8, **kwargs
    )


def test_stratified_nested_selection_preserves_metadata_and_originals(inputs, tmp_path):
    cleaned, media, original = inputs
    before = {
        p: p.read_bytes() for root in (cleaned, media) for p in root.rglob("*") if p.is_file()
    }
    output = tmp_path / "pilots"
    audit = _prepare(inputs, output)
    small, large, val = (_read(output / name) for name in ("train8.csv", "train16.csv", "val8.csv"))
    assert small == large[:8]
    for rows, expected in ((small, 2), (large, 4), (val, 2)):
        assert Counter(row["modify_type"] for row in rows) == {
            category: expected for category in pilots.CATEGORY_ORDER
        }
        assert [row["modify_type"] for row in rows[:4]] == list(pilots.CATEGORY_ORDER)
    lookup = {(r["split"], r["file"]): r for rows in original.values() for r in rows}
    assert all(row == lookup[(row["split"], row["file"])] for row in large + val)
    assert not {r["source_group"] for r in large} & {r["source_group"] for r in original["val"]}
    assert audit["subject_identity_separation_asserted"] is False
    assert audit["full_validation_source_overlap"] == 0
    assert all(p.read_bytes() == value for p, value in before.items())
    saved = {p.name: p.read_bytes() for p in output.iterdir()}
    assert _prepare(inputs, output) == audit
    assert {p.name: p.read_bytes() for p in output.iterdir()} == saved


def test_selection_is_independent_of_input_row_order(inputs, tmp_path):
    cleaned, _, rows = inputs
    _prepare(inputs, tmp_path / "first")
    _save_clean(cleaned, {split: list(reversed(values)) for split, values in rows.items()})
    _prepare(inputs, tmp_path / "reordered")
    for name in ("train8.csv", "train16.csv", "val8.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "reordered" / name
        ).read_bytes()


def test_rejects_changed_inputs_before_selection(inputs, tmp_path):
    cleaned, _, _ = inputs
    with (cleaned / "train.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed since its audit"):
        _prepare(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("problem", ["duplicate", "alias", "overlap", "shortage", "interval"])
def test_rejects_invalid_source_populations(inputs, tmp_path, problem):
    cleaned, _, rows = inputs
    first, second = rows["train"][:2]
    if problem == "duplicate":
        second["original"] = second["source_group"] = first["source_group"]
    elif problem == "alias":
        first["source_group"] = first["original"] = "vox_celeb_2/VoxCeleb2/dev/source.mp4"
    elif problem == "overlap":
        first["source_group"] = first["original"] = rows["val"][-1]["source_group"]
    elif problem == "shortage":
        rows["train"] = [r for r in rows["train"] if r["modify_type"] != "both_modified"]
    else:
        first["fake_segments_json"] = "[[0.5, 0.1]]"
    _save_clean(cleaned, rows)
    with pytest.raises(ValueError):
        _prepare(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_selected_media_is_not_replaced(inputs, tmp_path):
    _, media, _ = inputs
    _prepare(inputs, tmp_path / "first")
    selected = _read(tmp_path / "first/train8.csv")[0]
    (media / "train" / selected["file"]).unlink()
    with pytest.raises(ValueError, match="Missing media"):
        _prepare(inputs, tmp_path / "second")
    assert not (tmp_path / "second").exists()


def test_refuses_modified_outputs_and_new_seed_without_overwriting(inputs, tmp_path):
    output = tmp_path / "pilots"
    _prepare(inputs, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="new output directory"):
        _prepare(inputs, output, seed=42)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    (output / "val8.csv").write_text("preserve modified output")
    (output / "audit.json").unlink()
    with pytest.raises(FileExistsError, match="new output directory"):
        _prepare(inputs, output)
    assert not (output / "audit.json").exists()


@pytest.mark.parametrize("destination", ["cleaned", "media"])
def test_output_must_not_be_inside_inputs(inputs, destination):
    cleaned, media, _ = inputs
    with pytest.raises(ValueError, match="outside"):
        _prepare(inputs, {"cleaned": cleaned, "media": media}[destination] / "pilots")


@pytest.mark.parametrize("size", [0, 3, True])
def test_invalid_pilot_sizes_are_rejected(inputs, tmp_path, size):
    cleaned, media, _ = inputs
    with pytest.raises(ValueError, match="divisible by four"):
        pilots.prepare_local_pilots(cleaned, media, tmp_path / "out", train_sizes=(size,))


def test_pilot_profiles_have_matching_readiness_and_isolated_calibration(monkeypatch):
    from seepat.config import load_pipeline_settings
    from seepat.pipeline import PIPELINE_VERSION
    from seepat.training.runtime import load_runtime_options
    from seepat.workflow import _model_training_options, load_workflow_settings

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    val = load_pipeline_settings(Path("configs/local_pilot_val200.yaml"), PIPELINE_VERSION)
    assert val.dataset.pilot_manifest.name == "val200.csv"
    calibration_dirs = set()
    for size, epochs in ((500, 3), (1000, 5)):
        stem = f"local_pilot{size}"
        train = load_pipeline_settings(Path(f"configs/{stem}_train.yaml"), PIPELINE_VERSION)
        assert train.dataset.pilot_manifest.name == f"train{size}.csv"
        assert train.preprocessing.output_dir != val.preprocessing.output_dir
        full = load_workflow_settings(Path(f"configs/{stem}_training.yaml")).model_training_jobs[0]
        options = asdict(_model_training_options(full))
        assert options["epochs"] == epochs and options["max_train_batches"] is None
        assert options["max_validation_batches"] is None
        assert options["supervision"] == "video_max"
        for suffix, ready_epochs in (("readiness", 1), ("readiness_resume", 2)):
            ready = load_workflow_settings(Path(f"configs/{stem}_{suffix}.yaml")).model_training_jobs[0]
            actual = asdict(_model_training_options(ready))
            assert actual["epochs"] == ready_epochs
            for key in ("epochs", "max_train_batches", "max_validation_batches"):
                actual.pop(key)
            assert actual == {k: v for k, v in options.items() if k in actual}
            assert ready.output_dir == full.readiness_dir
            assert ready.train_manifest == full.train_manifest
            assert ready.validation_manifest == full.validation_manifest
            assert load_runtime_options(ready.runtime, ready.project_root).record() == (
                load_runtime_options(full.runtime, full.project_root).record()
            )
        preparation = load_workflow_settings(Path(f"configs/{stem}_prepare.yaml"))
        calibration = preparation.numerical_calibration_jobs[0]
        score_manifests = dict(calibration.score_manifests)
        assert calibration.train_manifest == score_manifests["train"]
        assert "val200" in str(score_manifests["val"])
        calibration_dirs.add(calibration.output_dir)
        decision = load_workflow_settings(Path(f"configs/{stem}_validation.yaml")).decision_jobs[0]
        assert decision.checkpoint.parent == full.output_dir
        assert decision.source_manifest == val.dataset.pilot_manifest
    assert len(calibration_dirs) == 2
