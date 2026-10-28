"""Public boundaries for the frozen-feature engineering diagnostic."""

import importlib.util
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
from test_fusion_model import EVIDENCE_WIDTH, _tiny_model
from test_sequence_pilot_cli import selection

from seepat.artifacts import atomic_write_json, file_sha256, read_csv_rows
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.training import experiment as runtime

SPEC = importlib.util.spec_from_file_location(
    "feature_probe", Path(__file__).parents[1] / "scripts/probe_features.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def sample():
    return {
        "video": torch.ones(3, 4, 8, 8) / 2,
        "frame_mask": torch.tensor([True, True, False, False]),
        "evidence_features": torch.full((EVIDENCE_WIDTH,), float("nan")),
        "evidence_feature_mask": torch.zeros(EVIDENCE_WIDTH, dtype=torch.bool),
        "label": torch.tensor(-1),
    }


def test_frozen_extraction_preserves_masks_weights_and_native_prediction():
    torch.manual_seed(29)
    model = _tiny_model().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    item = sample()
    extracted = probe.extract_clip(model, item, "cpu")
    with torch.inference_mode():
        expected = model(
            item["video"][None], item["frame_mask"][None],
            item["evidence_features"][None], item["evidence_feature_mask"][None],
        ).softmax(-1)[0, 1].item()
    assert extracted["probability"] == pytest.approx(expected, abs=1e-7)
    assert set(extracted["features"]) == {"swin", "temporal", "evidence", "fused"}
    assert extracted["frame_mask"].tolist() == [True, True, False, False]
    assert not extracted["evidence_mask"].any()
    assert all(np.isfinite(v).all() for v in extracted["features"].values())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_fixed_pooling_keeps_one_video_unit_and_known_mean_max_order():
    clips = {k: np.array([[1., 2.], [3., 4.]]) for k in (*probe.BRANCHES, "fused")}
    pooled = probe.pool_features(clips)
    np.testing.assert_array_equal(pooled["swin"], [2., 3., 3., 4.])
    np.testing.assert_array_equal(pooled["concatenated"], [2., 3., 3., 4.] * 3)
    changed = {**clips, "evidence": np.array([[np.nan, 1.], [2., 3.]])}
    with pytest.raises(ValueError, match="finite"):
        probe.pool_features(changed)


def test_probe_fits_train_scaling_only_and_metrics_handle_probability_ties():
    x = np.array([[-2., 7.], [-1., 7.], [1., 7.], [2., 7.]])
    fitted = probe.fit_probe(x, [0, 0, 1, 1])
    assert fitted["mean"] == [0., 7.]
    assert fitted["scale"][1] == 1.
    assert probe.predict_probe(fitted, [[-1000., 7.], [1000., 7.]]).tolist() == pytest.approx([0., 1.])
    assert fitted["mean"] == [0., 7.]
    metrics = probe.classification_summary([0, 0, 1, 1], [.1, .8, .8, .9])
    assert metrics["balanced_accuracy"] == .75
    assert metrics["auc"] == .875
    assert metrics["true_positive"] == 2 and metrics["false_positive"] == 1


class PreparedClips:
    """An in-memory media boundary; files are synthetic provenance bytes only."""

    def __init__(self, project, role, count):
        self.rows, self.inventory = [], []
        for i in range(count):
            identifier = f"{role}-{i}"
            category = "real" if i % 2 == 0 else "visual_modified"
            self.inventory.append({"video_id": identifier, "source_group": identifier,
                                   "dataset_split": "train" if role == "train" else "val",
                                   "category": category, "status": "paired", "reason": "",
                                   "expanded_clips": "2"})
            for kind in ("bilabial", "context"):
                path = project / f"synthetic/{identifier}-{kind}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic input; not a video")
                self.rows.append({**self.inventory[-1], "kind": kind,
                                  "event_id": f"{identifier}-{kind}",
                                  "manipulation_modality": category,
                                  "mouth_clip_path": path.relative_to(project).as_posix(),
                                  "class_id": "-1" if kind == "context" else "0",
                                  "phoneme": "" if kind == "context" else "p"})
        self.inventory.append({"video_id": f"{role}-unavailable", "source_group": f"{role}-missing",
                               "dataset_split": "train" if role == "train" else "val",
                               "category": "real", "status": "unevaluated", "reason": "fixture",
                               "expanded_clips": "0"})

    def __getitem__(self, index):
        row = self.rows[index]
        item = sample()
        item["video"] = item["video"] * (1 if row["category"] == "real" else .1)
        return {**item, **{k: row[k] for k in ("video_id", "event_id", "source_group")},
                "label": torch.tensor(int(row["class_id"])),
                "video_label": torch.tensor(int(row["category"] != "real"))}


def test_export_seals_exact_video_bags_and_reuses_complete_outputs_without_a_model(tmp_path):
    datasets = {r: PreparedClips(tmp_path, r, n) for r, n in (("train", 8), ("inner", 4))}
    inventories = {r: d.inventory for r, d in datasets.items()}
    output = tmp_path / "features"
    result = probe.export_features(
        tmp_path, output, datasets, inventories, lambda: _tiny_model(), {"fixture": "frozen"},
        device="cpu",
    )
    assert result["roles"]["train"] == {"requested": 9, "evaluated": 8, "unavailable": 1, "clips": 16}
    assert result["encoder_optimizer_updates"] == 0
    first_seal = file_sha256(output / "seal.json")
    repeated = probe.export_features(
        tmp_path, output, datasets, inventories,
        lambda: pytest.fail("Completed exports must not load a model"),
        {"fixture": "frozen"}, device="cpu",
    )
    assert repeated == result and file_sha256(output / "seal.json") == first_seal
    assert runtime.verify_output(output)["status"] == "complete"
    with pytest.raises(ValueError, match="binding|settings"):
        probe.export_features(tmp_path, output, datasets, inventories,
                              lambda: _tiny_model(), {"fixture": "changed"}, device="cpu")


def test_explicit_resume_keeps_sealed_videos_and_rejects_partial_video_records(tmp_path):
    datasets = {r: PreparedClips(tmp_path, r, 2) for r in probe.ROLES}
    inventories = {r: d.inventory for r, d in datasets.items()}
    config = tmp_path / "config.json"
    config.write_text("{}")
    reference = _tiny_model()

    class StopBetweenVideos(WorkflowProgress):
        def __call__(self, phase, finished=0, total=0, item=""):
            super().__call__(phase, finished, total, item)
            if phase == "extract frozen branches" and finished == 1:
                raise KeyboardInterrupt

    output = tmp_path / "features"
    with pytest.raises(KeyboardInterrupt):
        probe.export_features(tmp_path, output, datasets, inventories, lambda: deepcopy(reference), {},
                              progress=StopBetweenVideos(config, tmp_path / "status.json", 2))
    first = file_sha256(output / "train/train-0/seal.json")
    with pytest.raises(ValueError, match="resume"):
        probe.export_features(tmp_path, output, datasets, inventories, lambda: _tiny_model(), {})
    with pytest.raises(ValueError, match="weights"):
        probe.export_features(tmp_path, output, datasets, inventories,
                              lambda: _tiny_model(), {}, resume=True)
    result = probe.export_features(tmp_path, output, datasets, inventories,
                                   lambda: deepcopy(reference), {}, resume=True)
    assert result["reused_videos"] == 1
    assert file_sha256(output / "train/train-0/seal.json") == first
    # A partial record is retained; recovery must never silently replace its contents.
    (output / "seal.json").unlink()
    (output / "inner/inner-1/seal.json").unlink()
    with pytest.raises(ValueError, match="Partial video"):
        probe.export_features(tmp_path, output, datasets, inventories,
                              lambda: _tiny_model(), {}, resume=True)


def test_comparison_fits_only_fixed_train_probes_and_keeps_unavailable_inner_requests(tmp_path):
    datasets = {r: PreparedClips(tmp_path, r, n) for r, n in (("train", 8), ("inner", 4))}
    inventories = {r: d.inventory for r, d in datasets.items()}
    features = tmp_path / "features"
    probe.export_features(tmp_path, features, datasets, inventories, lambda: _tiny_model(), {})
    result = probe.compare_probes(features, tmp_path / "comparison", {})
    assert result["head_fits"] == 20
    assert result["encoder_optimizer_updates"] == 0
    assert result["gate"] == "stop_and_review_signal"
    assert set(result["views"]) == {*probe.VIEWS, "checkpoint_max"}
    assert all(r["inner_reused"]["samples"] == 4 for r in result["views"].values())
    category = result["views"]["swin"]["inner_reused"]["categories"]["real"]
    assert category["requested"] == 3 and category["evaluated"] == 2
    assert category["unavailable"] == 1
    rows = read_csv_rows(tmp_path / "comparison/predictions.csv")
    missing = [r for r in rows if r["role"] == "inner" and r["status"] == "not_evaluated"]
    assert len(missing) == 6 and all(r["probability"] == r["label"] == "" for r in missing)
    cache = probe.load_features(features, {})
    fitted = runtime.read_json(tmp_path / "comparison/swin_probe.json")["all_train"]
    np.testing.assert_allclose(fitted["mean"], cache["train"]["features"]["swin"].mean(axis=0))
    first = file_sha256(tmp_path / "comparison/seal.json")
    assert probe.compare_probes(features, tmp_path / "comparison", {}) == result
    assert file_sha256(tmp_path / "comparison/seal.json") == first


def test_cli_help_and_terminal_tracker_never_start_work_or_require_a_config(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        probe.main(["--help"])
    assert stopped.value.code == 0
    status = live_progress_path(tmp_path / probe.OUTPUT / "features_status.json")
    atomic_write_json(status, {"status": "complete", "jobs_total": 2, "job_index": 2,
                               "phase": "complete", "finished": 2, "total": 2})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    probe.main(["watch", "--project", str(tmp_path), "--stage", "features"])
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert "complete" in capsys.readouterr().out.lower()


def test_protocol_rejects_external_roles_changed_design_and_changed_pinned_bytes(tmp_path):
    old = runtime.read_json(Path(__file__).parents[1] / "configs/sequence_evidence_pilot_v2.json")
    data = b"synthetic protocol binding fixture\n"
    digest = sha256(data).hexdigest()
    codes = {**old["code_bindings"], "scripts/probe_features.py": digest}
    config = tmp_path / "config.json"
    definition = {"design": deepcopy(probe.DESIGN),
                  "metadata_bindings": dict.fromkeys(probe.METADATA, digest),
                  "code_bindings": dict.fromkeys(codes, digest)}
    for name in (*definition["metadata_bindings"], *codes):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    atomic_write_json(config, definition)
    assert probe.load_protocol(tmp_path, config)[0] == definition
    for change in (
        lambda p: p["design"]["roles"]["inner"].update(events="data/deepfake_eval/events.csv"),
        lambda p: p["design"].update(cutoff=.4),
        lambda p: p["design"]["probe"].update(C=100),
        lambda p: p["design"]["checkpoint"].update(path="another.pt"),
    ):
        changed = deepcopy(definition)
        change(changed)
        atomic_write_json(config, changed)
        with pytest.raises(ValueError, match="design"):
            probe.load_protocol(tmp_path, config)
    atomic_write_json(config, definition)
    (tmp_path / "scripts/probe_features.py").write_bytes(b"changed code")
    with pytest.raises(ValueError, match="source"):
        probe.load_protocol(tmp_path, config)
    (tmp_path / "scripts/probe_features.py").write_bytes(data)
    (tmp_path / next(iter(probe.METADATA))).write_bytes(b"changed metadata")
    with pytest.raises(ValueError, match="metadata"):
        probe.load_protocol(tmp_path, config)


def test_unavailable_source_leaks_and_unknown_video_categories_stop_before_extraction(tmp_path):
    datasets = {r: PreparedClips(tmp_path, r, 2) for r in probe.ROLES}
    inventory = {r: deepcopy(d.inventory) for r, d in datasets.items()}
    inventory["inner"][-1]["source_group"] = "train-0"
    with pytest.raises(ValueError, match="separation"):
        probe.validate_inputs(datasets, inventory)
    inventory = {r: deepcopy(d.inventory) for r, d in datasets.items()}
    inventory["inner"][1]["category"] = "unknown_category"
    with pytest.raises(ValueError, match="modality|category"):
        probe.validate_inputs(datasets, inventory)


def test_nonfinite_native_prediction_stops_instead_of_becoming_a_cached_score():
    model = _tiny_model().eval().requires_grad_(False)
    model.classifier.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="probability"):
        probe.extract_clip(model, sample(), "cpu")


def test_cli_preflight_keeps_all_requests_and_verify_missing_results_never_fits(tmp_path, monkeypatch):
    definition = runtime.read_json(Path(__file__).parents[1] / "configs/frozen_feature_probe_v1.json")
    selection(tmp_path)
    data = b"synthetic CLI fixture; not real media or checkpoint bytes"
    for group in ("code_bindings", "metadata_bindings"):
        for name in definition[group]:
            path = tmp_path / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            raw = path.read_bytes().replace(b"\r\n", b"\n") if group == "code_bindings" else path.read_bytes()
            definition[group][name] = sha256(raw).hexdigest()
    checkpoint = tmp_path / definition["design"]["checkpoint"]["path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(data)
    config = tmp_path / "configs/frozen_feature_probe_v1.json"
    atomic_write_json(config, definition)
    monkeypatch.chdir(tmp_path)
    result = probe.main(["preflight", "--project", str(tmp_path)])
    assert result == {"train": {"requested": 128, "usable": 125},
                      "inner": {"requested": 64, "usable": 62}}
    assert not (tmp_path / probe.OUTPUT).exists()
    with pytest.raises(FileNotFoundError):
        probe.main(["verify", "--stage", "probes", "--project", str(tmp_path)])
    assert not (tmp_path / probe.OUTPUT).exists()
