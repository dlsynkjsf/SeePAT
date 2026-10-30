from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_fusion_model import _tiny_model
from test_sequence_training import StudyClips

from seepat.models.visual_auxiliary import AuxiliaryVisualFusion
from seepat.training import auxiliary_comparison as study


class Clips(StudyClips):
    def __init__(self, role):
        super().__init__(role, False)
        self.auxiliary_balance = {"class_weights": [1., 1.]}
        for sample in self.samples:
            if role == "train":
                target = int(sample["kind"] == "context") if sample["video_label"] else -1
                if target == 1:
                    sample["video"] = sample["video"] + .1
                sample.update(auxiliary_target=torch.tensor(target), auxiliary_mask=torch.tensor(target != -1))


def test_both_fits_resume_at_sealed_epochs_and_match_uninterrupted_native_training(tmp_path):
    torch.manual_seed(14)
    template = AuxiliaryVisualFusion(_tiny_model(freeze_backbone=True))
    def datasets(_arm):
        return Clips("train"), Clips("inner")
    def model(_arm, _resume):
        return deepcopy(template)
    options = replace(study.fixed_options(), epochs=1)
    one = study.fit_epochs(tmp_path / "pair", options, datasets, model, {"synthetic": True}, stage="training")
    two = study.fit_epochs(tmp_path / "pair", replace(options, epochs=2), datasets, model,
                           {"synthetic": True}, stage="training")
    assert one["initial_state_sha256"] == two["initial_state_sha256"]
    assert {a["optimizer_updates"] for a in two["arms"].values()} == {2}
    assert two["arms"]["control"]["auxiliary_head_l1"] == 0
    assert two["arms"]["candidate"]["auxiliary_head_l1"] > 0
    # Independent native two-epoch fit: same objective, no study epoch snapshots.
    for arm in ("control", "candidate"):
        study.train_arm(tmp_path / ("native-" + arm), arm, model(arm, False), *datasets(arm),
                        replace(options, epochs=2), {"synthetic": True}, one["initial_state_sha256"],
                        stage="training")
        a = torch.load(tmp_path / "pair" / arm / "epoch-2/checkpoint_last.pt", weights_only=False)
        b = torch.load(tmp_path / ("native-" + arm) / "checkpoint_last.pt", weights_only=False)
        assert a["global_step"] == b["global_step"] == 2
        for key, value in a["model_state"].items():
            assert torch.equal(value, b["model_state"][key]) if isinstance(value, torch.Tensor) else value == b["model_state"][key]
    assert study.fit_epochs(tmp_path / "pair", replace(options, epochs=2), None, None,
                            {"synthetic": True}, stage="training") == two
    with pytest.raises(ValueError, match="binding|settings"):
        study.fit_epochs(tmp_path / "pair", options, None, None, {"changed": True}, stage="training")


def test_identical_results_cannot_pass_and_new_gate_rejects_a_recall_loss():
    labels = [0] * 4 + [1] * 4
    control = [.1, .6, .7, .8, .2, .51, .52, .9]
    candidate = [.1, .2, .3, .4, .45, .49, .8, .9]
    result = study.compare_predictions(labels, control, candidate, control)
    assert result["arms"]["candidate"]["auc"] > result["arms"]["control"]["auc"]
    assert result["arms"]["candidate"]["balanced_accuracy"] > result["arms"]["control"]["balanced_accuracy"]
    assert result["checks"]["no_lost_total_detections"] is False
    assert result["gate"] == "stop_and_review"
    assert study.compare_predictions(labels, control, control, control)["gate"] == "stop_and_review"


def test_partial_epoch_is_preserved_without_retraining(tmp_path):
    output = tmp_path / "pair/control/epoch-1"
    output.mkdir(parents=True)
    sentinel = output / "worker.lock"
    sentinel.write_text("interrupted")
    with pytest.raises(ValueError, match="interrupted"):
        study.fit_epochs(tmp_path / "pair", replace(study.fixed_options(), epochs=1), None, None,
                         {"synthetic": True}, stage="training")
    assert sentinel.read_text() == "interrupted"
    assert not (tmp_path / "pair/candidate").exists()


def test_help_and_finished_watcher_do_not_open_inputs_or_models(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        study.main(["--help"])
    assert stopped.value.code == 0
    from seepat.artifacts import atomic_write_json
    from seepat.live_progress import live_progress_path
    path = live_progress_path(tmp_path / study.OUTPUT / "inputs_status.json")
    atomic_write_json(path, {"status": "complete", "phase": "done", "finished": 3, "total": 3, "jobs_total": 3})
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    study.main(["watch", "--project", str(tmp_path), "--stage", "inputs"])
    assert before == sorted(str(p) for p in tmp_path.rglob("*"))
    assert "complete" in capsys.readouterr().out.lower()


def test_three_frozen_models_score_the_same_cases_and_cache_completed_outputs(tmp_path):
    class Data:
        def __init__(self):
            self.rows = [{"event_id": str(n), "video_id": str(n), "source_group": str(n), "kind": "bilabial",
                          "class_id": "0", "phoneme": "p", "manipulation_modality": "real" if n < 2 else "both_modified"}
                         for n in range(4)]
        def __len__(self):
            return 4
        def __getitem__(self, n):
            return {**self.rows[n], "video": torch.full((3, 1, 1, 1), float(n)),
                    "label": torch.tensor(0), "video_label": torch.tensor(int(n >= 2))}
    class Frozen(torch.nn.Module):
        def __init__(self, probabilities):
            super().__init__()
            self.register_buffer("values", torch.logit(torch.tensor(probabilities)))
        def forward(self, video):
            p = self.values[video[:, 0, 0, 0, 0].long()]
            return torch.stack([torch.zeros_like(p), p], 1)
    selected = [{"video_id": str(n), "source_group": str(n), "category": "real" if n < 2 else "both_modified",
                 "reason": "", "status": "paired"} for n in range(4)]
    selected.append({"video_id": "absent", "source_group": "absent", "category": "audio_modified",
                     "reason": "original_missing", "status": "unevaluated"})
    def model(name):
        return Frozen([.1, .2, .8, .9] if name == "candidate" else [.1, .6, .2, .8])
    output = tmp_path / "evaluation"
    result = study.evaluate_models(output, lambda _: Data(), model, selected, {"fixture": True})
    assert result["gate"] == "review_inference_integration_only"
    assert result["optimizer_updates"] == 0 and result["automatic_promotion"] is False
    assert {a["unavailable"] for a in result["arms"].values()} == {1}
    assert result["paired_changes"]["false_alarm_corrected"] == 1
    assert result["paired_changes"]["detection_gained"] == 1
    assert study.evaluate_models(output, None, None, selected, {"fixture": True}) == result
    with pytest.raises(ValueError, match="binding"):
        study.evaluate_models(output, None, None, selected, {"changed": True})


def test_anchor_guard_blocks_a_candidate_that_only_beats_a_degraded_control():
    result = study.compare_predictions([0, 0, 1, 1], [.8, .9, .1, .2], [.1, .6, .2, .8], [.1, .2, .8, .9])
    assert result["pair_passed"] is True and result["anchor_guard_passed"] is False
    assert result["gate"] == "stop_and_review"


def test_new_profile_binds_current_code_and_preserves_every_historical_source_pin():
    from hashlib import sha256
    from pathlib import Path

    from seepat.training import experiment as runtime
    project = Path(__file__).parents[1]
    profile, binding = study.load_protocol(project, project / study.CONFIG, metadata=False)
    old = runtime.read_json(project / "configs/auxiliary_visual_pilot_v1.json")
    assert len(profile["code_bindings"]) == 65
    assert len(profile["metadata_bindings"]) == 24
    assert all(profile["code_bindings"][name] == digest for name, digest in old["code_bindings"].items())
    assert binding["protocol_sha256"] == sha256((project / study.CONFIG).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_unplanned_protocol_and_missing_outputs_stop_without_launching_workers(tmp_path, monkeypatch):
    from seepat.artifacts import atomic_write_json

    config = tmp_path / study.CONFIG
    monkeypatch.chdir(tmp_path)
    atomic_write_json(config, {"design": {**study.DESIGN, "threshold": .4}, "code_bindings": {}, "metadata_bindings": {}})
    with pytest.raises(ValueError, match="protocol"):
        study.main(["run", "--project", str(tmp_path)])
    assert not (tmp_path / study.OUTPUT).exists()
