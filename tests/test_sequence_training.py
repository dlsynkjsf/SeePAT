from copy import deepcopy
from dataclasses import replace
from typing import ClassVar

import pytest
import torch
from test_sequence_evidence import model_input, tiny_base

from seepat.models.sequence_evidence import SequenceAwareFusion
from seepat.training import experiment as runtime
from seepat.training import sequence_pilot as pilot
from seepat.training.train import TrainingOptions, _forward_logits


def test_original_trainer_dispatches_the_versioned_sequence_batch():
    model = SequenceAwareFusion(tiny_base()).eval()
    values = model_input()
    batch = {"frame_mask": values["frame_mask"], "evidence_features": values["features"],
             "evidence_feature_mask": values["feature_mask"], "timing_features": values["timing_features"],
             "timing_mask": values["timing_mask"], "timing_version": ["fusion-vild-sequence-v1"] * 2}
    assert torch.equal(model(**values), _forward_logits(model, values["video"], batch, torch.device("cpu")))


def test_initial_digest_accepts_version_metadata_and_detects_it_changing():
    model = SequenceAwareFusion(tiny_base())
    assert runtime.state_digest(model) == runtime.state_digest(deepcopy(model))


class StudyClips:
    input_contract: ClassVar = {"fixed": "synthetic paired inputs"}
    calibration_contract: ClassVar = {"fixed": "synthetic population"}

    def __init__(self, role, enabled):
        self.rows, self.samples = [], []
        for label in (0, 1):
            for index, kind in enumerate(("bilabial", "context")):
                event = f"{role}-{label}-{index}"
                context = kind == "context"
                self.rows.append({"event_id": event, "video_id": f"{role}-{label}",
                                  "source_group": f"{role}-{label}", "class_id": "-1" if context else "0",
                                  "manipulation_modality": "both_modified" if label else "real",
                                  "kind": kind, "phoneme": "" if context else "p"})
                self.samples.append({
                    "event_id": event, "video_id": f"{role}-{label}", "kind": kind,
                    "phoneme": "" if context else "p", "label": torch.tensor(-1 if context else 0),
                    "video_label": torch.tensor(label), "video": torch.full((3, 16, 8, 8), 0.2 + 0.3 * label),
                    "frame_mask": torch.ones(16, dtype=torch.bool),
                    "features": torch.zeros(8), "feature_mask": torch.zeros(8, dtype=torch.bool),
                    "evidence_features": torch.zeros(7), "evidence_feature_mask": torch.zeros(7, dtype=torch.bool),
                    "timing_features": torch.ones(16, 3) * (0.1 + label) if enabled and not context else torch.zeros(16, 3),
                    "timing_mask": torch.full((16,), enabled and not context, dtype=torch.bool),
                    "timing_version": "fusion-vild-sequence-v1",
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def evidence_coverage(self):
        return {}


def test_native_paired_training_resumes_both_arms_from_identical_starts(tmp_path):
    torch.manual_seed(73)
    template = SequenceAwareFusion(tiny_base())
    options = TrainingOptions(epochs=1, amp=False, freeze_backbone=True, supervision="video_max",
                              class_weighting="balanced_global", gradient_accumulation_steps=1,
                              early_stopping_patience=0, max_train_batches=16, max_validation_batches=4)
    def datasets(arm):
        return StudyClips("train", arm == "candidate"), StudyClips("inner", arm == "candidate")
    def model(_arm, _resuming):
        return deepcopy(template)
    output = tmp_path / "readiness"
    first = pilot.fit_pair(output, options, datasets, model, {"fixed": "test"}, stage="readiness")
    assert {x["optimizer_updates"] for x in first["arms"].values()} == {2}
    resumed = pilot.fit_pair(output, replace(options, epochs=2), datasets, model,
                             {"fixed": "test"}, stage="readiness", resume=True)
    assert {x["optimizer_updates"] for x in resumed["arms"].values()} == {4}
    assert resumed["arms"]["control"]["timing_projection_l1"] == 0
    assert resumed["arms"]["candidate"]["timing_projection_l1"] > 0
    assert first["initial_state_sha256"] == resumed["initial_state_sha256"]
    for arm in ("control", "candidate"):
        saved = torch.load(output / arm / "checkpoint_last.pt", weights_only=False)
        assert saved["completed_epoch"] == 2
    fresh = tmp_path / "uninterrupted"
    pilot.fit_pair(fresh, replace(options, epochs=2), datasets, model,
                   {"fixed": "test"}, stage="readiness")
    for arm in ("control", "candidate"):
        resumed_state = torch.load(output / arm / "checkpoint_last.pt", weights_only=False)["model_state"]
        fresh_state = torch.load(fresh / arm / "checkpoint_last.pt", weights_only=False)["model_state"]
        for key, value in resumed_state.items():
            assert torch.equal(value, fresh_state[key]) if isinstance(value, torch.Tensor) else value == fresh_state[key]
    with pytest.raises(ValueError):
        pilot.fit_pair(output, options, datasets, model, {"changed": "binding"}, stage="readiness")


def test_paired_evaluation_retains_unavailable_and_stops_on_identical_models(tmp_path):
    torch.manual_seed(73)
    template = SequenceAwareFusion(tiny_base())
    selected = [{"video_id": f"inner-{label}", "source_group": f"inner-{label}",
                 "category": "real" if not label else "both_modified", "status": "paired", "reason": ""}
                for label in (0, 1)]
    selected.append({"video_id": "missing", "source_group": "missing", "category": "audio_modified",
                     "status": "unevaluated", "reason": "no_original_bilabials"})
    result = pilot.evaluate_pair(
        tmp_path / "evaluation", lambda arm: StudyClips("inner", arm == "candidate"),
        lambda _arm: deepcopy(template), selected, {"fixed": "test"},
    )
    assert result["gate"] == "stop_and_review" and result["optimizer_updates"] == 0
    for arm in ("control", "candidate"):
        assert result["arms"][arm]["evaluated"] == 2 and result["arms"][arm]["unavailable"] == 1


def test_tensor_only_initial_digest_preserves_its_historical_value():
    model = torch.nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [2.0]]))
        model.bias.copy_(torch.tensor([0.1, 0.2]))
    assert runtime.state_digest(model) == "0c469deb038ab4c7e2e34549cc5674c5e4d63085fd332bc8c15637c14241ca5b"
