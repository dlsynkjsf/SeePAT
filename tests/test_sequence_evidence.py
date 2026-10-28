from copy import deepcopy
from io import BytesIO

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from seepat.models import sequence_evidence as seq
from seepat.models.fusion import HybridFusionEventClassifier
from seepat.training.video_objective import VideoBag, scored_video_bags


def example():
    ids = [10, 12, 15]
    row = {
        "event_id": "video-0", "video_id": "video", "file": "real/video.mp4",
        "dataset_split": "train", "source_group": "real/video", "phoneme": "p",
        "phone_start_s": "1.0", "phone_end_s": "1.2",
        "video_phone_start_s": "1.1", "video_phone_end_s": "1.3",
        "audio_video_start_offset_s": "0.1",
        "mouth_crop_frame_indices_json": "[10, 12, 15]",
    }
    trace = {
        "artifact_version": "vild-trace-v1", "video_id": "video",
        "source_file": row["file"], "dataset_split": "train",
        "source_group": "real/video",
        "timing": {"audio_video_start_offset_s": 0.1, "fps": 10},
        "bilabial_event_windows": [{
            "event_id": "video-0", "phoneme": "p",
            "audio_phone_start_s": 1.0, "audio_phone_end_s": 1.2,
            "video_phone_start_s": 1.1, "video_phone_end_s": 1.3,
            "window_start_s": 0.9, "window_end_s": 1.5,
            "mouth_clip_frame_indices": ids,
        }],
        "frames": [
            {"frame_index": i, "timestamp_s": t, "normalized_vild": v, "face_count": 1}
            for i, t, v in zip(ids, [1.0, 1.13, 1.41], [0.04, 0.0, 0.08], strict=True)
        ],
    }
    return row, trace


def test_sequence_follows_actual_crop_frames_and_irregular_clock():
    row, trace = example()
    result = seq.event_timing_sequence(row, trace, decoded_frame_count=3, sequence_length=4)
    assert result["timing_frame_indices"] == (10, 12, 15, 15)
    assert result["timing_mask"].tolist() == [True, True, True, False]
    torch.testing.assert_close(result["timing_features"], torch.tensor([
        [0.04, -0.10, -0.30], [0.0, 0.03, -0.17], [0.08, 0.31, 0.11], [0, 0, 0],
    ]))


def test_missing_measurements_are_masked_without_inventing_closure():
    row, trace = example()
    trace["frames"][0]["normalized_vild"] = float("nan")
    trace["frames"][2]["face_count"] = 2
    result = seq.event_timing_sequence(row, trace, decoded_frame_count=3, sequence_length=4)
    assert result["timing_mask"].tolist() == [False, True, False, False]
    assert not result["timing_features"][[0, 2, 3]].any()
    assert result["timing_features"][1, 0] == 0  # Observed zero is still available.


def rounded_window_example(fps=25):
    """Recorded case-001 clock numbers with generated trace measurements."""
    import json

    from seepat.preprocessing.face import event_frame_bounds

    row, trace = example()
    first, last = event_frame_bounds(1.27, 1.33, fps, 0.2, 0.2)
    ids = list(range(first, last + 1))
    row.update(phone_start_s="1.27", phone_end_s="1.33", video_phone_start_s="1.27",
               video_phone_end_s="1.33", audio_video_start_offset_s="0",
               mouth_crop_frame_indices_json=json.dumps(ids))
    trace["timing"].update(fps=fps, audio_video_start_offset_s=0)
    trace["bilabial_event_windows"][0].update(audio_phone_start_s=1.27, audio_phone_end_s=1.33,
        video_phone_start_s=1.27, video_phone_end_s=1.33, window_start_s=1.07, window_end_s=1.53,
        mouth_clip_frame_indices=ids)
    trace["frames"] = [{"frame_index": i, "timestamp_s": i / fps, "normalized_vild": 0.02, "face_count": 1}
                       for i in range(last + 2)]
    return row, trace, ids


def test_sequence_accepts_recorded_rounded_edge_frames_and_preserves_phone_clock():
    row, trace, ids = rounded_window_example()
    result = seq.event_timing_sequence(row, trace, decoded_frame_count=len(ids))
    assert result["timing_frame_indices"] == (*range(26, 40), 39, 39)
    assert result["timing_mask"].tolist() == [True] * 14 + [False] * 2
    torch.testing.assert_close(result["timing_features"][0], torch.tensor([0.02, -0.23, -0.29]))
    torch.testing.assert_close(result["timing_features"][13], torch.tensor([0.02, 0.29, 0.23]))
    assert not result["timing_features"][14:].any()


@pytest.mark.parametrize("fps", [25, 30000 / 1001, 24000 / 1001])
def test_sequence_uses_producer_frame_bounds_at_integer_and_fractional_fps(fps):
    import json

    row, trace, ids = rounded_window_example(fps)
    result = seq.event_timing_sequence(row, trace, decoded_frame_count=len(ids))
    assert result["timing_frame_indices"][0] == ids[0] and result["timing_mask"].any()
    for wrong in ([ids[0] - 1, *ids], [*ids, ids[-1] + 1]):
        row["mouth_crop_frame_indices_json"] = json.dumps(wrong)
        trace["bilabial_event_windows"][0]["mouth_clip_frame_indices"] = wrong
        with pytest.raises(ValueError, match="rounded frame bounds"):
            seq.event_timing_sequence(row, trace, decoded_frame_count=len(wrong))


@pytest.mark.parametrize("fps", [None, 0, -25, float("nan"), float("inf")])
def test_sequence_rejects_missing_nonpositive_and_nonfinite_fps(fps):
    row, trace, ids = rounded_window_example()
    trace["timing"]["fps"] = fps
    with pytest.raises(ValueError):
        seq.event_timing_sequence(row, trace, decoded_frame_count=len(ids))


def test_rounded_bounds_still_reject_displaced_clock_with_matching_frame_ids():
    row, trace, ids = rounded_window_example()
    trace["frames"][ids[-1]]["timestamp_s"] = 1.59
    with pytest.raises(ValueError, match="recorded event window"):
        seq.event_timing_sequence(row, trace, decoded_frame_count=len(ids))


def test_unavailable_trace_still_requires_a_development_role():
    row, _ = example()
    row["dataset_split"] = "external"
    with pytest.raises(ValueError, match="development"):
        seq.event_timing_sequence(row, None, decoded_frame_count=3)


def test_encoder_starts_neutral_learns_and_preserves_missing_rows():
    torch.manual_seed(73)
    encoder = seq.MaskedTimingEncoder(8)
    values = torch.tensor([
        [[0.03, -0.2, -0.3], [0.0, 0.0, -0.1], [float("nan")] * 3],
        [[float("nan")] * 3] * 3,
    ], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    assert torch.equal(encoder(values, mask), torch.zeros(2, 8))
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    (encoder(values, mask)[0] - 1).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    optimizer.step()
    optimizer.zero_grad()
    values.grad = None
    result = encoder(values, mask)
    assert result[0].abs().sum() > 0 and torch.equal(result[1], torch.zeros(8))
    result[0].sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad[0, :2].abs().sum() > 0
    assert not values.grad[~mask].any()
    contaminated = values.detach().clone()
    contaminated[~mask] = 999999
    assert torch.equal(result.detach(), encoder(contaminated, mask).detach())


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

    def forward(self, video):
        return self.projection(video.mean((2, 3, 4)))


class TinyFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection, self.classifier = nn.Linear(3, 6), nn.Linear(6, 2)

    def forward(self, frames):
        return self.classifier(self.projection(frames.mean((2, 3))))


def tiny_base():
    return HybridFusionEventClassifier(
        pretrained=False, freeze_backbone=True,
        swin_backbone=TinySwin(), frame_backbone=TinyFrame(),
        evidence_hidden=8, temporal_channels=8, embedding_features=7,
        fusion_dimension=8, fusion_heads=2, dropout=0.0,
    )


def model_input():
    return {
        "video": torch.rand(2, 3, 4, 8, 8),
        "frame_mask": torch.ones(2, 4, dtype=torch.bool),
        "features": torch.rand(2, 7),
        "feature_mask": torch.tensor([[True] * 7, [False] * 7]),
        "timing_features": torch.rand(2, 4, 3),
        "timing_mask": torch.tensor([[True] * 4, [False] * 4]),
    }


def test_candidate_starts_exactly_like_base_and_keeps_missing_branch_neutral():
    torch.manual_seed(73)
    base = tiny_base().eval()
    model = seq.SequenceAwareFusion(deepcopy(base)).eval()
    inputs = model_input()
    original = {key: value for key, value in inputs.items() if not key.startswith("timing_")}
    assert torch.equal(base(**original), model(**inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.nn.functional.cross_entropy(model(**inputs), torch.tensor([1, 0])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.timing_encoder.parameters())
    optimizer.step()
    assert torch.equal(model.base(**original)[1], model(**inputs)[1])
    assert not torch.equal(model.base(**original)[0], model(**inputs)[0])
    with pytest.raises(ValueError, match="explicit timing"):
        model(**original)


def test_candidate_state_requires_its_version_and_round_trips_without_mutation():
    torch.manual_seed(73)
    model = seq.SequenceAwareFusion(tiny_base()).eval()
    state = deepcopy(model.state_dict())
    assert state["_extra_state"]["timing_version"] == seq.TIMING_VERSION
    buffer = BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    other = seq.SequenceAwareFusion(tiny_base()).eval()
    other.load_state_dict(torch.load(buffer, weights_only=True))
    inputs = model_input()
    assert torch.equal(model(**inputs), other(**inputs))
    wrong = deepcopy(state)
    wrong["_extra_state"]["timing_version"] = "another-input-order"
    for value in wrong.values():
        if isinstance(value, torch.Tensor):
            value.fill_(17)
    for incompatible in (wrong, model.base.state_dict()):
        with pytest.raises(ValueError, match="contract"):
            model.load_state_dict(incompatible)
    with pytest.raises(ValueError, match="strict"):
        model.load_state_dict(state, strict=False)
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, state[name])


def test_batch_forward_requires_the_sequence_version():
    model = seq.SequenceAwareFusion(tiny_base()).eval()
    inputs = model_input()
    batch = {
        "frame_mask": inputs["frame_mask"],
        "evidence_features": inputs["features"],
        "evidence_feature_mask": inputs["feature_mask"],
        "timing_features": inputs["timing_features"],
        "timing_mask": inputs["timing_mask"],
        "timing_version": [seq.TIMING_VERSION] * 2,
    }
    assert torch.equal(
        model(**inputs),
        seq.forward_sequence_model(model, inputs["video"], batch, torch.device("cpu")),
    )
    batch["timing_version"][1] = "unknown"
    with pytest.raises(ValueError, match="version"):
        seq.forward_sequence_model(model, inputs["video"], batch, torch.device("cpu"))


@pytest.mark.parametrize("change", [
    lambda r, t: t.update(video_id="different-video"),
    lambda r, t: t.update(source_group="different-source"),
    lambda r, t: t.update(dataset_split="val"),
    lambda r, t: r.update(phoneme="a"),
    lambda r, t: r.update(video_phone_start_s="1.2"),
    lambda r, t: t["timing"].update(audio_video_start_offset_s=0.2),
    lambda r, t: t["bilabial_event_windows"][0].update(mouth_clip_frame_indices=[10, 12, 16]),
    lambda r, t: t["frames"][1].update(frame_index=10),
    lambda r, t: t["frames"][1].update(timestamp_s=0.5),
    lambda r, t: t["frames"][2].update(timestamp_s=1.6),
], ids=["video", "source", "split", "phoneme", "onset", "offset", "crop-ids", "duplicate", "clock-order", "window"])
def test_adapter_rejects_inconsistent_provenance(change):
    row, trace = example()
    change(row, trace)
    with pytest.raises(ValueError):
        seq.event_timing_sequence(row, trace, decoded_frame_count=3)


def test_rgb_decoder_count_must_reconcile_before_sampling():
    row, trace = example()
    with pytest.raises(ValueError, match="decoded frame count"):
        seq.event_timing_sequence(row, trace, decoded_frame_count=2)


def test_clip_subsampling_and_labels_do_not_change_timing_selection():
    row, trace = example()
    ids = list(range(100, 120))
    trace["timing"]["fps"] = 100
    row["mouth_crop_frame_indices_json"] = str(ids)
    trace["bilabial_event_windows"][0]["mouth_clip_frame_indices"] = ids
    trace["frames"] = [
        {"frame_index": i, "timestamp_s": 1 + n / 100, "normalized_vild": n / 100, "face_count": 1}
        for n, i in enumerate(ids)
    ]
    first = seq.event_timing_sequence(row, trace, decoded_frame_count=20, sequence_length=4)
    row.update(class_id="1", event_label="fake", manipulation_modality="both_modified",
               fake_segments="[[0,999]]")
    second = seq.event_timing_sequence(row, trace, decoded_frame_count=20, sequence_length=4)
    assert first["timing_frame_indices"] == second["timing_frame_indices"] == (100, 106, 113, 119)
    assert first["timing_mask"].all() and torch.equal(first["timing_features"], second["timing_features"])
    torch.testing.assert_close(first["timing_features"][:, 0], torch.tensor([0.0, 0.06, 0.13, 0.19]))


def test_context_and_missing_trace_have_explicit_unavailable_reasons():
    row, _ = example()
    missing = seq.event_timing_sequence(row, None, decoded_frame_count=3)
    assert missing["timing_unavailable_reason"] == "missing_verified_trace"
    row.update(kind="context", phoneme="")
    context = seq.event_timing_sequence(row, None, decoded_frame_count=16)
    assert context["timing_unavailable_reason"] == "unanchored_context"
    assert not context["timing_mask"].any() and not context["timing_features"].any()
    row["phoneme"] = "p"
    with pytest.raises(ValueError, match="fabricate"):
        seq.event_timing_sequence(row, None, decoded_frame_count=16)


def test_available_and_unavailable_inputs_collate_in_one_batch():
    row, trace = example()
    anchored = seq.event_timing_sequence(row, trace, decoded_frame_count=3, sequence_length=4)
    row.update(kind="context", phoneme="")
    context = seq.event_timing_sequence(row, None, decoded_frame_count=16, sequence_length=4)
    batch = next(iter(DataLoader([anchored, context], batch_size=2)))
    assert batch["timing_features"].shape == (2, 4, 3)
    assert not batch["timing_mask"][1].any()
    assert context["timing_frame_indices"] == (-1, -1, -1, -1)


def test_model_rejects_unmasked_padding_and_tensor_contract_mismatch():
    model = seq.SequenceAwareFusion(tiny_base()).eval()
    inputs = model_input()
    inputs["frame_mask"][0, -1] = False
    with pytest.raises(ValueError, match="Padded"):
        model(**inputs)
    state = deepcopy(model.state_dict())
    name = next(key for key, value in state.items() if isinstance(value, torch.Tensor) and value.ndim == 2)
    state[name] = state[name][:1]
    with pytest.raises(ValueError, match="tensor contract"):
        model.load_state_dict(state)
    with pytest.raises(RuntimeError):
        model.base.load_state_dict(model.state_dict())


def test_encoder_rejects_nonfinite_available_input_and_invalid_mask():
    encoder = seq.MaskedTimingEncoder(8)
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.full((1, 4, 3), float("nan")), torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        encoder(torch.zeros(1, 4, 3), torch.ones(1, 4))


def test_actual_video_bag_forward_keeps_context_unlabelled_and_backpropagates():
    torch.manual_seed(73)
    model = seq.SequenceAwareFusion(tiny_base()).train()
    assert not model.base.backbone.training and not model.base.frame_backbone.training
    inputs = model_input()
    row, trace = example()
    timing = seq.event_timing_sequence(row, trace, decoded_frame_count=3, sequence_length=4)
    samples = []
    for index, kind in enumerate(("bilabial", "context")):
        available = kind == "bilabial"
        samples.append({
            "video": inputs["video"][index],
            "frame_mask": torch.tensor([True, True, True, not available]),
            "features": torch.zeros(8), "feature_mask": torch.zeros(8, dtype=torch.bool),
            "evidence_features": inputs["features"][index] if available else torch.zeros(7),
            "evidence_feature_mask": torch.full((7,), available, dtype=torch.bool),
            "timing_features": timing["timing_features"] if available else torch.zeros(4, 3),
            "timing_mask": timing["timing_mask"] if available else torch.zeros(4, dtype=torch.bool),
            "timing_version": seq.TIMING_VERSION, "kind": kind,
            "label": torch.tensor(0 if available else -1), "video_label": torch.tensor(1),
            "event_id": f"synthetic-{index}", "video_id": "synthetic",
            "phoneme": "p" if available else "",
        })
    bag = VideoBag("synthetic", 1, (0, 1), ("synthetic-0", "synthetic-1"))
    scored, = scored_video_bags(
        model, DataLoader(samples, batch_size=1), [bag], torch.device("cpu"),
        seq.forward_sequence_model,
    )
    assert scored["event_labels"] == [0, -1] and len(scored["event_probabilities"]) == 2
    nn.functional.cross_entropy(scored["logits"], scored["label"]).backward()
    assert model.base.classifier.weight.grad.abs().sum() > 0
