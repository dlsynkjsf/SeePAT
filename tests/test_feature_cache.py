"""Frozen-feature cache contracts and training parity on deterministic tiny encoders."""

import json
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from seepat.artifacts import file_sha256
from seepat.evidence import FUSION_EVIDENCE_FIELDS
from seepat.models.fusion import HybridFusionEventClassifier
from seepat.training import feature_cache
from seepat.training.clip_cache import sequence_frame_indices

WIDTH = len(FUSION_EVIDENCE_FIELDS)


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.head = nn.Linear(6, 2)
        self.calls = 0

    def forward(self, video):
        self.calls += 1
        return self.projection(video.mean(dim=(2, 3, 4)))


class TinyFrames(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(6, 2))
        self.calls = 0

    def forward(self, frames):
        self.calls += 1
        return self.classifier(self.projection(frames.mean(dim=(2, 3))))


def tiny_model(**options):
    return HybridFusionEventClassifier(
        pretrained=False, swin_backbone=TinySwin(), frame_backbone=TinyFrames(),
        evidence_hidden=8, temporal_channels=8, temporal_layers=2,
        embedding_features=7, fusion_dimension=8, fusion_heads=2, dropout=0.25,
        **{"freeze_backbone": True, **options},
    ).train()


class ClipSource:
    """Fixed pixel inputs plus independently mutable manifest metadata."""

    def __init__(self, path, *, context=False, image_size=8):
        self.path = path
        path.write_bytes(b"original-source-clip")
        self.sequence_length = 16 if context else 4
        self.image_size = image_size
        self.context = context
        self.context_mask = [True] * 15 + [False] if context else None
        self.expected_hash = file_sha256(path) if context else None
        self.decoded_count = 16 if context else 3
        self.rows = [{"event_id": "event-1", "video_id": "video-1", "class_id": "0",
                      "source_group": "group-1", "dataset_split": "train"}]
        self.calibration_contract = {"version": "test-only"}
        self.input_contract = {"context": context}
        self.decode_calls = 0
        self.forbid_decode = False
        self.consume_rng = True

    def __len__(self):
        return len(self.rows)

    def clip_source(self, index):
        return self.path, self.expected_hash, self.context_mask

    def sample_metadata(self, index):
        row = self.rows[index]
        return {
            "features": torch.arange(8, dtype=torch.float32) / 10,
            "feature_mask": torch.ones(8, dtype=torch.bool),
            "evidence_features": torch.arange(WIDTH, dtype=torch.float32) / 5,
            "evidence_feature_mask": torch.tensor([True, False] + [True] * (WIDTH - 2)),
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "video_label": torch.tensor(1, dtype=torch.long),
            "event_id": row["event_id"], "video_id": row["video_id"],
            "source_group": row["source_group"], "manipulation_modality": "visual_modified",
            "phoneme": "" if self.context else "p",
            **({"kind": "context"} if self.context else {}),
        }

    def evidence_coverage(self):
        return {"test": 1.0}

    def __getitem__(self, index):
        if self.forbid_decode:
            raise AssertionError("Feature cache decoded its source again")
        self.decode_calls += 1
        # Exercise preservation of all RNG states during a cold preparation.
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        count = 3 * self.sequence_length * self.image_size**2
        video = torch.linspace(0.05, 0.95, count).reshape(
            3, self.sequence_length, self.image_size, self.image_size,
        )
        mask = self.context_mask or sequence_frame_indices(self.decoded_count, self.sequence_length)[1]
        if self.context:
            video[:, ~torch.tensor(mask)] = 0
        else:
            video[:, self.decoded_count:] = video[:, self.decoded_count - 1:self.decoded_count]
        return {**self.sample_metadata(index), "video": video,
                "frame_mask": torch.tensor(mask), "decoded_frame_count": self.decoded_count}


def prepared(model, source, directory):
    datasets, summary = feature_cache.prepare_feature_datasets(
        model, [source], directory, torch.device("cpu"),
    )
    return datasets[0], summary


def cached_forward(model, sample):
    return model.forward_from_encoder_features(
        sample["frozen_swin_features"].unsqueeze(0),
        sample["frozen_frame_features"].unsqueeze(0),
        sample["frame_mask"].unsqueeze(0),
        sample["evidence_features"].unsqueeze(0),
        sample["evidence_feature_mask"].unsqueeze(0),
    )


def rgb_forward(model, sample):
    return model(sample["video"].unsqueeze(0), sample["frame_mask"].unsqueeze(0),
                 sample["evidence_features"].unsqueeze(0),
                 sample["evidence_feature_mask"].unsqueeze(0))


def test_cold_and_warm_caches_bypass_pixels_and_both_encoders(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin")
    cached, summary = prepared(model, source, tmp_path / "cache")
    assert summary["built"] == 1 and summary["hits"] == 0
    assert source.decode_calls == model.backbone.calls == model.frame_backbone.calls == 1
    assert cached.rows is source.rows
    assert cached.calibration_contract == source.calibration_contract
    assert cached.input_contract == source.input_contract
    assert cached.evidence_coverage() == source.evidence_coverage()
    first = cached[0]
    assert first["video"].numel() == 0
    assert first["frozen_swin_features"].shape == (6,)
    assert first["frozen_frame_features"].shape == (4, 6)
    assert first["frame_mask"].tolist() == [True, True, True, False]
    source.forbid_decode = True
    warm, summary = prepared(model, source, tmp_path / "cache")
    assert summary["hits"] == 1 and summary["built"] == 0
    assert source.decode_calls == model.backbone.calls == model.frame_backbone.calls == 1
    assert torch.equal(warm[0]["frozen_swin_features"], first["frozen_swin_features"])
    source.rows[0]["class_id"] = "1"
    source.rows[0]["source_group"] = "updated-live-group"
    assert warm[0]["label"].item() == 1
    assert warm[0]["source_group"] == "updated-live-group"
    with np.load(warm.entries[0][0], allow_pickle=False) as saved:
        assert "label" not in saved.files


def test_cache_preparation_restores_python_numpy_and_torch_rng(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin")
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    prepared(model, source, tmp_path / "cache")
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0] and np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_cached_forward_gradients_dropout_and_optimizer_step_match_rgb(tmp_path):
    torch.manual_seed(17)
    direct = tiny_model()
    cache_model = deepcopy(direct)
    source = ClipSource(tmp_path / "clip.bin")
    cached, _ = prepared(cache_model, source, tmp_path / "cache")
    rgb, feature_sample = source[0], cached[0]
    # Trainable dropout is active; preparation must not freeze its activations.
    assert direct.training and cache_model.training
    assert direct.evidence_branch[3].p > 0
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (direct, cache_model)]
    state = torch.get_rng_state()
    first_logits = rgb_forward(direct, rgb)
    first_rng = torch.get_rng_state()
    first_loss = nn.functional.cross_entropy(first_logits, torch.tensor([1]))
    first_loss.backward()
    torch.set_rng_state(state)
    second_logits = cached_forward(cache_model, feature_sample)
    second_rng = torch.get_rng_state()
    second_loss = nn.functional.cross_entropy(second_logits, torch.tensor([1]))
    second_loss.backward()
    torch.testing.assert_close(second_logits, first_logits, rtol=0, atol=0)
    torch.testing.assert_close(second_loss, first_loss, rtol=0, atol=0)
    assert torch.equal(first_rng, second_rng)
    gradients = []
    for (name, a), (other_name, b) in zip(direct.named_parameters(), cache_model.named_parameters(), strict=True):
        assert name == other_name
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            gradients.append(name)
    assert gradients and any("temporal_encoder" in name for name in gradients)
    assert any("evidence_branch" in name for name in gradients)
    for optimizer in optimizers:
        optimizer.step()
    for a, b in zip(direct.parameters(), cache_model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["source", "encoder_weights", "normalization", "resolution", "context_mask"])
def test_feature_entries_invalidate_when_their_inputs_change(tmp_path, change):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin", context=change == "context_mask")
    first, _ = prepared(model, source, tmp_path / "cache")
    if change == "source":
        source.path.write_bytes(b"changed-source-clip")
    elif change == "encoder_weights":
        with torch.no_grad():
            model.backbone.projection.weight.add_(0.1)
    elif change == "normalization":
        model.pixel_mean.add_(0.1)
    elif change == "resolution":
        source.image_size += 1
    else:
        source.context_mask = [False] + [True] * 15
    second, summary = prepared(model, source, tmp_path / "cache")
    assert summary["built"] == 1 and summary["hits"] == 0
    assert first.entries[0][0] != second.entries[0][0]
    assert source.decode_calls == 2


def test_trainable_head_changes_reuse_raw_encoder_features(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin")
    first, _ = prepared(model, source, tmp_path / "cache")
    with torch.no_grad():
        model.swin_projection.weight.add_(0.1)
        model.classifier.weight.mul_(1.1)
    source.forbid_decode = True
    second, summary = prepared(model, source, tmp_path / "cache")
    assert summary["hits"] == 1 and summary["built"] == 0
    assert first.entries[0][0] == second.entries[0][0]


@pytest.mark.parametrize("corruption", ["truncated", "payload", "mask", "header",
                                       "nonobject_header", "fields"])
def test_corrupt_feature_entries_rebuild_without_changing_outputs(tmp_path, corruption):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin")
    cached, _ = prepared(model, source, tmp_path / "cache")
    original = cached[0]
    path = cached.entries[0][0]
    if corruption == "truncated":
        path.write_bytes(b"not-an-npz")
    else:
        with np.load(path, allow_pickle=False) as entry:
            arrays = {name: entry[name] for name in entry.files}
        if corruption == "payload":
            arrays["swin"][0] += 1
        elif corruption == "mask":
            arrays["mask"][:] = False
        elif corruption == "header":
            arrays["header"] = np.frombuffer(b"not-json", dtype=np.uint8)
        elif corruption == 'nonobject_header':
            arrays['header'] = np.frombuffer(b'null', dtype=np.uint8)
        else:
            arrays.pop("frames")
        np.savez(path, **arrays)
    rebuilt, summary = prepared(model, source, tmp_path / "cache")
    assert summary["built"] == 1 and summary["hits"] == 0
    assert source.decode_calls == 2
    torch.testing.assert_close(rebuilt[0]["frozen_swin_features"], original["frozen_swin_features"], rtol=0, atol=0)


def test_changed_sources_are_rejected_during_cached_training(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin")
    cached, _ = prepared(model, source, tmp_path / "cache")
    source.path.write_bytes(b"a-new-and-longer-source")
    with pytest.raises(ValueError, match="Source clip changed"):
        cached[0]


def test_context_frozen_hash_is_still_required_for_warm_features(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin", context=True)
    prepared(model, source, tmp_path / "cache")
    source.path.write_bytes(b"tampered-source")
    with pytest.raises(ValueError, match="Context clip hash mismatch"):
        prepared(model, source, tmp_path / "cache")


@pytest.mark.parametrize("invalid", ["model_type", "subclass", "unfrozen", "partial", "manual_grad", "training_encoder"])
def test_feature_cache_refuses_unsupported_or_unfrozen_models(tmp_path, invalid):
    model = tiny_model()
    if invalid == "model_type":
        model = nn.Linear(2, 2)
    elif invalid == "subclass":
        class Derived(HybridFusionEventClassifier):
            pass
        model.__class__ = Derived
    elif invalid == "unfrozen":
        model = tiny_model(freeze_backbone=False)
    elif invalid == "partial":
        model = tiny_model(unfreeze_final_backbone_stages=True)
    elif invalid == "manual_grad":
        model.backbone.projection.weight.requires_grad = True
    else:
        model.backbone.train()
    with pytest.raises(ValueError, match="base hybrid|frozen|evaluation-mode"):
        prepared(model, ClipSource(tmp_path / "clip.bin"), tmp_path / "cache")


def test_read_checks_semantic_context_mask_even_with_consistent_checksums(tmp_path):
    model = tiny_model()
    source = ClipSource(tmp_path / "clip.bin", context=True)
    cached, _ = prepared(model, source, tmp_path / "cache")
    path, _, _, _ = cached.entries[0]
    with np.load(path, allow_pickle=False) as entry:
        arrays = {name: entry[name] for name in entry.files}
    arrays["mask"][:] = True
    header = json.loads(arrays["header"].tobytes().decode())
    header["sha256"]["mask"] = feature_cache._digest(arrays["mask"])
    arrays["header"] = np.frombuffer(feature_cache._json(header), dtype=np.uint8)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="mask|shape"):
        cached[0]


class TrainingClips(ClipSource):
    def __init__(self, directory, prefix):
        directory.mkdir()
        super().__init__(directory / "original.bin")
        self.consume_rng = False
        self.paths = []
        self.rows = []
        for index in range(4):
            video, label = index // 2, index // 2
            path = directory / f"clip-{index}.bin"
            path.write_bytes(f"fixed source {prefix} {index}".encode())
            self.paths.append(path)
            self.rows.append({
                "event_id": f"{prefix}-event-{index}", "video_id": f"{prefix}-video-{video}",
                "source_group": f"{prefix}-group-{video}", "class_id": str(label),
                "dataset_split": "train" if prefix == "train" else "val",
                "manipulation_modality": "real" if label == 0 else "visual_modified",
                "phoneme": "p",
            })

    def clip_source(self, index):
        return self.paths[index], None, None

    def sample_metadata(self, index):
        row = self.rows[index]
        return {**super().sample_metadata(index),
                "video_label": torch.tensor(int(row["class_id"])),
                "manipulation_modality": row["manipulation_modality"]}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["video"] = sample["video"] * (0.6 + index / 10)
        return sample


def test_bounded_feature_plan_never_decodes_unselected_clips(tmp_path):
    source = TrainingClips(tmp_path / 'source', 'train')
    (cached,), report = feature_cache.prepare_feature_datasets(
        tiny_model(), [source], tmp_path / 'features', torch.device('cpu'),
        index_plans=([0, 2],),
    )
    assert source.decode_calls == report['built'] == 2
    assert len(cached) == 4 and cached.rows is source.rows
    assert cached[0]['event_id'] == source.rows[0]['event_id']
    assert cached[2]['event_id'] == source.rows[2]['event_id']
    with pytest.raises(ValueError, match='outside.*plan'):
        cached[1]


@pytest.mark.parametrize('invalid', [{'batch_size': 2}, {'amp': True}, {'freeze_backbone': False}])
def test_training_refuses_incompatible_feature_cache_execution(tmp_path, invalid):
    from seepat.training.runtime import TrainingRuntimeOptions
    from seepat.training.train import TrainingOptions, train_model

    training = TrainingClips(tmp_path / 'train', 'train')
    validation = TrainingClips(tmp_path / 'val', 'val')
    options = TrainingOptions(**{'amp': False, 'freeze_backbone': True, **invalid})
    with pytest.raises(ValueError, match='Feature caching requires'):
        train_model(tiny_model(), training, validation, tmp_path / 'run', options,
                    torch.device('cpu'), {'model': 'guard-test'},
                    runtime=TrainingRuntimeOptions(feature_cache_dir=tmp_path / 'features'))


@pytest.mark.parametrize('workers', [0, 1])
def test_bounded_cached_event_readiness_limits_prefetch_and_preparation(tmp_path, workers):
    from seepat.training.runtime import TrainingRuntimeOptions
    from seepat.training.train import TrainingOptions, train_model

    training = TrainingClips(tmp_path / 'train', 'train')
    validation = TrainingClips(tmp_path / 'val', 'val')
    options = TrainingOptions(
        epochs=1, batch_size=1, workers=workers, sequence_length=4, image_size=8,
        amp=False, freeze_backbone=True, class_weighting='balanced_global',
        max_train_batches=2, max_validation_batches=2,
        selection_metric='validation_video_balanced_accuracy',
    )
    report = train_model(
        tiny_model(), training, validation, tmp_path / 'run', options, torch.device('cpu'),
        {'model': 'bounded-tiny-fusion'},
        runtime=TrainingRuntimeOptions(feature_cache_dir=tmp_path / 'features'),
    )
    assert report['runtime']['feature_cache']['built'] == 4
    assert training.decode_calls == validation.decode_calls == 2
    assert report['processed_train_events'] == report['processed_validation_events'] == 2


def test_actual_matched_contexts_keep_missing_evidence_and_training_parity(tmp_path):
    from test_matched_training import synthetic_export

    from seepat.training.matched_inputs import MatchedClipDataset
    from seepat.training.runtime import TrainingRuntimeOptions
    from seepat.training.train import TrainingOptions, train_model

    _, directory, profile = synthetic_export(tmp_path)
    datasets = [MatchedClipDataset(directory / f'candidate_{split}.csv', tmp_path,
                                   split, profile, image_size=8) for split in ('train', 'val')]
    torch.manual_seed(501)
    initial = tiny_model()
    options = TrainingOptions(
        epochs=1, batch_size=1, sequence_length=16, image_size=8, amp=False,
        freeze_backbone=True, supervision='video_max', class_weighting='balanced_global',
        gradient_accumulation_steps=2, early_stopping_patience=0, seed=46,
    )
    contract = {'model': 'matched-tiny-fusion', 'clip_input_contract': datasets[0].input_contract}
    for name, runtime in [('rgb', None), ('cached', TrainingRuntimeOptions(
            feature_cache_dir=tmp_path / 'features', persistent_train_workers=True))]:
        train_model(deepcopy(initial), *datasets, tmp_path / name, options, torch.device('cpu'),
                    contract, runtime=runtime)
    reference, optimized = [torch.load(tmp_path / name / 'checkpoint_last.pt', weights_only=False)
                            for name in ('rgb', 'cached')]
    for key in ('model_state', 'optimizer_state', 'history', 'rng_state'):
        assert_nested_equal(reference[key], optimized[key])
    assert optimized['history'][0]['train']['context_clips_processed'] == 4
    assert optimized['history'][0]['validation']['context_clips_processed'] == 4
    (cached,), _ = feature_cache.prepare_feature_datasets(
        deepcopy(initial), [datasets[0]], tmp_path / 'features', torch.device('cpu'),
    )
    index = next(i for i, row in enumerate(cached.rows) if row['kind'] == 'context')
    sample = cached[index]
    assert sample['label'].item() == -1 and sample['phoneme'] == ''
    assert not sample['evidence_features'].any() and not sample['evidence_feature_mask'].any()
    assert not sample['features'].any() and not sample['feature_mask'].any()


def assert_nested_equal(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_cached_training_and_resume_match_rgb_with_dropout_and_complete_videos(tmp_path):
    from dataclasses import replace

    from seepat.training.runtime import TrainingRuntimeOptions
    from seepat.training.train import TrainingOptions, train_model

    training = TrainingClips(tmp_path / "train-data", "train")
    validation = TrainingClips(tmp_path / "val-data", "val")
    torch.manual_seed(301)
    initial = tiny_model()
    options = TrainingOptions(
        epochs=2, batch_size=1, workers=0, sequence_length=4, image_size=8,
        amp=False, freeze_backbone=True, supervision="video_max",
        class_weighting="balanced_global", gradient_accumulation_steps=2,
        early_stopping_patience=0, seed=101,
        selection_metric="validation_video_balanced_accuracy",
    )
    runtime = TrainingRuntimeOptions(feature_cache_dir=tmp_path / "features",
                                     persistent_train_workers=True)
    direct_dir, cached_dir, resume_dir = (tmp_path / name for name in ("direct", "cached", "resumed"))
    contract = {"model": "swin3d_b_vild_fusion", "purpose": "synthetic-parity-only"}
    train_model(deepcopy(initial), training, validation, direct_dir, options,
                torch.device("cpu"), contract)
    train_model(deepcopy(initial), training, validation, cached_dir, options,
                torch.device("cpu"), contract, runtime=runtime)
    train_model(deepcopy(initial), training, validation, resume_dir, replace(options, epochs=1),
                torch.device("cpu"), contract, runtime=runtime)
    train_model(deepcopy(initial), training, validation, resume_dir, options,
                torch.device("cpu"), contract,
                resume_from=resume_dir / "checkpoint_last.pt", runtime=runtime)
    checkpoints = [torch.load(directory / "checkpoint_last.pt", weights_only=False)
                   for directory in (direct_dir, cached_dir, resume_dir)]
    for other in checkpoints[1:]:
        for key in ("model_state", "optimizer_state", "rng_state", "global_step"):
            assert_nested_equal(checkpoints[0][key], other[key])
    histories = [json.loads((directory / "history.json").read_text())
                 for directory in (direct_dir, cached_dir, resume_dir)]
    assert histories[0] == histories[1] == histories[2]
    assert checkpoints[0]["global_step"] == 2
