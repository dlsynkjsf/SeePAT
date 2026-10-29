"""Content-bound FP32 encoder features; labels and numerical evidence stay live."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from seepat.artifacts import file_sha256
from seepat.models.fusion import HybridFusionEventClassifier
from seepat.training.clip_cache import _contract as clip_contract
from seepat.training.clip_cache import sequence_frame_indices

FEATURE_CACHE_VERSION = "frozen-visual-encoders-v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode('utf-8')


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def encoder_binding(model, device):
    if type(model) is not HybridFusionEventClassifier:
        raise ValueError("Feature caching supports the base hybrid fusion model only")
    if (not model.freeze_backbone or model.unfreeze_final_backbone_stages
            or any(p.requires_grad for branch in (model.backbone, model.frame_backbone)
                   for p in branch.parameters())):
        raise ValueError("Feature caching requires fully frozen encoders")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Feature caching requires FP32 model parameters")
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in model.buffers()):
        raise ValueError("Feature caching requires FP32 model buffers")
    branches = {}
    for name in ('backbone', 'frame_backbone'):
        branch = getattr(model, name)
        if branch.training:
            raise ValueError("Feature caching requires evaluation-mode encoders")
        digest = hashlib.sha256()
        for key, value in branch.state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(_json([key, str(array.dtype), list(array.shape)]))
            digest.update(array.tobytes())
        branches[name] = {
            'class': f'{type(branch).__module__}.{type(branch).__qualname__}',
            'architecture': repr(branch),
            'sha256': digest.hexdigest(),
        }
    return {
        'version': FEATURE_CACHE_VERSION, 'encoders': branches,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'device': str(device), 'cuda': torch.version.cuda,
        'implementation_sha256': file_sha256(Path(__file__).parents[1] / 'models' / 'fusion.py'),
        'cache_implementation_sha256': file_sha256(Path(__file__)),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'normalization': {
            name: getattr(model, name).detach().cpu().flatten().tolist()
            for name in ('pixel_mean', 'pixel_std', 'frame_pixel_mean', 'frame_pixel_std')
        },
        'dtype': 'float32', 'input_policy': 'selected-rgb-v1; deterministic; no augmentation',
        'swin_width': model.swin_projection.in_features,
        'frame_width': model.temporal_encoder.input_projection[0].in_features,
    }


def _read(path, identity):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {'header', 'swin', 'frames', 'mask', 'decoded'}:
            raise ValueError("Invalid encoder cache fields")
        header = json.loads(stored['header'].tobytes().decode('utf-8'))
        arrays = {name: stored[name] for name in ('swin', 'frames', 'mask', 'decoded')}
    if not isinstance(header, dict):
        raise TypeError("Invalid encoder cache header")
    if header.get('identity') != identity:
        raise ValueError("Encoder cache identity changed")
    if header.get('sha256') != {k: _digest(v) for k, v in arrays.items()}:
        raise ValueError("Encoder cache payload checksum mismatch")
    swin, frames, mask, decoded = (arrays[k] for k in ('swin', 'frames', 'mask', 'decoded'))
    length = identity['sequence_length']
    if (swin.dtype != np.float32 or swin.shape != (identity['swin_width'],)
            or frames.dtype != np.float32 or frames.shape != (length, identity['frame_width'])
            or mask.dtype != np.bool_ or mask.shape != (length,)
            or decoded.dtype != np.int64 or decoded.shape != (1,) or decoded[0] < 1
            or not np.isfinite(swin).all() or not np.isfinite(frames).all()):
        raise ValueError("Invalid encoder cache shape, dtype or values")
    expected_mask = identity['context_frame_mask']
    if expected_mask is None:
        expected_mask = sequence_frame_indices(int(decoded[0]), length)[1]
    elif decoded[0] != 16:
        raise ValueError("Context cache must contain exactly 16 decoded slots")
    if mask.tolist() != expected_mask:
        raise ValueError("Encoder cache frame mask differs from the clip policy")
    return arrays


def _write(path, identity, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {'identity': identity, 'sha256': {k: _digest(v) for k, v in arrays.items()}}
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez(stream, header=np.frombuffer(_json(header), dtype=np.uint8), **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FrozenFeatureDataset(Dataset):
    """Read compact CPU features; never carry a GPU model into worker processes."""

    def __init__(self, source, entries, binding):
        self.source, self.entries, self.binding = source, entries, binding
        self.rows = source.rows
        self.calibration_contract = getattr(source, 'calibration_contract', None)
        self.input_contract = getattr(source, 'input_contract', None)

    def __len__(self):
        return len(self.source)

    def evidence_coverage(self):
        return self.source.evidence_coverage()

    def __getitem__(self, index):
        if self.entries[index] is None:
            raise ValueError("Clip is outside the prepared feature-cache run plan")
        path, identity, source_path, source_stat = self.entries[index]
        current_stat = source_path.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != source_stat:
            raise ValueError("Source clip changed after encoder-cache preparation")
        if file_sha256(source_path) != identity['clip_sha256']:
            raise ValueError("Source clip hash changed after encoder-cache preparation")
        arrays = _read(path, identity)
        return {
            **self.source.sample_metadata(index),
            # No pixel allocation or decode on the feature path. The trainer
            # dispatches through forward_from_encoder_features instead.
            'video': torch.empty((3, identity['sequence_length'], 0, 0)),
            'frozen_swin_features': torch.from_numpy(arrays['swin']),
            'frozen_frame_features': torch.from_numpy(arrays['frames']),
            'frame_mask': torch.from_numpy(arrays['mask']),
            'decoded_frame_count': int(arrays['decoded'][0]),
        }


def prepare_feature_datasets(model, datasets, directory: Path, device, progress=None,
                             *, index_plans=None):
    """Build missing entries once after warm-start/resume restores encoder weights."""
    binding = encoder_binding(model, device)
    binding_hash = hashlib.sha256(_json(binding)).hexdigest()
    namespace = directory / binding_hash
    results, hits, misses = [], 0, 0
    started = perf_counter()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if index_plans is None:
            index_plans = [range(len(source)) for source in datasets]
        if len(index_plans) != len(datasets):
            raise ValueError("Feature-cache plans must match dataset roles")
        for source, plan in zip(datasets, index_plans, strict=True):
            if not all(callable(getattr(source, name, None))
                       for name in ('sample_metadata', 'clip_source')):
                raise ValueError("Feature cache requires a deterministic clip dataset")
            entries = [None] * len(source)
            plan = sorted(set(plan))
            if any(type(index) is not int or not 0 <= index < len(source) for index in plan):
                raise ValueError("Feature-cache plan has invalid indices")
            for completed, index in enumerate(plan, 1):
                source_path, expected, context_mask = source.clip_source(index)
                clip_hash = file_sha256(source_path)
                if expected is not None and clip_hash != expected:
                    raise ValueError("Context clip hash mismatch")
                identity = {
                    'encoder_binding_sha256': binding_hash, 'clip_sha256': clip_hash,
                    'sequence_length': source.sequence_length, 'image_size': source.image_size,
                    'context_frame_mask': context_mask,
                    'swin_width': binding['swin_width'], 'frame_width': binding['frame_width'],
                    'decoded_input': clip_contract(clip_hash, source.sequence_length,
                                                   source.image_size, context_mask),
                }
                path = namespace / (hashlib.sha256(_json(identity)).hexdigest() + '.npz')
                try:
                    _read(path, identity)
                except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):
                    sample = source[index]
                    with torch.no_grad():
                        swin, frames = model.extract_frozen_features(
                            # Match DataLoader's stacked, contiguous BCTHW input.
                            # Direct unsqueeze can retain the RGB array's
                            # channels-last strides and select different kernels.
                            torch.stack([sample['video']], dim=0).to(device)
                        )
                    arrays = {
                        'swin': swin[0].detach().cpu().float().numpy().copy(),
                        'frames': frames[0].detach().cpu().float().numpy().copy(),
                        'mask': sample['frame_mask'].cpu().numpy().copy(),
                        'decoded': np.array([sample['decoded_frame_count']], dtype=np.int64),
                    }
                    if file_sha256(source_path) != clip_hash:
                        raise ValueError("Source clip changed during encoder-cache preparation")
                    _write(path, identity, arrays)
                    _read(path, identity)
                    misses += 1
                else:
                    hits += 1
                stat = source_path.stat()
                entries[index] = (path, identity, source_path, (stat.st_size, stat.st_mtime_ns))
                if progress is not None:
                    progress('prepare frozen encoder cache', completed, len(plan),
                             str(source.rows[index]['event_id']))
            results.append(FrozenFeatureDataset(source, entries, binding))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return results, {
        'version': FEATURE_CACHE_VERSION, 'binding': binding, 'hits': hits, 'built': misses,
        'preparation_seconds': round(perf_counter() - started, 6),
    }
