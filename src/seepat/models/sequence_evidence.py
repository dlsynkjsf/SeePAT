"""Isolated sequence-evidence candidate; no exports, downloads or experiment runner."""

from __future__ import annotations

import json
import math
from itertools import pairwise

import torch
from torch import nn

from seepat.data.sampling import canonical_source_group
from seepat.evidence import EVIDENCE_VERSION, FUSION_EVIDENCE_FIELDS
from seepat.preprocessing.face import timestamp_to_frame
from seepat.training.dataset import sequence_frame_indices

TIMING_VERSION = "fusion-vild-sequence-v1"
MODEL_VERSION = "swin-vild-sequence-candidate-v1"
TIMING_FIELDS = ("normalized_vild", "from_phone_start_s", "from_phone_end_s")
FRAME_WINDOW_POLICY = "inclusive_producer_frame_and_clock_bounds_v2"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Timing metadata must be finite numbers") from error
    if not math.isfinite(result):
        raise ValueError("Timing metadata must be finite numbers")
    return result


def _frame_ids(value):
    result = json.loads(value) if isinstance(value, str) else value
    if (not isinstance(result, list) or not result
            or any(type(i) is not int or i < 0 for i in result)
            or any(a >= b for a, b in pairwise(result))):
        raise ValueError("Crop frame indices must be nonempty, ordered and unique")
    return result


def event_timing_sequence(row, trace, *, decoded_frame_count, sequence_length=16):
    """Bind recorded VILD to the actual decoded RGB slots, without reading labels.

    The caller supplies an already verified trace and the decoder's original
    frame count. This function checks their event/clock/index agreement; it does
    not authenticate a source manifest or hash/read any project file.
    """
    if type(decoded_frame_count) is not int or type(sequence_length) is not int:
        raise ValueError("Frame counts must be integers")
    slots, unpadded = sequence_frame_indices(decoded_frame_count, sequence_length)
    if row.get("dataset_split") not in {"train", "val"}:
        raise ValueError("Candidate preparation accepts Train/Validation development only")
    result = {
        "timing_version": TIMING_VERSION,
        "timing_features": torch.zeros(sequence_length, len(TIMING_FIELDS)),
        "timing_mask": torch.zeros(sequence_length, dtype=torch.bool),
        "timing_frame_indices": (-1,) * sequence_length,
        "timing_unavailable_reason": "",
    }
    if row.get("kind", "bilabial") == "context":
        if row.get("phoneme"):
            raise ValueError("Context cannot fabricate a phoneme anchor")
        result["timing_unavailable_reason"] = "unanchored_context"
        return result
    if row.get("kind", "bilabial") != "bilabial" or row.get("phoneme", "").lower() not in {"m", "b", "p"}:
        raise ValueError("Timing evidence requires a bilabial event")
    if trace is None:
        result["timing_unavailable_reason"] = "missing_verified_trace"
        return result
    for field, source in (("video_id", "video_id"), ("dataset_split", "dataset_split"), ("file", "source_file")):
        if not row.get(field) or row[field] != trace.get(source):
            raise ValueError(f"Trace identity mismatch: {field}")
    if not row.get("source_group") or not trace.get("source_group") or canonical_source_group(
        row["source_group"], None,
    ) != canonical_source_group(trace["source_group"], None):
        raise ValueError("Trace source group mismatch")
    if trace.get("artifact_version") not in {"vild-trace-v1", "vild-trace-v2"}:
        raise ValueError("Unsupported trace version")
    key = row.get("vild_trace_event_key") or row["event_id"]
    windows = [w for w in trace["bilabial_event_windows"] if w["event_id"] == key]
    if len(windows) != 1 or windows[0]["phoneme"].lower() != row["phoneme"].lower():
        raise ValueError("Trace event/phoneme mismatch")
    window = windows[0]
    ids = _frame_ids(row.get("mouth_crop_frame_indices_json"))
    if ids != _frame_ids(window["mouth_clip_frame_indices"]) or len(ids) != decoded_frame_count:
        raise ValueError("Recorded crop indices disagree with trace or decoded frame count")
    onset, end = (_number(row[f"video_phone_{p}_s"]) for p in ("start", "end"))
    offset = _number(row["audio_video_start_offset_s"])
    for part, actual in (("start", onset), ("end", end)):
        audio = _number(row[f"phone_{part}_s"])
        expected = (_number(window[f"audio_phone_{part}_s"]),
                    _number(window[f"video_phone_{part}_s"]), audio + offset)
        if (abs(audio - expected[0]) > 1e-6 or abs(actual - expected[1]) > 1e-6
                or abs(actual - expected[2]) > 1e-6):
            raise ValueError("Audio/video phoneme clocks disagree")
    if end < onset or abs(offset - _number(trace["timing"]["audio_video_start_offset_s"])) > 1e-6:
        raise ValueError("Audio/video phoneme clocks disagree")
    frames = trace["frames"]
    frame_ids = [f["frame_index"] for f in frames]
    times = [_number(f["timestamp_s"]) for f in frames]
    if (any(type(i) is not int or i < 0 for i in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or any(a >= b for a, b in pairwise(frame_ids))
            or any(a >= b for a, b in pairwise(times))):
        raise ValueError("Trace frames must have unique ordered indices and timestamps")
    lookup = dict(zip(frame_ids, frames, strict=True))
    if any(i not in lookup for i in ids):
        raise ValueError("A recorded crop frame is absent from the trace")
    start, stop = (_number(window[f"window_{p}_s"]) for p in ("start", "end"))
    fps = _number(trace["timing"].get("fps"))
    if fps <= 0 or not 0 <= start <= stop:
        raise ValueError("Trace FPS/window bounds must be positive and ordered")
    # Match the crop producer's inclusive floor/ceil bounds, including edge frames.
    first, last = timestamp_to_frame(start, fps), timestamp_to_frame(stop, fps, round_up=True)
    if any(not first <= i <= last for i in ids):
        raise ValueError("Crop frame is outside its recorded rounded frame bounds")
    if any(not first / fps - 1e-6 <= _number(lookup[i]["timestamp_s"]) <= last / fps + 1e-6 for i in ids):
        raise ValueError("Crop frame is outside its recorded event window")
    result["timing_frame_indices"] = tuple(ids[i] for i in slots)
    for slot, (index, valid) in enumerate(zip(slots, unpadded, strict=True)):
        frame = lookup[ids[index]]
        opening = frame.get("normalized_vild")
        if not valid or type(frame.get("face_count")) is not int or frame["face_count"] != 1:
            continue
        try:
            opening = _number(opening)
        except ValueError:
            continue
        if opening < 0:
            raise ValueError("Observed normalized VILD cannot be negative")
        time = _number(frame["timestamp_s"])
        result["timing_features"][slot] = torch.tensor([opening, time - onset, time - end])
        result["timing_mask"][slot] = True
    if not result["timing_mask"].any():
        result["timing_unavailable_reason"] = "no_valid_sampled_measurements"
    return result


class MaskedTimingEncoder(nn.Module):
    """Encode local VILD/timing patterns; missing slots contribute exactly zero."""

    def __init__(self, output_features: int):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(3, 16, 3, padding=1), nn.GELU())
        self.output = nn.Linear(16, output_features, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, values, mask):
        if (values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != 3
                or mask.shape != values.shape[:2] or mask.dtype != torch.bool
                or mask.device != values.device or not values.is_floating_point()):
            raise ValueError("Timing input requires floating B x T x 3 and boolean B x T mask")
        clean = torch.where(mask.unsqueeze(-1), values, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Available timing measurements must be finite")
        local = self.local(clean.transpose(1, 2)).transpose(1, 2)
        pooled = torch.where(mask.unsqueeze(-1), local, 0.0).sum(1)
        pooled = pooled / mask.sum(1, keepdim=True).clamp_min(1)
        projected = self.output(pooled)
        return torch.where(mask.any(1, keepdim=True), projected, 0.0)


class SequenceAwareFusion(nn.Module):
    """Add timing evidence to an explicitly supplied original fusion model."""

    uses_evidence_features = True
    uses_frame_mask = True
    uses_timing_features = True
    evidence_fields = FUSION_EVIDENCE_FIELDS

    def __init__(self, base):
        super().__init__()
        if not base.uses_evidence_features or tuple(base.evidence_fields) != FUSION_EVIDENCE_FIELDS:
            raise ValueError("Sequence candidate requires the original seven-field evidence branch")
        self.base = base
        self.timing_encoder = MaskedTimingEncoder(base.classifier.in_features)

    def get_extra_state(self):
        return {
            "model_version": MODEL_VERSION, "timing_version": TIMING_VERSION,
            "timing_fields": list(TIMING_FIELDS), "evidence_version": EVIDENCE_VERSION,
            "fusion_dimension": self.base.classifier.in_features,
            "encoder": "conv3x16-k3-gelu-masked-mean-zero-linear-no-bias-v1",
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Sequence candidate input contract differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True:
            raise ValueError("Sequence candidate loading requires strict=True")
        self.set_extra_state(state_dict.get("_extra_state"))
        expected = self.state_dict()
        if expected.keys() != state_dict.keys():
            raise ValueError("Sequence candidate state keys differ from its contract")
        for name, value in expected.items():
            supplied = state_dict[name]
            if isinstance(value, torch.Tensor) and (
                not isinstance(supplied, torch.Tensor)
                or supplied.shape != value.shape or supplied.dtype != value.dtype
            ):
                raise ValueError(f"Sequence candidate tensor contract differs: {name}")
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def forward(self, video, frame_mask=None, features=None, feature_mask=None,
                *, timing_features=None, timing_mask=None):
        if timing_features is None or timing_mask is None:
            raise ValueError("Sequence candidate requires explicit timing features and mask")
        if not isinstance(timing_features, torch.Tensor) or not isinstance(timing_mask, torch.Tensor):
            raise TypeError("Timing features and mask must be tensors")
        if video.ndim != 5 or timing_features.shape[:2] != (video.shape[0], video.shape[2]):
            raise ValueError("Timing slots must match the RGB batch and frames")
        if frame_mask is not None:
            if frame_mask.shape != timing_mask.shape or frame_mask.dtype != torch.bool:
                raise ValueError("RGB and timing frame masks must share shape B x T")
            if (timing_mask & ~frame_mask).any():
                raise ValueError("Padded RGB slots cannot contain available timing evidence")
        contribution = self.timing_encoder(timing_features, timing_mask)
        tokens = self.base.extract_modality_tokens(video, frame_mask, features, feature_mask)
        tokens["evidence"] = tokens["evidence"] + contribution
        return self.base.classifier(self.base.fusion(torch.stack(list(tokens.values()), dim=1)))


def forward_sequence_model(model, videos, batch, device):
    """Forward only explicitly versioned batches for a timing-aware model."""
    versions = batch.get("timing_version")
    if (not isinstance(model, SequenceAwareFusion) or not isinstance(versions, (list, tuple))
            or list(versions) != [TIMING_VERSION] * videos.shape[0]):
        raise ValueError("Every sequence batch must declare the candidate input version")
    names = {
        "frame_mask": "frame_mask", "features": "evidence_features",
        "feature_mask": "evidence_feature_mask", "timing_features": "timing_features",
        "timing_mask": "timing_mask",
    }
    kwargs = {}
    for argument, field in names.items():
        value = batch.get(field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Sequence batch requires tensor {field}")
        if "mask" in field and value.dtype != torch.bool:
            raise ValueError(f"Sequence batch requires boolean {field}")
        kwargs[argument] = value.to(device=device)
    return model(videos, **kwargs)
