"""Audio-Visual Temporal Fusion classifier (Figure 4.1 center block).

The fusion model combines three modality branches:

1. **Video Swin-Base** (3D patch embedding + shifted-window transformer) over
   the bilabial mouth-event clip, producing Swin temporal features.
2. **CNN-temporal embedding** (EfficientNetV2-S frame features + TempCNN),
   producing the spatial-temporal embedding of the same clip.
3. **Biological alignment evidence** (VILD regression residual, phoneme-viseme
   residual, Isolation Forest anomaly score, timing and closure measurements)
   as a masked numerical branch.

A cross-modal fusion block attends over the three modality tokens with a
dedicated fusion token. A classifier emits binary event logits (softmax at
inference). The measured closure offset is descriptive timing evidence in
seconds; there is no separately supervised sync-gap anomaly head.

The evidence vector contract lives in :mod:`seepat.evidence`; field order and
width are frozen per training version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.models.video import Swin3D_B_Weights, swin3d_b

from seepat.artifacts import atomic_write_json
from seepat.evidence import CLOSURE_OFFSET_DEFINITION, EVIDENCE_VERSION, FUSION_EVIDENCE_FIELDS
from seepat.models.cnn_temporal import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TemporalConvEncoder,
    _classifier_input_features,
)

FUSION_MODEL_NAME = "swin3d_b_vild_fusion"
KINETICS_MEAN = (0.4850, 0.4560, 0.4060)
KINETICS_STD = (0.2290, 0.2240, 0.2250)


class CrossModalFusionBlock(nn.Module):
    """Self-attention across modality tokens with a dedicated fusion token."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dimension < 1 or heads < 1:
            raise ValueError("dimension and heads must be positive")
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.normalization = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )
        self.output_normalization = nn.LayerNorm(dimension)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape B x M x D")
        query = self.fusion_token.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        fused = self.normalization(query + attended)
        fused = self.output_normalization(fused + self.feed_forward(fused))
        return fused.squeeze(1)


class HybridFusionEventClassifier(nn.Module):
    """Hybrid CNN-Transformer temporal fusion classifier for mouth events."""

    uses_frame_mask = True
    uses_evidence_features = True
    evidence_fields = FUSION_EVIDENCE_FIELDS

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        swin_backbone: nn.Module | None = None,
        frame_backbone: nn.Module | None = None,
        evidence_features: int = len(FUSION_EVIDENCE_FIELDS),
        evidence_hidden: int = 64,
        temporal_channels: int = 256,
        temporal_layers: int = 2,
        temporal_kernel_size: int = 3,
        embedding_features: int = 256,
        fusion_dimension: int = 256,
        fusion_heads: int = 8,
        dropout: float = 0.2,
        use_evidence: bool = True,
    ) -> None:
        super().__init__()
        if evidence_features < 1 or evidence_hidden < 1:
            raise ValueError("evidence feature counts must be positive")

        if swin_backbone is None:
            weights = Swin3D_B_Weights.KINETICS400_V1 if pretrained else None
            swin_backbone = swin3d_b(weights=weights)
        swin_head = getattr(swin_backbone, "head", None)
        swin_feature_count = getattr(swin_head, "in_features", None)
        if not isinstance(swin_feature_count, int):
            raise TypeError("Swin backbone must expose head.in_features")
        swin_backbone.head = nn.Identity()

        if frame_backbone is None:
            frame_weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            frame_backbone = efficientnet_v2_s(weights=frame_weights)
        frame_feature_count = _classifier_input_features(frame_backbone)
        frame_backbone.classifier = nn.Identity()

        self.backbone = swin_backbone
        self.frame_backbone = frame_backbone
        self.evidence_feature_count = evidence_features
        self.uses_evidence_features = use_evidence
        self.evidence_fields = FUSION_EVIDENCE_FIELDS if use_evidence else ()
        self.temporal_encoder = TemporalConvEncoder(
            input_features=frame_feature_count,
            temporal_channels=temporal_channels,
            embedding_features=embedding_features,
            layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.swin_projection = nn.Linear(swin_feature_count, fusion_dimension)
        self.temporal_projection = nn.Linear(embedding_features, fusion_dimension)
        self.evidence_branch = nn.Sequential(
            nn.Linear(2 * evidence_features, evidence_hidden),
            nn.LayerNorm(evidence_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_hidden, fusion_dimension),
            nn.GELU(),
        ) if use_evidence else None
        self.fusion = CrossModalFusionBlock(fusion_dimension, fusion_heads, dropout)
        self.classifier = nn.Linear(fusion_dimension, 2)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "frame_pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "frame_pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for branch in (self.backbone, self.frame_backbone):
                for parameter in branch.parameters():
                    parameter.requires_grad = False
                branch.eval()

    def train(self, mode: bool = True) -> HybridFusionEventClassifier:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            self.frame_backbone.eval()
        return self

    def _swin_features(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape B x 3 x T x H x W")
        normalized = (video - self.pixel_mean) / self.pixel_std
        features = self.backbone(normalized)
        if features.ndim != 2:
            raise RuntimeError("Swin backbone returned an unexpected feature shape")
        return features

    def _temporal_features(self, video: Tensor, frame_mask: Tensor | None) -> Tensor:
        batch_size, _, frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames,
            3,
            height,
            width,
        )
        normalized = (frame_batch - self.frame_pixel_mean) / self.frame_pixel_std
        gradients_enabled = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.set_grad_enabled(gradients_enabled):
            frame_features = self.frame_backbone(normalized)
        if frame_features.ndim != 2:
            raise RuntimeError("Frame backbone returned an unexpected feature shape")
        return self.temporal_encoder(
            frame_features.reshape(batch_size, frames, -1),
            frame_mask,
        )

    def _evidence_embedding(self, features: Tensor, feature_mask: Tensor) -> Tensor:
        if features.ndim != 2 or feature_mask.shape != features.shape:
            raise ValueError("evidence features and mask must share shape B x E")
        if features.shape[1] != self.evidence_feature_count:
            raise ValueError(
                "evidence features must have width "
                f"{self.evidence_feature_count}, received {features.shape[1]}"
            )
        mask = feature_mask.to(device=features.device, dtype=torch.bool)
        masked = torch.where(mask, features, 0.0)
        if not torch.isfinite(masked).all():
            raise ValueError("Available evidence features must be finite")
        return self.evidence_branch(torch.cat([masked, mask.to(features.dtype)], dim=1))

    def extract_modality_tokens(
        self,
        video: Tensor,
        frame_mask: Tensor | None = None,
        features: Tensor | None = None,
        feature_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return the projected Swin, CNN-temporal and evidence tokens."""
        evidence_token = None
        if self.uses_evidence_features:
            if features is None or feature_mask is None:
                raise ValueError("Fusion requires explicit evidence features and per-feature masks")
            if features.shape[0] != video.shape[0]:
                raise ValueError("Evidence and video batch sizes must match")
            evidence_token = self._evidence_embedding(features, feature_mask)
        swin_token = self.swin_projection(self._swin_features(video))
        temporal_token = self.temporal_projection(self._temporal_features(video, frame_mask))
        tokens = {
            "swin": swin_token,
            "temporal": temporal_token,
        }
        if evidence_token is not None:
            tokens["evidence"] = evidence_token
        return tokens

    def forward(
        self,
        video: Tensor,
        frame_mask: Tensor | None = None,
        features: Tensor | None = None,
        feature_mask: Tensor | None = None,
    ) -> Tensor:
        tokens = self.extract_modality_tokens(video, frame_mask, features, feature_mask)
        stacked = torch.stack(list(tokens.values()), dim=1)
        fused = self.fusion(stacked)
        return self.classifier(fused)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.models.fusion")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.frames < 1 or args.image_size < 1:
        raise ValueError("frames and image-size must be positive")

    device = torch.device(args.device)
    model = HybridFusionEventClassifier(pretrained=False).to(device).eval()
    event_id = None
    label = None
    frame_mask = None
    if args.manifest is not None:
        from seepat.training.dataset import MouthEventDataset

        dataset = MouthEventDataset(
            manifest_path=args.manifest,
            project_root=args.project_root,
            sequence_length=args.frames,
            image_size=args.image_size,
            require_calibration=True,
        )
        event = dataset[0]
        sample = event["video"].unsqueeze(0).to(device)
        features = event["evidence_features"].unsqueeze(0).to(device)
        feature_mask = event["evidence_feature_mask"].unsqueeze(0).to(device)
        frame_mask = event["frame_mask"].unsqueeze(0).to(device)
        event_id = event["event_id"]
        label = int(event["label"].item())
    else:
        sample = torch.zeros(
            (1, 3, args.frames, args.image_size, args.image_size),
            device=device,
        )
        features = torch.zeros((1, len(FUSION_EVIDENCE_FIELDS)), device=device)
        feature_mask = torch.zeros_like(features, dtype=torch.bool)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        logits = model(
            sample,
            frame_mask=frame_mask,
            features=features,
            feature_mask=feature_mask,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report = {
        "purpose": "wiring_check_only_not_a_reported_experiment",
        "model": FUSION_MODEL_NAME,
        "pretrained": False,
        "device": str(device),
        "input_shape": list(sample.shape),
        "evidence_fields": list(FUSION_EVIDENCE_FIELDS),
        "evidence_version": EVIDENCE_VERSION,
        "evidence_shape": list(features.shape),
        "evidence_mask": feature_mask.squeeze(0).tolist(),
        "valid_frames": int(frame_mask.sum().item()) if frame_mask is not None else args.frames,
        "closure_offset_definition": CLOSURE_OFFSET_DEFINITION,
        "output_shape": list(logits.shape),
        "event_id": event_id,
        "label": label,
        "parameters": parameter_counts(model),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    if not torch.isfinite(logits).all():
        raise RuntimeError("Fusion forward pass returned non-finite logits")
    if args.report is not None:
        atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
