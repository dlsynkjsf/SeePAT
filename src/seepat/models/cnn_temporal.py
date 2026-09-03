from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from seepat.artifacts import atomic_write_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _classifier_input_features(backbone: nn.Module) -> int:
    classifier = getattr(backbone, "classifier", None)
    if classifier is None:
        raise TypeError("EfficientNet backbone must expose a classifier")
    linear_layers = [module for module in classifier.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise TypeError("EfficientNet classifier must contain a linear layer")
    return linear_layers[-1].in_features


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


class TemporalResidualBlock(nn.Module):
    """Length-preserving temporal convolution with per-frame normalization."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.normalization = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        update = self.convolution(sequence).transpose(1, 2)
        update = self.normalization(update)
        update = self.dropout(self.activation(update)).transpose(1, 2)
        return sequence + update


class TemporalConvEncoder(nn.Module):
    """TempCNN encoder for a sequence of per-frame spatial features."""

    def __init__(
        self,
        input_features: int,
        temporal_channels: int = 256,
        embedding_features: int = 256,
        layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        integer_values = {
            "input_features": input_features,
            "temporal_channels": temporal_channels,
            "embedding_features": embedding_features,
            "layers": layers,
        }
        if any(value < 1 for value in integer_values.values()):
            raise ValueError("TempCNN feature sizes and layer count must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_projection = nn.Sequential(
            nn.Linear(input_features, temporal_channels),
            nn.LayerNorm(temporal_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(
                channels=temporal_channels,
                kernel_size=kernel_size,
                dilation=2**index,
                dropout=dropout,
            )
            for index in range(layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(temporal_channels),
            nn.Linear(temporal_channels, embedding_features),
        )
        self.embedding_features = embedding_features

    @staticmethod
    def _validated_mask(frame_features: Tensor, frame_mask: Tensor | None) -> Tensor:
        batch_size, frames, _ = frame_features.shape
        if frame_mask is None:
            return torch.ones(
                (batch_size, frames),
                dtype=torch.bool,
                device=frame_features.device,
            )
        if frame_mask.shape != (batch_size, frames):
            raise ValueError("frame_mask must have shape B x T")
        mask = frame_mask.to(device=frame_features.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each event must contain at least one valid frame")
        return mask

    def forward(self, frame_features: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape B x T x F")
        mask = self._validated_mask(frame_features, frame_mask)
        numeric_mask = mask.to(dtype=frame_features.dtype).unsqueeze(-1)

        sequence = self.input_projection(frame_features) * numeric_mask
        sequence = sequence.transpose(1, 2)
        channel_mask = numeric_mask.transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence) * channel_mask

        pooled = sequence.sum(dim=2) / channel_mask.sum(dim=2).clamp_min(1)
        return self.output_projection(pooled)


class EfficientNetTempCNNEventClassifier(nn.Module):
    """EfficientNetV2-S spatial features followed by a TempCNN event encoder."""

    uses_frame_mask = True

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        temporal_channels: int = 256,
        embedding_features: int = 256,
        temporal_layers: int = 2,
        temporal_kernel_size: int = 3,
        dropout: float = 0.2,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_v2_s(weights=weights)
        frame_features = _classifier_input_features(backbone)
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.frame_feature_count = frame_features
        self.temporal_encoder = TemporalConvEncoder(
            input_features=frame_features,
            temporal_channels=temporal_channels,
            embedding_features=embedding_features,
            layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_features, 2)
        self.freeze_backbone = freeze_backbone
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True) -> EfficientNetTempCNNEventClassifier:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_frame_features(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape B x 3 x T x H x W")
        batch_size, _, frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames,
            3,
            height,
            width,
        )
        normalized = (frame_batch - self.pixel_mean) / self.pixel_std
        gradients_enabled = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.set_grad_enabled(gradients_enabled):
            features = self.backbone(normalized)
        if features.ndim != 2 or features.shape[1] != self.frame_feature_count:
            raise RuntimeError("EfficientNet backbone returned an unexpected feature shape")
        return features.reshape(batch_size, frames, self.frame_feature_count)

    def extract_features(self, video: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        frame_features = self.extract_frame_features(video)
        return self.temporal_encoder(frame_features, frame_mask)

    def forward(self, video: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        return self.classifier(self.extract_features(video, frame_mask))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.models.cnn_temporal")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.frames < 1 or args.image_size < 1:
        raise ValueError("frames and image-size must be positive")

    device = torch.device(args.device)
    torch.hub.set_dir(str(args.project_root / ".cache" / "torch"))
    model = EfficientNetTempCNNEventClassifier(
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device).eval()
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
        )
        event = dataset[0]
        sample = event["video"].unsqueeze(0).to(device)
        frame_mask = event["frame_mask"].unsqueeze(0).to(device)
        event_id = event["event_id"]
        label = int(event["label"].item())
    else:
        sample = torch.zeros(
            (1, 3, args.frames, args.image_size, args.image_size),
            device=device,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        spatial_features = model.extract_frame_features(sample)
        embedding = model.temporal_encoder(spatial_features, frame_mask)
        logits = model.classifier(embedding)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report: dict[str, object] = {
        "purpose": "wiring_check_only_not_a_reported_experiment",
        "model": "torchvision.efficientnet_v2_s+tempcnn",
        "pretrained": args.pretrained,
        "pretrained_weights": (
            "EfficientNet_V2_S_Weights.IMAGENET1K_V1" if args.pretrained else None
        ),
        "backbone_frozen": args.freeze_backbone,
        "device": str(device),
        "input_shape": list(sample.shape),
        "spatial_feature_shape": list(spatial_features.shape),
        "embedding_shape": list(embedding.shape),
        "output_shape": list(logits.shape),
        "event_id": event_id,
        "label": label,
        "valid_frames": int(frame_mask.sum().item()) if frame_mask is not None else args.frames,
        "parameters": parameter_counts(model),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    if args.report is not None:
        report["report"] = args.report.as_posix()
        atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
