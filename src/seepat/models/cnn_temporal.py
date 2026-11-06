from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

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


class MaskedFrameMean(nn.Module):
    """Parameter-free temporal pooling for the spatial-only control."""

    def forward(self, features: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        mask = TemporalConvEncoder._validated_mask(features, frame_mask).unsqueeze(-1)
        return torch.where(mask, features, 0.0).sum(1) / mask.sum(1)


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
        temporal: bool = True,
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
        ) if temporal else MaskedFrameMean()
        self.classifier = nn.Linear(embedding_features if temporal else frame_features, 2)
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


class TempCNNOnlyEventClassifier(nn.Module):
    """TempCNN over fixed grayscale 16x16 frames; no learned spatial encoder."""

    uses_frame_mask = True

    def __init__(self) -> None:
        super().__init__()
        self.temporal_encoder = TemporalConvEncoder(input_features=256)
        self.classifier = nn.Linear(256, 2)

    def extract_frame_features(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape B x 3 x T x H x W")
        grayscale = video[:, 0] * 0.2989 + video[:, 1] * 0.5870 + video[:, 2] * 0.1140
        batch, frames, height, width = grayscale.shape
        return F.adaptive_avg_pool2d(
            grayscale.reshape(batch * frames, 1, height, width), (16, 16)
        ).reshape(batch, frames, 256)

    def forward(self, video: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        return self.classifier(self.temporal_encoder(self.extract_frame_features(video), frame_mask))
