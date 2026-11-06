from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.video import Swin3D_B_Weights, swin3d_b

KINETICS_MEAN = (0.4850, 0.4560, 0.4060)
KINETICS_STD = (0.2290, 0.2240, 0.2250)


class SwinBaseEventClassifier(nn.Module):
    """Binary mouth-event baseline using the panel-required Video Swin Base."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            weights = Swin3D_B_Weights.KINETICS400_V1 if pretrained else None
            backbone = swin3d_b(weights=weights)
        head = getattr(backbone, "head", None)
        feature_count = getattr(head, "in_features", None)
        if not isinstance(feature_count, int):
            raise TypeError("Swin backbone must expose head.in_features")

        backbone.head = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_count, 2)
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

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def extract_features(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape B x 3 x T x H x W")
        normalized = (video - self.pixel_mean) / self.pixel_std
        return self.backbone(normalized)

    def forward(self, video: Tensor) -> Tensor:
        return self.classifier(self.extract_features(video))
