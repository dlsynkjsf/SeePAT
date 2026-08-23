from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.models.swin_baseline")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.frames < 1 or args.image_size < 1:
        raise ValueError("frames and image-size must be positive")

    device = torch.device(args.device)
    model = SwinBaseEventClassifier(pretrained=False).to(device).eval()
    event_id = None
    label = None
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
        logits = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report = {
        "model": "torchvision.swin3d_b",
        "pretrained": False,
        "device": str(device),
        "input_shape": list(sample.shape),
        "output_shape": list(logits.shape),
        "event_id": event_id,
        "label": label,
        "parameters": parameter_counts(model),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
