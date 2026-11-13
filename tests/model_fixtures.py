"""Small synthetic model and bag fixtures shared by training tests."""

from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from seepat.models.fusion import HybridFusionEventClassifier


class TinySwinBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.head = nn.Linear(6, 10)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.projection(video.mean(dim=(2, 3, 4)))



class TinyFrameBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(6, 10))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projection(frames.mean(dim=(2, 3))))



def tiny_fusion_model(
    freeze_backbone: bool = False,
    unfreeze_final_backbone_stages: bool = False,
) -> HybridFusionEventClassifier:
    return HybridFusionEventClassifier(
        pretrained=False,
        freeze_backbone=freeze_backbone,
        unfreeze_final_backbone_stages=unfreeze_final_backbone_stages,
        swin_backbone=TinySwinBackbone(),
        frame_backbone=TinyFrameBackbone(),
        evidence_hidden=8,
        temporal_channels=8,
        embedding_features=7,
        fusion_dimension=8,
        fusion_heads=2,
        dropout=0.0,
    )



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

