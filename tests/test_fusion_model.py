from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
model_module = pytest.importorskip("seepat.models.fusion")
from seepat.evidence import FUSION_EVIDENCE_FIELDS

nn = torch.nn
HybridFusionEventClassifier = model_module.HybridFusionEventClassifier
parameter_counts = model_module.parameter_counts
EVIDENCE_WIDTH = len(FUSION_EVIDENCE_FIELDS)


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


def _tiny_model(freeze_backbone: bool = False) -> HybridFusionEventClassifier:
    return HybridFusionEventClassifier(
        pretrained=False,
        freeze_backbone=freeze_backbone,
        swin_backbone=TinySwinBackbone(),
        frame_backbone=TinyFrameBackbone(),
        evidence_hidden=8,
        temporal_channels=8,
        embedding_features=7,
        fusion_dimension=8,
        fusion_heads=2,
        dropout=0.0,
    )


def test_fusion_returns_logits_and_sync_gap_scores() -> None:
    model = _tiny_model().eval()
    video = torch.rand(2, 3, 4, 8, 8)
    frame_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    features = torch.rand(2, EVIDENCE_WIDTH)
    feature_mask = torch.tensor([[True] * EVIDENCE_WIDTH, [False] * EVIDENCE_WIDTH])

    logits, sync_gap = model.forward_with_sync_gap(video, frame_mask, features, feature_mask)
    plain_logits = model(video, frame_mask, features, feature_mask)

    assert logits.shape == (2, 2)
    assert sync_gap.shape == (2,)
    assert torch.allclose(logits, plain_logits)
    assert torch.all(sync_gap > 0) and torch.all(sync_gap < 1)


def test_fusion_ignores_masked_evidence_values() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    video = torch.rand(1, 3, 4, 8, 8)
    features = torch.rand(1, EVIDENCE_WIDTH)
    feature_mask = torch.tensor([[True, True, False, False, False, False, False]])
    altered = features.clone()
    altered[0, 2:] = 1000.0

    first_logits, first_sync = model.forward_with_sync_gap(
        video,
        None,
        features,
        feature_mask,
    )
    second_logits, second_sync = model.forward_with_sync_gap(
        video,
        None,
        altered,
        feature_mask,
    )

    assert torch.allclose(first_logits, second_logits)
    assert torch.allclose(first_sync, second_sync)


def test_fusion_runs_without_evidence_branch_inputs() -> None:
    model = _tiny_model().eval()
    video = torch.rand(1, 3, 4, 8, 8)

    logits, sync_gap = model.forward_with_sync_gap(video)

    assert logits.shape == (1, 2)
    assert sync_gap.shape == (1,)


def test_fusion_rejects_wrong_evidence_width() -> None:
    model = _tiny_model().eval()
    video = torch.rand(1, 3, 4, 8, 8)
    features = torch.zeros(1, 3)
    feature_mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="evidence features must have width"):
        model(video, None, features, feature_mask)


def test_frozen_fusion_backbones_leave_fusion_branch_trainable() -> None:
    model = _tiny_model(freeze_backbone=True).train()
    counts = parameter_counts(model)

    assert model.backbone.training is False
    assert model.frame_backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.frame_backbone.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.temporal_encoder.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.evidence_branch.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(parameter.requires_grad for parameter in model.sync_gap_head.parameters())
    assert 0 < counts["trainable"] < counts["total"]


def test_fusion_contract_metadata_defaults() -> None:
    assert HybridFusionEventClassifier.uses_frame_mask is True
    assert HybridFusionEventClassifier.uses_evidence_features is True
    assert HybridFusionEventClassifier.evidence_fields == FUSION_EVIDENCE_FIELDS
    assert model_module.FUSION_MODEL_NAME == "swin3d_b_vild_fusion"
