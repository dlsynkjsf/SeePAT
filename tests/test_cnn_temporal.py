from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
model_module = pytest.importorskip("seepat.models.cnn_temporal")
nn = torch.nn
EfficientNetTempCNNEventClassifier = model_module.EfficientNetTempCNNEventClassifier
TemporalConvEncoder = model_module.TemporalConvEncoder
parameter_counts = model_module.parameter_counts


class TinyFrameBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(6, 10))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pooled = frames.mean(dim=(2, 3))
        return self.classifier(self.projection(pooled))


def _tiny_model(freeze_backbone: bool = False) -> EfficientNetTempCNNEventClassifier:
    return EfficientNetTempCNNEventClassifier(
        pretrained=False,
        freeze_backbone=freeze_backbone,
        temporal_channels=8,
        embedding_features=7,
        temporal_layers=2,
        dropout=0.0,
        backbone=TinyFrameBackbone(),
    )


def test_cnn_temporal_wrapper_returns_features_and_binary_logits() -> None:
    model = _tiny_model().eval()
    video = torch.zeros((2, 3, 4, 8, 8))
    frame_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    spatial = model.extract_frame_features(video)
    embedding = model.extract_features(video, frame_mask)
    logits = model(video, frame_mask)

    assert spatial.shape == (2, 4, 6)
    assert embedding.shape == (2, 7)
    assert logits.shape == (2, 2)


def test_temporal_encoder_ignores_masked_frame_values() -> None:
    torch.manual_seed(7)
    encoder = TemporalConvEncoder(
        input_features=3,
        temporal_channels=5,
        embedding_features=4,
        layers=2,
        dropout=0.0,
    ).eval()
    first = torch.randn(1, 4, 3)
    second = first.clone()
    second[:, 2:] = 1000
    mask = torch.tensor([[True, True, False, False]])

    first_embedding = encoder(first, mask)
    second_embedding = encoder(second, mask)

    assert torch.allclose(first_embedding, second_embedding)


def test_temporal_encoder_rejects_an_event_without_valid_frames() -> None:
    encoder = TemporalConvEncoder(input_features=3, temporal_channels=4)

    with pytest.raises(ValueError, match="at least one valid frame"):
        encoder(torch.zeros((1, 2, 3)), torch.zeros((1, 2), dtype=torch.bool))


def test_frozen_efficientnet_leaves_temporal_branch_trainable() -> None:
    model = _tiny_model(freeze_backbone=True).train()
    counts = parameter_counts(model)

    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.temporal_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert 0 < counts["trainable"] < counts["total"]
