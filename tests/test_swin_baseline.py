from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
model_module = pytest.importorskip("seepat.models.swin_baseline")
nn = torch.nn
SwinBaseEventClassifier = model_module.SwinBaseEventClassifier
parameter_counts = model_module.parameter_counts


class TinyVideoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.head = nn.Linear(4, 10)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pooled = video.mean(dim=(2, 3, 4))
        return self.head(self.projection(pooled))


def test_swin_wrapper_returns_binary_logits() -> None:
    model = SwinBaseEventClassifier(pretrained=False, backbone=TinyVideoBackbone())

    logits = model(torch.zeros((2, 3, 4, 8, 8)))

    assert logits.shape == (2, 2)


def test_frozen_backbone_leaves_classifier_trainable() -> None:
    model = SwinBaseEventClassifier(
        pretrained=False,
        freeze_backbone=True,
        backbone=TinyVideoBackbone(),
    )

    counts = parameter_counts(model)

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert counts["trainable"] == 10
