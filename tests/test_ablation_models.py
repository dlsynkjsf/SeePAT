import pytest
import torch
from test_cnn_temporal import TinyFrameBackbone
from test_fusion_model import TinySwinBackbone

from seepat.models import cnn_temporal, fusion
from seepat.training import train


@pytest.mark.parametrize("model_name", [train.EFFICIENTNET_MODEL, train.TEMPCNN_MODEL, train.VISUAL_FUSION_MODEL])
def test_ablation_builder_masks_and_real_optimizer_updates(model_name, monkeypatch):
    monkeypatch.setattr(cnn_temporal, "efficientnet_v2_s", lambda **_: TinyFrameBackbone())
    monkeypatch.setattr(fusion, "efficientnet_v2_s", lambda **_: TinyFrameBackbone())
    monkeypatch.setattr(fusion, "swin3d_b", lambda **_: TinySwinBackbone())
    model = train.build_event_classifier(model_name, pretrained=False, freeze_backbone=False).eval()
    video = torch.rand(2, 3, 4, 16, 16)
    mask = torch.tensor([[True, True, False, False]] * 2)
    logits = train._forward_logits(model, video, {"frame_mask": mask}, torch.device("cpu"))
    assert logits.shape == (2, 2)
    if model_name != train.VISUAL_FUSION_MODEL:
        altered = video.clone()
        altered[:, :, 2:] = 100
        assert torch.allclose(logits, model(altered, mask))
    if model_name == train.EFFICIENTNET_MODEL:
        assert not any(isinstance(m, cnn_temporal.TemporalConvEncoder) for m in model.modules())
        assert isinstance(model.temporal_encoder, cnn_temporal.MaskedFrameMean)
    if model_name == train.TEMPCNN_MODEL:
        assert not hasattr(model, "backbone")
        assert model.extract_frame_features(video).shape == (2, 4, 256)
    if model_name == train.VISUAL_FUSION_MODEL:
        assert model.evidence_branch is None
        assert not model.uses_evidence_features and model.evidence_fields == ()
        assert set(model.extract_modality_tokens(video, mask)) == {"swin", "temporal"}
        assert torch.equal(logits, model(video, mask, torch.full((2, 7), float("nan")), torch.ones(2, 7)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before = model.classifier.weight.detach().clone()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    assert all(p in optimizer.state and int(optimizer.state[p]["step"]) == 1
               for p in model.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="valid frame"):
        model(video, torch.zeros_like(mask))


def test_temporal_only_rejects_nonexistent_pretraining():
    for pretrained, frozen in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="no pretrained backbone"):
            train.build_event_classifier(train.TEMPCNN_MODEL, pretrained, frozen)


def test_standalone_spatial_pool_is_order_invariant(monkeypatch):
    monkeypatch.setattr(cnn_temporal, "efficientnet_v2_s", lambda **_: TinyFrameBackbone())
    model = train.build_event_classifier(train.EFFICIENTNET_MODEL, False, True).eval()
    video = torch.rand(1, 3, 4, 8, 8)
    assert torch.allclose(model(video), model(video.flip(2)))
    assert all(not p.requires_grad for p in model.backbone.parameters())
