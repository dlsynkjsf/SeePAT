"""Opt-in visual-exposure training head; video inference needs no edit labels."""

import torch
from torch import nn

from seepat.evidence import FUSION_EVIDENCE_FIELDS

MODEL_VERSION = "fusion-visual-auxiliary-v1"


class AuxiliaryVisualFusion(nn.Module):
    uses_frame_mask = True
    uses_evidence_features = True
    evidence_fields = FUSION_EVIDENCE_FIELDS

    def __init__(self, base):
        super().__init__()
        if not base.uses_evidence_features or tuple(base.evidence_fields) != FUSION_EVIDENCE_FIELDS:
            raise ValueError("Auxiliary model requires the original seven-field fusion model")
        self.base = base
        self.auxiliary_head = nn.Linear(2 * base.classifier.in_features, 2)
        nn.init.zeros_(self.auxiliary_head.weight)
        nn.init.zeros_(self.auxiliary_head.bias)

    def get_extra_state(self):
        return {"model_version": MODEL_VERSION, "visual_dimension": self.auxiliary_head.in_features,
                "auxiliary_target": "sampled visual-edit exposure; not authenticity or localization"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Auxiliary model contract differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True:
            raise ValueError("Auxiliary model contract requires strict loading")
        self.set_extra_state(state_dict.get("_extra_state"))
        expected = self.state_dict()
        if expected.keys() != state_dict.keys():
            raise ValueError("Auxiliary model contract keys differ")
        for name, value in expected.items():
            supplied = state_dict[name]
            if isinstance(value, torch.Tensor) and (
                not isinstance(supplied, torch.Tensor) or value.shape != supplied.shape or value.dtype != supplied.dtype
            ):
                raise ValueError(f"Auxiliary model tensor contract differs: {name}")
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def forward(self, video, frame_mask=None, features=None, feature_mask=None):
        return self.base(video, frame_mask, features, feature_mask)

    def forward_with_auxiliary(self, video, frame_mask=None, features=None, feature_mask=None):
        tokens = self.base.extract_modality_tokens(video, frame_mask, features, feature_mask)
        auxiliary = self.auxiliary_head(torch.cat([tokens["swin"], tokens["temporal"]], dim=1))
        logits = self.base.classifier(self.base.fusion(torch.stack(list(tokens.values()), dim=1)))
        return logits, auxiliary
