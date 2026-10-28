from copy import deepcopy

import pytest
import torch
from test_fusion_model import _tiny_model

from seepat.models.visual_auxiliary import AuxiliaryVisualFusion


def test_auxiliary_head_preserves_video_prediction_and_uses_only_visual_tokens():
    torch.manual_seed(17)
    base = _tiny_model(freeze_backbone=True).eval()
    model = AuxiliaryVisualFusion(deepcopy(base)).eval()
    video = torch.rand(1, 3, 4, 8, 8)
    evidence = torch.ones(1, 7)
    mask = torch.ones(1, 7, dtype=torch.bool)
    values = {'features': evidence, 'feature_mask': mask}
    video_logits, auxiliary = model.forward_with_auxiliary(video, **values)
    assert torch.equal(video_logits, base(video, **values))
    assert torch.equal(model(video, **values), video_logits)
    assert torch.equal(auxiliary, torch.zeros(1, 2))
    with torch.no_grad():
        model.auxiliary_head.weight.fill_(0.3)
    _, first = model.forward_with_auxiliary(video, **values)
    _, second = model.forward_with_auxiliary(video, features=evidence * 10, feature_mask=mask)
    assert torch.equal(first, second)
    saved = deepcopy(model.state_dict())
    restored = AuxiliaryVisualFusion(deepcopy(base)).eval()
    restored.load_state_dict(saved)
    assert torch.equal(restored(video, **values), video_logits)
    saved['_extra_state']['model_version'] = 'old-production-model'
    with pytest.raises(ValueError, match='contract'):
        restored.load_state_dict(saved)


def test_per_video_balancing_and_masked_loss_keep_unknowns_out_of_denominator():
    from seepat.training.auxiliary_visual import balance_targets, masked_auxiliary_loss

    records = [{'video_id': 'a', 'target': y, 'target_mask': y != -1} for y in (0, 1, -1)]
    records += [{'video_id': 'b', 'target': 0, 'target_mask': True}] * 3
    records += [{'video_id': 'unknown', 'target': -1, 'target_mask': False}]
    balance = balance_targets(records)
    assert balance == {'known_videos': 2, 'class_mass': ['3/2', '1/2'],
                       'class_weights': [2 / 3, 2.0]}
    logits = torch.zeros(3, 2, requires_grad=True)
    loss = masked_auxiliary_loss(logits, torch.tensor([0, 1, -1]),
                                 torch.tensor([True, True, False]), balance['class_weights'])
    assert loss.item() == pytest.approx(0.92419624)
    loss.backward()
    assert torch.equal(logits.grad[2], torch.zeros(2))
    unknown = masked_auxiliary_loss(logits, torch.full((3,), -1), torch.zeros(3, dtype=torch.bool),
                                    balance['class_weights'])
    assert unknown.item() == 0
    with pytest.raises(ValueError, match='target'):
        masked_auxiliary_loss(logits, torch.tensor([0, 1, -1]), torch.ones(3, dtype=torch.bool), [1., 1.])


def test_two_arm_training_has_equal_starts_exact_resume_and_label_free_validation(tmp_path):
    from dataclasses import replace

    from test_sequence_training import StudyClips

    from seepat.training import auxiliary_pilot as pilot
    from seepat.training.train import TrainingOptions

    class Clips(StudyClips):
        def __init__(self, role):
            super().__init__(role, False)
            self.auxiliary_balance = {'class_weights': [1., 1.]}
            for sample in self.samples:
                if role == 'train':
                    label = int(sample['video_label']) if sample['kind'] == 'context' else 0
                    known = bool(sample['video_label'])
                    if known and sample['kind'] == 'context':
                        sample['video'] = sample['video'] + 0.1
                    sample.update(auxiliary_target=torch.tensor(label if known else -1),
                                  auxiliary_mask=torch.tensor(known))

    torch.manual_seed(11)
    template = AuxiliaryVisualFusion(_tiny_model(freeze_backbone=True))
    options = TrainingOptions(epochs=1, amp=False, freeze_backbone=True, supervision='video_max',
                              class_weighting='balanced_global', gradient_accumulation_steps=1,
                              early_stopping_patience=0, max_train_batches=16, max_validation_batches=4)
    def inputs(_arm):
        return Clips('train'), Clips('inner')
    def model(_arm, _resume):
        return deepcopy(template)
    output = tmp_path / 'readiness'
    first = pilot.fit_pair(output, options, inputs, model, {'synthetic': 'bound'}, stage='readiness')
    second = pilot.fit_pair(output, replace(options, epochs=2), inputs, model,
                            {'synthetic': 'bound'}, stage='readiness', resume=True)
    assert first['initial_state_sha256'] == second['initial_state_sha256']
    assert {arm['optimizer_updates'] for arm in second['arms'].values()} == {4}
    assert second['arms']['control']['auxiliary_head_l1'] == 0
    assert second['arms']['candidate']['auxiliary_head_l1'] > 0
    from seepat.training import experiment as runtime
    history = runtime.read_json(output / 'candidate/history.json')
    assert history[-1]['train']['videos_processed'] == 2
    assert history[-1]['train']['auxiliary_known_clips'] == 2
    fresh = tmp_path / 'fresh'
    pilot.fit_pair(fresh, replace(options, epochs=2), inputs, model, {'synthetic': 'bound'}, stage='readiness')
    for arm in ('control', 'candidate'):
        resumed = torch.load(output / arm / 'checkpoint_last.pt', weights_only=False)
        whole = torch.load(fresh / arm / 'checkpoint_last.pt', weights_only=False)
        assert resumed['global_step'] == 4 and resumed['completed_epoch'] == 2
        for key, value in resumed['model_state'].items():
            other = whole['model_state'][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    with pytest.raises(ValueError, match='binding'):
        pilot.fit_pair(output, options, inputs, model, {'changed': True}, stage='readiness')
