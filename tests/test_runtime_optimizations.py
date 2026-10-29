"""Execution tuning must preserve the original deterministic training order."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_train import TinyEventDataset, TinyVideoClassifier

from seepat.training import train as trainer
from seepat.training.runtime import TrainingRuntimeOptions, load_runtime_options
from seepat.workflow import load_workflow_settings


@pytest.mark.parametrize('seed', [5, 17])
@pytest.mark.parametrize('balanced_limit', [None, 2])
def test_reused_event_plan_matches_original_loader(seed, balanced_limit):
    dataset = TinyEventDataset('train')
    for index, row in enumerate(dataset.rows):
        row['event_id'] = str(index)
        dataset.samples[index]['event_id'] = str(index)
    old = trainer._loader(dataset, 2, 0, torch.device('cpu'), True, seed,
                          balanced_limit=balanced_limit)
    expected = [identifier for batch in old for identifier in batch['event_id']]
    indices = trainer._epoch_indices(dataset, 2, seed, balanced_limit)
    assert [dataset.rows[index]['event_id'] for index in indices] == expected


def _run(directory, model, options, runtime, resume=None):
    trainer.train_model(model, TinyEventDataset('train'), TinyEventDataset('val'),
                        directory, options, torch.device('cpu'), {'model': 'tiny'},
                        resume_from=resume, runtime=runtime)
    return torch.load(directory / 'checkpoint_last.pt', weights_only=False)


def _same_state(first, second):
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, second[key], rtol=0, atol=0)
        elif isinstance(value, dict):
            _same_state(value, second[key])
        else:
            assert value == second[key]


def test_reused_loader_preserves_updates_and_resume(tmp_path):
    torch.manual_seed(23)
    initial = TinyVideoClassifier()
    options = trainer.TrainingOptions(epochs=2, batch_size=2, amp=False, seed=12,
                                       early_stopping_patience=0)
    reference = _run(tmp_path / 'reference', deepcopy(initial), options, None)
    optimized = _run(tmp_path / 'optimized', deepcopy(initial), options,
                     TrainingRuntimeOptions(persistent_train_workers=True))
    for key in ('model_state', 'optimizer_state'):
        _same_state(reference[key], optimized[key])
    assert reference['history'] == optimized['history']
    directory = tmp_path / 'resumed'
    _run(directory, deepcopy(initial), replace(options, epochs=1),
         TrainingRuntimeOptions(persistent_train_workers=True))
    resumed = _run(directory, deepcopy(initial), options,
                   TrainingRuntimeOptions(persistent_train_workers=True),
                   directory / 'checkpoint_last.pt')
    for key in ('model_state', 'optimizer_state'):
        _same_state(reference[key], resumed[key])
    assert reference['history'] == resumed['history']


def test_runtime_controls_leave_scientific_options_unchanged(tmp_path):
    before = asdict(trainer.TrainingOptions())
    runtime = load_runtime_options({'clip_cache_dir': 'pixels', 'feature_cache_dir': 'features',
                                    'persistent_train_workers': True}, tmp_path)
    assert runtime.clip_cache_dir == tmp_path / 'pixels'
    assert runtime.feature_cache_dir == tmp_path / 'features'
    assert runtime.persistent_train_workers is True
    assert asdict(trainer.TrainingOptions()) == before
    with pytest.raises(TypeError):
        load_runtime_options({'persistent_train_workers': 'yes'})
    with pytest.raises(TypeError):
        load_runtime_options({'unknown': True})


def test_workflow_parses_inference_workers_without_coercion(tmp_path):
    profile = tmp_path / 'workflow.json'
    decision = {'name': 'example', 'validation_manifest': 'events.csv',
                'output_dir': 'decisions', 'workers': 2, 'pin_memory': True}
    profile.write_text(json.dumps({'decisions': [decision]}))
    job, = load_workflow_settings(profile).decision_jobs
    assert job.workers == 2 and job.pin_memory is True
    decision['workers'] = '2'
    profile.write_text(json.dumps({'decisions': [decision]}))
    with pytest.raises(ValueError, match='workers'):
        load_workflow_settings(profile)


def test_optimized_profiles_keep_frozen_phase5_inputs_and_separate_outputs():
    configs = Path(__file__).resolve().parents[1] / 'configs'
    for suffix in ('', '_readiness', '_readiness_resume'):
        original = load_workflow_settings(configs / f'fusion_phase5{suffix}.yaml')
        optimized = load_workflow_settings(configs / f'fusion_phase5_optimized{suffix}.yaml')
        assert original.report_path != optimized.report_path
        for old, new in zip(original.model_training_jobs, optimized.model_training_jobs, strict=True):
            assert old.train_manifest == new.train_manifest
            assert old.validation_manifest == new.validation_manifest
            assert old.input_profile == new.input_profile
            assert old.initialize_from == new.initialize_from
            assert {k: v for k, v in old.options.items() if k != 'workers'} == {
                k: v for k, v in new.options.items() if k != 'workers'
            }
            assert new.options['workers'] == 2
            assert old.output_dir != new.output_dir
            assert old.runtime is None and new.runtime['persistent_train_workers'] is True
            from seepat.workflow import _model_training_options

            _model_training_options(new)
