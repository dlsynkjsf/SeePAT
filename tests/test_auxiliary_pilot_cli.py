from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from test_fusion_model import _tiny_model
from test_sequence_training import StudyClips

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256
from seepat.live_progress import live_progress_path
from seepat.models.visual_auxiliary import AuxiliaryVisualFusion
from seepat.training import auxiliary_pilot as pilot
from seepat.training import experiment as runtime


@pytest.mark.parametrize('field,value', [('auxiliary_coefficient', .5), ('threshold', .3),
                                        ('epochs', 10), ('inner_events', 'data/deepfake_eval/events.csv')])
def test_protocol_rejects_unplanned_training_or_test_access(field, value):
    protocol = {'design': deepcopy(pilot.DESIGN), 'code_bindings': {}, 'metadata_bindings': {}}
    protocol['design'][field] = value
    with pytest.raises(ValueError, match='design'):
        pilot.validate_protocol(protocol)


def test_load_checks_bound_source_and_metadata_with_synthetic_files_only(tmp_path):
    data = b'synthetic source fixture'
    expected = sha256(data).hexdigest()
    protocol = {'design': deepcopy(pilot.DESIGN),
                'code_bindings': dict.fromkeys(pilot.REQUIRED_SOURCE, expected),
                'metadata_bindings': dict.fromkeys(pilot.METADATA, expected)}
    for name in (*protocol['code_bindings'], *protocol['metadata_bindings']):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    config = tmp_path / 'config.json'
    atomic_write_json(config, protocol)
    assert pilot.load_protocol(tmp_path, config)[0] == protocol
    path = tmp_path / next(iter(protocol['metadata_bindings']))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='metadata'):
        pilot.load_protocol(tmp_path, config)


def test_label_free_evaluation_retains_unavailable_and_reuses_complete_output(tmp_path):
    torch.manual_seed(7)
    template = AuxiliaryVisualFusion(_tiny_model(freeze_backbone=True)).eval()
    selected = [{'video_id': f'inner-{n}', 'source_group': f'inner-{n}', 'category': category,
                 'status': 'paired', 'reason': ''} for n, category in ((0, 'real'), (1, 'both_modified'))]
    selected.append({'video_id': 'missing', 'source_group': 'missing', 'category': 'audio_modified',
                     'status': 'unevaluated', 'reason': 'no_original_inputs'})
    output = tmp_path / 'evaluation'
    def dataset(_arm):
        return StudyClips('inner', False)
    def model(_arm):
        return deepcopy(template)
    result = pilot.evaluate_pair(output, dataset, model, selected, {'synthetic': True})
    assert result['optimizer_updates'] == 0 and result['gate'] == 'stop_and_review'
    assert result['paired_changes']['changed'] == 0
    assert result['arms']['candidate']['unavailable'] == 1
    assert result['arms']['candidate']['by_category']['audio_modified']['evaluated'] == 0
    assert pilot.evaluate_pair(output, None, None, selected, {'synthetic': True}) == result
    with pytest.raises(ValueError, match='binding'):
        pilot.evaluate_pair(output, None, None, selected, {'changed': True})


def test_help_and_terminal_watcher_never_start_workers(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stopped:
        pilot.main(['--help'])
    assert stopped.value.code == 0 and not list(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    status = live_progress_path(tmp_path / pilot.OUTPUT / 'inputs_status.json')
    atomic_write_json(status, {'status': 'complete', 'jobs_total': 2, 'job_index': 2,
                               'phase': 'complete', 'finished': 2, 'total': 2})
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    pilot.main(['watch', '--project', str(tmp_path), '--stage', 'inputs'])
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before
    assert 'complete' in capsys.readouterr().out.lower()


def test_profile_source_membership_is_frozen_and_has_no_old_protocol_mutation():
    config = Path(__file__).parents[1] / 'configs/auxiliary_visual_pilot_v1.json'
    profile = runtime.read_json(config)
    pilot.validate_protocol(profile)
    source = Path(__file__).parents[1]
    for relative, digest in profile['code_bindings'].items():
        assert sha256((source / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest


@pytest.mark.parametrize('extra_false_alarm,expected_gate', [(False, 'review_larger_design_only'),
                                                          (True, 'stop_and_review')])
def test_evaluation_enforces_all_three_fixed_criteria(tmp_path, extra_false_alarm, expected_gate):
    class Clips:
        def __init__(self):
            self.rows = [{'video_id': str(n), 'event_id': str(n), 'source_group': str(n), 'kind': 'bilabial',
                          'class_id': '0', 'phoneme': 'p', 'manipulation_modality': 'real' if n < 4 else 'both_modified'}
                         for n in range(6)]
        def __len__(self):
            return 6
        def __getitem__(self, index):
            return {**self.rows[index], 'video': torch.full((3, 1, 1, 1), float(index)),
                    'label': torch.tensor(0), 'video_label': torch.tensor(int(index >= 4))}
    class Scorer(torch.nn.Module):
        def __init__(self, probabilities):
            super().__init__()
            self.register_buffer('scores', torch.logit(torch.tensor(probabilities)))
        def forward(self, video):
            value = self.scores[video[:, 0, 0, 0, 0].long()]
            return torch.stack([torch.zeros_like(value), value], dim=1)
    original = [.1, .2, .3, .4, .05, .8]
    candidate = [.1, .2, .3, .7 if extra_false_alarm else .4, .8, .9]
    selected = [{'video_id': str(n), 'source_group': str(n), 'category': 'real' if n < 4 else 'both_modified',
                 'status': 'paired', 'reason': ''} for n in range(6)]
    result = pilot.evaluate_pair(tmp_path / 'evaluation', lambda _arm: Clips(),
                                 lambda arm: Scorer(original if arm == 'control' else candidate), selected, {})
    assert result['arms']['candidate']['auc'] > result['arms']['control']['auc']
    assert result['arms']['candidate']['balanced_accuracy'] > result['arms']['control']['balanced_accuracy']
    assert result['gate'] == expected_gate


def test_partial_output_and_live_lock_are_preserved_without_replay(tmp_path):
    output = tmp_path / 'evaluation'
    output.mkdir()
    sentinel = output / 'worker.lock'
    sentinel.write_text('synthetic interrupted worker')
    with pytest.raises(ValueError, match='interrupted'):
        pilot.evaluate_pair(output, None, None, [], {})
    assert sentinel.read_text() == 'synthetic interrupted worker'
    assert not (output / 'job.json').exists()


def test_cli_training_requires_bound_inputs_before_any_model_or_stage_output(tmp_path, monkeypatch):
    directory = tmp_path / pilot.SELECTION
    inventories, indexes = {}, {}
    for role, split, count, missing, extra in [('train', 'train', 32, 3, 90), ('inner', 'val', 16, 2, 19)]:
        inventories[role], indexes[role], usable = [], [], 0
        for category in ('real', 'audio_modified', 'visual_modified', 'both_modified'):
            for n in range(count):
                identifier = f'{role}-{category}-{n}'
                absent = category == 'real' and n < missing
                clips = 0 if absent else 10 + int(usable < extra)
                inventories[role].append({'video_id': identifier, 'source_group': identifier, 'category': category,
                    'dataset_split': split, 'status': 'unevaluated' if absent else 'paired',
                    'reason': 'missing' if absent else '', 'expanded_clips': str(clips)})
                for clip in range(clips):
                    indexes[role].append({'video_id': identifier, 'source_group': identifier, 'dataset_split': split,
                        'clip_id': f'{identifier}-{clip}', 'kind': 'bilabial' if clip < 6 else 'context'})
                usable += int(not absent)
    with runtime.output_stage(directory, {'synthetic': 'selection'}) as complete:
        for role in pilot.ROLES:
            atomic_write_csv(directory / f'inventory_{role}.csv', inventories[role])
            atomic_write_csv(directory / f'index_{role}.csv', indexes[role])
        complete({'status': 'complete'})
    for name in (*pilot.METADATA, *pilot.REQUIRED_SOURCE):
        path = tmp_path / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic fixture')
    profile = {'design': deepcopy(pilot.DESIGN),
               'metadata_bindings': {name: file_sha256(tmp_path / name) for name in pilot.METADATA},
               'code_bindings': {name: file_sha256(tmp_path / name) for name in pilot.REQUIRED_SOURCE}}
    config = tmp_path / 'profile.json'
    atomic_write_json(config, profile)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        pilot.main(['train', '--project', str(tmp_path), '--config', str(config)])
    assert not (tmp_path / pilot.OUTPUT).exists()
    assert not (tmp_path / pilot.DESIGN['initial_checkpoint']).exists()
