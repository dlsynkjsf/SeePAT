import pytest
from test_sequence_training import StudyClips

from seepat.artifacts import file_sha256
from seepat.training import experiment as runtime
from seepat.training.auxiliary_inputs import AuxiliaryInputDataset, export_inputs


class SavedClips(StudyClips):
    def __init__(self, root, role):
        super().__init__(role, False)
        self.root = root
        self.records = {}
        for row, sample in zip(self.rows, self.samples, strict=True):
            path = root / (row['event_id'] + '.bin')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic pixels')
            row.update(mouth_clip_path=path.name, dataset_split='train' if role == 'train' else 'val')
            sample.update(decoded_frame_count=16)
            self.records[row['event_id']] = {
                'clip_sha256': file_sha256(path), 'trace_sha256': '', 'augmentation_sha256': '',
                'decoded_frame_count': 16,
                **{k: row[k] for k in ('video_id', 'kind', 'source_group', 'dataset_split')},
            }


def targets(dataset):
    result = {}
    for row in dataset.rows:
        label = int(row['manipulation_modality'] != 'real') if row['kind'] == 'context' else 0
        result[row['event_id']] = {
            'target': label, 'target_mask': True, 'rgb_mask': [True] * 16,
            'declared_clip_sha256': dataset.records[row['event_id']]['clip_sha256'],
            'declared_trace_sha256': '', 'original_event_class_id': row['class_id'],
            'original_phoneme': row['phoneme'], 'video_id': row['video_id'],
            'source_group': row['source_group'], 'kind': row['kind'],
        }
    return result


def test_export_binds_exact_clips_and_masks_without_inner_auxiliary_labels(tmp_path):
    datasets = {role: SavedClips(tmp_path / 'media', role) for role in ('train', 'inner')}
    saved = {role: data.records for role, data in datasets.items()}
    inventories = {role: [{'video_id': f'{role}-{label}', 'status': 'paired'} for label in (0, 1)]
                   for role in datasets}
    output = tmp_path / 'prepared'
    result = export_inputs(datasets, inventories, saved, targets(datasets['train']), output, {'fixture': True})
    assert result['roles']['train']['clips'] == 4 and result['new_pixels'] == 0
    train = AuxiliaryInputDataset(datasets['train'], output, 'train')
    inner = AuxiliaryInputDataset(datasets['inner'], output, 'inner')
    assert train.input_contract == inner.input_contract
    assert train[0]['auxiliary_target'].item() == 0
    assert 'auxiliary_target' not in inner[0] and 'auxiliary_mask' not in inner[0]
    assert all('target' not in r and 'target_mask' not in r
               for r in runtime.read_json(output / 'inner.json').values())
    assert export_inputs(datasets, inventories, saved, targets(datasets['train']), output, {'fixture': True}) == result
    with pytest.raises(ValueError, match='binding'):
        export_inputs(datasets, inventories, saved, targets(datasets['train']), output, {'changed': True})
    (datasets['train'].root / datasets['train'].rows[0]['mouth_clip_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='clip'):
        train[0]


@pytest.mark.parametrize('changed', ['mask', 'identity', 'clip', 'unknown'])
def test_export_rejects_target_provenance_or_mask_changes(tmp_path, changed):
    datasets = {role: SavedClips(tmp_path / 'media', role) for role in ('train', 'inner')}
    sidecar = targets(datasets['train'])
    item = sidecar[next(iter(sidecar))]
    if changed == 'mask':
        item['rgb_mask'][0] = False
    elif changed == 'identity':
        item['video_id'] = 'another-video'
    elif changed == 'clip':
        item['declared_clip_sha256'] = '0' * 64
    else:
        item['target'] = -1
    with pytest.raises(ValueError):
        export_inputs(datasets, {role: [{'video_id': f'{role}-{n}', 'status': 'paired'} for n in (0, 1)]
                                 for role in datasets},
                      {role: data.records for role, data in datasets.items()}, sidecar,
                      tmp_path / 'prepared', {'fixture': True})
