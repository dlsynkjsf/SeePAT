"""Bind a reviewed Train sidecar to unchanged, label-free inference inputs."""

import torch

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.training import experiment as runtime
from seepat.training.auxiliary_visual import balance_targets
from seepat.training.matched_inputs import inside

IDENTITY = ("video_id", "source_group", "kind", "dataset_split")


def verified_sample(base, index, record):
    row = base.rows[index]
    if (any(row[k] != record[k] for k in IDENTITY)
            or row['class_id'] != record['original_event_class_id'] or row['phoneme'] != record['original_phoneme']
            or row['kind'] == 'bilabial' and (
                row.get('vild_trace_sha256', '') != record['trace_sha256']
                or row.get('vild_augmentation_sha256', '') != record['augmentation_sha256'])
            or file_sha256(inside(base.root, row['mouth_clip_path'])) != record['clip_sha256']):
        raise ValueError("Auxiliary input clip identity/hash/provenance changed")
    sample = base[index]
    mask = sample['frame_mask']
    if (sample['decoded_frame_count'] != record['decoded_frame_count'] or mask.shape != (16,)
            or mask.dtype != torch.bool or not mask.any()
            or 'frame_mask' in record and mask.tolist() != record['frame_mask']):
        raise ValueError("Auxiliary input decoded count/RGB mask changed")
    return sample


def export_inputs(datasets, inventories, saved, targets, output, binding, *, progress=None):
    job = {'stage': 'inputs', 'study': 'auxiliary-visual-pilot-v1', 'binding': binding}
    if output.exists():
        return runtime.verify_output(output, job)
    if set(datasets) != {'train', 'inner'} or set(inventories) != set(datasets) or set(saved) != set(datasets):
        raise ValueError('Auxiliary inputs require exactly Train and inner roles')
    results, balance = {}, None
    with runtime.output_stage(output, job, progress) as complete:
        for number, (role, base) in enumerate(datasets.items(), 1):
            if progress:
                progress.start_job(number, role)
            expected = {r['video_id'] for r in inventories[role] if r['status'] == 'paired'}
            identifiers = [r['event_id'] for r in base.rows]
            if (set(saved[role]) != set(identifiers) or len(set(identifiers)) != len(identifiers)
                    or {r['video_id'] for r in base.rows} != expected
                    or role == 'train' and set(targets) != set(identifiers)):
                raise ValueError('Auxiliary inputs changed original usable/clip membership')
            records = {}
            for index, row in enumerate(base.rows):
                clip = row['event_id']
                if progress:
                    progress('verify existing clip bytes, decoder and target bindings', index, len(base), clip)
                original = saved[role][clip]
                record = {k: original[k] for k in (*IDENTITY, 'clip_sha256', 'trace_sha256',
                                                  'augmentation_sha256', 'decoded_frame_count')}
                record.update(original_event_class_id=row['class_id'], original_phoneme=row['phoneme'])
                sample = verified_sample(base, index, record)
                record['frame_mask'] = sample['frame_mask'].tolist()
                if role == 'train':
                    target = targets[clip]
                    if (any(target[k] != record[k] for k in ('video_id', 'source_group', 'kind',
                                                             'original_event_class_id', 'original_phoneme'))
                            or target['declared_clip_sha256'] != record['clip_sha256']
                            or target['declared_trace_sha256'] != record['trace_sha256']
                            or target['rgb_mask'] != record['frame_mask']
                            or any(type(v) is not bool for v in target['rgb_mask'])):
                        raise ValueError('Auxiliary target clip/mask/provenance differs')
                    record.update(target=target['target'], target_mask=target['target_mask'])
                records[clip] = record
            if role == 'train':
                balance = balance_targets(records.values())
            atomic_write_json(output / f'{role}.json', records)
            results[role] = {'requested': len(inventories[role]), 'usable': len(expected),
                             'unavailable': len(inventories[role]) - len(expected), 'clips': len(records)}
            if progress:
                progress('verify existing clip bytes, decoder and target bindings', len(base), len(base), '')
        return complete({'status': 'complete', 'roles': results, 'auxiliary_balance': balance,
                         'new_pixels': 0, 'model_calls': 0, 'optimizer_updates': 0,
                         'gate': 'ready_for_readiness', 'audit_gate_changed': False})


class AuxiliaryInputDataset:
    def __init__(self, base, directory, role):
        if role not in {'train', 'inner'}:
            raise ValueError('Unsupported auxiliary input role')
        summary = runtime.verify_output(directory)
        if summary.get('gate') != 'ready_for_readiness':
            raise ValueError('Auxiliary input preparation is not ready')
        self.base, self.rows = base, base.rows
        self.role = role
        self.records = runtime.read_json(directory / f'{role}.json')
        if set(self.records) != {r['event_id'] for r in self.rows}:
            raise ValueError('Auxiliary prepared clip membership differs')
        if role == 'inner' and any('target' in r or 'target_mask' in r for r in self.records.values()):
            raise ValueError('Inner inference inputs must not contain auxiliary targets')
        self.auxiliary_balance = summary['auxiliary_balance']
        if role == 'train' and balance_targets(self.records.values()) != self.auxiliary_balance:
            raise ValueError('Auxiliary Train class balance differs')
        self.calibration_contract = base.calibration_contract
        self.input_contract = {'version': 'auxiliary-visual-inputs-v1',
                               'inputs_seal_sha256': file_sha256(directory / 'seal.json'),
                               'auxiliary_targets': 'Train loss only; unknown masked; no inner targets'}

    def __len__(self):
        return len(self.base)

    def evidence_coverage(self):
        return self.base.evidence_coverage()

    def __getitem__(self, index):
        record = self.records[self.rows[index]['event_id']]
        sample = verified_sample(self.base, index, record)
        if self.role == 'train':
            return {**sample, 'auxiliary_target': torch.tensor(record['target'], dtype=torch.long),
                    'auxiliary_mask': torch.tensor(record['target_mask'], dtype=torch.bool)}
        return sample
