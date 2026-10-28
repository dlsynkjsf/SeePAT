"""One fixed auxiliary visual-supervision pilot; execution is explicit."""

import argparse
import hashlib
import os
import platform
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from seepat.artifacts import atomic_write_csv, file_sha256, read_csv_rows
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.models.visual_auxiliary import MODEL_VERSION, AuxiliaryVisualFusion
from seepat.training import experiment as runtime
from seepat.training.auxiliary_inputs import AuxiliaryInputDataset, export_inputs
from seepat.training.auxiliary_visual import balance_targets, train_auxiliary_epoch
from seepat.training.context_dataset import CONTEXT_INPUTS
from seepat.training.matched_inputs import inside
from seepat.training.metrics import binary_classification_metrics
from seepat.training.sequence_pilot import (
    ROLES,
    SELECTION,
    base_dataset,
    fixed_options,
    selected_inputs,
)
from seepat.training.train import (
    FUSION_MODEL,
    VideoObjective,
    _forward_logits,
    build_event_classifier,
    train_model,
)
from seepat.training.video_objective import (
    VIDEO_SUPERVISION_CONTRACT,
    scored_video_bags,
    train_video_epoch,
    video_bags,
    video_loader,
)

ARMS = ("control", "candidate")
VERSION = "auxiliary-visual-pilot-v1"
OUTPUT = "outputs/engineering/auxiliary_visual_pilot_v1"
AUDIT = "outputs/engineering/clip_target_audit_v1"
SAVED = "outputs/engineering/sequence_evidence_pilot_v2/inputs"
DESIGN = {
    "version": VERSION, "model": MODEL_VERSION, "output": OUTPUT, "selection": SELECTION,
    "audit": AUDIT, "saved_inputs": SAVED,
    "audit_seal_sha256": "b310dd0cb0b3552c2b5172780ee49c035f92ca3aac79b1b22bafc4ade1a4fe59",
    "audit_protocol_sha256": "08bfef02a52e686b06e654a3211b25f23bc780832f4c0ea4ff8107abfbc5c4fc",
    "initial_checkpoint": "outputs/training/fusion_phase5_v1/experiments/candidate/checkpoint_best.pt",
    "initial_sha256": "46fb132230fb2292ecadab63ac84aca34882174d6dbd4680ade5adab82dc4b07",
    "initial_run": "outputs/training/fusion_phase5_v1/experiments/candidate/run.json",
    "train_events": "outputs/numerical_calibration/guarded_per_video_v3/events_train_calibrated.csv",
    "inner_events": "outputs/numerical_calibration/guarded_per_video_v3/events_val_calibrated.csv",
    "export": "outputs/phase5_context_inputs_v1", "auxiliary_coefficient": 0.25,
    "balance": "Train-only inverse per-video exposure; known-clip mean; unknown zero",
    "epochs": 4, "training": asdict(fixed_options()), "readiness_limits": [16, 4],
    "threshold": 0.5, "checkpoint": "fixed last epoch4; never best",
    "gate": "strict_auc_and_ba_improvement_with_no_extra_false_positives",
    "scope": "small reused-inner development; not H1, significance or promotion",
}
REQUIRED_SOURCE = {f"src/seepat/{name}.py" for name in (
    "models/visual_auxiliary", "models/fusion", "training/auxiliary_pilot", "training/auxiliary_inputs",
    "training/auxiliary_visual", "training/train", "training/video_objective", "training/experiment",
    "training/sequence_pilot", "training/context_dataset", "training/dataset", "artifacts",
)}
METADATA = {
    DESIGN['train_events'], DESIGN['inner_events'], DESIGN['initial_run'],
    'outputs/numerical_calibration/guarded_per_video_v3/calibration.json',
    'outputs/numerical_calibration/guarded_per_video_v3/summary.json',
    *(f'{CONTEXT_INPUTS}/{name}' for name in ('inventory_train.csv', 'inventory_inner.csv',
        'expanded_train.csv', 'expanded_inner.csv', 'records_train.json', 'records_inner.json', 'summary.json', 'seal.json')),
    *(f'{SELECTION}/{name}' for name in ('inventory_train.csv', 'inventory_inner.csv',
        'index_train.csv', 'index_inner.csv', 'seal.json')),
    *(f'{SAVED}/{name}' for name in ('train.json', 'inner.json', 'job.json', 'summary.json', 'seal.json')),
    *(f'{AUDIT}/{name}' for name in ('job.json', 'summary.json', 'seal.json')),
}


def validate_protocol(protocol):
    if (set(protocol) != {'design', 'metadata_bindings', 'code_bindings'} or protocol['design'] != DESIGN
            or set(protocol['metadata_bindings']) != METADATA
            or not REQUIRED_SOURCE <= protocol['code_bindings'].keys()):
        raise ValueError('Unsupported auxiliary pilot design; preserve this frozen version')


def load_protocol(project, config):
    protocol = runtime.read_json(config)
    validate_protocol(protocol)
    actual = {p.relative_to(project).as_posix() for p in (project / 'src/seepat').rglob('*.py')}
    if set(protocol['code_bindings']) != actual:
        raise ValueError('Frozen auxiliary source membership changed')
    for relative, expected in protocol['code_bindings'].items():
        if hashlib.sha256(inside(project, relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError(f'Frozen auxiliary source changed: {relative}')
    for relative, expected in protocol['metadata_bindings'].items():
        if file_sha256(inside(project, relative)) != expected:
            raise ValueError(f'Frozen auxiliary metadata changed: {relative}')
    return protocol, {'protocol_sha256': hashlib.sha256(config.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


def audit_targets(project, inventories, indexes):
    directory = inside(project, AUDIT)
    summary = runtime.verify_output(directory, {'stage': 'clip-target-audit-v1',
        'binding': {'protocol_sha256': DESIGN['audit_protocol_sha256']}})
    if (file_sha256(directory / 'seal.json') != DESIGN['audit_seal_sha256']
            or summary['targets'] != {'-1': 34, '0': 1230, '1': 76}
            or summary['training_ready'] is not False or summary['gate'] != 'review_auxiliary_supervision_design'
            or read_csv_rows(directory / 'inventory.csv') != inventories['train']):
        raise ValueError('Original target-audit provenance/counts/gate differ')
    csv_rows = read_csv_rows(directory / 'clips.csv')
    rows = {r['clip_id']: r for r in csv_rows}
    expected = {r['clip_id']: r for r in indexes['train']}
    if len(rows) != len(csv_rows) or set(rows) != set(expected) or len(rows) != 1340:
        raise ValueError('Original target-audit clip membership differs')
    targets = {}
    for original in inventories['train']:
        identifier = original['video_id']
        case = runtime.read_json(inside(directory, f'cases/{identifier}.json'))
        planned = [r['clip_id'] for r in indexes['train'] if r['video_id'] == identifier]
        if case['inventory'] != original or [c['clip_id'] for c in case['clips']] != planned:
            raise ValueError('Original target-audit case/order differs')
        for clip in case['clips']:
            row = rows[clip['clip_id']]
            if (any(str(v) != row[k] for k, v in clip.items() if k != 'support')
                    or row['video_id'] != identifier or row['source_group'] != original['source_group']
                    or row['category'] != original['category'] or row['kind'] != expected[clip['clip_id']]['kind']
                    or len(clip['support']) != 16
                    or type(clip['aux_visual_target']) is not int
                    or type(clip['aux_target_mask']) is not bool):
                raise ValueError('Original target-audit CSV/support/provenance differs')
            targets[clip['clip_id']] = {**{k: clip[k] for k in ('declared_clip_sha256', 'declared_trace_sha256',
                'original_event_class_id', 'original_phoneme', 'kind')},
                'video_id': identifier, 'source_group': original['source_group'],
                'target': clip['aux_visual_target'], 'target_mask': clip['aux_target_mask'],
                'rgb_mask': [s['rgb_valid'] for s in clip['support']]}
    expected_balance = {'known_videos': 124, 'class_mass': ['14108177/120120', '786703/120120'],
                        'class_weights': [0.5278811004426723, 9.466647515008841]}
    if balance_targets(targets.values()) != expected_balance:
        raise ValueError('Original Train-only auxiliary balance differs')
    return targets


def fit_pair(output, options, inputs_loader, model_loader, binding, *, stage,
             resume=False, device="cpu", progress=None):
    job = {"stage": stage, "study": VERSION, "binding": binding, "options": asdict(options),
           "environment": {"device": device, "torch": torch.__version__, "python": platform.python_version()}}

    def train_arm(arm, model, train, validation, initial, directory):
        coefficient = 0.25 if arm == "candidate" else 0.0
        weights = train.auxiliary_balance["class_weights"]
        supervision = {**VIDEO_SUPERVISION_CONTRACT, "version": VERSION,
                       "auxiliary_coefficient": coefficient, "auxiliary_weights": weights,
                       "auxiliary_reduction": "known-clip mean within video; all-unknown zero"}
        objective = VideoObjective(
            "video maximum plus masked visual exposure", supervision,
            partial(train_auxiliary_epoch, weights=weights) if coefficient else train_video_epoch,
        )
        contract = {"pilot": VERSION, "model": MODEL_VERSION, "arm": arm, "stage": stage,
                    "binding": binding, "initial_state_sha256": initial,
                    "clip_input_contract": train.input_contract, "fusion_inputs": train.calibration_contract,
                    "auxiliary_coefficient": coefficient, "checkpoint": "fixed last epoch, never best"}
        run = train_model(model, train, validation, directory, options, torch.device(device), contract,
                          resume_from=directory / "checkpoint_last.pt" if resume else None,
                          progress=progress, video_objective=objective)
        history = runtime.read_json(directory / "history.json")
        runtime.check_history(run, history, train, validation, options)
        if any(e["validation"]["selection"]["threshold"] != 0.5 for e in history):
            raise ValueError("Auxiliary pilot must keep its fixed diagnostic cutoff")
        head = torch.cat([p.detach().flatten() for p in model.auxiliary_head.parameters()])
        if not torch.isfinite(head).all() or (coefficient == 0 and head.any()) or (coefficient and not head.any()):
            raise ValueError("Auxiliary head failed its zero/finite-learning contract; preserve for review")
        return {"completed_epochs": run["completed_epochs"], "optimizer_updates": run["global_step"],
                "skipped_updates": run["skipped_optimizer_steps"], "initial_state_sha256": initial,
                "final_validation": history[-1]["validation"], "elapsed_seconds": run["elapsed_seconds"],
                "peak_cuda_memory_bytes": run.get("peak_cuda_memory_bytes"),
                "auxiliary_head_l1": float(head.abs().sum()), "checkpoint": f"{arm}/checkpoint_last.pt"}

    return runtime.run_training_pair(output, job, inputs_loader, model_loader, train_arm,
                                     arms=ARMS, resume=resume, progress=progress,
                                     summary_fields={"scope": "small reused-inner development; no promotion"})


def evaluate_pair(output, dataset_loader, model_loader, selected, binding, *, device='cpu', progress=None):
    job = {'stage': 'evaluation', 'study': VERSION, 'binding': binding, 'threshold': 0.5,
           'optimizer_updates': 0, 'device': device, 'checkpoint': 'fixed last epoch4'}
    if output.exists():
        return runtime.verify_output(output, job)
    results, signatures, paired = {}, [], {}
    with runtime.output_stage(output, job, progress) as complete:
        for number, arm in enumerate(ARMS, 1):
            if progress:
                progress.start_job(number, arm)
            started = perf_counter()
            dataset, model = dataset_loader(arm), model_loader(arm)
            model.to(device).eval().requires_grad_(False)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            bags = video_bags(dataset.rows, True)
            loader = video_loader(dataset, bags, 0, torch.device(device), 20261025)
            signatures.append([(r['event_id'], r['video_id'], r['source_group'], r['kind'], r['class_id'])
                               for r in dataset.rows])
            scores, clips = {}, []
            with torch.inference_mode():
                for bag, scored in zip(bags, scored_video_bags(model, loader, bags, torch.device(device),
                    _forward_logits, progress, 'score original video classifier; fixed maximum/cutoff'), strict=True):
                    scores[bag.video_id] = {'probability': max(scored['event_probabilities']), 'label': bag.label}
                    clips.extend({'video_id': bag.video_id, 'clip_id': identifier, 'probability': probability}
                                 for identifier, probability in zip(bag.event_ids, scored['event_probabilities'], strict=True))
            if (set(scores) != {r['video_id'] for r in selected if r['status'] == 'paired'}
                    or any(scores[r['video_id']]['label'] != int(r['category'] != 'real')
                           for r in selected if r['video_id'] in scores)):
                raise ValueError('Evaluation changed original usable membership or labels')
            rows = [{**{k: r[k] for k in ('video_id', 'source_group', 'category', 'reason')},
                     'status': 'evaluated' if r['video_id'] in scores else 'not_evaluated',
                     **scores.get(r['video_id'], {'probability': '', 'label': ''})} for r in selected]
            available = [r for r in rows if r['status'] == 'evaluated']
            labels, probabilities = [r['label'] for r in available], [r['probability'] for r in available]
            metrics = binary_classification_metrics(labels, probabilities, 0.5)
            positive = [p for y, p in zip(labels, probabilities, strict=True) if y]
            negative = [p for y, p in zip(labels, probabilities, strict=True) if not y]
            pairs = len(positive) * len(negative)
            if not pairs:
                raise ValueError('Evaluation requires both video classes')
            wins = sum(2 * (p > n) + (p == n) for p in positive for n in negative)
            categories = {}
            for category in sorted({r['category'] for r in selected}):
                category_rows = [r for r in rows if r['category'] == category]
                usable = [r for r in category_rows if r['status'] == 'evaluated']
                categories[category] = {'requested': len(category_rows), 'evaluated': len(usable),
                    'unavailable': len(category_rows) - len(usable),
                    'predicted_manipulated': sum(r['probability'] >= 0.5 for r in usable),
                    'correct': sum(int(r['probability'] >= 0.5) == r['label'] for r in usable)}
            results[arm] = {'metrics': metrics, 'auc': wins / (2 * pairs), 'auc_twice_pair_wins': wins,
                'auc_pair_count': pairs, 'balanced_accuracy': (metrics['recall'] + metrics['specificity']) / 2,
                'requested': len(rows), 'evaluated': len(available), 'unavailable': len(rows) - len(available),
                'by_category': categories, 'elapsed_seconds': perf_counter() - started,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
            paired[arm] = scores
            atomic_write_csv(output / f'{arm}_videos.csv', rows)
            atomic_write_csv(output / f'{arm}_clips.csv', clips)
            del model, dataset
            if device == 'cuda':
                torch.cuda.empty_cache()
        if signatures[0] != signatures[1]:
            raise ValueError('Evaluation arms changed original clips, identities or labels')
        a, b = (results[arm] for arm in ARMS)
        def ba_numerator(result):
            m = result['metrics']
            return m['true_positive'] * (m['true_negative'] + m['false_positive']) + m['true_negative'] * (m['true_positive'] + m['false_negative'])
        passes = (b['auc_twice_pair_wins'] > a['auc_twice_pair_wins'] and ba_numerator(b) > ba_numerator(a)
                  and b['metrics']['false_positive'] <= a['metrics']['false_positive'])
        changes = Counter()
        changed_rows = []
        for identifier, before in paired['control'].items():
            after = paired['candidate'][identifier]
            if (before['probability'] >= 0.5) != (after['probability'] >= 0.5):
                key = ('detection_gained' if after['probability'] >= 0.5 else 'detection_lost') if before['label'] else (
                    'false_alarm_added' if after['probability'] >= 0.5 else 'false_alarm_corrected')
                changes[key] += 1
                changed_rows.append({'video_id': identifier, 'label': before['label'], 'change': key,
                                     'control': before['probability'], 'candidate': after['probability']})
        return complete({'status': 'complete', 'arms': results, 'optimizer_updates': 0,
            'paired_changes': {'changed': len(changed_rows), **dict(changes), 'videos': changed_rows},
            'gate': 'review_larger_design_only' if passes else 'stop_and_review', 'scope': DESIGN['scope']})


def stage_path(project, stage):
    return inside(project, f"{OUTPUT}/{'readiness' if stage.startswith('readiness') else stage}")


def verify_stage(project, stage, binding):
    directory = stage_path(project, stage)
    summary = runtime.verify_output(directory)
    job = runtime.read_json(directory / 'job.json')
    if job.get('binding') != binding or job.get('study') != VERSION:
        raise ValueError('Auxiliary stage belongs to different inputs/protocol')
    if stage == 'inputs':
        expected = {'train': {'requested': 128, 'usable': 125, 'unavailable': 3, 'clips': 1340},
                    'inner': {'requested': 64, 'usable': 62, 'unavailable': 2, 'clips': 639}}
        if summary['roles'] != expected or summary['gate'] != 'ready_for_readiness' or summary['model_calls'] != 0:
            raise ValueError('Auxiliary prepared membership/gate differs')
    elif stage != 'evaluation':
        epoch = int(stage[-1]) if stage.startswith('readiness') else 4
        expected_updates = epoch * (4 if stage.startswith('readiness') else 32)
        if (summary['through_epoch'] != epoch or set(summary['arms']) != set(ARMS)
                or any(a['optimizer_updates'] != expected_updates or a['skipped_updates']
                       or a['initial_state_sha256'] != summary['initial_state_sha256'] for a in summary['arms'].values())
                or summary['arms']['control']['auxiliary_head_l1'] != 0
                or not summary['arms']['candidate']['auxiliary_head_l1'] > 0):
            raise ValueError('Auxiliary epoch/update/equal-start/head accounting differs')
    elif (summary['optimizer_updates'] != 0 or any(a['evaluated'] != 62 or a['unavailable'] != 2
                                                 for a in summary['arms'].values())):
        raise ValueError('Auxiliary evaluation changed membership or fitted parameters')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fixed auxiliary visual-supervision pilot; no automatic promotion.')
    parser.add_argument('command', choices=('preflight', 'export', 'verify', 'readiness1', 'readiness2', 'train', 'evaluate', 'watch'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=Path('configs/auxiliary_visual_pilot_v1.json'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--stage', choices=('inputs', 'readiness1', 'readiness2', 'training', 'evaluation'), default='inputs')
    args = parser.parse_args(argv)
    project = args.project.resolve()
    os.chdir(project)
    if args.command == 'watch':
        return runtime.watch_progress(live_progress_path(inside(project, f'{OUTPUT}/{args.stage}_status.json')), 'AUXILIARY PILOT')
    config = args.config.resolve()
    _protocol, binding = load_protocol(project, config)
    inventories, indexes = selected_inputs(project)
    if {r: len(v) for r, v in indexes.items()} != {'train': 1340, 'inner': 639}:
        raise ValueError('Original fixed clip counts differ')

    def base(role):
        definition = {'roles': {r: {'events': DESIGN[f'{r}_events'], 'export': DESIGN['export']} for r in ROLES}}
        return base_dataset(project, definition, role, indexes[role])

    if args.command in {'preflight', 'export'}:
        targets = audit_targets(project, inventories, indexes)
        runtime.verify_output(inside(project, SAVED))
        if args.command == 'preflight':
            print('AUXILIARY PREFLIGHT PASSED | source/saved metadata/Train targets only | no model/media load')
            return {'status': 'passed', 'auxiliary_balance': balance_targets(targets.values())}
        progress = WorkflowProgress(config, inside(project, f'{OUTPUT}/inputs_status.json'), 2)
        try:
            result = export_inputs({r: base(r) for r in ROLES}, inventories,
                {r: runtime.read_json(inside(project, f'{SAVED}/{r}.json')) for r in ROLES},
                targets, stage_path(project, 'inputs'), binding, progress=progress)
            verify_stage(project, 'inputs', binding)
            progress.finish()
            print('AUXILIARY INPUTS COMPLETE | existing crops verified; zero new pixels or model calls')
            return result
        except BaseException as error:
            progress.finish(error)
            raise
    verify_stage(project, 'inputs', binding)
    bound = {**binding, 'inputs_seal_sha256': file_sha256(stage_path(project, 'inputs') / 'seal.json')}
    evaluation_binding = None
    if args.command == 'evaluate' or args.command == 'verify' and args.stage == 'evaluation':
        verify_stage(project, 'training', bound)
        evaluation_binding = {**bound, 'training_seal_sha256': file_sha256(stage_path(project, 'training') / 'seal.json')}
    if args.command == 'verify':
        expected = binding if args.stage == 'inputs' else evaluation_binding if args.stage == 'evaluation' else bound
        result = verify_stage(project, args.stage, expected)
        print(f'AUXILIARY PILOT VERIFIED | {args.stage} | checkpoint bytes not rehashed')
        return result

    def dataset(role):
        return AuxiliaryInputDataset(base(role), stage_path(project, 'inputs'), role)

    def fresh_model(resuming):
        if resuming:
            original = build_event_classifier(FUSION_MODEL, pretrained=False, freeze_backbone=True)
        else:
            contract = runtime.read_json(inside(project, DESIGN['initial_run']))['resume_contract']
            original = runtime.warm_start(inside(project, DESIGN['initial_checkpoint']), contract, DESIGN['initial_sha256'])
        torch.manual_seed(DESIGN['training']['seed'])
        return AuxiliaryVisualFusion(original)

    stage = 'training' if args.command == 'train' else 'evaluation' if args.command == 'evaluate' else args.command
    if (stage_path(project, stage) / 'worker.lock').exists():
        raise ValueError('Stage active or interrupted; preserve and inspect before recovery')
    progress = WorkflowProgress(config, inside(project, f'{OUTPUT}/{stage}_status.json'), 2)
    try:
        if stage != 'evaluation':
            options, resume = fixed_options(), False
            if stage.startswith('readiness'):
                options = replace(options, epochs=int(stage[-1]), max_train_batches=16, max_validation_batches=4)
                if stage == 'readiness2':
                    previous = runtime.verify_output(stage_path(project, stage))
                    if previous['through_epoch'] not in {1, 2}:
                        raise ValueError('Only sealed readiness1->2 resume is supported')
                    resume = previous['through_epoch'] == 1
            else:
                verify_stage(project, 'readiness2', bound)
            result = fit_pair(stage_path(project, stage), options,
                lambda _arm: (dataset('train'), dataset('inner')), lambda _arm, resuming: fresh_model(resuming),
                bound, stage='training' if stage == 'training' else 'readiness', resume=resume,
                device=args.device, progress=progress)
        else:
            directory = stage_path(project, 'training')
            seal = runtime.read_json(directory / 'seal.json')
            def evaluated_model(arm):
                path = directory / arm / 'checkpoint_last.pt'
                if file_sha256(path) != seal[f'{arm}/checkpoint_last.pt']['sha256']:
                    raise ValueError('Auxiliary checkpoint bytes changed')
                saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                declared = runtime.read_json(directory / arm / 'run.json')['resume_contract']
                if (saved['checkpoint_type'] != 'seepat_resumable_training' or saved['completed_epoch'] != 4
                        or saved['resume_contract'] != declared or declared['binding'] != bound
                        or declared['pilot'] != VERSION or declared['arm'] != arm or declared['model'] != MODEL_VERSION
                        or declared['initial_state_sha256'] != runtime.read_json(directory / 'summary.json')['initial_state_sha256']):
                    raise ValueError('Auxiliary checkpoint/arm/model/input contract changed')
                model = fresh_model(True)
                model.load_state_dict(saved['model_state'], strict=True)
                return model
            result = evaluate_pair(stage_path(project, stage), lambda _arm: dataset('inner'), evaluated_model,
                                   inventories['inner'], evaluation_binding, device=args.device, progress=progress)
        progress.finish()
        print(f"AUXILIARY PILOT {stage.upper()} COMPLETE | {result.get('gate', 'both arms complete')} | {DESIGN['scope']}")
        return result
    except BaseException as error:
        progress.finish(error)
        raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        raise SystemExit(f'AUXILIARY PILOT STOPPED | {error}') from error
