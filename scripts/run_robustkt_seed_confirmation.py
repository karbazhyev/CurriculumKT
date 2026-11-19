"""Freeze and run selected RobustKT settings at seeds84/126; reuse seed42.

The active screening source is imported unchanged. Preparation and training are
gated on terminal screening, immutable identities and exclusive worker locks.
No calibration or test arrays are loaded by this stage.
"""
import argparse
import copy
import fcntl
import json
import os
import platform
import signal
import sys
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch

import run_robustkt_sweeps as base

ROOT, SOURCE, sha = base.ROOT, base.SOURCE, base.sha
SCREEN = ROOT / 'results/robustkt_sweeps_20261107'
OUT = ROOT / 'results/robustkt_seed_confirmation_20261107'
QUEUE = ROOT / 'results/model_sequence_20261107'
SEEDS = [84, 126]
CELLS = [(ds, tax) for ds in ['assist2017', 'assist2012']
         for tax in ['original', 'induced']]


def select_screening(registry):
    """Selection is possible only after every planned trial has a disposition."""
    trials = registry['trials']
    assert len(trials) == 32
    assert registry['state'] in ['complete', 'complete_with_failures']
    assert registry['active_run'] is None
    assert all(e['state'] in ['complete', 'failed'] for e in trials.values())
    selected = {}
    for ds, tax in CELLS:
        cell = f'{ds}_robustkt_{tax}'
        entries = [(k, e) for k, e in trials.items() if e['cell'] == cell]
        assert len(entries) == 8
        assert {e['candidate']['id'] for _, e in entries} == {f'c{i:02}' for i in range(8)}
        complete = [(k, e) for k, e in entries if e['state'] == 'complete']
        assert complete, f'No completed screening candidate for {cell}'
        key, entry = min(complete, key=lambda v: (
            -v[1]['best_validation_auc'], v[1]['best_validation_loss'],
            v[1]['parameter_count'], v[1]['candidate']['id']))
        assert entry['validation_reload_passed']
        selected[cell] = {'trial': key, 'candidate': copy.deepcopy(entry['candidate']),
                          'validation_auc': entry['best_validation_auc'],
                          'checkpoint_sha256': entry['checkpoint_sha256'],
                          'all_eight_candidates_completed': len(complete) == 8}
    return selected


def verify_screening():
    protocol = base.read(SCREEN / 'protocol.json')
    assert sha(SCREEN / 'protocol.json') == base.read(SCREEN / 'protocol_identity.json')['sha256']
    for path, expected in protocol['source_sha256'].items():
        assert sha(Path(path)) == expected, f'Frozen screening source changed: {path}'
    assert sha(SCREEN / 'runtime_checks.json') == protocol['runtime_checks_sha256']
    return protocol


def verify_stage():
    protocol = base.read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == base.read(OUT / 'protocol_identity.json')['sha256']
    for path, expected in protocol['source_sha256'].items():
        assert sha(Path(path)) == expected, f'Frozen confirmation source changed: {path}'
    for path, expected in protocol['parent_artifact_sha256'].items():
        assert sha(Path(path)) == expected, f'Frozen screening artifact changed: {path}'
    for selection in protocol['selected_settings'].values():
        checkpoint = SCREEN / 'trials' / selection['trial'] / 'best.ckpt'
        assert sha(checkpoint) == selection['checkpoint_sha256']
    return protocol


def prepare():
    if (OUT / 'protocol.json').exists():
        return verify_stage()
    original = verify_screening()
    registry = base.read(SCREEN / 'run_registry.json')
    selected = select_screening(registry)
    assert selected == base.read(SCREEN / 'selected_configurations.json')
    for ds in original['datasets'].values():
        for record in [*ds['files'].values(), ds['mapping']]:
            # Identity check only; calibration and test responses are not read.
            assert sha(Path(record['path'])) == record['sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    base.dump(OUT / 'screening_registry_frozen.json', registry)
    protocol = copy.deepcopy(original)
    for key in ['seed', 'candidates', 'planned_sweeps', 'proposal_sha256']:
        protocol.pop(key, None)
    protocol.update({
        'created_utc': base.now(), 'stage': 'selected_setting_seed_confirmation',
        'authorization': 'User authorized all compatible models sequentially, including selected seeds.',
        'scope': 'Eight validation-only fits at development-selected settings; seed42 reused unchanged. No held-out/readout fitting.',
        'planned_trials': 8, 'seeds': SEEDS, 'reused_seed': 42,
        'selected_settings': selected,
        'selection': 'Settings frozen from completed screening by validation AUC, validation loss, parameter count and candidate ID. Every new seed is retained.',
        'parent_artifact_sha256': {str(p): sha(p) for p in [
            SCREEN / 'protocol.json', SCREEN / 'selected_configurations.json',
            OUT / 'screening_registry_frozen.json']},
    })
    protocol['wandb']['group'] = 'currkt-robustkt-seed-confirmation-20261107'
    protocol['source_sha256'][str(Path(__file__).resolve())] = sha(Path(__file__))
    protocol['source_sha256'][str(ROOT / 'research/MODEL_EXECUTION_QUEUE.md')] = sha(ROOT / 'research/MODEL_EXECUTION_QUEUE.md')
    trials = {}
    for ds, tax in CELLS:
        cell = f'{ds}_robustkt_{tax}'
        choice = selected[cell]
        for seed in SEEDS:
            key = f"{cell}_{choice['candidate']['id']}_seed{seed}"
            trials[key] = {'state': 'pending', 'cell': cell, 'dataset': ds,
                'taxonomy': tax, 'model': 'robustkt', 'candidate': choice['candidate'],
                'seed': seed, 'epochs': 0, 'selected_from_trial': choice['trial']}
    base.dump(OUT / 'run_registry.json', {'state': 'prepared', 'active_run': None,
        'created_utc': base.now(), 'trials': trials})
    diagnoses = dict(Counter(e['diagnosis']['status'] for e in registry['trials'].values()
                            if e['state'] == 'complete'))
    base.dump(OUT / 'screening_review.json', {
        'reviewed_utc': base.now(), 'all_32_accounted_for': True,
        'counts': dict(Counter(e['state'] for e in registry['trials'].values())),
        'convergence_counts': diagnoses, 'selected_settings': selected,
        'all_completed_checkpoint_replays_passed': all(e['validation_reload_passed']
            for e in registry['trials'].values() if e['state'] == 'complete'),
        'disposition': 'Retain failed, plateau, overfitting and epoch-budget-limited runs; do not extend epoch caps or retune. Freeze selected settings for new seeds.',
        'online_history_verification': 'Pending separate post-screening audit; training logs already online.',
        'test_used': False})
    base.dump(OUT / 'protocol.json', protocol)
    base.dump(OUT / 'protocol_identity.json', {'sha256': sha(OUT / 'protocol.json')})
    return verify_stage()


def summarize():
    protocol = verify_stage()
    registry = base.read(OUT / 'run_registry.json')
    counts = dict(Counter(e['state'] for e in registry['trials'].values()))
    models = {}
    for ds, tax in CELLS:
        cell = f'{ds}_robustkt_{tax}'
        selected = protocol['selected_settings'][cell]
        models[cell] = {'42': {'checkpoint': str(SCREEN / 'trials' / selected['trial'] / 'best.ckpt'),
            'checkpoint_sha256': selected['checkpoint_sha256'], 'candidate': selected['candidate'],
            'source': 'screening_reused', 'trial': selected['trial']}}
    for key, entry in registry['trials'].items():
        if entry['state'] == 'complete':
            checkpoint = OUT / 'trials' / key / 'best.ckpt'
            assert sha(checkpoint) == entry['checkpoint_sha256']
            models[entry['cell']][str(entry['seed'])] = {
                'checkpoint': str(checkpoint), 'checkpoint_sha256': entry['checkpoint_sha256'],
                'candidate': entry['candidate'], 'source': 'selected_seed_confirmation', 'trial': key}
    complete = counts.get('complete', 0) == 8
    base.dump(OUT / 'selected_models.json', {'models': models, 'all_twelve_available': complete,
        'protocol_sha256': sha(OUT / 'protocol.json'), 'test_used': False})
    lines = ['# RobustKT selected-seed confirmation', '',
        f'Counts: {json.dumps(counts)}. Seed42 checkpoints are reused; eight additional fits were planned.', '',
        'Validation-only training. No held-out or mapped-back AUC was computed in this stage.', '',
        '| Trial | State | Best validation AUC | Best epoch | Diagnosis |', '|---|---|---:|---:|---|']
    for key, e in registry['trials'].items():
        value = f"{e['best_validation_auc']:.6f}" if e.get('best_validation_auc') is not None else ''
        lines.append(f"| {key} | {e['state']} | {value} | {e.get('best_epoch', '')} | {e.get('diagnosis', {}).get('status', '')} |")
    lines += ['', 'Next: verify online histories and the MPS query adapter, then freeze the readout protocol and fit on independent calibration students before held-out scoring.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return counts


def worker(protocol):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Explicitly disable CPU fallback'
    assert torch.backends.mps.is_available(), 'Actual MPS is required'
    torch.set_num_threads(1)
    base.OUT = OUT
    base.DATA_CACHE.clear()
    wandb = base.login()
    verify_stage()
    base.STOP = False
    signal.signal(signal.SIGTERM, lambda *_: setattr(base, 'STOP', True))
    base.dump(OUT / 'worker.json', {'pid': os.getpid(), 'started_utc': base.now(),
        'python': sys.executable, 'python_version': platform.python_version(),
        'torch': torch.__version__, 'wandb': wandb.__version__,
        'mps_available': True, 'cpu_fallback': False, 'state': 'running'})
    state = 'worker_error'
    try:
        for key in base.read(OUT / 'run_registry.json')['trials']:
            if base.STOP:
                raise InterruptedError('Stop requested')
            registry = base.read(OUT / 'run_registry.json')
            entry = registry['trials'][key]
            if entry['state'] in ['complete', 'failed']:
                continue
            run = None
            if not entry.get('wandb_run_id'):
                entry['wandb_run_id'] = wandb.util.generate_id()
                base.dump(OUT / 'run_registry.json', registry)
            try:
                config = {**entry['candidate'], 'dataset': entry['dataset'], 'taxonomy': entry['taxonomy'],
                    'model': 'robustkt', 'seed': entry['seed'], 'stage': 'selected_setting_seed_confirmation',
                    'selection_trial': entry['selected_from_trial']}
                run = wandb.init(entity=protocol['wandb']['entity'], project=protocol['wandb']['project'],
                    id=entry['wandb_run_id'], resume='allow', config=config, settings=base.settings(),
                    group=protocol['wandb']['group'], job_type='selected_setting_seed_confirmation')
                base.fit(run, protocol, key, entry)
                run.finish(exit_code=0)
            except BaseException as exc:
                registry = base.read(OUT / 'run_registry.json')
                current = registry['trials'][key]
                # A logging failure after completed fitting must not erase its result.
                if current['state'] != 'complete':
                    interrupted = run is None or isinstance(exc, (KeyboardInterrupt, InterruptedError))
                    current.update(state='interrupted' if interrupted else 'failed',
                        error_type=type(exc).__name__, error=str(exc), updated_utc=base.now())
                else:
                    current.update(wandb_sync='unverified_after_finish_error', logging_error_type=type(exc).__name__)
                registry['active_run'] = None
                base.dump(OUT / 'run_registry.json', registry)
                folder = OUT / 'trials' / key
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'failure.log').write_text(traceback.format_exc())
                if run:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                if run is None or isinstance(exc, (KeyboardInterrupt, InterruptedError)):
                    raise
                traceback.print_exc()
            finally:
                torch.mps.empty_cache()
                summarize()
        registry = base.read(OUT / 'run_registry.json')
        counts = Counter(e['state'] for e in registry['trials'].values())
        state = 'complete' if counts['complete'] == 8 else 'complete_with_failures'
        registry.update(state=state, active_run=None, updated_utc=base.now())
        base.dump(OUT / 'run_registry.json', registry)
    finally:
        record = base.read(OUT / 'worker.json')
        record.update(state=state, finished_utc=base.now())
        base.dump(OUT / 'worker.json', record)
        summarize()


def selfcheck():
    """Exercise consequential selection and premature-start guards on synthetic scores."""
    trials = {}
    for ds, tax in CELLS:
        cell = f'{ds}_robustkt_{tax}'
        for i in range(8):
            trials[f'{cell}_c{i:02}'] = {'cell': cell, 'state': 'complete',
                'candidate': {'id': f'c{i:02}'}, 'best_validation_auc': .75,
                'best_validation_loss': .5, 'parameter_count': 100,
                'checkpoint_sha256': 'synthetic', 'validation_reload_passed': True}
    registry = {'state': 'complete', 'active_run': None, 'trials': trials}
    assert all(v['candidate']['id'] == 'c00' for v in select_screening(registry).values())
    first = next(iter(trials))
    trials[first]['best_validation_loss'] = .6
    assert select_screening(registry)[trials[first]['cell']]['candidate']['id'] == 'c01'
    trials[first]['best_validation_auc'] = .76
    assert select_screening(registry)[trials[first]['cell']]['candidate']['id'] == 'c00'
    trials[first]['state'] = 'pending'
    try:
        select_screening(registry)
    except AssertionError:
        pass
    else:
        raise AssertionError('Premature selection was accepted')
    verify_screening()
    live = base.read(SCREEN / 'run_registry.json')
    report = {'checked_utc': base.now(), 'synthetic_selection_checks_passed': True,
        'frozen_screening_sources_unchanged': True,
        'screening_counts': dict(Counter(e['state'] for e in live['trials'].values())),
        'stage_ready': live['state'] in ['complete', 'complete_with_failures'],
        'training_launched': False, 'heldout_arrays_loaded': False,
        'runner_sha256': sha(Path(__file__)),
        'remaining_checks': 'Actual MPS and W&B initialization are checked at launch after the screening lock is released.'}
    base.dump(QUEUE / 'robustkt_seed_confirmation_preparation.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'prepare', 'worker', 'summarize'])
    action = parser.parse_args().action
    if action == 'check':
        return selfcheck()
    if action == 'summarize':
        return summarize()
    # Use the old worker's actual lock as well as a queue-wide GPU lock. An
    # unfinished screening worker cannot overlap even during its final reports.
    with ExitStack() as stack:
        for path in [SCREEN / 'worker.lock', QUEUE / 'gpu_worker.lock',
                     ROOT / 'results/simplekt_sweeps_20261107/worker.lock']:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare()
        if action == 'worker':
            worker(protocol)


if __name__ == '__main__':
    main()
