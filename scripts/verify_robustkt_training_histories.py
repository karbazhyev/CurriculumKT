"""Incremental, read-only W&B audit of completed RobustKT training runs.

No GPU operations, fitting, test scoring or writes to the active sweep occur.
Verified finished runs are reused only while local history/checkpoint identities
match; unfinished remote synchronization is recorded separately from mismatch.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import wandb

import run_robustkt_sweeps as base

OUT = base.ROOT / 'results/model_sequence_20261107/robustkt_online_history_checks.json'
STAGES = ['robustkt_sweeps_20261107', 'robustkt_seed_confirmation_20261107']
KEYS = ['epoch', 'train/auc', 'train/loss', 'validation/auc', 'validation/loss',
        'validation/best_auc', 'validation/best_epoch', 'optimization/training_bce',
        'optimization/grad_norm_mean', 'optimization/grad_norm_max',
        'optimization/learning_rate', 'runtime/epoch_seconds']


def compare(local, remote):
    assert all(all(k in row and row[k] is not None for k in KEYS) for row in local)
    local_by_epoch = {int(r['epoch']): r for r in local}
    remote_by_epoch = {int(r['epoch']): r for r in remote}
    assert len(local_by_epoch) == len(local), 'Duplicate local epochs'
    assert len(remote_by_epoch) == len(remote), 'Duplicate online epochs'
    assert sorted(local_by_epoch) == list(range(1, len(local) + 1))
    assert set(local_by_epoch) == set(remote_by_epoch), 'Local/online epoch set differs'
    delta = {}
    for key in KEYS:
        pairs = [(float(row[key]), float(remote_by_epoch[epoch][key]))
                 for epoch, row in local_by_epoch.items()]
        assert all(math.isfinite(v) for pair in pairs for v in pair), key
        delta[key] = max(abs(a-b) for a, b in pairs)
    assert max(delta.values()) < 1e-10, f'History values differ: {delta}'
    return delta


def main(require_complete=False):
    previous = base.read(OUT) if OUT.exists() else {'runs': {}}
    jobs, stage_counts = [], {}
    for name in STAGES:
        directory = base.ROOT / 'results' / name
        registry_file = directory / 'run_registry.json'
        if not registry_file.exists():
            stage_counts[name] = {'not_started': True}
            continue
        registry = base.read(registry_file)
        protocol = base.read(directory / 'protocol.json')
        assert base.sha(directory / 'protocol.json') == base.read(directory / 'protocol_identity.json')['sha256']
        stage_counts[name] = {'state': registry['state'],
            'counts': dict(Counter(e['state'] for e in registry['trials'].values()))}
        for key, entry in registry['trials'].items():
            if entry['state'] == 'complete':
                jobs.append((directory, protocol, key, entry))
    api_key = base.read(base.SOURCE / 'configs/wandb.json')['api_key']

    def verify(job):
        directory, protocol, key, entry = job
        record_key = f'{directory.name}/{key}'
        path = f"{protocol['wandb']['entity']}/{protocol['wandb']['project']}/{entry['wandb_run_id']}"
        result = {'trial': key, 'stage': directory.name, 'path': path,
                  'url': entry['wandb_url'], 'checked_utc': base.now()}
        try:
            folder = directory / 'trials' / key
            history = base.read(folder / 'history.json')
            identity = {'history_sha256': base.sha(folder / 'history.json'),
                        'checkpoint_sha256': base.sha(folder / 'best.ckpt'),
                        'protocol_sha256': base.sha(directory / 'protocol.json')}
            assert identity['checkpoint_sha256'] == entry['checkpoint_sha256']
            old = previous['runs'].get(record_key, {})
            if old.get('status') == 'verified' and old.get('identity') == identity and old.get('path') == path:
                return record_key, old
            result['identity'] = identity
            assert len(history) == entry['epochs']
            best = max(history, key=lambda h: h['validation/auc'])
            assert best['epoch'] == entry['best_epoch']
            assert abs(best['validation/auc'] - entry['best_validation_auc']) < 1e-12
            assert entry['validation_reload_passed']
            api = wandb.Api(api_key=api_key, timeout=30)
            run = api.run(path)
            result['online_state'] = run.state
            if run.state != 'finished':
                result['status'] = 'pending_online_finish'
                return record_key, result
            online = list(run.scan_history(keys=KEYS, page_size=200))
            delta = compare(history, online)
            config = dict(run.config)
            expected_config = {**{k: entry['candidate'][k] for k in ['d_model', 'learning_rate', 'dropout']},
                'seed': entry['seed'], 'dataset': entry['dataset'], 'taxonomy': entry['taxonomy'],
                'model': 'robustkt', 'device': 'mps', 'n_pid': 0,
                'protocol_sha256': identity['protocol_sha256'], 'test_evaluated': False,
                'data_sha256': protocol['datasets'][entry['dataset']]['files']['train']['sha256'],
                'validation_data_sha256': protocol['datasets'][entry['dataset']]['files']['validation']['sha256'],
                'mapping_sha256': protocol['datasets'][entry['dataset']]['mapping']['sha256'],
                'source_sha256': protocol['source_sha256'][str(base.SOURCE/'pykt/models/robustkt.py')]}
            for k, value in expected_config.items():
                assert config[k] == value, f'W&B configuration mismatch: {k}'
            summary = dict(run.summary)
            assert summary['result/state'] == 'complete'
            assert summary['result/checkpoint_sha256'] == identity['checkpoint_sha256']
            assert summary['result/best_epoch'] == entry['best_epoch']
            assert abs(summary['result/best_validation_auc'] - entry['best_validation_auc']) < 1e-12
            assert summary['result/validation_reload_passed']
            diagnostic = base.diagnose(history, entry['diagnosis']['stop_reason'])
            assert diagnostic == entry['diagnosis']
            result.update(status='verified', epochs=len(history),
                max_metric_difference=delta, best_epoch=entry['best_epoch'],
                best_validation_auc=entry['best_validation_auc'], diagnosis=diagnostic,
                actual_mps_config=True, checkpoint_validation_replayed=True,
                checkpoint_and_protocol_identities_match=True)
        except Exception as exc:
            # Do not include SDK request bodies or credentials in diagnostics.
            result.update(status='audit_error', error_type=type(exc).__name__)
            if isinstance(exc, AssertionError):
                result['detail'] = str(exc)
        return record_key, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = dict(pool.map(verify, jobs))
    counts = dict(Counter(r['status'] for r in records.values()))
    report = {'checked_utc': base.now(), 'stages': stage_counts, 'counts': counts,
        'runs': records, 'diagnosis_counts': dict(Counter(r['diagnosis']['status']
            for r in records.values() if r['status'] == 'verified')),
        'scope': 'Aggregate history/config/summary audit of completed runs only; no GPU work or held-out evaluation.',
        'all_currently_completed_runs_verified': len(records) == counts.get('verified', 0),
        'script_sha256': base.sha(Path(__file__))}
    base.dump(OUT, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)
    if require_complete:
        assert len(records) == 40, 'Expected32 screening plus8 selected-seed fits'
        assert report['all_currently_completed_runs_verified']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)
