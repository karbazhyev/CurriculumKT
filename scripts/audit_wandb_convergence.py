"""Read existing W&B journals without opening source files for writing."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
OUT = ROOT / 'results/model_readiness_20261107'
METRICS = {'epoch', 'train_loss', 'validauc', 'validacc', 'trainauc', 'trainacc', 'best_auc', 'best_epoch', '_step'}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def journal(path, history=True):
    # SDK open_for_scan uses r+b. Use its identical reader with an rb handle.
    ds = DataStore()
    ds._fname = str(path)
    ds._fp = path.open('rb')
    ds._index = 0
    ds._size_bytes = path.stat().st_size
    ds._opened_for_scan = True
    ds._read_header()
    info, rows = {}, []
    try:
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = Record()
            rec.ParseFromString(data)
            if rec.HasField('run'):
                info = {'entity': rec.run.entity, 'project': rec.run.project, 'run_id': rec.run.run_id}
                if not history:
                    break
            if rec.HasField('history'):
                row = {}
                keys = {item.key or '/'.join(item.nested_key) for item in rec.history.item}
                for item in rec.history.item:
                    key = item.key or '/'.join(item.nested_key)
                    if key in METRICS:
                        row[key] = json.loads(item.value_json)
                # Final checkpoint summaries are not epoch measurements.
                if {'epoch', 'train_loss', 'validauc'} <= row.keys():
                    row['checkpoint_summary_merged_into_epoch'] = 'testauc' in keys or 'model_save_path' in keys
                    rows.append(row)
    finally:
        ds.close()
    return info, rows


def diagnose(rows, selected_epoch=None):
    if not rows:
        return {'state': 'epoch_history_unavailable'}
    mixed = [r['epoch'] for r in rows if r.get('checkpoint_summary_merged_into_epoch')]
    epochs = sorted({int(r['epoch']): r for r in rows if not r.get('checkpoint_summary_merged_into_epoch')}.values(), key=lambda r: r['epoch'])
    if not epochs:
        return {'state': 'uncontaminated_epoch_history_unavailable', 'excluded_mixed_summary_epochs': mixed}
    finite = all(np.isfinite(r[k]) for r in epochs for k in ['train_loss', 'validauc'])
    if not finite:
        return {'state': 'nonfinite_metrics', 'epochs': len(epochs)}
    best = max(epochs, key=lambda r: r['validauc'])
    last = epochs[-1]
    selected = next((r for r in epochs if r['epoch'] == selected_epoch), None)
    tail = epochs[-min(10, len(epochs)):]
    tail_slope = float(np.polyfit([r['epoch'] for r in tail], [r['validauc'] for r in tail], 1)[0]) if len(tail) > 1 else None
    gap = last.get('trainauc', last['validauc']) - last['validauc'] if 'trainauc' in last else None
    drop = best['validauc'] - last['validauc']
    state = ('validation_degradation_with_training_improvement'
             if drop > .002 and last['train_loss'] < best['train_loss']
             else 'best_near_end_review_epoch_budget' if last['epoch'] - best['epoch'] < 5
             else 'validation_plateau_or_no_further_gain')
    return {'state': state, 'epochs': len(epochs), 'first_epoch': epochs[0]['epoch'],
            'last_epoch': last['epoch'], 'actual_best_epoch': best['epoch'],
            'actual_best_validation_auc': best['validauc'], 'last_validation_auc': last['validauc'],
            'validation_auc_drop_from_peak': drop, 'last_train_validation_auc_gap': gap,
            'training_loss_at_peak': best['train_loss'], 'last_training_loss': last['train_loss'],
            'tail_10_epoch_auc_slope': tail_slope, 'saved_selected_epoch': selected_epoch,
            'saved_selected_epoch_validation_auc': selected['validauc'] if selected else None,
            'finite_metrics': finite, 'excluded_mixed_summary_epochs': mixed,
            'interpretation_limit': 'Descriptive learning-curve diagnosis; not a proof of mathematical optimizer convergence.'}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    projects, inventory, errors = collections.Counter(), [], []
    for d in sorted((SOURCE / 'examples/wandb').glob('run-*')):
        file = next(d.glob('*.wandb'), None)
        if file is None:
            continue
        try:
            info, _ = journal(file, False)
            if info:
                projects[(info['entity'], info['project'])] += 1
                inventory.append({**info, 'local_directory': str(d)})
        except Exception as e:
            errors.append({'path': str(file), 'error_type': type(e).__name__})
    selected = []
    p = json.loads((ROOT / 'results/curriculum_kt_assist2012_k1000_20261107/protocol.json').read_text())
    for r in p['runs']:
        selected.append({'dataset': 'assist2012', 'taxonomy': r['kind'], 'seed': r['seed'],
                         'summary': r['log_summary'], 'num_concepts': r['num_concepts']})
    audit = json.loads((ROOT / 'results/curriculum_kt_readout_followup_20261107/wandb_reuse_audit.json').read_text())
    for r in audit['wandb_logs_matching_fixed_settings']:
        selected.append({'dataset': 'assist2017', 'taxonomy': 'original_legacy', 'summary': r['summary'], 'num_concepts': 68})
    results = []
    for row in selected:
        summary_path = Path(row['summary'])
        summary = json.loads(summary_path.read_text())
        file = next(summary_path.parent.parent.glob('*.wandb'), None)
        info, history = journal(file) if file else ({}, [])
        if info:
            info['url'] = f"https://wandb.ai/{info['entity']}/{info['project']}/runs/{info['run_id']}"
        result = {**row, **info, 'journal_sha256': hashlib.sha256(file.read_bytes()).hexdigest() if file else None,
                  'logged_test_equals_validation': summary.get('testauc') == summary.get('validauc'),
                  'diagnosis': diagnose(history, summary.get('best_epoch')), 'history': history}
        results.append(result)
    # The six matched 2017 fits were local-only, so identify them explicitly.
    local = []
    for f in sorted((ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017').glob('*_seed*/history.json')):
        data = json.loads(f.read_text())
        normalized = [{**r, 'validauc': r['validation_auc']} for r in data]
        local.append({'path': str(f), 'wandb_run': None, 'history': data,
                      'diagnosis': diagnose(normalized, data[-1]['best_epoch'])})
    dump(OUT / 'local_wandb_inventory.json', {'projects': [{'entity': e, 'project': p, 'local_runs': n} for (e,p),n in projects.most_common()], 'runs': inventory, 'errors': errors})
    dump(OUT / 'selected_run_convergence.json', {'historical_wandb_runs': results, 'matched_2017_local_logs': local})
    print(json.dumps({'inventory_runs': len(inventory), 'errors': len(errors),
        'selected': [{k:v for k,v in r.items() if k not in ['history', 'summary', 'journal_sha256']} for r in results],
        'matched_2017_local_logs': len(local)}), flush=True)


if __name__ == '__main__':
    main()
