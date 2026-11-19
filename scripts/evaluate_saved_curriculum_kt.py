"""Read-only retrospective evaluation of the saved ASSIST2012 K=1000 DKT runs.

All generated artifacts stay in CurriculumKT. The legacy source, its datasets,
and its checkpoints are read only. Run with the established Python and -B.
"""
import argparse
import hashlib
import json
import platform
import runpy
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
OUT = ROOT / 'results/curriculum_kt_assist2012_k1000_20261107'
STUDY = SOURCE / 'studies/dkt_influence_graph_stability_20260818'
INDUCED_DATA = SOURCE / 'data/assist2012'
ORIGINAL_DATA = STUDY / 'runtime/data/assist2012'
INDUCED_MODELS = SOURCE / 'examples/models/dkt_tiaocan_assist2012/assist2012_dkt_qid_models'
ORIGINAL_MODELS = STUDY / 'runtime/examples/models/assist2012/assist2012_dkt_qid_models'
RAW_NAME = '2012-2013-data-with-predictions-4-final.csv'
METHODS = ['item_overlap', 'practice_frequency', 'equal_children']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def numeric_id(value):
    d = Decimal(str(value))
    assert d.is_finite() and d == d.to_integral_value(), value
    return int(d)


def vocabulary(path):
    keys = json.loads(path.read_text())
    return {name: {numeric_id(k): int(v) for k, v in keys[name].items()}
            for name in ['questions', 'concepts', 'uid']}


def inverse(vocab):
    a = np.empty(len(vocab), np.int64)
    assert set(vocab.values()) == set(range(len(vocab)))
    for k, v in vocab.items():
        a[v] = k
    return a


def load_sequences(path, vocab):
    fields = ['questions', 'concepts', 'responses', 'selectmasks', 'timestamps']
    df = pd.read_csv(path, usecols=['uid', 'fold'] + fields)
    d = {k: np.stack([np.fromstring(s, sep=',', dtype=np.int64) for s in df[k]])
         for k in fields}
    d['uid'] = inverse(vocab['uid'])[df.uid.to_numpy()]
    d['fold'] = df.fold.to_numpy()
    d['row'] = np.arange(len(df))
    assert d['questions'].shape[1] == 200
    assert set(np.unique(d['selectmasks'])) <= {-1, 1}
    valid = d['questions'] >= 0
    assert np.array_equal(valid, d['concepts'] >= 0)
    assert np.array_equal(valid, d['responses'] >= 0)
    assert np.array_equal(valid, d['selectmasks'] == 1)
    assert set(np.unique(d['responses'])) <= {-1, 0, 1}
    assert np.all(d['questions'][valid] < len(vocab['questions']))
    assert np.all(d['concepts'][valid] < len(vocab['concepts']))
    assert np.all(np.diff(valid.astype(int), axis=1) <= 0), 'Interior padding'
    return d


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def item_labels(path):
    df = pd.read_csv(path, usecols=['problem_id', 'skill_id'])[['problem_id', 'skill_id']].dropna().drop_duplicates()
    assert not df.problem_id.duplicated().any(), 'Conflicting item labels'
    return {numeric_id(q): numeric_id(c) for q, c in df.itertuples(index=False, name=None)}


def verify_item_labels(d, q_to_c):
    valid = d['questions'] >= 0
    assert np.array_equal(q_to_c[d['questions'][valid]], d['concepts'][valid])


def run_list():
    runs = []
    for kind, parent, log_root in [
        ('induced', INDUCED_MODELS, SOURCE / 'examples/wandb'),
        ('original', ORIGINAL_MODELS, STUDY / 'wandb/assist2012/wandb')]:
        summaries = {}
        for f in log_root.glob('run-*/files/wandb-summary.json'):
            obj = json.loads(f.read_text())
            if 'model_save_path' in obj:
                summaries[Path(obj['model_save_path']).parent.name] = (f, obj)
        for config_path in sorted(parent.glob('*/config.json')):
            config = json.loads(config_path.read_text())
            checkpoint = config_path.parent / 'qid_model.ckpt'
            log, summary = summaries[config_path.parent.name]
            runs.append({'kind': kind, 'seed': config['params']['seed'],
                         'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                         'config': str(config_path), 'config_sha256': sha(config_path),
                         'log_summary': str(log), 'log_summary_sha256': sha(log),
                         'expected_valid_auc': summary['validauc'],
                         'logged_test_auc': summary['testauc'],
                         'best_epoch': summary['best_epoch'],
                         'num_concepts': config['data_config']['num_c'],
                         'embedding_size': config['model_config']['emb_size']})
    return sorted(runs, key=lambda x: (x['kind'], x['seed']))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    iv = vocabulary(INDUCED_DATA / 'keyid2idx.json')
    ov = vocabulary(ORIGINAL_DATA / 'keyid2idx.json')
    train = load_sequences(INDUCED_DATA / 'train_valid_sequences.csv', iv)
    test = load_sequences(INDUCED_DATA / 'test_sequences.csv', iv)
    otr = load_sequences(ORIGINAL_DATA / 'train_valid_sequences.csv', ov)
    ote = load_sequences(ORIGINAL_DATA / 'test_sequences.csv', ov)
    il = item_labels(INDUCED_DATA / RAW_NAME)
    ol = item_labels(SOURCE / 'clustering/data_subsets/assist2012' / RAW_NAME)
    rawq = inverse(iv['questions'])
    qc = np.array([ov['concepts'][ol[q]] for q in rawq], np.int64)
    qk = np.array([iv['concepts'][il[q]] for q in rawq], np.int64)
    oqc = np.array([ov['concepts'][ol[q]] for q in inverse(ov['questions'])], np.int64)
    for d in [train, test]:
        verify_item_labels(d, qk)
    for d in [otr, ote]:
        verify_item_labels(d, oqc)
    n_c, n_k = len(ov['concepts']), len(iv['concepts'])
    counts = np.zeros((n_c, n_k), np.float64)
    np.add.at(counts, (qc, qk), 1)
    assert np.all(counts.sum(1) > 0)
    w_item = counts / counts.sum(1, keepdims=True)
    practice_q = train['questions'][train['fold'] != 0]
    qcounts = np.bincount(practice_q[practice_q >= 0], minlength=len(rawq))
    pcounts = np.zeros_like(counts)
    np.add.at(pcounts, (qc, qk), qcounts)
    empty = pcounts.sum(1) == 0
    pcounts[empty] = counts[empty]
    w_practice = pcounts / pcounts.sum(1, keepdims=True)
    support = counts > 0
    w_equal = support / support.sum(1, keepdims=True)
    weights = dict(zip(METHODS, [w_item, w_practice, w_equal]))
    for w in weights.values():
        assert np.allclose(w.sum(1), 1) and np.all(w >= 0)
        assert np.all(w[~support] == 0)
    # Independent calculation: averaging item predictions equals Wp, even after
    # a permutation of the cluster vocabulary. These use no observed responses.
    rng = np.random.default_rng(20261107)
    mock = rng.uniform(size=n_k)
    explicit = np.array([mock[qk[qc == c]].mean() for c in range(n_c)])
    assert np.allclose(w_item @ mock, explicit)
    perm = rng.permutation(n_k)
    assert np.allclose(w_item[:, perm] @ mock[perm], explicit)
    iset, oset = set(test['uid']), set(ote['uid'])
    assert not iset & set(train['uid'])
    assert not oset & set(otr['uid'])
    common = sorted(iset & oset)
    paired = subset(test, np.isin(test['uid'], common))
    # Re-encode exactly the induced test histories under the original taxonomy.
    # Both models therefore receive the same questions, responses, and resets.
    paired_original = {k: v.copy() for k, v in paired.items()}
    valid = paired['questions'] >= 0
    paired_original['concepts'][valid] = qc[paired['questions'][valid]]
    # Compare complete raw item/response/timestamp chunks for provenance.
    def signatures(d, qraw):
        signatures = set()
        for i, u in enumerate(d['uid']):
            good = d['questions'][i] >= 0
            h = hashlib.sha256()
            for a in [qraw[d['questions'][i, good]], d['responses'][i, good],
                      d['timestamps'][i, good]]:
                h.update(a.tobytes())
            signatures.add((int(u), h.hexdigest()))
        return signatures
    shared_chunks = len(signatures(paired, rawq) & signatures(ote, inverse(ov['questions'])))
    runs = run_list()
    files = [Path(__file__), SOURCE / 'pykt/models/dkt.py',
             SOURCE / 'pykt/datasets/data_loader.py', SOURCE / 'pykt/models/evaluate_model.py',
             INDUCED_DATA / RAW_NAME, SOURCE / 'clustering/data_subsets/assist2012' / RAW_NAME]
    for parent in [INDUCED_DATA, ORIGINAL_DATA]:
        files += [parent / f for f in ['keyid2idx.json', 'train_valid_sequences.csv', 'test_sequences.csv']]
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved ASSIST2012 K1000 condition; separate from the paper K300 result',
        'methods_frozen_before_test': METHODS,
        'weights': {'item_overlap': 'Unique model-vocabulary items in each original/induced intersection',
                    'practice_frequency': 'All nonpadding events in training folds 1,2,3,4; no correctness used; item weights if original concept has zero training events',
                    'equal_children': 'Equal weight to every induced cluster intersecting an original concept; sensitivity analysis'},
        'prediction_timing': 'Input positions 0..198 predict responses 1..199; reset at each saved 200-step chunk; source selectmasks',
        'validation_gate_absolute_auc_tolerance': 0.00001,
        'test_protocol': 'All fixed methods and seeds retained. Induced full test and shared held-out students; original models only on same induced histories for shared held-out students.',
        'comparison_limit': 'Shared test events and histories, but training populations, splits, hyperparameters, dates and seed sets differ; exploratory saved-model comparison, not a controlled taxonomy effect or paper replication.',
        'no_dkt_training': True, 'device_required': 'mps', 'runs': runs,
        'input_sha256': {str(f): sha(f) for f in files},
        'counts': {'original_concepts': n_c, 'induced_concepts': n_k,
                   'reference_items': len(rawq), 'mapping_edges': int(support.sum()),
                   'practice_zero_rows_fallback': int(empty.sum()),
                   'induced_train_students': len(set(train['uid'][train['fold'] != 0])),
                   'induced_validation_students': len(set(train['uid'][train['fold'] == 0])),
                   'induced_test_students': len(iset), 'original_test_students': len(oset),
                   'shared_test_students': len(common), 'shared_test_chunks': len(paired['uid']),
                   'shared_chunks_also_exact_in_original_export': shared_chunks,
                   'induced_test_students_in_original_train_or_validation': len(iset & set(otr['uid']))},
        'checks': ['Unique item labels in both taxonomies', 'All source sequence labels match raw item/vocabulary joins',
                   'No train/development student overlaps with own test', 'Weights nonnegative and normalized',
                   'Direct item average equals matrix map-back', 'Cluster-index permutation invariance',
                   'Common evaluation uses identical question/response histories and chunk boundaries']}
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous['input_sha256'] == protocol['input_sha256'], 'Input changed after protocol freeze'
        assert previous['runs'] == protocol['runs'], 'Runs changed after protocol freeze'
        assert previous['methods_frozen_before_test'] == METHODS
    else:
        dump(protocol_path, protocol)
    np.savez_compressed(OUT / 'mapping.npz', **weights, counts=counts, practice_counts=pcounts,
                        q_to_original=qc, q_to_induced=qk, raw_question_ids=rawq,
                        original_concept_ids=inverse(ov['concepts']),
                        induced_concept_ids=inverse(iv['concepts']))
    # Every cache is written here, never through the legacy Dataset constructor.
    for name, d in [('induced_valid', subset(train, train['fold'] == 0)), ('induced_test', test),
                    ('original_valid', subset(otr, otr['fold'] == 0)), ('original_shared', paired_original)]:
        np.savez_compressed(OUT / f'{name}_sequences.npz', **d)
    np.save(OUT / 'shared_student_ids.npy', np.array(common, np.int64))
    print(json.dumps(protocol['counts']), flush=True)


def metrics(y, p, concepts):
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    eligible = []
    for c in np.unique(concepts):
        idx = concepts == c
        if len(np.unique(y[idx])) == 2:
            eligible.append((int(c), int(idx.sum()), float(roc_auc_score(y[idx], p[idx]))))
    return {'auc': float(roc_auc_score(y, p)), 'brier': float(np.mean((p.astype(float) - y)**2)),
            'log_loss': float(log_loss(y, np.clip(p.astype(float), 1e-7, 1 - 1e-7))),
            'events': len(y), 'positive_responses': int(y.sum()),
            'within_original_macro_auc': float(np.mean([a[2] for a in eligible])),
            'within_original_weighted_auc': float(np.average([a[2] for a in eligible], weights=[a[1] for a in eligible])),
            'within_original_eligible_concepts': len(eligible), 'per_original_concept': eligible}


def predict(run, split, batch_size=64):
    start = time.monotonic()
    assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
    d = dict(np.load(OUT / f'{split}_sequences.npz'))
    mapping = np.load(OUT / 'mapping.npz')
    config = json.loads(Path(run['config']).read_text())
    model_class = runpy.run_path(str(SOURCE / 'pykt/models/dkt.py'))['DKT']
    model = model_class(run['num_concepts'], config['model_config']['emb_size'],
                        config['model_config']['dropout'])
    model.load_state_dict(torch.load(run['checkpoint'], map_location='cpu', weights_only=True))
    model = model.to('mps').eval()
    assert all(p.device.type == 'mps' for p in model.parameters())
    qs, cs, rs = d['questions'], d['concepts'], d['responses']
    mask = (cs[:, :-1] != -1) & (cs[:, 1:] != -1)
    selected = d['selectmasks'][:, 1:] != -1
    assert np.all(~selected | mask)
    rows, positions = np.where(selected)
    original = mapping['q_to_original'][qs[rows, positions + 1]] if run['kind'] == 'induced' or split == 'original_shared' else cs[rows, positions + 1]
    output = {'uid': d['uid'][rows], 'row': d['row'][rows], 'position': positions + 1,
              'label': rs[rows, positions + 1].astype(np.int8), 'original_concept': original,
              'question_index': qs[rows, positions + 1]}
    names = ['raw_induced'] + METHODS if run['kind'] == 'induced' else ['original_dkt']
    accumulated = {name: [] for name in names}
    weights = {name: torch.as_tensor(mapping[name].astype(np.float32).T.copy(), device='mps') for name in METHODS} if run['kind'] == 'induced' else {}
    checked = False
    with torch.inference_mode():
        for lo in range(0, len(cs), batch_size):
            hi = min(lo + batch_size, len(cs))
            inputs_c = torch.as_tensor(cs[lo:hi, :-1] * mask[lo:hi], device='mps')
            inputs_r = torch.as_tensor(rs[lo:hi, :-1] * mask[lo:hi], device='mps')
            preds = model(inputs_c, inputs_r)
            local_selected = selected[lo:hi]
            target = torch.as_tensor((cs[lo:hi, 1:] * mask[lo:hi])[:, :, None], device='mps')
            direct = preds.gather(-1, target).squeeze(-1).cpu().numpy()[local_selected]
            accumulated[names[0]].append(direct)
            if run['kind'] == 'induced':
                original_targets = mapping['q_to_original'][np.maximum(qs[lo:hi, 1:], 0)]
                target_original = torch.as_tensor(original_targets[:, :, None], device='mps')
                for name, w in weights.items():
                    back = preds @ w
                    values = back.gather(-1, target_original).squeeze(-1).cpu().numpy()[local_selected]
                    accumulated[name].append(values)
                if not checked:
                    # Independent numpy check for mapping and causality: change
                    # response at input j; predictions strictly before j stay fixed.
                    first = preds[0, 0].cpu().numpy()
                    c0 = original_targets[0, 0]
                    assert np.isclose(accumulated['item_overlap'][-1][0], np.dot(mapping['item_overlap'][c0], first), atol=2e-6)
                    changed = inputs_r[:1].clone()
                    j = min(2, int((cs[lo] >= 0).sum()) - 1)
                    changed[0, j] = 1 - changed[0, j]
                    altered = model(inputs_c[:1], changed)
                    assert torch.allclose(altered[:, :j], preds[:1, :j], atol=2e-6, rtol=1e-5)
                    checked = True
            if lo == 0 or hi == len(cs) or (lo // batch_size) % 20 == 0:
                print(f"{run['kind']} seed={run['seed']} {split}: {hi}/{len(cs)} chunks, {time.monotonic()-start:.1f}s", flush=True)
    output.update({name: np.concatenate(values) for name, values in accumulated.items()})
    target_path = OUT / f"{run['kind']}_seed{run['seed']}_{split}_predictions.npz"
    np.savez_compressed(target_path, **output)
    all_metrics = {name: metrics(output['label'], output[name], original) for name in names}
    summary = {'kind': run['kind'], 'seed': run['seed'], 'split': split, 'device': 'mps',
               'checkpoint_sha256': run['checkpoint_sha256'], 'prediction_sha256': sha(target_path),
               'students': len(np.unique(output['uid'])), 'seconds': time.monotonic() - start,
               'metrics': all_metrics}
    if split.endswith('valid'):
        delta = all_metrics[names[0]]['auc'] - run['expected_valid_auc']
        summary['validation_auc_difference_from_logged'] = delta
        summary['validation_reproduced'] = abs(delta) <= 1e-5
    dump(target_path.with_suffix('.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'metrics'}) + '\n' + json.dumps({k: v['auc'] for k, v in all_metrics.items()}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'validate', 'test'])
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for run in protocol['runs']:
        assert sha(Path(run['checkpoint'])) == run['checkpoint_sha256']
    dump(OUT / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
                                   'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available(),
                                   'device': 'mps', 'script_sha256': sha(Path(__file__))})
    for run in protocol['runs']:
        validation = OUT / f"{run['kind']}_seed{run['seed']}_{run['kind']}_valid_predictions.json"
        if args.stage == 'validate':
            if validation.exists():
                result = json.loads(validation.read_text())
            else:
                result = predict(run, run['kind'] + '_valid', args.batch_size)
            assert result['validation_reproduced'], 'Checkpoint/data validation failed; test not authorized by this protocol'
        else:
            # All six reproduction checks must pass before any test prediction.
            for r in protocol['runs']:
                f = OUT / f"{r['kind']}_seed{r['seed']}_{r['kind']}_valid_predictions.json"
                assert json.loads(f.read_text())['validation_reproduced']
            split = 'induced_test' if run['kind'] == 'induced' else 'original_shared'
            result_file = OUT / f"{run['kind']}_seed{run['seed']}_{split}_predictions.json"
            if not result_file.exists():
                predict(run, split, args.batch_size)


if __name__ == '__main__':
    main()
