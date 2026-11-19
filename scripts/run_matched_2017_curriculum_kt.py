"""Bounded six-fit 2017 comparison, prepared separately from the main campaign.

Preparation is read-only with respect to legacy sources. Training must be
explicitly requested by the user; the current clarification is recorded in chat.
"""
import argparse
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from curriculum_readouts import dump, table, fit_readouts, predict_readouts
from evaluate_saved_curriculum_kt import ROOT, SOURCE, sha

OUT = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017'
DATA = SOURCE / 'data/assist2017'
RAW = SOURCE / 'clustering/data_subsets/assist2017/anonymized_full_release_competition_dataset.csv'
TAXONOMY = SOURCE / 'results/taxonomy_comparison/cluster_assignments_assist2017_openai_text-embedding-3-large_1536_pca10_unique_problem_mean_rs42_ninit10_gridb5d922a9.csv'
DKT_SOURCE = SOURCE / 'pykt/models/dkt.py'
SEEDS = [42, 84, 126]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    keys = json.loads((DATA / 'keyid2idx.json').read_text())
    uid_reverse = {int(v): int(k) for k, v in keys['uid'].items()}
    q_lookup = {int(k): int(v) for k, v in keys['questions'].items()}
    original_lookup = keys['concepts']
    assignments = pd.read_csv(TAXONOMY, usecols=['problem_id', 'cluster_k100'])
    assert not assignments.problem_id.duplicated().any()
    assert set(assignments.problem_id) == set(q_lookup)
    k_lookup = dict(assignments.itertuples(index=False, name=None))
    raw = pd.read_csv(RAW, usecols=['studentId', 'action_num', 'problemId', 'skill', 'correct', 'startTime'])
    assert raw.notna().all().all()
    assert not raw.duplicated(['studentId', 'action_num']).any()
    assert set(raw.correct) == {0, 1}
    splits = {}
    for name in ['train_valid.csv', 'test.csv']:
        source = pd.read_csv(DATA / name, usecols=['uid', 'fold'])
        for uid, fold in source[['uid', 'fold']].itertuples(index=False, name=None):
            true_uid = uid_reverse[int(uid)]
            assert true_uid not in splits
            splits[true_uid] = int(fold)
    dropped = raw[~raw.studentId.isin(splits)]
    assert (dropped.groupby('studentId').size() < 3).all()
    raw = raw[raw.studentId.isin(splits)].copy()
    raw['fold'] = raw.studentId.map(splits)
    raw['original_concept'] = raw.skill.map(original_lookup)
    raw['induced_concept'] = raw.problemId.map(k_lookup)
    raw['question_index'] = raw.problemId.map(q_lookup)
    assert raw[['fold', 'original_concept', 'induced_concept', 'question_index']].notna().all().all()
    raw.sort_values(['studentId', 'startTime', 'action_num'], kind='stable', inplace=True)
    raw.to_csv(OUT / 'canonical_events.csv', index=False)
    split_names = {-1: 'test', 0: 'validation', 1: 'calibration', 2: 'train', 3: 'train', 4: 'train'}
    pd.DataFrame([{'uid': u, 'legacy_fold': f, 'split': split_names[f]} for u, f in sorted(splits.items())]).to_csv(OUT / 'student_split.csv', index=False)
    pairs = raw[['original_concept', 'question_index', 'induced_concept']].drop_duplicates().sort_values(['original_concept', 'question_index']).to_numpy(dtype=np.int64)
    assert len(pairs) == len(np.unique(pairs[:, :2], axis=0))
    n_c, n_k = len(original_lookup), 100
    counts = np.zeros((n_c, n_k), float)
    np.add.at(counts, (pairs[:, 0], pairs[:, 2]), 1)
    assert (counts.sum(1) > 0).all()
    practice_counts = np.zeros_like(counts)
    tr = raw[raw.fold.isin([2, 3, 4])]
    np.add.at(practice_counts, (tr.original_concept, tr.induced_concept), 1)
    empty = practice_counts.sum(1) == 0
    practice_counts[empty] = counts[empty]
    support = counts > 0
    weights = {'item_overlap': counts / counts.sum(1, keepdims=True),
        'practice_frequency': practice_counts / practice_counts.sum(1, keepdims=True),
        'equal_children': support / support.sum(1, keepdims=True)}
    for w in weights.values():
        assert (w >= 0).all() and np.allclose(w.sum(1), 1)
    np.savez_compressed(OUT / 'mapping.npz', counts=counts, practice_counts=practice_counts, pairs=pairs, **weights)
    sizes = {}
    for split in ['train', 'validation', 'calibration', 'test']:
        df = raw[raw.fold.map(split_names) == split]
        rows = []
        for uid, events in df.groupby('studentId', sort=False):
            for lo in range(0, len(events), 200):
                part = events.iloc[lo:lo + 200]
                row = {'uid': int(uid)}
                for key, column in [('questions', 'question_index'), ('original_concepts', 'original_concept'),
                                    ('induced_concepts', 'induced_concept'), ('responses', 'correct'),
                                    ('timestamps', 'startTime'), ('event_id', 'action_num')]:
                    a = np.full(200, -1, np.int64)
                    a[:len(part)] = part[column].to_numpy(dtype=np.int64)
                    row[key] = a
                rows.append(row)
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        valid = arrays['questions'] >= 0
        assert np.array_equal(valid, arrays['original_concepts'] >= 0)
        assert np.array_equal(valid, arrays['induced_concepts'] >= 0)
        for j in range(len(rows)):
            for c, k in zip(arrays['original_concepts'][j, valid[j]], arrays['induced_concepts'][j, valid[j]]):
                assert counts[c, k] > 0
        np.savez_compressed(OUT / f'{split}_sequences.npz', **arrays)
        sizes[split] = {'students': int(df.studentId.nunique()), 'events': len(df),
                        'chunks': len(rows), 'targets': int(valid[:, 1:].sum())}
    student_sets = {s: set(np.load(OUT / f'{s}_sequences.npz')['uid']) for s in sizes}
    for i, a in enumerate(sizes):
        for b in list(sizes)[i + 1:]:
            assert not student_sets[a] & student_sets[b]
    protocol = {'dataset': 'assist2017', 'scope': 'new matched K100 DKT comparison; not historical paper checkpoint recovery',
        'seeds': SEEDS, 'taxonomies': ['original', 'induced'], 'planned_dkt_fits': 6,
        'training': {'hidden_size': 256, 'dropout': 0.5, 'learning_rate': 0.0005,
                     'batch_size': 256, 'optimizer': 'Adam', 'max_epochs': 100,
                     'early_stop_patience': 10, 'select': 'highest validation AUC', 'seq_len': 200},
        'split': 'Reuse legacy student memberships: folds2/3/4 train, fold0 validation, fold1 independent calibration, fold-1 test',
        'events': 'Unique (studentId,action_num) raw events, sorted startTime then action_num; no exploding labels or many-to-many joins',
        'original_taxonomy': 'Preserve each event skill; a question can have different original labels in different contexts',
        'reference_bank': 'Unique (original concept, question) pairs across the known item bank; no correctness used',
        'induced_taxonomy': 'Saved text-embedding-3-large 1536 / PCA10 / K100 / unique-problem-mean / random-state42, n_init10; no clustering selection in this batch',
        'embedding_scope': 'Legacy content artifact over known item bank, transductive; not a claim of unseen-item generalization or exact paper provenance',
        'readouts': ['item_overlap', 'direct_item_average', 'practice_frequency', 'equal_children', 'learned_monotone',
                     'parent_calibration', 'intersection_item_overlap', 'intersection_practice_frequency',
                     'item_aware_reference_average', 'and_product', 'noisy_or'],
        'readout_fit': 'Independent calibration students only; fixed MPS Adam lr0.03,1000 steps. Parent eligibility50 events/5 each class; intersection20/2; item5/1; ridge weights0.01,intercept0.001,item0.1. No test tuning.',
        'test': 'Evaluate every selected model and every specified readout; retain null results and failures',
        'device_required': 'mps', 'counts': sizes, 'original_concepts': n_c, 'induced_concepts': n_k,
        'reference_items': len(q_lookup), 'reference_pairs': len(pairs), 'mapping_edges': int(support.sum()),
        'items_with_context_dependent_original_label': int((raw.groupby('problemId').skill.nunique() > 1).sum()),
        'excluded_events_in_three_short_students': len(dropped),
        'source_sha256': {str(p): sha(p) for p in [RAW, TAXONOMY, DATA / 'keyid2idx.json', DATA / 'train_valid.csv', DATA / 'test.csv', DKT_SOURCE]},
        'derived_sha256': {p.name: sha(p) for p in OUT.glob('*.npz')},
        'event_manifest_sha256': sha(OUT / 'canonical_events.csv'), 'split_manifest_sha256': sha(OUT / 'student_split.csv'),
        'readout_source_sha256': sha(Path(__file__).with_name('curriculum_readouts.py'))}
    file = OUT / 'protocol.json'
    if file.exists():
        assert json.loads(file.read_text()) == protocol
    else:
        dump(file, protocol)
    registry = OUT / 'run_registry.json'
    if not registry.exists():
        dump(registry, {f'{kind}_seed{seed}': {'state': 'prepared', 'kind': kind, 'seed': seed} for kind in ['original', 'induced'] for seed in SEEDS})
    print(json.dumps({'counts': sizes, 'reference_pairs': len(pairs), 'mapping_edges': int(support.sum())}), flush=True)


def model(kind, seed):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    cls = runpy.run_path(str(DKT_SOURCE))['DKT']
    return cls(68 if kind == 'original' else 100, 256, 0.5).to('mps')


def tensors(d, kind):
    c, r = d[kind + '_concepts'], d['responses']
    mask = (c[:, :-1] >= 0) & (c[:, 1:] >= 0)
    return (torch.tensor(c[:, :-1] * mask, device='mps'), torch.tensor(r[:, :-1] * mask, device='mps'),
            torch.tensor(np.maximum(c[:, 1:], 0), device='mps'),
            torch.tensor(np.maximum(r[:, 1:], 0), dtype=torch.float32, device='mps'), torch.tensor(mask, device='mps'))


def scores(m, values):
    cs, rs, target, y, mask = values
    result = []
    m.eval()
    with torch.inference_mode():
        for lo in range(0, len(cs), 256):
            sl = slice(lo, lo + 256)
            p = m(cs[sl], rs[sl]).gather(2, target[sl, :, None]).squeeze(2)
            result.append(p[mask[sl]].cpu().numpy())
    return np.concatenate(result), y[mask].cpu().numpy()


def train():
    assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert sha(Path(path)) == digest
    dump(OUT / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'mps', 'mps_available': True, 'platform': platform.platform(),
        'training_script_sha256': sha(Path(__file__))})
    train_data = dict(np.load(OUT / 'train_sequences.npz'))
    valid_data = dict(np.load(OUT / 'validation_sequences.npz'))
    registry = json.loads((OUT / 'run_registry.json').read_text())
    for key, entry in registry.items():
        if entry['state'] == 'complete':
            continue
        kind, seed = entry['kind'], entry['seed']
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        m = model(kind, seed)
        optimizer = torch.optim.Adam(m.parameters(), lr=0.0005)
        tr, va = tensors(train_data, kind), tensors(valid_data, kind)
        start = time.monotonic()
        history, best_auc, best_epoch, first_epoch = [], -1.0, 0, 1
        latest = folder / 'latest.ckpt'
        if latest.exists():
            saved = torch.load(latest, map_location='cpu', weights_only=True)
            m.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng'])
            torch.mps.set_rng_state(saved['mps_rng'])
            best_auc, best_epoch, first_epoch = saved['best_auc'], saved['best_epoch'], saved['epoch'] + 1
            history = json.loads((folder / 'history.json').read_text())
        entry['state'] = 'running'
        dump(OUT / 'run_registry.json', registry)
        try:
            for epoch in range(first_epoch, 101):
                m.train()
                order = np.random.default_rng(seed * 1000 + epoch).permutation(len(tr[0]))
                losses, events = 0.0, 0
                for lo in range(0, len(order), 256):
                    ix = torch.tensor(order[lo:lo + 256], device='mps')
                    c, r, target, y, mask = [a[ix] for a in tr]
                    if not mask.any():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    p = m(c, r).gather(2, target[:, :, None]).squeeze(2)
                    loss = torch.nn.functional.binary_cross_entropy(p[mask], y[mask])
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    n = int(mask.sum().cpu())
                    losses += float(loss.detach().cpu()) * n
                    events += n
                prediction, truth = scores(m, va)
                auc = float(roc_auc_score(truth, prediction))
                if auc > best_auc:
                    best_auc, best_epoch = auc, epoch
                    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, folder / 'best.ckpt')
                row = {'epoch': epoch, 'train_loss': losses / events, 'validation_auc': auc,
                       'best_auc': best_auc, 'best_epoch': best_epoch, 'elapsed_seconds_this_session': time.monotonic() - start}
                history.append(row)
                dump(folder / 'history.json', history)
                torch.save({'model': m.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                            'best_auc': best_auc, 'best_epoch': best_epoch,
                            'cpu_rng': torch.get_rng_state(), 'mps_rng': torch.mps.get_rng_state()}, latest)
                entry.update({'epoch': epoch, 'best_epoch': best_epoch, 'best_validation_auc': best_auc})
                dump(OUT / 'run_registry.json', registry)
                print(f'{key} ' + json.dumps(row), flush=True)
                if epoch - best_epoch >= 10:
                    break
            m.load_state_dict(torch.load(folder / 'best.ckpt', map_location='cpu', weights_only=True))
            pred, truth = scores(m, va)
            assert abs(roc_auc_score(truth, pred) - best_auc) < 1e-6
            # Causality and serialization checks use development inputs only.
            m.eval()
            with torch.inference_mode():
                c, r = va[0][:1], va[1][:1]
                original = m(c, r)
                changed = r.clone()
                changed[0, 2] = 1 - changed[0, 2]
                assert torch.allclose(m(c, changed)[:, :2], original[:, :2], atol=2e-6)
            entry.update({'state': 'complete', 'checkpoint_sha256': sha(folder / 'best.ckpt'),
                          'seconds_this_session': time.monotonic() - start, 'validation_reload_passed': True})
            dump(OUT / 'run_registry.json', registry)
        except Exception as exc:
            entry.update({'state': 'failed', 'failure': repr(exc)})
            dump(OUT / 'run_registry.json', registry)
            raise


def extract(m, d, kind, mapping):
    vals = tensors(d, kind)
    mask = vals[4].cpu().numpy()
    rows, pos = np.where(mask)
    data = {'uid': d['uid'][rows], 'row': rows, 'position': pos + 1,
        'event_id': d['event_id'][rows, pos + 1], 'label': d['responses'][rows, pos + 1],
        'question_index': d['questions'][rows, pos + 1], 'original_concept': d['original_concepts'][rows, pos + 1],
        'induced_concept': d['induced_concepts'][rows, pos + 1]}
    children = table(mapping)[0]
    outputs, features, cursor = [], [], 0
    m.eval()
    with torch.inference_mode():
        for lo in range(0, len(vals[0]), 64):
            sl = slice(lo, lo + 64)
            p = m(vals[0][sl], vals[1][sl])
            full = p[vals[4][sl]]
            direct = p.gather(2, vals[2][sl, :, None]).squeeze(2)[vals[4][sl]]
            outputs.append(direct.cpu().numpy())
            if kind == 'induced':
                original = data['original_concept'][cursor:cursor + len(full)]
                ix = torch.tensor(children[original], device='mps')
                features.append(torch.logit(full.gather(1, ix).clamp(1e-6, 1 - 1e-6)).cpu().numpy())
            cursor += len(full)
    data['raw_induced' if kind == 'induced' else 'original_dkt'] = np.concatenate(outputs)
    return data, np.concatenate(features) if features else None


def evaluate():
    assert torch.backends.mps.is_available()
    registry = json.loads((OUT / 'run_registry.json').read_text())
    assert len(registry) == 6 and all(x['state'] == 'complete' for x in registry.values())
    mapping = dict(np.load(OUT / 'mapping.npz'))
    for key, entry in registry.items():
        kind, seed = entry['kind'], entry['seed']
        destination = OUT / f'{key}_test_predictions.npz'
        if destination.exists():
            continue
        m = model(kind, seed)
        checkpoint = OUT / key / 'best.ckpt'
        assert sha(checkpoint) == entry['checkpoint_sha256']
        m.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        for p in m.parameters():
            p.requires_grad_(False)
        if kind == 'induced':
            calibration = dict(np.load(OUT / 'calibration_sequences.npz'))
            dev, x = extract(m, calibration, kind, mapping)
            np.savez_compressed(OUT / f'seed{seed}_calibration_predictions.npz', **dev)
            np.save(OUT / f'seed{seed}_calibration_features.npy', x)
            coefficients = fit_readouts(x, dev, mapping, mapping['pairs'], OUT, seed)
            frozen = sha(OUT / f'seed{seed}_readout_coefficients.npz')
        test = dict(np.load(OUT / 'test_sequences.npz'))
        data, x = extract(m, test, kind, mapping)
        if kind == 'induced':
            np.save(OUT / f'seed{seed}_test_features.npy', x)
            data.update(predict_readouts(x, data, mapping, coefficients))
            assert frozen == sha(OUT / f'seed{seed}_readout_coefficients.npz')
        np.savez_compressed(destination, **data)
        entry['test_predictions_sha256'] = sha(destination)
        dump(OUT / 'run_registry.json', registry)
        print(f'{key}: test predictions saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.stage]()
