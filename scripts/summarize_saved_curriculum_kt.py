"""Summarize map-back scores and paired student bootstrap on shared holdouts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate_saved_curriculum_kt import OUT, METHODS, dump, metrics, sha


def weighted_auc_function(y, p):
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    ys = y[order]

    def score(weights):
        ws = weights[order]
        positive = np.add.reduceat(ws * ys, starts)
        negative = np.add.reduceat(ws * (1 - ys), starts)
        numerator = (positive * (np.cumsum(negative) - negative + 0.5 * negative)).sum()
        return numerator / (positive.sum() * negative.sum())

    # Independent sklearn verification, including nonuniform clustered weights.
    w = 1 + np.arange(len(y)) % 3
    assert abs(score(w) - roc_auc_score(y, p, sample_weight=w)) < 1e-12
    return score


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    common = np.load(OUT / 'shared_student_ids.npy')
    rows, per_concept, shared_scores = [], [], {}
    reference = None
    methods = ['raw_induced'] + METHODS + ['learned_monotone']
    for run in protocol['runs']:
        kind, seed = run['kind'], run['seed']
        split = 'induced_test' if kind == 'induced' else 'original_shared'
        d = dict(np.load(OUT / f'{kind}_seed{seed}_{split}_predictions.npz'))
        if kind == 'induced':
            learned = np.load(OUT / f'learned_seed{seed}_induced_test_predictions.npz')
            for key in ['uid', 'row', 'position', 'label', 'original_concept']:
                assert np.array_equal(d[key], learned[key]), key
            d['learned_monotone'] = learned['learned_monotone']
            populations = {'full_induced_test': np.ones(len(d['label']), dtype=bool),
                           'shared_heldout_students': np.isin(d['uid'], common)}
            names = methods
        else:
            populations = {'shared_heldout_students': np.ones(len(d['label']), dtype=bool)}
            names = ['original_dkt']
        for population, selected in populations.items():
            labels, concepts = d['label'][selected], d['original_concept'][selected]
            for method in names:
                result = metrics(labels, d[method][selected], concepts)
                by_concept = result.pop('per_original_concept')
                rows.append({'population': population, 'kind': kind, 'seed': seed,
                             'method': method, 'students': len(np.unique(d['uid'][selected])), **result})
                per_concept.extend({'population': population, 'seed': seed, 'method': method,
                                    'original_concept_index': c, 'events': n, 'auc': auc}
                                   for c, n, auc in by_concept)
                if population == 'shared_heldout_students':
                    shared_scores.setdefault(method, []).append(d[method][selected])
            if population == 'shared_heldout_students':
                identifiers = {k: d[k][selected] for k in ['uid', 'row', 'position', 'label', 'original_concept']}
                if reference is None:
                    reference = identifiers
                else:
                    for key, value in reference.items():
                        assert np.array_equal(value, identifiers[key]), f'Misaligned {key}'
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'auc_by_seed_and_population.csv', index=False)
    pd.DataFrame(per_concept).to_csv(OUT / 'auc_by_original_concept.csv', index=False)
    summary = []
    for (population, method), group in frame.groupby(['population', 'method'], sort=False):
        summary.append({'population': population, 'method': method, 'seeds': list(group.seed),
                        'auc_mean': float(group.auc.mean()), 'auc_seed_sd': float(group.auc.std(ddof=1)),
                        'auc_min': float(group.auc.min()), 'auc_max': float(group.auc.max()),
                        'brier_mean': float(group.brier.mean()),
                        'log_loss_mean': float(group.log_loss.mean()),
                        'within_original_macro_auc_mean': float(group.within_original_macro_auc.mean()),
                        'within_original_eligible_concepts': int(group.within_original_eligible_concepts.iloc[0]),
                        'events': int(group.events.iloc[0]), 'students': int(group.students.iloc[0])})
    y = reference['label']
    fns = {method: [weighted_auc_function(y, p) for p in predictions]
           for method, predictions in shared_scores.items()}
    _, uid_index = np.unique(reference['uid'], return_inverse=True)
    rng = np.random.default_rng(20261107)
    n_students = len(common)
    bootstrap = {method: [] for method in methods}
    baseline_auc = np.mean([f(np.ones(len(y))) for f in fns['original_dkt']])
    for _ in range(500):
        counts = np.bincount(rng.integers(n_students, size=n_students), minlength=n_students)
        weights = counts[uid_index]
        baseline = np.mean([f(weights) for f in fns['original_dkt']])
        for method in methods:
            bootstrap[method].append(np.mean([f(weights) for f in fns[method]]) - baseline)
    comparison = []
    for method in methods:
        score = np.mean([f(np.ones(len(y))) for f in fns[method]])
        # Extra same-seed comparison avoids pairing seed126 with seed255.
        sub = frame[(frame.population == 'shared_heldout_students') & frame.seed.isin([42, 84])]
        matched_delta = sub[sub.method == method].auc.mean() - sub[sub.method == 'original_dkt'].auc.mean()
        comparison.append({'method': method, 'mean_auc': float(score),
            'original_dkt_mean_auc': float(baseline_auc), 'delta_auc': float(score - baseline_auc),
            'paired_student_bootstrap_95ci': np.quantile(bootstrap[method], [0.025, 0.975]).tolist(),
            'same_seeds_42_84_delta_auc': float(matched_delta)})
    result = {'scope': protocol['scope'], 'comparison_limit': protocol['comparison_limit'],
              'aggregation': 'Mean of per-seed AUCs, not AUC of averaged predictions',
              'summary': summary, 'shared_test_comparison': comparison,
              'bootstrap': {'replicates': 500, 'unit': 'student', 'seed': 20261107,
                  'uncertainty': 'Conditional on the six saved checkpoints and fitted mappings; excludes retraining uncertainty and training-design confounds'},
              'checks': ['All compared target labels and event identifiers match exactly',
                  'Nonuniform weighted AUC implementation agrees with sklearn to 1e-12',
                  'All six base validation AUCs reproduced within 1e-5',
                  'All seeds and fixed methods retained'],
              'artifacts': {f.name: sha(f) for f in OUT.glob('*predictions.npz')}}
    dump(OUT / 'summary.json', result)
    pd.DataFrame(summary).to_csv(OUT / 'auc_summary.csv', index=False)
    pd.DataFrame(comparison).to_csv(OUT / 'paired_comparison.csv', index=False)
    dump(OUT / 'paired_bootstrap_deltas.json', bootstrap)
    print(json.dumps({'summary': summary, 'comparison': comparison}, indent=2))


if __name__ == '__main__':
    main()
