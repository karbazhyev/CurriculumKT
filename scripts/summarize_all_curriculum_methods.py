"""Score every prespecified readout, preserving stable AND/OR ranking scores."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from curriculum_readouts import dump
from evaluate_saved_curriculum_kt import ROOT, OUT as OLD, metrics, sha
from summarize_saved_curriculum_kt import weighted_auc_function

METHODS = ['item_overlap', 'direct_item_average', 'practice_frequency', 'equal_children',
           'learned_monotone', 'parent_calibration', 'intersection_item_overlap',
           'intersection_practice_frequency', 'item_aware_reference_average', 'and_product', 'noisy_or', 'raw_induced']


def rank_scores(data, method):
    return data[{'and_product': 'and_log_score', 'noisy_or': 'noisy_or_log_score'}.get(method, method)]


def calculate(data, method, select):
    y, p, c = data['label'][select], data[method][select], data['original_concept'][select]
    s = rank_scores(data, method)[select]
    value = metrics(y, p, c)
    value.pop('per_original_concept')
    value['auc'] = float(roc_auc_score(y, s))
    aucs, ns = [], []
    for k in np.unique(c):
        ix = c == k
        if len(np.unique(y[ix])) == 2:
            aucs.append(roc_auc_score(y[ix], s[ix]))
            ns.append(ix.sum())
    value['within_original_macro_auc'] = float(np.mean(aucs))
    value['within_original_weighted_auc'] = float(np.average(aucs, weights=ns))
    value['auc_score_scale'] = 'logarithmic ranking score to avoid probability saturation' if method in ['and_product', 'noisy_or'] else 'probability'
    return value


def main(dataset):
    out = ROOT / f'results/curriculum_kt_all_methods_20261107/{dataset}'
    common = np.load(OLD / 'shared_student_ids.npy') if dataset == 'assist2012' else None
    rows, predictions = [], {}
    reference = None
    for seed in [42, 84, 126]:
        file = out / (f'seed{seed}_test_predictions.npz' if dataset == 'assist2012' else f'induced_seed{seed}_test_predictions.npz')
        d = dict(np.load(file))
        assert np.array_equal(d['item_overlap'], d['direct_item_average'])
        selected = np.isin(d['uid'], common) if common is not None else np.ones(len(d['uid']), bool)
        ids = {k: d[k][selected] for k in ['uid', 'row', 'position', 'label', 'original_concept']}
        if reference is None:
            reference = ids
        else:
            assert all(np.array_equal(reference[k], ids[k]) for k in ids)
        for method in METHODS:
            for population, selection in [('comparison_test', selected)] + ([('full_induced_test', np.ones(len(d['uid']), bool))] if common is not None else []):
                result = calculate(d, method, selection)
                rows.append({'dataset': dataset, 'population': population, 'seed': seed, 'method': method,
                             'students': len(np.unique(d['uid'][selection])), **result})
            predictions.setdefault(method, []).append(rank_scores(d, method)[selected])
    baseline_seeds = [42, 84, 255] if dataset == 'assist2012' else [42, 84, 126]
    for seed in baseline_seeds:
        file = OLD / f'original_seed{seed}_original_shared_predictions.npz' if dataset == 'assist2012' else out / f'original_seed{seed}_test_predictions.npz'
        d = dict(np.load(file))
        assert all(np.array_equal(reference[k], d[k]) for k in reference)
        result = calculate(d, 'original_dkt', np.ones(len(d['label']), bool))
        rows.append({'dataset': dataset, 'population': 'comparison_test', 'seed': seed,
                     'method': 'original_dkt', 'students': len(np.unique(d['uid'])), **result})
        predictions.setdefault('original_dkt', []).append(d['original_dkt'])
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'scores_by_seed.csv', index=False)
    aggregate = frame.groupby(['population', 'method']).agg(
        auc=('auc', 'mean'), seed_sd=('auc', 'std'), events=('events', 'first'), students=('students', 'first'),
        brier=('brier', 'mean'), log_loss=('log_loss', 'mean'), macro_original_auc=('within_original_macro_auc', 'mean')).reset_index()
    y = reference['label']
    functions = {method: [weighted_auc_function(y, p) for p in ps] for method, ps in predictions.items()}
    _, student = np.unique(reference['uid'], return_inverse=True)
    n_students = student.max() + 1
    rng = np.random.default_rng(20261107)
    bootstrap = {method: [] for method in METHODS}
    for i in range(500):
        counts = np.bincount(rng.integers(n_students, size=n_students), minlength=n_students)
        w = counts[student]
        baseline = np.mean([fn(w) for fn in functions['original_dkt']])
        for method in METHODS:
            bootstrap[method].append(np.mean([fn(w) for fn in functions[method]]) - baseline)
    baseline = aggregate[(aggregate.population == 'comparison_test') & (aggregate.method == 'original_dkt')].auc.iloc[0]
    aggregate['delta_auc'] = np.where(aggregate.population == 'comparison_test', aggregate.auc - baseline, np.nan)
    aggregate['ci_low'] = np.nan
    aggregate['ci_high'] = np.nan
    for method in METHODS:
        idx = (aggregate.population == 'comparison_test') & (aggregate.method == method)
        aggregate.loc[idx, ['ci_low', 'ci_high']] = np.quantile(bootstrap[method], [.025, .975])
    aggregate.to_csv(out / 'method_comparison.csv', index=False)
    dump(out / 'bootstrap_deltas.json', bootstrap)
    rows_json = json.loads(aggregate.to_json(orient='records'))
    dump(out / 'summary.json', {'dataset': dataset, 'results': rows_json,
        'bootstrap': '500 paired student resamples; conditional on saved/fitted models; not simultaneous multiple-comparison intervals',
        'selection': 'All methods retained. Highest observed test AUC is exploratory; not a selection/confirmation claim.',
        'and_or': 'Diagnostic products of correctness marginals, not validated mastery models. AUC uses stable logarithmic scores.',
        'direct_average': 'Exactly identical to item-overlap weighting for this concept-only DKT',
        'prediction_sha256': {f.name: sha(f) for f in out.glob('*test_predictions.npz')}})
    print(aggregate[aggregate.population == 'comparison_test'].sort_values('auc', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['assist2012', 'assist2017'])
    main(parser.parse_args().dataset)
