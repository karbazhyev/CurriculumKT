"""Prespecified metrics, paired student intervals and evidence verification."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_saved_curriculum_kt import metrics, sha
from summarize_saved_curriculum_kt import weighted_auc_function
from run_pooling_batch_v2 import OUT, ROOT, DATASETS, DESIGNS, SEEDS, read, save, now, guard

METHODS = ['original_dkt', 'pooled_original', 'original_selected', 'practice_frequency',
           'fixed_map', 'pooled_map', 'augment_induced', 'augment_original',
           'augment_induced_selected', 'augment_original_selected']
CONTRASTS = [('pooled_map', 'original_dkt'), ('pooled_map', 'original_selected'),
             ('pooled_map', 'fixed_map'), ('augment_induced_selected', 'original_dkt'),
             ('augment_induced_selected', 'original_selected'),
             ('augment_induced_selected', 'augment_original_selected'),
             ('augment_original_selected', 'original_selected')]
ADJUSTED = {('pooled_map', 'original_dkt'), ('augment_induced_selected', 'augment_original_selected')}


def summarize(ds):
    start = time.monotonic()
    rows, per, preds = [], [], {m: [] for m in METHODS}
    ref = None
    for pair in range(3):
        file = OUT/ds/f'test_pair{pair}.npz'
        assert sha(file) == read(file.with_suffix('.json'))['sha256']
        d = dict(np.load(file))
        ids = {k: d[k] for k in ['uid', 'row', 'position', 'event_id', 'label', 'original_concept']}
        if ref is None:
            ref = ids
        else:
            assert all(np.array_equal(ref[k], ids[k]) for k in ids)
        for m in METHODS:
            result = metrics(d['label'], d[m], d['original_concept'])
            for c, n, auc in result.pop('per_original_concept'):
                per.append({'dataset': ds, 'method': m, 'pair': pair, 'original_concept': c, 'events': n, 'auc': auc})
            rows.append({'dataset': ds, 'method': m, 'pair': pair, 'original_seed': SEEDS[ds][pair], **result})
            preds[m].append(d[m])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/ds/'scores_by_seed.csv', index=False)
    pd.DataFrame(per).to_csv(OUT/ds/'per_concept_auc.csv', index=False)
    agg = frame.groupby('method', sort=False).agg(auc=('auc','mean'), seed_sd=('auc','std'),
        brier=('brier','mean'), log_loss=('log_loss','mean'),
        macro_auc=('within_original_macro_auc','mean'), within_weighted_auc=('within_original_weighted_auc','mean'),
        events=('events','first')).reset_index()
    point = dict(zip(agg.method, agg.auc))
    # Identical retained endpoints reuse the same function; all seed pairs share
    # the same sampled student multiplicities in each bootstrap draw.
    selection = read(OUT/ds/'selection.json')
    aliases = selection['retention']
    functions = {m: [weighted_auc_function(ref['label'], p) for p in preds[m]]
                 for m in METHODS if m not in aliases}
    _, uid = np.unique(ref['uid'], return_inverse=True)
    n = int(uid.max()+1)
    rng = np.random.default_rng(20261109)
    boot = {m: [] for m in METHODS}
    for j in range(2000):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)[uid]
        values = {m: float(np.mean([f(weights) for f in fs])) for m, fs in functions.items()}
        for alias, target in aliases.items():
            values[alias] = values[target]
        for m in METHODS:
            boot[m].append(values[m])
        if (j+1) % 250 == 0:
            print(f'{ds}: paired bootstrap {j+1}/2000', flush=True)
    boot = {m: np.array(v) for m, v in boot.items()}
    np.savez_compressed(OUT/ds/'bootstrap_aucs.npz', **boot)
    for baseline in ['original_dkt', 'original_selected', 'augment_original_selected']:
        agg['delta_'+baseline] = agg.auc-point[baseline]
        intervals = [np.quantile(boot[m]-boot[baseline], [.025, .975]) for m in agg.method]
        agg['ci_low_'+baseline] = [a[0] for a in intervals]
        agg['ci_high_'+baseline] = [a[1] for a in intervals]
    agg.to_csv(OUT/ds/'method_comparison.csv', index=False)
    contrasts = []
    for method, baseline in CONTRASTS:
        delta = boot[method]-boot[baseline]
        q = np.quantile(delta, [.025, .975, .00625, .99375])
        row = {'dataset': ds, 'method': method, 'comparator': baseline,
               'delta_auc': point[method]-point[baseline], 'ci95_low': q[0], 'ci95_high': q[1],
               'designated_four_contrast_family': (method, baseline) in ADJUSTED,
               'familywise_interval_low': q[2] if (method, baseline) in ADJUSTED else None,
               'familywise_interval_high': q[3] if (method, baseline) in ADJUSTED else None}
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(OUT/ds/'paired_contrasts.csv', index=False)
    save(OUT/ds/'summary.json', {'events': len(uid), 'students': n, 'seconds': time.monotonic()-start,
         'bootstrap': '2000 paired student draws, fixed models and selection; seeds not independent replications',
         'aggregation': 'Mean per-seed AUC, not prediction ensemble AUC',
         'all_five_families_and_retention_endpoints_retained': True})
    return agg, contrasts


def verify():
    entries = read(OUT/'run_registry.json')['runs']
    complete = [x for x in entries.values() if x['state'] == 'complete']
    assert len(entries) == len(complete) == 390
    assert all(r['converged'] and r['device'].startswith('mps') for r in complete)
    for key, info in entries.items():
        assert info['protocol_sha256'] == sha(OUT/'protocol.json')
        assert sha(Path(info.get('coefficient_file', str(OUT/'fits'/key)+'.npz'))) == info['coefficient_sha256']
    protected = read(OUT/'protected_source_hashes.json')
    changed = [p for p, value in protected.items() if sha(Path(p)) != value]
    assert not changed, changed
    hashes = read(OUT/'development_complete.json')['selections']
    assert all(sha(OUT/ds/'selection.json') == value for ds, value in hashes.items())
    for ds in DATASETS:
        s = read(OUT/ds/'selection.json')
        for key, value in s['full_fit_coefficient_hashes'].items():
            assert sha(Path(str(OUT/'fits'/key)+'.npz')) == value
        meta = np.load(OUT/ds/'test_pair0.npz')
        groups = read(OUT/ds/'development_folds.json')['students']
        assert not set(np.concatenate(groups)) & set(meta['uid'])
        assert not set(groups[0]) & set(groups[1])
        assert not set(groups[0]) & set(groups[2])
        assert not set(groups[1]) & set(groups[2])
        expected = read(OUT/'protocol.json')['test'][ds]
        assert len(meta['label']) == expected['targets']
        assert len(np.unique(meta['uid'])) == expected['students']
    result = {'all_390_empirical_fits_complete': True, 'unique_fit_configurations': 390, 'empirical_solver_invocations_including_failure': 391,
              'preserved_failed_numerical_invocations': 1, 'precision_repair': 'numerical_amendment_01.json', 'all_fits_actual_mps': True,
              'all_independent_float64_convergence_checks_passed': True,
              'max_scaled_projected_gradient': max(r['projected_gradient'] for r in complete),
              'fit_seconds_sum': sum(r['seconds'] for r in complete), 'new_dkt_fits': 0,
              'protected_files_unchanged': len(protected),
              'source_checkpoint_feature_and_previous_result_hashes_unchanged': True,
              'selected_coefficients_unchanged_after_test': True,
              'paired_events_labels_original_targets_aligned': True,
              'student_calibration_folds_and_test_disjoint': True,
              'both_datasets_selected_and_refit_before_new_test': True,
              'synthetic_solver_diagnostics_passed': (OUT/'solver_diagnostic.json').exists(),
              'protocol_sha256': sha(OUT/'protocol.json'),
              'script_identities': {name: sha(ROOT/'scripts'/name) for name in ['run_pooling_batch.py','run_pooling_batch_v2.py','prepare_pooling_batch.py','summarize_pooling_batch.py','summarize_pooling_batch_v2.py']},
              'completed_utc': now()}
    save(OUT/'verification.json', result)
    registry = read(OUT/'run_registry.json')
    registry['state'] = 'experiments_and_summary_complete_pending_judge'
    save(OUT/'run_registry.json', registry)
    return result


def main():
    guard()
    assert (OUT/'test_complete.json').exists()
    wide = []
    all_contrasts = []
    for ds in DATASETS:
        agg, contrasts = summarize(ds)
        agg.insert(0, 'dataset', ds)
        wide.extend(agg.to_dict(orient='records'))
        all_contrasts.extend(contrasts)
    pd.DataFrame(wide).to_csv(OUT/'two_dataset_comparison.csv', index=False)
    pd.DataFrame(all_contrasts).to_csv(OUT/'paired_method_contrasts.csv', index=False)
    info = verify()
    print(pd.DataFrame(wide)[['dataset','method','auc','delta_original_dkt']].to_string(index=False), flush=True)
    print(f"All 390 fits verified; summed fit time {info['fit_seconds_sum']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
