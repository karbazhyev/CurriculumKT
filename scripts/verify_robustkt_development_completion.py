"""Read-only replay of calibration selections before held-out extraction."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import wandb

from extract_robustkt_curriculum import guard, require_test_ready
from prepare_robustkt_curriculum import OUT, ROOT, DESIGNS, LAMBDAS, SEEDS, sha
from run_robustkt_curriculum_readouts import design_inputs, fit_id, predict
from run_robustkt_sweeps import dump, now, read


def verify():
    protocol = guard()
    require_test_ready()
    registry = read(OUT / 'run_registry.json')
    assert not registry['heldout_started']
    worker = read(OUT / 'readout_worker.json')
    assert worker['state'] == 'complete' and worker['stage'] == 'development'
    try:
        os.kill(worker['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('Development worker still alive')

    fits = registry['fits']
    assert len(fits) == 312
    for key, fit in fits.items():
        assert fit['state'] == 'complete' and fit['converged']
        assert fit['device'] in ('mps', 'mps:0') and fit['mps_available_in_fit']
        assert fit['projected_gradient'] <= 0.0002 and fit['test_used'] is False
        assert fit['protocol_sha256'] == sha(OUT / 'protocol.json')
        assert sha(Path(fit['coefficient_file'])) == fit['coefficient_sha256']
        marker = read(Path(fit['coefficient_file']).with_suffix('.json'))
        assert all(fit[k] == v for k, v in marker.items())
        with np.load(fit['coefficient_file']) as coefficients:
            assert all(np.isfinite(coefficients[k]).all() for k in coefficients.files)

    maximum_auc_difference = 0.
    datasets = {}
    for ds in protocol['datasets']:
        selection = read(OUT / ds / 'selection.json')
        table = pd.read_csv(OUT / ds / 'development_scores.csv')
        assert len(table) == 144 and selection['test_used'] is False
        saved_folds = read(OUT / ds / 'development_folds.json')
        groups = [np.asarray(g) for g in saved_folds['students']]
        assert len(set(np.concatenate(groups))) == sum(map(len, groups))
        native_scores = []
        selected_scores = {}
        for design in DESIGNS:
            replayed = {lam: [] for lam in LAMBDAS}
            for pair in range(3):
                meta, x, _, _, _, native = design_inputs(protocol, ds, pair, 'calibration', design)
                assert set(np.concatenate(groups)) == set(meta['uid'])
                expected_groups = np.array_split(np.random.default_rng(20261108).permutation(np.unique(meta['uid'])), 3)
                assert all(np.array_equal(a, b) for a, b in zip(groups, expected_groups))
                for fold, group in enumerate(groups):
                    valid = np.isin(meta['uid'], group)
                    assert valid.sum() == saved_folds['events'][fold]
                    if design == 'pooled_original':
                        native_scores.append(roc_auc_score(meta['label'][valid], native[valid]))
                    for lam in LAMBDAS:
                        key = fit_id(ds, design, pair, lam, fold)
                        assert fits[key]['fit_students'] == len(np.unique(meta['uid'][~valid]))
                        assert fits[key]['fit_events'] == int((~valid).sum())
                        with np.load(fits[key]['coefficient_file']) as coefficients:
                            prediction = predict(x[valid], meta['original_concept'][valid], dict(coefficients))
                        auc = float(roc_auc_score(meta['label'][valid], prediction))
                        row = table[(table.design == design) & (table.pair == pair) &
                                    (table['lambda'] == lam) & (table.fold == fold)]
                        assert len(row) == 1
                        difference = abs(auc - float(row.iloc[0].auc))
                        maximum_auc_difference = max(maximum_auc_difference, difference)
                        assert difference < 1e-12
                        replayed[lam].append(auc)
            means = {lam: float(np.mean(values)) for lam, values in replayed.items()}
            winner = max(lam for lam, score in means.items() if max(means.values()) - score <= 1e-6)
            choice = selection['families'][design]
            assert winner == choice['lambda']
            assert all(abs(score - choice['mean_fold_seed_auc'][str(lam)]) < 1e-12 for lam, score in means.items())
            assert abs(means[winner] - choice['selected_auc']) < 1e-12
            selected_scores[design] = means[winner]
            for pair in range(3):
                fit = fits[fit_id(ds, design, pair)]
                assert fit['lambda_value'] == winner
                assert fit['fit_events'] == saved_folds['calibration_events']
                assert fit['fit_students'] == saved_folds['calibration_students']
        native_auc = float(np.mean(native_scores))
        assert abs(native_auc - selection['native_cv_auc']) < 1e-12
        original = 'pooled_original' if selected_scores['pooled_original'] > native_auc + 1e-6 else 'original_kt'
        control = selected_scores['pooled_original'] if original == 'pooled_original' else native_auc
        retention = {'original_selected': original}
        for design in ('augment_induced', 'augment_original'):
            retention[design + '_selected'] = design if selected_scores[design] > control + 1e-6 else original
        assert retention == selection['retention']
        datasets[ds] = {'calibration_events': saved_folds['calibration_events'],
                        'calibration_students': saved_folds['calibration_students'],
                        'student_folds_disjoint': True, 'all_144_cv_aucs_replayed': True,
                        'selected_lambdas': {d: selection['families'][d]['lambda'] for d in DESIGNS},
                        'retention_reproduced': retention}

    legacy = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
    credential = json.loads((legacy / 'configs/wandb.json').read_text())
    api = wandb.Api(api_key=credential['api_key'], timeout=30)
    run = api.run('/'.join([protocol['wandb']['entity'], protocol['wandb']['project'],
                           registry['wandb_runs']['development']['run_id']]))
    assert run.state == 'finished'
    assert run.config['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert run.config['test_used_for_selection'] is False
    assert run.summary['readout/completed_fits'] == 312
    assert run.summary['readout/all_convergence_checks_passed'] is True
    rows = list(run.scan_history(keys=['readout/fit', 'readout/lambda', 'readout/projected_gradient',
                                      'readout/fit_seconds', 'readout/converged'], page_size=500))
    assert len(rows) == 312 and {r['readout/fit'] for r in rows} == set(fits)
    for row in rows:
        fit = fits[row['readout/fit']]
        assert row['readout/converged'] is True
        for online, local in [('lambda', 'lambda_value'), ('projected_gradient', 'projected_gradient'),
                              ('fit_seconds', 'seconds')]:
            assert abs(float(row['readout/' + online]) - float(fit[local])) < 1e-12
    for ds in datasets:
        selection = read(OUT / ds / 'selection.json')
        for design, value in selection['families'].items():
            assert run.summary[f'calibration/{ds}/{design}/selected_lambda'] == value['lambda']
            assert abs(run.summary[f'calibration/{ds}/{design}/mean_cv_auc'] - value['selected_auc']) < 1e-12

    result = {'checked_utc': now(), 'status': 'passed', 'protocol_sha256': sha(OUT / 'protocol.json'),
              'verifier_sha256': sha(Path(__file__)), 'all_312_converged_on_mps': True,
              'all_312_coefficient_hashes_verified': True,
              'maximum_projected_gradient': max(f['projected_gradient'] for f in fits.values()),
              'all_288_cv_aucs_replayed': True, 'maximum_cv_auc_difference': maximum_auc_difference,
              'datasets': datasets, 'online_state': run.state, 'online_fit_records_verified': len(rows),
              'wandb_url': run.url, 'test_used': False, 'heldout_started': False}
    guard(); require_test_ready()
    dump(OUT / 'development_completion_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with ExitStack() as stack:
        for path in [ROOT / 'results/robustkt_sweeps_20261107/worker.lock', ROOT/'results/simplekt_sweeps_20261107/worker.lock',
                     ROOT / 'results/model_sequence_20261107/gpu_worker.lock']:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
