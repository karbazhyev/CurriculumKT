"""Independent CPU audit of completed RobustKT reports and saved predictions."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from extract_robustkt_curriculum import guard, require_test_ready
from prepare_robustkt_curriculum import OUT, ROOT, sha
from run_robustkt_sweeps import read, dump, now


def main():
    protocol = guard()
    require_test_ready()
    registry = read(OUT / 'run_registry.json')
    assert registry['state'] == 'complete'
    assert read(OUT / 'summary_worker.json')['state'] == 'complete'
    verification = read(OUT / 'verification.json')
    assert verification['online_final_verification'] == 'passed'
    assert verification['backbone_fits'] == 40 and verification['selected_checkpoints'] == 12
    assert verification['all_312_readout_fits_complete_and_converged']
    history_path = ROOT / 'results/model_sequence_20261107/robustkt_online_history_checks.json'
    history = read(history_path)
    assert history['all_currently_completed_runs_verified'] and history['counts'] == {'verified': 40}
    query_path = ROOT / 'results/model_sequence_20261107/robustkt_stable_queries_mps_checks.json'
    query = read(query_path)
    assert query['status'] == 'passed' and query['all_twelve_selected_checkpoints']
    assert query['device'] == 'mps' and query['cpu_fallback'] is False
    online = read(OUT / 'online_verification.json')
    assert online['aggregate_metrics_and_protocols_match']
    assert {r['stage'] for r in online['runs']} == {'development', 'evaluate', 'report'}
    assert all(r['verified'] and r['state'] == 'finished' for r in online['runs'])
    prediction_checks = read(OUT / 'test_prediction_verification.json')
    assert prediction_checks['status'] == 'passed' and len(prediction_checks['checks']) == 6
    for name in ('calibration_feature_verification.json', 'test_feature_verification.json'):
        check = read(OUT / name)
        assert check['status'] == 'passed' and len(check['checks']) == 12
    headline = pd.read_csv(OUT / 'three_way_auc_comparison.csv')
    aggregate = pd.read_csv(OUT / 'two_dataset_comparison.csv')
    contrasts = pd.read_csv(OUT / 'paired_method_contrasts.csv')
    assert len(headline) == 2 and len(aggregate) == 20 and len(contrasts) == 8
    max_bootstrap_delta = 0.
    for ds in ('assist2012', 'assist2017'):
        folder = OUT / ds
        summary = read(folder / 'summary.json')
        assert summary['protocol_sha256'] == sha(OUT / 'protocol.json')
        for name, expected in summary['output_sha256'].items():
            assert sha(folder / name) == expected
        preds = []
        scores = pd.read_csv(folder / 'scores_by_seed.csv')
        assert len(scores) == 30
        for pair in range(3):
            path = folder / f'test_pair{pair}.npz'
            assert sha(path) == summary['prediction_sha256'][path.name]
            checked = next(r for r in prediction_checks['checks'] if r['dataset'] == ds and r['pair'] == pair)
            assert sha(path) == checked['prediction_sha256']
            with np.load(path) as data:
                d = dict(data)
            preds.append(d)
            for method in protocol['methods']:
                row = scores[(scores.pair == pair) & (scores.method == method)].iloc[0]
                assert abs(row.auc - checked['aucs'][method]) < 1e-12
                probabilities = np.asarray(d[method], dtype=np.float64)
                assert abs(row.brier - np.mean((probabilities - d['label'])**2)) < 1e-12
                # The frozen scorer clips probabilities at 1e-7 for proper scores.
                clipped = np.clip(probabilities, 1e-7, 1-1e-7)
                assert abs(row.log_loss - log_loss(d['label'], clipped)) < 1e-12
        means = {}
        for method in protocol['methods']:
            values = scores[scores.method == method].auc.to_numpy()
            means[method] = float(np.mean(values))
            row = aggregate[(aggregate.dataset == ds) & (aggregate.method == method)].iloc[0]
            assert abs(row.auc - means[method]) < 1e-12
            assert abs(row.seed_sd - np.std(values, ddof=1)) < 1e-12
        row = headline[headline.dataset == ds].iloc[0]
        assert row.model == 'robustkt'
        for column, method in [('baseline_auc', 'original_kt'), ('induced_auc', 'raw_induced_diagnostic'),
                               ('currkt_pure_map_back_auc', 'pooled_map')]:
            assert abs(row[column] - means[method]) < 1e-12
        with np.load(folder / 'bootstrap_aucs.npz') as data:
            bootstrap = dict(data)
        assert set(bootstrap) == set(protocol['methods']) - {'raw_induced_diagnostic'}
        assert all(v.shape == (2000,) and np.isfinite(v).all() for v in bootstrap.values())
        selection = read(folder / 'selection.json')
        for alias, method in selection['retention'].items():
            assert np.array_equal(bootstrap[alias], bootstrap[method])
        # Verify actual weighted-bootstrap draws through an independent AUC path.
        _, students = np.unique(preds[0]['uid'], return_inverse=True)
        n = int(students.max()+1)
        rng = np.random.default_rng(protocol['bootstrap']['seed'])
        assert summary['rejected_degenerate_draws'] == 0
        for draw in range(3):
            weights = np.bincount(rng.integers(n, size=n), minlength=n)[students]
            for method in bootstrap:
                if method in selection['retention']:
                    continue
                value = np.mean([roc_auc_score(d['label'], d[method], sample_weight=weights) for d in preds])
                delta = abs(value - bootstrap[method][draw])
                assert delta < 1e-12, (ds, method, draw, delta)
                max_bootstrap_delta = max(max_bootstrap_delta, float(delta))
        for record in contrasts[contrasts.dataset == ds].to_dict('records'):
            a, b = record['method'], record['comparator']
            assert abs(record['delta_auc'] - (means[a] - means[b])) < 1e-12
            ci = np.quantile(bootstrap[a] - bootstrap[b], [.025, .975, .00625, .99375])
            assert abs(record['ci95_low'] - ci[0]) < 1e-12
            assert abs(record['ci95_high'] - ci[1]) < 1e-12
            if record['designated_four_contrast_family']:
                assert abs(record['familywise_interval_low'] - ci[2]) < 1e-12
                assert abs(record['familywise_interval_high'] - ci[3]) < 1e-12
        for record in aggregate[aggregate.dataset == ds].to_dict('records'):
            if record['method'] not in bootstrap:
                continue
            ci = np.quantile(bootstrap[record['method']] - bootstrap['original_kt'], [.025, .975])
            assert abs(record['ci95_low_original_kt'] - ci[0]) < 1e-12
            assert abs(record['ci95_high_original_kt'] - ci[1]) < 1e-12
    guard()
    require_test_ready()
    names = ['RESULTS.md', 'three_way_auc_comparison.csv', 'two_dataset_comparison.csv',
             'paired_method_contrasts.csv', 'verification.json', 'online_verification.json',
             'test_prediction_verification.json', 'test_feature_verification.json',
             'calibration_feature_verification.json', 'development_completion_verification.json']
    completion = {'checked_utc': now(), 'model': 'robustkt', 'state': 'complete',
                  'protocol_sha256': sha(OUT / 'protocol.json'),
                  'verifier_sha256': sha(Path(__file__)),
                  'training_history_audit_sha256': sha(history_path),
                  'selected_mps_query_audit_sha256': sha(query_path),
                  'final_artifact_sha256': {name: sha(OUT / name) for name in names},
                  'frozen_source_check_passed': True, 'all_dataset_summary_hashes_passed': True,
                  'main_auc_columns_match_full_report': True,
                  'all_seed_auc_and_proper_scores_reproduced': True,
                  'all_saved_interval_quantiles_reproduced': True,
                  'independent_first_three_student_draws_per_dataset_max_difference': max_bootstrap_delta,
                  'auc_rows': headline.to_dict('records'),
                  'report_wandb_url': registry['wandb_runs']['report']['url']}
    dump(ROOT / 'results/model_sequence_20261107/robustkt_completion.json', completion)
    print({'status': 'passed', 'model': 'robustkt', 'auc_rows': completion['auc_rows']})


if __name__ == '__main__':
    main()
