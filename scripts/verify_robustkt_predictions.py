"""Independently replay frozen RobustKT readouts on CPU; no fitting or inference."""
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from extract_robustkt_curriculum import guard, require_test_ready, load_features
from prepare_robustkt_curriculum import OUT, SEEDS, sha
from run_robustkt_sweeps import read, dump, now


def main():
    protocol = guard()
    require_test_ready()
    assert read(OUT / 'test_feature_verification.json')['status'] == 'passed'
    assert read(OUT / 'test_complete.json')['six_paired_prediction_files']
    development = read(OUT / 'development_complete.json')
    registry = read(OUT / 'run_registry.json')
    assert development['completed_utc'] < registry['heldout_started_utc']
    report = {'checked_utc': now(), 'status': 'running', 'device': 'cpu',
              'model_calls': 0, 'empirical_fits': 0,
              'protocol_sha256': sha(OUT / 'protocol.json'),
              'verifier_sha256': sha(Path(__file__)), 'checks': []}
    for ds, spec in protocol['datasets'].items():
        selection = read(OUT / ds / 'selection.json')
        assert selection['test_used'] is False
        assert sha(OUT / ds / 'selection.json') == development['selections'][ds]
        assert sha(Path(spec['mapping']['path'])) == spec['mapping']['sha256']
        with np.load(spec['mapping']['path']) as mapping:
            support = mapping['counts'] > 0
            weights = mapping['practice_frequency']
        width = int(support.sum(1).max())
        compact = np.zeros((len(support), width), dtype=np.float64)
        active = np.zeros_like(compact, dtype=bool)
        for concept in range(len(support)):
            children = np.flatnonzero(support[concept])
            compact[concept, :len(children)] = weights[concept, children]
            active[concept, :len(children)] = True
        norm = np.sqrt(np.sum(compact**2, axis=1) + 1e-12)
        reference_meta = None
        for pair, seed in enumerate(SEEDS):
            path = OUT / ds / f'test_pair{pair}.npz'
            marker = read(path.with_suffix('.json'))
            assert sha(path) == marker['sha256']
            assert marker['protocol_sha256'] == report['protocol_sha256']
            assert marker['selection_sha256'] == development['selections'][ds]
            with np.load(path) as saved:
                predictions = dict(saved)
            meta, original = load_features(ds, 'original', seed, 'test')
            ind_meta, induced = load_features(ds, 'induced', seed, 'test')
            other_meta, other = load_features(ds, 'original', SEEDS[(pair+1) % 3], 'test')
            if reference_meta is None:
                reference_meta = meta
            for k in meta:
                assert np.array_equal(meta[k], reference_meta[k])
                assert np.array_equal(meta[k], predictions[k])
                assert np.array_equal(meta[k], ind_meta[k])
                assert np.array_equal(meta[k], other_meta[k])
            assert set(predictions) == set(meta) | set(protocol['methods'])
            c = meta['original_concept']
            expected = {'original_kt': original['native_probability'],
                        'raw_induced_diagnostic': induced['native_probability'],
                        'practice_frequency': induced['practice_frequency']}
            max_delta = {}
            for design in ('pooled_map', 'pooled_original', 'augment_induced', 'augment_original'):
                key = f'{ds}/{design}/pair{pair}/refit'
                coef = Path(str(OUT / 'fits' / key) + '.npz')
                assert sha(coef) == selection['full_fit_coefficient_hashes'][key]
                with np.load(coef) as values:
                    q, b = values['q'], values['b']
                assert np.isfinite(q).all() and np.isfinite(b).all()
                assert (q >= 0).all()
                replay = np.empty(len(c), dtype=np.float64)
                # Reconstruct each design independently of the fitting runner.
                for lo in range(0, len(c), 4096):
                    sl = slice(lo, lo+4096)
                    cc = c[sl]
                    z = np.asarray(original['native_logit'][sl], dtype=np.float32)
                    if design == 'pooled_map':
                        x = np.array(induced['child_logits'][sl], dtype=np.float32)
                        x *= active[cc]
                        x *= norm[cc, None].astype(np.float32)
                    elif design == 'pooled_original':
                        x = z[:, None]
                    else:
                        second = (logit(np.clip(induced['practice_frequency'][sl], 1e-6, 1-1e-6))
                                  if design == 'augment_induced' else other['native_logit'][sl])
                        x = np.column_stack((z, second)).astype(np.float32)
                    replay[sl] = expit(np.sum(x.astype(np.float64) * q[cc], axis=1) + b[cc])
                expected[design] = replay
            for endpoint, chosen in selection['retention'].items():
                expected[endpoint] = expected[chosen]
                assert np.array_equal(predictions[endpoint], predictions[chosen])
            aucs = {}
            for method in protocol['methods']:
                p = predictions[method]
                assert p.shape == meta['label'].shape
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                max_delta[method] = float(np.max(np.abs(p - expected[method])))
                assert max_delta[method] < 1e-12, (ds, pair, method, max_delta[method])
                aucs[method] = float(roc_auc_score(meta['label'], p))
            report['checks'].append({'dataset': ds, 'seed': seed, 'pair': pair,
                'prediction_sha256': sha(path), 'events': len(c),
                'students': len(np.unique(meta['uid'])), 'metadata_exact': True,
                'fixed_coefficients_replayed': True, 'max_prediction_difference': max_delta,
                'aucs': aucs})
    assert len(report['checks']) == 6
    guard()
    require_test_ready()
    report.update(status='passed', completed_utc=now())
    dump(OUT / 'test_prediction_verification.json', report)
    print({'status': 'passed', 'predictions_verified': 6,
           'maximum_replay_difference': max(v for c in report['checks'] for v in c['max_prediction_difference'].values())})


if __name__ == '__main__':
    main()
