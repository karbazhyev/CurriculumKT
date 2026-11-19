"""CPU-only artifact verification; never runs a model or fits a readout."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from extract_robustkt_curriculum import guard, require_test_ready, directory, load_features
from prepare_robustkt_curriculum import OUT, SEEDS, sha
from run_robustkt_sweeps import read, dump, now


def verify(split, allow_incomplete=False):
    protocol = guard()
    if split == 'test':
        require_test_ready()
        review = read(OUT / 'development_completion_verification.json')
        assert review['status'] == 'passed'
        assert review['protocol_sha256'] == sha(OUT / 'protocol.json')
    report = {'checked_utc': now(), 'split': split, 'device': 'cpu',
              'model_calls': 0, 'empirical_fits': 0,
              'heldout_arrays_loaded': split == 'test',
              'protocol_sha256': sha(OUT / 'protocol.json'),
              'verifier_sha256': sha(Path(__file__)), 'checks': [], 'pending': []}
    for ds, spec in protocol['datasets'].items():
        source = spec['files'][split]
        assert sha(Path(source['path'])) == source['sha256']
        with np.load(source['path']) as data:
            oc, ic, responses = (data[k] for k in
                                 ('original_concepts', 'induced_concepts', 'responses'))
            assert np.array_equal(oc >= 0, ic >= 0)
            assert np.array_equal(oc >= 0, responses >= 0)
            row, prior = np.nonzero((oc[:, 1:] >= 0) & (oc[:, :-1] >= 0))
            pos = prior + 1
            expected = {'row': row, 'position': pos, 'uid': data['uid'][row],
                        'event_id': data['event_id'][row, pos],
                        'label': responses[row, pos], 'original_concept': oc[row, pos]}
        events = len(row)
        students = np.unique(expected['uid'])
        assert events == spec['counts'][split]['predictable_targets']
        assert len(students) == spec['counts'][split]['students']
        # Calibration verification never opens test arrays. Frozen input hashes
        # and the parent protocol separately protect the complete split identity.
        others = ['train', 'validation'] + (['calibration'] if split == 'test' else [])
        for other in others:
            rec = spec['files'][other]
            assert sha(Path(rec['path'])) == rec['sha256']
            with np.load(rec['path']) as data:
                assert not np.intersect1d(students, data['uid']).size
        mapping = spec['mapping']
        assert sha(Path(mapping['path'])) == mapping['sha256']
        with np.load(mapping['path']) as data:
            support = data['counts'] > 0
            weights = np.asarray(data['practice_frequency'], dtype=np.float64)
        assert support.any(axis=1).all()
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert (weights[~support] == 0).all()
        assert np.allclose(weights.sum(1), 1., atol=1e-6, rtol=0)
        width = int(support.sum(1).max())
        compact = np.zeros((len(support), width))
        active = np.zeros_like(compact, dtype=bool)
        for concept in range(len(support)):
            indices = np.flatnonzero(support[concept])
            compact[concept, :len(indices)] = weights[concept, indices]
            active[concept, :len(indices)] = True
        for taxonomy in ('original', 'induced'):
            for seed in SEEDS:
                key = f'{ds}/{taxonomy}/seed{seed}/{split}'
                path = directory(ds, taxonomy, seed, split) / 'complete.json'
                if not path.exists():
                    report['pending'].append(key)
                    continue
                mark = read(path)
                checkpoint = protocol['selected_models'][f'{ds}_robustkt_{taxonomy}'][str(seed)]
                assert sha(Path(checkpoint['checkpoint'])) == checkpoint['checkpoint_sha256']
                identity = {'checkpoint_sha256': checkpoint['checkpoint_sha256'],
                            'data_sha256': source['sha256'], 'mapping_sha256': mapping['sha256'],
                            'split': split, 'dataset': ds, 'taxonomy': taxonomy, 'seed': seed,
                            'device': 'mps', 'cpu_fallback': False,
                            'selected_query_check_sha256': protocol['selected_query_check_sha256']}
                assert all(mark[k] == v for k, v in identity.items())
                meta, features = load_features(ds, taxonomy, seed, split)
                assert set(meta) == set(expected)
                assert all(np.array_equal(meta[k], v) for k, v in expected.items())
                assert all(np.isfinite(v).all() for v in features.values())
                native = features['native_probability']
                assert native.shape == features['native_logit'].shape == (events,)
                assert ((native >= 0) & (native <= 1)).all()
                native_delta = float(np.max(np.abs(expit(features['native_logit']) - native)))
                assert native_delta < 2e-5 and mark['max_native_query_difference'] < 2e-5
                mapped_delta = 0.
                if taxonomy == 'induced':
                    child = features['child_logits']
                    assert child.shape == (events, width)
                    c = expected['original_concept']
                    assert (child[~active[c]] == 0).all()
                    assert int(active[c].sum()) == mark['independent_child_queries']
                    # Reconstruct the saved training-frequency aggregate in
                    # chunks without invoking the extraction implementation.
                    for lo in range(0, events, 4096):
                        cs = c[lo:lo + 4096]
                        pred = (expit(np.asarray(child[lo:lo + 4096], dtype=np.float64)) * compact[cs]).sum(1)
                        mapped_delta = max(mapped_delta, float(np.max(np.abs(
                            pred - features['practice_frequency'][lo:lo + 4096]))))
                    assert mapped_delta < 1e-12
                else:
                    assert mark['independent_child_queries'] == 0
                report['checks'].append({'key': key, 'events': events, 'students': len(students),
                                         'metadata_exactly_matches_frozen_source': True,
                                         'students_disjoint_from_other_development_splits': True,
                                         'files_and_checkpoint_verified': True,
                                         'native_query_max_probability_difference': native_delta,
                                         'frequency_aggregate_max_difference': mapped_delta,
                                         'complete_sha256': sha(path)})
    complete = not report['pending']
    if complete:
        manifest = read(OUT / f'{split}_features_complete.json')
        assert manifest['all_twelve_complete'] and len(manifest['manifests']) == 12
        assert manifest['protocol_sha256'] == sha(OUT / 'protocol.json')
        for path, expected_hash in manifest['manifests'].items():
            assert sha(Path(path)) == expected_hash
        assert len(report['checks']) == 12
    else:
        assert allow_incomplete, f'Feature extraction still pending: {report["pending"]}'
    guard()
    report.update(status='passed' if complete else 'partial', all_twelve_complete=complete)
    suffix = '' if complete else '_partial'
    dump(OUT / f'{split}_feature_verification{suffix}.json', report)
    print(json.dumps({'status': report['status'], 'verified': len(report['checks']),
                      'pending': len(report['pending']), 'split': split}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['calibration', 'test'])
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    verify(args.split, args.allow_incomplete)
