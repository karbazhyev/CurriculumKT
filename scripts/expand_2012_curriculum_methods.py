"""Apply all discussed readout families to the saved 2012 condition."""
import json
from pathlib import Path

import numpy as np
import torch

from curriculum_readouts import dump, fit_readouts, predict_readouts
from evaluate_saved_curriculum_kt import OUT as OLD, ROOT, sha

OUT = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2012'


def main():
    assert torch.backends.mps.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = dict(np.load(OLD / 'mapping.npz'))
    pairs = np.column_stack([mapping['q_to_original'], np.arange(len(mapping['q_to_original'])), mapping['q_to_induced']])
    protocol = {'dataset': 'assist2012', 'condition': 'historical saved K1000, previously evaluated',
        'status': 'exploratory expansion after first test results were observed; no confirmation claim',
        'seeds': [42, 84, 126], 'fit_split': 'historical validation fold0, also used for checkpoint selection',
        'methods': ['item_overlap', 'direct_item_average', 'practice_frequency', 'equal_children',
                    'learned_monotone', 'parent_calibration', 'intersection_item_overlap',
                    'intersection_practice_frequency', 'item_aware_reference_average', 'and_product', 'noisy_or'],
        'fit': {'device': 'mps', 'optimizer': 'Adam', 'lr': 0.03, 'steps': 1000,
                'parent_min_events': 50, 'parent_min_each_class': 5,
                'intersection_min_events': 20, 'intersection_min_each_class': 2,
                'item_min_events': 5, 'item_min_each_class': 1,
                'monotone_weight_ridge': 0.01, 'intercept_ridge': 0.001,
                'intersection_ridge': 0.01, 'item_ridge': 0.1},
        'selection': 'No test-based coefficient/hyperparameter/epoch/method selection; retain all results',
        'mastery_caveat': 'AND/OR applied only as diagnostic score rules; no latent mastery interpretation',
        'source_protocol_sha256': sha(OLD / 'protocol.json'),
        'source_mapping_sha256': sha(OLD / 'mapping.npz'),
        'readout_source_sha256': sha(Path(__file__).with_name('curriculum_readouts.py'))}
    if not (OUT / 'protocol.json').exists():
        dump(OUT / 'protocol.json', protocol)
    else:
        assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    for seed in protocol['seeds']:
        destination = OUT / f'seed{seed}_test_predictions.npz'
        if destination.exists():
            continue
        val = dict(np.load(OLD / f'induced_seed{seed}_induced_valid_predictions.npz'))
        val['induced_concept'] = mapping['q_to_induced'][val['question_index']]
        vx = np.load(OLD / f'learned_seed{seed}_induced_valid_features.npy')
        params = fit_readouts(vx, val, mapping, pairs, OUT, seed)
        # Preserve the already reported learned mapping exactly. An independent
        # refit can drift slightly through MPS floating-point optimization.
        prior_coef = np.load(OLD / f'learned_seed{seed}_coefficients.npz')
        assert np.array_equal(params['eligible'], prior_coef['eligible'])
        params['learned_monotone_a'] = prior_coef['a']
        params['learned_monotone_b'] = prior_coef['b']
        np.savez_compressed(OUT / f'seed{seed}_readout_coefficients.npz', **params)
        fit_record = json.loads((OUT / f'seed{seed}_readout_fit.json').read_text())
        fit_record['learned_monotone_reuses_prior_frozen_coefficients'] = sha(OLD / f'learned_seed{seed}_coefficients.npz')
        fit_record['learned_refit_trace_is_diagnostic_only'] = True
        dump(OUT / f'seed{seed}_readout_fit.json', fit_record)
        frozen = sha(OUT / f'seed{seed}_readout_coefficients.npz')
        test = dict(np.load(OLD / f'induced_seed{seed}_induced_test_predictions.npz'))
        test['induced_concept'] = mapping['q_to_induced'][test['question_index']]
        tx = np.load(OLD / f'learned_seed{seed}_induced_test_features.npy')
        scores = predict_readouts(tx, test, mapping, params)
        for key in ['item_overlap', 'practice_frequency', 'equal_children']:
            assert np.allclose(scores[key], test[key], atol=2e-6, rtol=1e-5)
        original_learned = np.load(OLD / f'learned_seed{seed}_induced_test_predictions.npz')['learned_monotone']
        assert np.allclose(scores['learned_monotone'], original_learned, atol=2e-5, rtol=1e-4)
        np.savez_compressed(destination, **{k: test[k] for k in ['uid', 'row', 'position', 'label', 'original_concept', 'question_index', 'induced_concept', 'raw_induced']}, **scores)
        assert frozen == sha(OUT / f'seed{seed}_readout_coefficients.npz')
        dump(OUT / f'seed{seed}_complete.json', {'seed': seed, 'status': 'complete',
            'coefficients_sha256': frozen, 'predictions_sha256': sha(destination),
            'fixed_and_prior_learned_scores_reproduced': True})
        print(f'2012 seed {seed} all readouts saved', flush=True)


if __name__ == '__main__':
    main()
