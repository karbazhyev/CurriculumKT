"""Prepare a reviewable draft; freezing is a separate explicit local action."""
from pathlib import Path
import numpy as np
from run_pooling_batch import OUT, ROOT, PRIOR, DATASETS, DESIGNS, LAMBDAS, SEEDS, INDUCED_SEEDS, FOLD_SEED, save, now, fit_id, runs, sha


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Never overwrite a frozen protocol'
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'created_utc': now(), 'frozen': False,
        'scope': 'One authorized two-dataset follow-up; no main milestone advancement; no DKT retraining',
        'authorization': 'User requested fresh multi-agent debate and one bounded experiment batch; later expanded to six AI perspectives plus independent statistics-scientist judge.',
        'panel': {'experts': ['sequence modeling', 'psychometrics', 'evaluation', 'hierarchical Bayesian modeling', 'representation transfer', 'alternative mappings'],
                  'judge': 'Separate independent AI statistical research scientist; pre-evaluation and post-result adjudication',
                  'temperature_control_available': False,
                  'temperature_values': None,
                  'diversity': 'Conservative, balanced, exploratory prompting; actual sampling temperatures were not changed; same inherited model/backend.'},
        'datasets': DATASETS, 'fit_designs': DESIGNS, 'lambda_grid': LAMBDAS,
        'backbones': {ds: runs(ds) for ds in DATASETS},
        'original_seeds': SEEDS, 'induced_seeds': INDUCED_SEEDS,
        'pairing': {ds: [{'pair': j, 'original': SEEDS[ds][j], 'induced': INDUCED_SEEDS[j],
                         'second_original': SEEDS[ds][(j+1)%3]} for j in range(3)] for ds in DATASETS},
        'calibration': {'assist2012': {'students': 596, 'targets': 25789, 'historical_checkpoint_validation_reused': True},
                        'assist2017': {'students': 272, 'targets': 36286, 'separate_from_training_and_checkpoint_selection': True}},
        'test': {'assist2012': {'students': 1464, 'targets': 63869}, 'assist2017': {'students': 340, 'targets': 48389},
                 'status': 'Already inspected in prior research: all new outcomes are exploratory, not fresh confirmation'},
        'target_information': 'Past frozen DKT history logits plus target original concept. No realized target question or induced label in any new predictor. Original+induced is augmentation; maps use induced encoder only.',
        'map': {'prediction': 'sigmoid(b_c + sum_{k in mapping support} a_ck zI_k), a_ck >= 0',
                'weights': 'Existing training-practice-frequency w_c; support is mapping counts>0, including mapped zero-frequency edges',
                'pooled_objective': 'mean_event_NLL + lambda/C sum_c[||a_c-s*w_c||^2/(||w_c||^2+1e-12)+0.1*(b_c-b0)^2] + 1e-6*((s-1)^2+b0^2); s>=0',
                'fixed_control': 'Identical objective/data/support/search; s=1 and b0=0 fixed',
                'numerical_coordinates': 'q_c=a_c/sqrt(||w_c||^2+1e-12); X_i=zI_mapped*sqrt(||w_c||^2+1e-12); nonnegative q equivalent to nonnegative a',
                'interpretation': 'Penalized partial pooling, not estimated Bayesian posterior or latent mastery; norm scaling equalizes pure-scale deviations only'},
        'original': {'prediction': 'sigmoid(b_c+a_c*zO_c), a_c>=0',
                     'objective': 'mean_event_NLL + lambda/C sum_c[(a_c-s)^2+0.1*(b_c-b0)^2] + 1e-6*((s-1)^2+b0^2); s>=0'},
        'augmentation': {'prediction': 'sigmoid(b_c+a_c*zO_c+g_c*x), a_c,g_c>=0',
                         'induced_input': 'x=logit(clip(sum_k w_ck*sigmoid(zI_k),1e-6,1-1e-6)); fixed probability aggregate, never a development-fitted input',
                         'original_input': 'x=native target-original logit from fixed cyclic other original seed',
                         'objective': 'mean_event_NLL + lambda/C sum_c[||(a_c,g_c)-(s0,s1)||^2+0.1*(b_c-b0)^2]+1e-6*((s0-1)^2+s1^2+b0^2); shared s0,s1>=0',
                         'controls': 'Original-only is nested at g=s1=0; identical model/grid/selection/retention for O+I and O+O',
                         'scaling': 'Raw logits; no standardization search. Penalty is coordinate dependent; O+O is an operational ensemble control, not a causal taxonomy isolation.'},
        'support_rule': 'All mapped original concepts; no 50/5 eligibility cutoff. Unseen fitting concepts set analytically to pooling center. Fixed-map unseen fallback is a=w,b=0. No-data monotone logit average differs from probability average.',
        'selection': {'folds': 3, 'unit': 'student', 'seed': FOLD_SEED,
                      'assignment': 'Sort unique calibration student IDs, permute with NumPy default_rng(seed), np.array_split into 3 groups; same folds for every family/seed',
                      'metric': 'Mean of 3 held-out fold pooled response AUCs, then mean over 3 model pairs; one lambda per dataset/design',
                      'tie': 'Within 1e-6 AUC choose strongest lambda',
                      'original_retention': 'Keep native unless selected pooled-original CV AUC exceeds native by >1e-6',
                      'augmentation_retention': 'For each augmentation keep same selected-original endpoint unless selected augmentation CV AUC exceeds it by >1e-6',
                      'refits': 'Refit all five selected families on all calibration students even if a retention endpoint wins; preserve standalone family scores',
                      'freeze': 'Both datasets selections and full-refit coefficient hashes written before new test prediction stage'},
        'runtime': {'python': '/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing/.venv/bin/python',
                    'flag': '-B', 'mps_required_for_all_fits': True,
                    'cpu_allowed': 'Feature preprocessing, SciPy optimizer control, final fixed-coefficient gradient diagnostics, frozen prediction application, metrics/bootstrap',
                    'cv_fit_cap': 360, 'refit_cap': 30, 'total_fit_cap': 390, 'new_dkt_fits': 0,
                    'optimizer': 'L-BFGS-B direct nonnegative coordinates, data products/gradients on MPS; independent float64 fixed-coefficient audit',
                    'objective_scale': 'Optimize C times the entire mean-event objective, unchanged minimizer',
                    'kkt_tolerance': 0.0002, 'solver_gtol': 0.00002,
                    'max_iterations_per_attempt': 1500, 'max_function_calls_per_attempt': 3000,
                    'max_attempts_same_objective': 3, 'optimizer_failures': 'Stop affected execution and preserve failure; no CPU fitting fallback or candidate substitution'},
        'primary': 'Pooled map minus native original DKT, separately for both datasets; retain equally fitted original control comparison',
        'key_secondary': 'Development-retained O+I minus development-retained O+O, separately for both datasets; also compare both to native and selected original-only',
        'additional_prespecified_contrasts': ['pooled_map - fixed_map', 'pooled_map - original_selected', 'augment_induced_selected - original_selected'],
        'metrics': ['mean per-seed pooled AUC', 'per-seed AUC and SD', 'Brier', 'log loss', 'event-weighted and equal-concept within-original AUC'],
        'uncertainty': {'resamples': 2000, 'seed': 20261109,
                        'unit': 'Paired students, identical multiplicities across every method and all three seed pairs; do not treat seed pairs as independent samples',
                        'pointwise': 'Percentile 95% intervals',
                        'multiplicity': 'Bonferroni 98.75% individual intervals (.00625,.99375) for four designated contrasts: pooled-map vs native in each dataset and retained O+I vs retained O+O in each dataset',
                        'interpretation': 'Approximate conditional bootstrap intervals; selection/training/taxonomy uncertainty not included; multiplicity adjustment does not erase prior test reuse'},
        'operational_practical_auc_threshold': 0.002,
        'threshold_note': 'Chosen before this batch as an operational effect-size reference, not a validated educational utility threshold',
        'stopping_rule': 'Run finite development list, refit selected settings, evaluate test once, retain all five families/endpoints/nulls/failures, produce judge report and stop. No grid extension, new family, gate, distillation, encoder training, milestone advancement, recurring monitor or Git push.',
        'limitations': ['2012 historical backbone training populations, widths and seed sets differ', '2012 calibration reused for checkpoint validation and repeated target positions remain', 'Both test sets previously inspected', 'Semantic-taxonomy causal attribution needs further controls', 'Prior higher raw induced AUC uses realized target induced concept and is not map-back performance'],
    }
    save(OUT/'protocol_draft.json', protocol)
    entries = {}
    for ds in DATASETS:
        for design in DESIGNS:
            for pair in range(3):
                for lam in LAMBDAS:
                    for fold in range(3):
                        entries[fit_id(ds, design, pair, lam, fold)] = {'state': 'prepared', 'dataset': ds, 'design': design, 'pair': pair, 'lambda': lam, 'fold': fold}
                entries[fit_id(ds, design, pair, 0)] = {'state': 'prepared', 'dataset': ds, 'design': design, 'pair': pair, 'lambda': 'development_selected', 'stage': 'refit'}
    assert len(entries) == 390
    save(OUT/'run_registry.json', {'state': 'prepared_for_review', 'runs': entries})
    # Hash prior artifacts before any new predictive result, without evaluating test.
    protected = {}
    for ds in DATASETS:
        for run in runs(ds):
            cp = Path(run['checkpoint'])
            assert sha(cp) == run['checkpoint_sha256']
            protected[str(cp)] = sha(cp)
            folder = PRIOR/ds/f"{run['kind']}_seed{run['seed']}"
            for path in folder.iterdir():
                if path.is_file():
                    protected[str(path)] = sha(path)
    for folder in [ROOT/'results/curriculum_kt_assist2012_k1000_20261107', ROOT/'results/curriculum_kt_all_methods_20261107', PRIOR]:
        for path in folder.iterdir():
            if path.is_file() and path.suffix in ['.json', '.csv', '.md', '.npz']:
                protected[str(path)] = sha(path)
    for ds in DATASETS:
        for name in ['protocol.json', 'mapping.npz']:
            folder = ROOT/('results/curriculum_kt_assist2012_k1000_20261107' if ds=='assist2012' else 'results/curriculum_kt_all_methods_20261107/assist2017')
            protected[str(folder/name)] = sha(folder/name)
    save(OUT/'protected_source_hashes.json', protected)
    print(f'Prepared draft with {len(entries)} bounded fits and {len(protected)} protected identities.')


if __name__ == '__main__':
    prepare()
