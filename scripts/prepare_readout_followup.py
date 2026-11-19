"""Freeze the user-authorized two-dataset follow-up and log-based reuse audit."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_saved_curriculum_kt import ROOT, SOURCE, OUT as OLD, sha, dump

OUT = ROOT / 'results/curriculum_kt_readout_followup_20261107'


def main():
    OUT.mkdir(exist_ok=True)
    audit = json.loads((ROOT / 'results/paper_curriculum_kt_audit_20261107/artifact_audit.json').read_text())
    archived = audit['datasets']['assist2017']['archived_condition']['target_rows']
    settings = {(int(r['emb_size']), float(r['dropout']), float(r['learning_rate'])) for r in archived}
    assert settings == {(256, 0.5, 0.0005)}
    configs = []
    for path in (SOURCE / 'examples/models/dkt_tiaocan_assist2017').glob('**/config.json'):
        c = json.loads(path.read_text())
        configs.append({'path': str(path), 'sha256': sha(path), 'num_c': c['data_config']['num_c'],
                        'fold': c['params']['fold'], 'seed': c['params']['seed'],
                        'hidden_size': c['model_config']['emb_size']})
    matched_logs = []
    all_2017 = 0
    for path in (SOURCE / 'examples/wandb').glob('run-*/files/wandb-summary.json'):
        s = json.loads(path.read_text())
        saved = s.get('model_save_path', '')
        if 'assist2017' not in saved:
            continue
        all_2017 += 1
        # Folder fields encode seed/fold/dropout/width/lr in these legacy runs.
        if '_0.5_256_0.0005_' not in saved:
            continue
        checkpoint = SOURCE / 'examples' / saved
        matched_logs.append({'summary': str(path), 'sha256': sha(path),
                             'checkpoint_exists': checkpoint.exists(),
                             'best_epoch': s.get('best_epoch'), 'last_epoch': s.get('epoch'),
                             'validation_auc': s.get('validauc'),
                             'logged_test_equals_validation': s.get('testauc') == s.get('validauc'),
                             'model_save_path': saved})
    cv = np.intersect1d(np.load(OLD / 'induced_valid_sequences.npz')['uid'],
                        np.load(OLD / 'original_valid_sequences.npz')['uid'])
    np.save(OUT / 'assist2012_common_calibration_students.npy', cv)
    reuse = {'generated_utc': datetime.now(timezone.utc).isoformat(),
             'wandb_2017_summaries_with_model_paths': all_2017,
             'wandb_logs_matching_fixed_settings': matched_logs,
             'archived_induced_condition_rows': archived,
             'existing_main_2017_configs': configs,
             'existing_main_2017_concept_counts': sorted({c['num_c'] for c in configs}),
             'existing_main_2017_folds': sorted({c['fold'] for c in configs}),
             'reuse_2012': 'All six saved DKT checkpoints; no DKT retraining',
             'new_2017': 'Three original and three induced fits; no hyperparameter sweep',
             'why_original_2017_not_reused': 'Saved fold0 models trained on fold1, which is now the independent readout-calibration split. Reusing them would break the held-out calibration design and matched training population.',
             'why_induced_2017_training_needed': 'No K100 checkpoint recovered; all main saved configurations have 68 outputs.',
             'configuration_selection': 'Use the fixed archived K100 condition settings, supported by local W&B logs. Do not select by logged test fields, many of which equal validation.',
             '2012_shared_calibration_students': len(cv)}
    dump(OUT / 'wandb_reuse_audit.json', reuse)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User explicitly requested execution of the proposed follow-up and 2017, using W&B logs to train only what is necessary.',
        'scope': 'Bounded two-dataset map-back follow-up; not the main campaign',
        'datasets': ['assist2012', 'assist2017'],
        'mps_required': True,
        '2012': {'source': str(OLD), 'dkt_retraining': False, 'induced_seeds': [42,84,126], 'original_seeds': [42,84,255],
                 'main_readout_calibration': '596 students held out for validation in both historical designs, encoded using identical induced-export histories',
                 'audit': 'Separately optimize the identical old monotone objective on all historical induced validation events; preserve old coefficients and rows',
                 'limitations': 'Historical base training and hidden sizes remain unmatched; checkpoint selection used the calibration students; previously inspected test; repeated saved positions retained'},
        '2017': {'prepared_data': str(ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017'),
                 'seeds': [42,84,126], 'dkt_fits': 6, 'hidden_size': 256, 'dropout':0.5,
                 'learning_rate':0.0005,'batch_size':256,'max_epochs':100,'patience':10,
                 'selection':'Highest validation AUC, never test',
                 'split':'folds2/3/4 encoder training; fold0 checkpoint validation; fold1 readout calibration; fold-1 test'},
        'main_readouts': ['item_overlap','practice_frequency','equal_children','monotone_converged','signed_mapped','soft_mapping','full_logits','induced_state_head'],
        'original_controls':['original_dkt','original_calibration','original_state_head','original_selected'],
        'penalties': {'mapped_coefficients':0.01,'soft_off_map_coefficients':0.1,'unrestricted_logit_coefficients':0.01,'hidden_coefficients':0.01,'intercepts':0.001},
        'anchors':'Frequency weights for mapped/logit readouts; their corresponding output-row average for induced hidden head; native output row for original hidden head; slope1 for original scalar calibration',
        'eligibility':{'min_events':50,'min_each_class':5,'induced_fallback':'training-frequency average','original_fallback':'native original prediction'},
        'optimizer':'Actual coefficients; bound-constrained L-BFGS-B with MPS loss/gradient data operations, CPU optimizer control and float64 objective accumulation; direct-coordinate projected gradient verification. No softplus.',
        'optimization_tolerance':{'projected_gradient_max':0.0002,'max_iterations':1500,'max_function_evaluations':3000,'ftol':1e-12},
        'selection':'No mapping hyperparameter sweep. Original-only retain/scalar/state comparator selected on a fixed student-level 20% calibration holdout, then fitted methods refit on all calibration students. All candidates reported.',
        'selection_seed':20261107,
        'primary':'Pooled response AUC for induced_state_head against development-selected original control, with native original retained; all other methods prespecified secondary comparisons',
        'secondary':['within-original macro AUC','within-original event-weighted AUC','Brier','log loss','per-concept support and changes'],
        'uncertainty':'500 paired student bootstrap resamples, mean per-seed AUC rather than prediction ensemble; conditional on fitted models, no multiple-comparison correction',
        'feature_identity':'Use exact pre-sigmoid logits and hidden states; for the old-objective audit use the original clipped feature cache unchanged',
        '2017_rank_note':'K100 outputs cannot span all 256 hidden dimensions; full-logit and hidden-state heads are distinct here, unlike full-rank K1000 saved 2012 outputs.',
        'protocol_frozen_before_new_test_scoring':True,
        'source_identities':{'saved_2012_protocol':sha(OLD/'protocol.json'), 'prepared_2017_protocol':sha(ROOT/'results/curriculum_kt_all_methods_20261107/assist2017/protocol.json'), 'wandb_reuse_audit':sha(OUT/'wandb_reuse_audit.json')}
    }
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Existing follow-up protocol: resume instead of overwriting')
    dump(OUT / 'protocol.json', protocol)
    dump(OUT / 'run_registry.json', {'state':'prepared','datasets':{d:{'features':'pending','readout_fit':'pending','test':'pending'} for d in protocol['datasets']},'2012_audit':'pending','2017_training_registry':str(ROOT/'results/curriculum_kt_all_methods_20261107/assist2017/run_registry.json')})
    print(json.dumps({'2017_log_summaries':all_2017,'matching_settings_logs':len(matched_logs),'saved_2017_output_counts':reuse['existing_main_2017_concept_counts'],'2012_shared_calibration_students':len(cv),'new_dkt_fits':6}),flush=True)


if __name__ == '__main__':
    main()
