"""Prepare a calibration/test protocol; freeze only after selected-model checks."""
import argparse
import copy
from pathlib import Path

from run_simplekt_seed_confirmation import ROOT, OUT as SEED_ROOT, SCREEN, sha, verify_stage
from run_simplekt_sweeps import read, dump, now

OUT = ROOT / 'results/simplekt_curriculum_20261107'
DESIGNS = ['pooled_map', 'pooled_original', 'augment_induced', 'augment_original']
LAMBDAS = [.001, .01, .1, 1.]
SEEDS = [42, 84, 126]
IMPLEMENTATIONS = ['prepare_simplekt_curriculum.py', 'simplekt_mapped_queries.py',
    'simplekt_curriculum_features.py', 'check_simplekt_selected_queries.py',
    'extract_simplekt_curriculum.py', 'run_simplekt_curriculum_readouts.py',
    'summarize_simplekt_curriculum.py', 'run_pooling_batch_v2.py',
    'evaluate_saved_curriculum_kt.py', 'summarize_saved_curriculum_kt.py',
    'check_attention_model_readiness.py']


def specification():
    parent = read(SCREEN/'protocol.json')
    assert sha(SCREEN/'protocol.json') == read(SCREEN/'protocol_identity.json')['sha256']
    return {'created_utc':now(), 'frozen':False, 'model':'simplekt',
        'scope':'Frozen selected SimpleKT backbones; original-taxonomy comparison on both datasets. No backbone fitting or taxonomy changes in this stage.',
        'authorization':'User requested all compatible models one by one including map-back/augmentation comparisons.',
        'datasets':copy.deepcopy(parent['datasets']), 'fixed_backbone':copy.deepcopy(parent['fixed']),
        'selected_models':None, 'seeds':SEEDS, 'designs':DESIGNS, 'lambdas':LAMBDAS,
        'methods':['original_kt','raw_induced_diagnostic','practice_frequency',*DESIGNS,
                   'original_selected','augment_induced_selected','augment_original_selected'],
        'native_induced_scope':'Diagnostic requiring realized target induced concept; excluded from map-back superiority claims.',
        'query':'Observed past interactions and original target concept only for mapped/augmented predictions. Query each overlapping child independently without altering observed past. No actual target item enters predictor.',
        'mapping':'Training-only practice_frequency on saved item-overlap support. Preserve fallback rows and saved taxonomy.',
        'feature_identity':'Store sparse supported-child logits, exact original scalar logits and frequency-weighted mean induced probabilities with paired original-target metadata. Do not invert clipped original probabilities to recover logits.',
        'positive_map':'Multiply child-logit features by ||w_c||_2; nonnegative coefficients shrink to shared slope times w_c/||w_c||_2. Unsupported entries fixed to zero.',
        'augmentation':'sigmoid(b_c + a_c z_original + g_c logit(clip(sum_k w_ck sigmoid(z_child_k),1e-6,1-1e-6))); a_c,g_c >= 0.',
        'original_control':'Replace induced aggregate with next original seed scalar logit in cyclic order42->84->126->42. Match calibration and selection capacity.',
        'objective':'Mean-event logistic NLL + lambda/C*(sum squared coefficient deviations from shared center +0.1*sum squared intercept deviations) +1e-6*(sum squared shared slope deviations from anchor +shared intercept squared). Optimize C times this same objective.',
        'anchors':{'pooled_map':[1.], 'pooled_original':[1.], 'augment_induced':[1.,0.], 'augment_original':[1.,0.]},
        'cv':{'folds':3,'unit':'student','seed':20261108,
            'selection':'One lambda per dataset/family by mean of nine fold/seed AUCs; within1e-6 of best choose larger lambda. Refit every selected family on all calibration students.',
            'retention':'Original-only endpoint retains native unless pooled_original improves mean CV AUC by >1e-6. Each augmentation endpoint retains selected original-only unless augmentation improves its CV AUC by >1e-6.',
            'test_used':False},
        'runtime':{'backend':'mps','cpu_training_fallback':False,
            'cpu_allowed':'Feature preprocessing, optimizer control/float64 reductions, fixed-coefficient independent audits, metrics/bootstrap and frozen coefficient application.',
            'cv_fits':288,'full_calibration_fits':24,'total_readout_fits':312,
            'optimizer':'Reuse compensated-MPS PooledLogistic implementation from run_pooling_batch_v2.py unchanged.',
            'kkt_tolerance':.0002,'max_attempts_same_objective':3,'max_iterations_per_attempt':1500,
            'failure_policy':'Preserve failed coefficients/records and stop affected stage; document any numerical amendment. No silent CPU fit, candidate substitution or test-driven retuning.'},
        'stage_boundary':'Complete all calibration selections/refits for BOTH datasets and hash their coefficients before any test feature extraction or scoring. Check hashes again after test.',
        'contrasts':[['pooled_map','original_kt'],['augment_induced_selected','original_kt'],
                     ['augment_induced_selected','original_selected'],['augment_induced_selected','augment_original_selected']],
        'metrics':['mean per-seed response AUC','per-seed AUC and SD','Brier','log loss','equal-concept within-original AUC','event-weighted within-original AUC'],
        'bootstrap':{'resamples':2000,'seed':20261109,'unit':'paired students',
            'same_draw_across_all_seeds_and_methods':True,'interval':'pointwise percentile95%',
            'degenerate_draws':'If a sampled student draw has only one response class, reject it for all methods and redraw; record rejection count, cap total attempts at10 times requested draws.',
            'additional_four_contrast_intervals':'98.75% per contrast for pooled_map-original_kt and retained O+I-retained O+O in each dataset; does not control the whole16-model campaign.',
            'conditional_on':'frozen backbones, taxonomy, selection and fitted coefficients'},
        'limitations':['Both historical test sets previously inspected; exploratory evidence.',
            '2012 preserves repeated sequence positions and K1000; not a reproduction of the paper K300 result.',
            'Matched new original/induced runs share populations/splits; historical baseline scores are not matched comparators.',
            'Per-model inference and local four-contrast intervals do not establish campaign-wide significance. Retain all model outcomes.'],
        'wandb':{**parent['wandb'],'group':'currkt-simplekt-curriculum-20261107'},
        'parent_screening_protocol_sha256':sha(SCREEN/'protocol.json')}


def validate_specification(protocol):
    assert protocol['designs'] == DESIGNS and protocol['seeds'] == SEEDS
    assert protocol['lambdas'] == LAMBDAS
    assert protocol['runtime']['cv_fits'] == 2*len(DESIGNS)*3*len(LAMBDAS)*3
    assert protocol['runtime']['full_calibration_fits'] == 2*len(DESIGNS)*3
    assert protocol['runtime']['total_readout_fits'] == 312
    assert all(d['student_split_disjoint'] and d['paired_events_responses_order_and_masks']
               for d in protocol['datasets'].values())
    assert protocol['cv']['test_used'] is False


def freeze():
    if (OUT/'protocol.json').exists():
        assert sha(OUT/'protocol.json') == read(OUT/'protocol_identity.json')['sha256']
        return read(OUT/'protocol.json')
    parent = verify_stage()
    selected = read(SEED_ROOT/'selected_models.json')
    assert selected['all_twelve_available'] and selected['protocol_sha256'] == sha(SEED_ROOT/'protocol.json')
    assert read(SEED_ROOT/'run_registry.json')['state'] == 'complete'
    audit = read(ROOT/'results/model_sequence_20261107/simplekt_online_history_checks.json')
    assert audit['counts'] == {'verified':40}, 'All32+8 training runs need verified W&B histories'
    query_path = ROOT/'results/model_sequence_20261107/simplekt_selected_query_mps_checks.json'
    query = read(query_path)
    assert query['status'] == 'passed' and query['all_twelve_selected_checkpoints']
    assert query['selected_models_sha256'] == sha(SEED_ROOT/'selected_models.json')
    assert query['heldout_queries'] == 0 and query['empirical_fits'] == 0
    for path, expected in query['implementation_sha256'].items():
        assert sha(Path(path)) == expected
    preparation_path=ROOT/'results/model_sequence_20261107/simplekt_curriculum_preparation_checks.json'
    preparation=read(preparation_path)
    assert preparation['status']=='passed' and preparation['empirical_fits']==0
    assert preparation['heldout_arrays_loaded'] is False
    for path,expected in preparation['implementation_sha256'].items():
        assert sha(Path(path))==expected, f'CPU preparation checks are stale: {path}'
    protocol = specification()
    protocol['fixed_backbone'] = copy.deepcopy(parent['fixed'])
    protocol.update(frozen=True, frozen_utc=now(), selected_models=selected['models'],
        selected_models_sha256=sha(SEED_ROOT/'selected_models.json'),
        seed_protocol_sha256=sha(SEED_ROOT/'protocol.json'),
        selected_query_check_sha256=sha(query_path),
        preparation_check_sha256=sha(preparation_path))
    implementation = {str(ROOT/'scripts'/s):sha(ROOT/'scripts'/s) for s in IMPLEMENTATIONS}
    for path, expected in parent['source_sha256'].items():
        assert sha(Path(path)) == expected
        implementation[path] = expected
    protocol['implementation_sha256'] = implementation
    for records in selected['models'].values():
        assert set(records) == {'42','84','126'}
        for record in records.values():
            assert sha(Path(record['checkpoint'])) == record['checkpoint_sha256']
    for ds in protocol['datasets'].values():
        for record in [*ds['files'].values(),ds['mapping']]:
            assert sha(Path(record['path'])) == record['sha256']
    validate_specification(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/'protocol.json', protocol)
    dump(OUT/'protocol_identity.json', {'sha256':sha(OUT/'protocol.json')})
    dump(OUT/'run_registry.json', {'state':'prepared','created_utc':now(),
        'feature_extraction':{},'fits':{},'test_predictions':{},'heldout_started':False})
    return protocol


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['draft','freeze'])
    args=parser.parse_args()
    if args.action == 'draft':
        assert not (OUT/'protocol.json').exists(), 'Do not overwrite a frozen protocol'
        protocol=specification();validate_specification(protocol)
        OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol_draft.json',protocol)
        print('Draft saved:312 readout fits planned; none executed, protocol not frozen.')
    else:
        freeze();print('Protocol frozen after training and independent query checks.')
