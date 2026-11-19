"""CPU-only checks of feature consolidation and unready-stage rejection."""
import ast
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from extract_robustkt_curriculum import consolidate, require_test_ready, save_npz
from prepare_robustkt_curriculum import OUT, ROOT, specification, validate_specification
from run_robustkt_sweeps import dump, now, sha
import run_robustkt_curriculum_readouts as readouts
import summarize_robustkt_curriculum as summary


def main():
    frozen_path=OUT/'protocol.json'
    report_path=ROOT/'results/model_sequence_20261107/robustkt_curriculum_preparation_checks.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text());assert sha(report_path)==frozen['preparation_check_sha256']
        previous=json.loads(report_path.read_text())
        for path,expected in previous['implementation_sha256'].items():assert sha(Path(path))==expected
        print('Existing frozen CPU preparation checks remain valid; no files overwritten.',flush=True)
        return
    specification_value = specification()
    validate_specification(specification_value)
    implementations = ['check_robustkt_selected_queries.py','prepare_robustkt_curriculum.py',
                       'extract_robustkt_curriculum.py','run_robustkt_curriculum_readouts.py','summarize_robustkt_curriculum.py',
                       'check_robustkt_curriculum_preparation.py']
    for name in implementations:
        ast.parse((ROOT/'scripts'/name).read_text())
    with tempfile.TemporaryDirectory(prefix='currkt-feature-check-',dir='/private/tmp') as folder:
        dest = Path(folder);(dest/'parts').mkdir()
        metadata = {'uid':np.array([10,10,10,20,20]), 'row':np.array([0,0,0,1,1]),
            'position':np.array([1,2,3,1,2]), 'event_id':np.array([1,2,3,201,202]),
            'label':np.array([0,1,1,0,1]), 'original_concept':np.array([0,0,1,1,0])}
        features = {'native_logit':np.array([-2,-1,0,1,2],dtype=np.float32),
            'native_probability':np.array([.1,.2,.5,.7,.9],dtype=np.float32),
            'child_logits':np.arange(15,dtype=np.float32).reshape(5,3),
            'practice_frequency':np.arange(5,dtype=np.float64)/5}
        parts=[]
        for j,(start,end) in enumerate([(0,3),(3,5)]):
            path=dest/'parts'/f'part{j}.npz'
            save_npz(path,{**{'meta__'+k:v[start:end] for k,v in metadata.items()},
                **{'feature__'+k:v[start:end] for k,v in features.items()}})
            parts.append({'file':path.name,'sha256':sha(path),'events':end-start})
        consolidate(dest,parts,5,{'protocol_sha256':'synthetic-no-empirical-data'})
        with np.load(dest/'metadata.npz') as data:
            assert all(np.array_equal(data[k],v) for k,v in metadata.items())
        assert all(np.array_equal(np.load(dest/f'{k}.npy'),v) for k,v in features.items())
        # A resumed assembly over the same completed chunks preserves values.
        consolidate(dest,parts,5,{'protocol_sha256':'synthetic-no-empirical-data'})
        with np.load(dest/'metadata.npz') as data:
            assert all(np.array_equal(data[k],v) for k,v in metadata.items())
        bad=[{**parts[0],'sha256':'wrong'},parts[1]]
        try:
            consolidate(dest,bad,5,{'protocol_sha256':'synthetic-no-empirical-data'})
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupted chunk identity accepted')
        weights=np.array([[.75,.25,0],[.2,.3,.5]])
        mapping=dest/'mapping.npz'
        save_npz(mapping,{'counts':(weights>0).astype(int),'practice_frequency':weights})
        selected_c=metadata['original_concept']
        child=np.array([[-2,1,0],[.2,-.5,0],[1,2,-1],[.1,.2,.3],[3,2,0]],dtype=np.float32)
        # Mask each synthetic event according to its original target.
        child*=weights[selected_c]>0
        q=(expit(child.astype(float))*weights[selected_c]).sum(1)
        original={seed:{'native_logit':np.arange(5,dtype=np.float32)+seed/100,
            'native_probability':np.linspace(.1,.9,5)} for seed in [42,84,126]}
        induced={'child_logits':child,'practice_frequency':q,
            'native_probability':np.linspace(.9,.1,5),'native_logit':np.full(5,999.)}
        previous_loader=readouts.load_features
        readouts.load_features=lambda ds,tax,seed,split:(metadata,original[seed] if tax=='original' else induced)
        protocol={'datasets':{'synthetic':{'mapping':{'path':str(mapping),'sha256':sha(mapping)}}}}
        try:
            designs={name:readouts.design_inputs(protocol,'synthetic',0,'calibration',name)
                     for name in readouts.DESIGNS}
            _,x,mask,basis,anchor,_=designs['pooled_map']
            assert np.allclose((x*basis[selected_c]).sum(1),(child*weights[selected_c]).sum(1),atol=1e-6)
            assert np.array_equal(designs['pooled_original'][1][:,0],original[42]['native_logit'])
            assert np.allclose(designs['augment_induced'][1][:,1],logit(q),atol=1e-6)
            assert np.array_equal(designs['augment_original'][1][:,1],original[84]['native_logit'])
            induced['native_probability'][:]=0;induced['native_logit'][:]=-999.
            for name,previous in designs.items():
                current=readouts.design_inputs(protocol,'synthetic',0,'calibration',name)
                assert np.array_equal(previous[1],current[1]), 'Native induced diagnostic leaked into readout'
        finally:
            readouts.load_features=previous_loader
        entries=readouts.fit_registry()
        assert len(entries)==312 and sum('fold' in e for e in entries.values())==288
    # Independent physical student resampling verifies weighted paired AUCs,
    # aliases, ties and averaging seed-level AUC instead of seed predictions.
    synthetic_rng=np.random.default_rng(812)
    uid=np.repeat(np.arange(8),3);label=np.tile([0,1,0],8)
    synthetic_meta={'uid':uid,'row':uid,'position':np.tile([1,2,3],8),
        'event_id':np.arange(24),'label':label,'original_concept':np.arange(24)%2}
    retention={'original_selected':'original_kt','augment_induced_selected':'augment_induced',
               'augment_original_selected':'augment_original'}
    predictions=[]
    for pair in range(3):
        p={**synthetic_meta,**{m:np.round(synthetic_rng.uniform(.05,.95,24),1)
                             for m in summary.METHODS if m not in retention}}
        p['pooled_map']=p['original_kt'].copy()
        for endpoint,target in retention.items():p[endpoint]=p[target].copy()
        predictions.append(p)
    result=summary.analyze_predictions(predictions,retention,resamples=32,seed=111)
    draws=np.random.default_rng(111)
    for iteration in range(32):
        students=draws.integers(8,size=8)
        indices=np.concatenate([np.flatnonzero(uid==student) for student in students])
        for method,values in result['bootstrap'].items():
            direct=np.mean([roc_auc_score(label[indices],p[method][indices]) for p in predictions])
            assert abs(values[iteration]-direct)<1e-12
    same=result['contrasts'].query("method == 'pooled_map'").iloc[0]
    assert same.delta_auc==same.ci95_low==same.ci95_high==0
    assert np.isnan(result['aggregate'].query("method == 'raw_induced_diagnostic'").delta_original_kt.iloc[0])
    for row in result['aggregate'].to_dict('records'):
        assert abs(row['auc']-np.mean([roc_auc_score(label,p[row['method']]) for p in predictions]))<1e-12
    assert any(abs(np.mean([roc_auc_score(label,p[m]) for p in predictions])-
        roc_auc_score(label,np.mean([p[m] for p in predictions],axis=0)))>1e-3 for m in summary.METHODS)
    heldout_blocked=None
    if not (OUT/'development_complete.json').exists():
        try:
            require_test_ready()
        except FileNotFoundError:
            heldout_blocked=True
        else:
            raise AssertionError('Held-out extraction allowed before calibration selections')
    report={'checked_utc':now(),'status':'passed','syntax_and_imports_passed':True,
        'synthetic_compact_feature_consolidation_and_resume_passed':True,
        'corrupted_chunk_rejected':True,'premature_heldout_access_blocked':heldout_blocked,
        'all_four_synthetic_readout_design_checks_passed':True,
        'realized_induced_diagnostic_excluded_from_readouts':True,
        'unique_fit_registry_288_cv_24_refit':True,
        'paired_student_bootstrap_matches_physical_resampling':True,
        'mean_per_seed_auc_and_identical_predictor_zero_difference_checked':True,
        'native_induced_diagnostic_excluded_from_map_back_contrasts':True,
        'planned_readout_fits':specification_value['runtime']['total_readout_fits'],
        'empirical_fits':0,'heldout_arrays_loaded':False,'gpu_operations':0,
        'protocol_frozen':(OUT/'protocol.json').exists(),
        'implementation_sha256':{str(ROOT/'scripts'/s):sha(ROOT/'scripts'/s) for s in implementations},
        'remaining':'Final code/provenance review, all-selected-checkpoint MPS verification, final protocol freeze and actual staged execution. Empirical readout solver, extraction and report execution remain unverified for RobustKT.'}
    dump(ROOT/'results/model_sequence_20261107/robustkt_curriculum_preparation_checks.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
