"""Frozen SimpleKT comparisons, paired-student uncertainty and W&B verification."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_simplekt_sweeps as base
from prepare_simplekt_curriculum import OUT, ROOT, SEEDS, sha
from extract_simplekt_curriculum import guard, require_test_ready
from evaluate_saved_curriculum_kt import metrics
from summarize_saved_curriculum_kt import weighted_auc_function

METHODS = ['original_kt','raw_induced_diagnostic','practice_frequency','pooled_map',
    'pooled_original','augment_induced','augment_original','original_selected',
    'augment_induced_selected','augment_original_selected']
CONTRASTS = [('pooled_map','original_kt'),('augment_induced_selected','original_kt'),
    ('augment_induced_selected','original_selected'),('augment_induced_selected','augment_original_selected')]
ADJUSTED = {('pooled_map','original_kt'),('augment_induced_selected','augment_original_selected')}
META = ['uid','row','position','event_id','label','original_concept']
DATASETS = ['assist2017','assist2012']


def analyze_predictions(predictions, retention, resamples=2000, seed=20261109, progress=None):
    """Pure CPU statistics, usable with synthetic arrays without reading test files."""
    assert len(predictions) == 3
    ref = {k:predictions[0][k] for k in META}
    assert all(np.array_equal(d[k],ref[k]) for d in predictions for k in META)
    rows,per_concept=[],[]
    for pair,d in enumerate(predictions):
        for method in METHODS:
            result=metrics(d['label'],d[method],d['original_concept'])
            for concept,n,auc in result.pop('per_original_concept'):
                per_concept.append({'method':method,'pair':pair,'original_concept':concept,'events':n,'auc':auc})
            rows.append({'method':method,'pair':pair,'seed':SEEDS[pair],**result})
        for endpoint,target in retention.items():
            assert np.array_equal(d[endpoint],d[target]), 'Retained endpoint differs from frozen choice'
    frame=pd.DataFrame(rows)
    aggregate=frame.groupby('method',sort=False).agg(auc=('auc','mean'),seed_sd=('auc','std'),
        brier=('brier','mean'),log_loss=('log_loss','mean'),
        macro_auc=('within_original_macro_auc','mean'),within_weighted_auc=('within_original_weighted_auc','mean'),
        eligible_concepts=('within_original_eligible_concepts','first'),events=('events','first')).reset_index()
    point=dict(zip(aggregate.method,aggregate.auc))
    compared=[m for m in METHODS if m!='raw_induced_diagnostic']
    functions={method:[weighted_auc_function(ref['label'],d[method]) for d in predictions]
               for method in compared if method not in retention}
    _,student=np.unique(ref['uid'],return_inverse=True)
    n=int(student.max()+1)
    rng=np.random.default_rng(seed)
    bootstrap={m:[] for m in compared}
    rejected=0;draws=0
    while len(bootstrap[compared[0]])<resamples:
        draws+=1
        assert draws<=resamples*10, 'Too many degenerate student draws'
        weights=np.bincount(rng.integers(n,size=n),minlength=n)[student]
        if np.sum(weights*ref['label'])==0 or np.sum(weights*(1-ref['label']))==0:
            rejected+=1;continue
        values={method:float(np.mean([f(weights) for f in fs])) for method,fs in functions.items()}
        for endpoint,target in retention.items():
            values[endpoint]=values[target]
        assert all(np.isfinite(value) for value in values.values())
        for method in compared:
            bootstrap[method].append(values[method])
        if progress and len(bootstrap[compared[0]])%250==0:
            progress(len(bootstrap[compared[0]]))
    bootstrap={method:np.array(value) for method,value in bootstrap.items()}
    aggregate['role']=aggregate.method.map(lambda m:'native induced diagnostic' if m=='raw_induced_diagnostic'
        else 'pure map-back' if m in ['pooled_map','practice_frequency']
        else 'original plus induced' if m.startswith('augment_induced')
        else 'original plus original control' if m.startswith('augment_original') else 'original taxonomy')
    aggregate['delta_original_kt']=[point[m]-point['original_kt'] if m in compared else np.nan for m in aggregate.method]
    intervals=[np.quantile(bootstrap[m]-bootstrap['original_kt'],[.025,.975]) if m in compared else [np.nan,np.nan]
               for m in aggregate.method]
    aggregate['ci95_low_original_kt']=[a[0] for a in intervals]
    aggregate['ci95_high_original_kt']=[a[1] for a in intervals]
    contrasts=[]
    for method,comparator in CONTRASTS:
        values=bootstrap[method]-bootstrap[comparator]
        ci=np.quantile(values,[.025,.975,.00625,.99375])
        contrasts.append({'method':method,'comparator':comparator,'delta_auc':point[method]-point[comparator],
            'ci95_low':ci[0],'ci95_high':ci[1],
            'designated_four_contrast_family':(method,comparator) in ADJUSTED,
            'familywise_interval_low':ci[2] if (method,comparator) in ADJUSTED else None,
            'familywise_interval_high':ci[3] if (method,comparator) in ADJUSTED else None})
    return {'scores':frame,'per_concept':pd.DataFrame(per_concept),'aggregate':aggregate,
        'contrasts':pd.DataFrame(contrasts),'bootstrap':bootstrap,
        'events':len(student),'students':n,'rejected_degenerate_draws':rejected}


def summarize_dataset(protocol,ds):
    folder=OUT/ds;marker=folder/'summary.json'
    prediction_identity={f'test_pair{pair}.npz':sha(folder/f'test_pair{pair}.npz') for pair in range(3)}
    if marker.exists():
        old=base.read(marker)
        assert old['protocol_sha256']==sha(OUT/'protocol.json')
        assert old['prediction_sha256']==prediction_identity
        for name,expected in old['output_sha256'].items():
            assert sha(folder/name)==expected
        return pd.read_csv(folder/'method_comparison.csv'),pd.read_csv(folder/'paired_contrasts.csv')
    predictions=[]
    for pair in range(3):
        path=folder/f'test_pair{pair}.npz';record=base.read(path.with_suffix('.json'))
        assert sha(path)==record['sha256']
        assert record['protocol_sha256']==sha(OUT/'protocol.json')
        assert record['selection_sha256']==sha(folder/'selection.json')
        with np.load(path) as d:
            predictions.append(dict(d))
    selection=base.read(folder/'selection.json')
    start=time.monotonic()
    result=analyze_predictions(predictions,selection['retention'],
        resamples=protocol['bootstrap']['resamples'],seed=protocol['bootstrap']['seed'],
        progress=lambda done:print(f'{ds}: paired student bootstrap {done}/2000',flush=True))
    for key,name in [('scores','scores_by_seed.csv'),('per_concept','per_concept_auc.csv'),
                     ('aggregate','method_comparison.csv'),('contrasts','paired_contrasts.csv')]:
        frame=result[key];frame.insert(0,'dataset',ds);frame.to_csv(folder/name,index=False)
    np.savez_compressed(folder/'bootstrap_aucs.npz',**result['bootstrap'])
    files=['scores_by_seed.csv','per_concept_auc.csv','method_comparison.csv','paired_contrasts.csv','bootstrap_aucs.npz']
    base.dump(marker,{'protocol_sha256':sha(OUT/'protocol.json'),'prediction_sha256':prediction_identity,
        'output_sha256':{name:sha(folder/name) for name in files},'completed_utc':base.now(),
        'seconds':time.monotonic()-start,'events':result['events'],'students':result['students'],
        'rejected_degenerate_draws':result['rejected_degenerate_draws'],
        'aggregation':'Mean of three seed AUCs, not AUC of averaged predictions.',
        'uncertainty':'Paired student draws shared across methods and seeds, conditional on fixed models/selection. No campaign-wide multiplicity claim.'})
    return result['aggregate'],result['contrasts']


def verify(protocol):
    guard();require_test_ready()
    assert base.read(OUT/'test_complete.json')['protocol_sha256']==sha(OUT/'protocol.json')
    entries=base.read(OUT/'run_registry.json')['fits']
    assert len(entries)==312
    for key,e in entries.items():
        assert e['state']=='complete' and e['converged'] and e['device']=='mps'
        assert e['projected_gradient']<=protocol['runtime']['kkt_tolerance']
        assert e['protocol_sha256']==sha(OUT/'protocol.json') and e['test_used'] is False
        assert sha(Path(e['coefficient_file']))==e['coefficient_sha256']
        assert e['invocations'] and e['invocations'][-1]['state']=='complete'
    checkpoint_count=0
    for cell in protocol['selected_models'].values():
        for model in cell.values():
            assert sha(Path(model['checkpoint']))==model['checkpoint_sha256'];checkpoint_count+=1
    assert checkpoint_count==12
    for ds,data in protocol['datasets'].items():
        for identity in [*data['files'].values(),data['mapping']]:
            assert sha(Path(identity['path']))==identity['sha256']
        with np.load(OUT/ds/'test_pair0.npz') as p:
            assert len(p['label'])==data['counts']['test']['predictable_targets']
            assert len(np.unique(p['uid']))==data['counts']['test']['students']
            groups=base.read(OUT/ds/'development_folds.json')['students']
            assert not set(np.concatenate(groups)) & set(p['uid'])
            for a in range(3):
                for b in range(a):assert not set(groups[a]) & set(groups[b])
    development=base.read(OUT/'development_complete.json')
    assert development['calibration_feature_manifest_sha256']==sha(OUT/'calibration_features_complete.json')
    screening=base.read(ROOT/'results/simplekt_sweeps_20261107/run_registry.json')
    seeds=base.read(ROOT/'results/simplekt_seed_confirmation_20261107/run_registry.json')
    complete=[e for r in [screening,seeds] for e in r['trials'].values() if e['state']=='complete']
    assert len(complete)==40 and all(e['validation_reload_passed'] for e in complete)
    verification={'completed_utc':base.now(),'protocol_sha256':sha(OUT/'protocol.json'),
        'all_312_readout_fits_complete_and_converged':True,'all_readout_fits_actual_mps':True,
        'solver_invocations':sum(len(e['invocations']) for e in entries.values()),
        'interrupted_solver_invocations':sum(i['state']=='interrupted' for e in entries.values() for i in e['invocations']),
        'max_scaled_projected_gradient':max(e['projected_gradient'] for e in entries.values()),
        'readout_fit_seconds_sum':sum(e['seconds'] for e in entries.values()),
        'backbone_fits':40,'selected_checkpoints':12,
        'backbone_epoch_seconds_sum':sum(e['total_epoch_seconds'] for e in complete),
        'backbone_convergence_counts':dict(Counter(e['diagnosis']['status'] for e in complete)),
        'paired_events_labels_and_seed_order_verified':True,'calibration_cv_students_and_test_disjoint':True,
        'coefficients_and_selected_models_unchanged_after_test':True,'frozen_source_data_mapping_identities_unchanged':True,
        'both_datasets_selected_before_test':True,'online_final_verification':'pending'}
    base.dump(OUT/'verification.json',verification)
    return verification


def sync_and_verify(protocol,aggregate):
    base.OUT=OUT;wandb=base.login()
    registry=base.read(OUT/'run_registry.json')
    entry=registry.setdefault('wandb_runs',{}).setdefault('report',{'run_id':wandb.util.generate_id()})
    base.dump(OUT/'run_registry.json',registry)
    run=wandb.init(entity=protocol['wandb']['entity'],project=protocol['wandb']['project'],
        id=entry['run_id'],resume='allow',settings=base.settings(),group=protocol['wandb']['group'],
        job_type='curriculum_report',name='simplekt_curriculum_report',
        config={'model':'simplekt','protocol_sha256':sha(OUT/'protocol.json'),'exploratory':True,
                'bootstrap_resamples':2000,'aggregation':'mean per-seed AUC'})
    expected={}
    try:
        for row in aggregate.to_dict('records'):
            for key in ['auc','seed_sd','brier','log_loss','macro_auc','within_weighted_auc',
                        'delta_original_kt','ci95_low_original_kt','ci95_high_original_kt']:
                if pd.notna(row[key]):
                    metric=f"report/{row['dataset']}/{row['method']}/{key}"
                    expected[metric]=float(row[key]);run.summary[metric]=float(row[key])
        run.summary['report/local_checks_passed']=True
        registry=base.read(OUT/'run_registry.json');registry['wandb_runs']['report']['url']=run.url
        base.dump(OUT/'run_registry.json',registry)
        run.finish(exit_code=0)
    except BaseException:
        try:run.finish(exit_code=1)
        except Exception:pass
        raise
    api=wandb.Api(timeout=30)
    project=f"{protocol['wandb']['entity']}/{protocol['wandb']['project']}"
    checked=[]
    registry=base.read(OUT/'run_registry.json')
    for stage in ['development','evaluate','report']:
        record=registry['wandb_runs'][stage]
        remote=api.run(project+'/'+record['run_id'])
        assert remote.state=='finished',f'W&B {stage} not yet finished'
        assert remote.config['protocol_sha256']==sha(OUT/'protocol.json')
        summary=dict(remote.summary)
        if stage=='report':
            assert summary['report/local_checks_passed']
            for key,value in expected.items():assert abs(float(summary[key])-value)<1e-12,key
        elif stage=='development':
            assert summary['readout/completed_fits']==312 and summary['readout/all_convergence_checks_passed']
            for ds in DATASETS:
                selected=base.read(OUT/ds/'selection.json')
                for design,choice in selected['families'].items():
                    assert float(summary[f'calibration/{ds}/{design}/selected_lambda'])==choice['lambda']
                    assert abs(float(summary[f'calibration/{ds}/{design}/mean_cv_auc'])-choice['selected_auc'])<1e-12
        else:
            for ds in DATASETS:
                for row in pd.read_csv(OUT/ds/'scores_by_seed.csv').to_dict('records'):
                    key=f"test/{ds}/seed{row['seed']}/{row['method']}/auc"
                    assert abs(float(summary[key])-float(row['auc']))<1e-12,key
        checked.append({'stage':stage,'url':remote.url,'state':remote.state,'verified':True})
    result={'checked_utc':base.now(),'runs':checked,'aggregate_metrics_and_protocols_match':True,
            'no_student_data_or_checkpoints_uploaded':True}
    base.dump(OUT/'online_verification.json',result)
    return result


def write_report(protocol,aggregate,contrasts,verification):
    labels={'original_kt':'Original SimpleKT','practice_frequency':'Fixed frequency map-back',
        'pooled_map':'Learned positive map-back','pooled_original':'Calibrated original',
        'original_selected':'Development-retained original','augment_induced':'Original + induced',
        'augment_original':'Original + original control','augment_induced_selected':'Development-retained original + induced',
        'augment_original_selected':'Development-retained original + original'}
    headline=(aggregate[aggregate.method.isin(['original_kt','raw_induced_diagnostic','pooled_map'])]
        .pivot(index='dataset',columns='method',values='auc')
        .reindex(index=['assist2012','assist2017'],columns=['original_kt','raw_induced_diagnostic','pooled_map'])
        .rename(columns={'original_kt':'baseline_auc','raw_induced_diagnostic':'induced_auc',
                         'pooled_map':'currkt_pure_map_back_auc'}).reset_index())
    assert headline.notna().all().all()
    headline.insert(0,'model','simplekt')
    headline.to_csv(OUT/'three_way_auc_comparison.csv',index=False)
    lines=['# SimpleKT Curriculum KT comparison','',
        'Separate original/induced SimpleKT models and sweeps use matched events and splits. All model settings and readout choices used development data. Values are means of three per-seed response AUCs on the same held-out correctness outcomes.',
        '', '| Dataset | Baseline AUC | Induced AUC (paper-style) | CurrKT AUC (pure map-back) |',
        '|---|---:|---:|---:|']
    for row in headline.to_dict('records'):
        lines.append(f"| {row['dataset']} | {row['baseline_auc']:.6f} | {row['induced_auc']:.6f} | {row['currkt_pure_map_back_auc']:.6f} |")
    lines+=['', 'The induced column queries the realized induced target concept. CurrKT uses the prespecified learned positive map-back to predict from the original target concept; it does not use the original model or realized induced target label. The columns are different prediction conditions on the same responses. Augmentation is reported separately below and is never substituted into the pure map-back column.',
        '', '| Method | ASSIST2012 AUC | ASSIST2017 AUC |','|---|---:|---:|']
    for method,label in labels.items():
        values=[float(aggregate[(aggregate.dataset==ds)&(aggregate.method==method)].auc.iloc[0]) for ds in ['assist2012','assist2017']]
        lines.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    lines+=['','| Dataset | Contrast | AUC difference | Paired student95% interval |','|---|---|---:|---|']
    for row in contrasts.to_dict('records'):
        lines.append(f"| {row['dataset']} | {row['method']} minus {row['comparator']} | {row['delta_auc']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    lines+=['','Raw induced AUC is a separate diagnostic using the realized target induced concept; it is not evidence of map-back improvement.']
    for ds in ['assist2012','assist2017']:
        value=float(aggregate[(aggregate.dataset==ds)&(aggregate.method=='raw_induced_diagnostic')].auc.iloc[0])
        lines.append(f'- {ds} raw induced diagnostic: {value:.6f}.')
    lines+=['', 'Pure map-back and original-plus-induced augmentation answer different questions. The original-plus-original control tests whether an augmentation gain exceeds combining two original-taxonomy models.',
        '',f"Training:40 backbone fits; convergence diagnoses:{verification['backbone_convergence_counts']}. All312 readout fits passed the independent convergence check; maximum scaled projected gradient{verification['max_scaled_projected_gradient']:.3g}.",
        '', 'Both datasets have prior test exposure. Results are exploratory. Student intervals condition on the fitted models, taxonomy and selection. The additional four-contrast intervals in the contrast CSV do not control comparisons across the16-model campaign. The2012 condition retains repeated sequence positions and K1000 and does not reproduce the paper K300 condition.',
        '', '[Three-way AUC comparison](three_way_auc_comparison.csv) · [All scores and proper-score diagnostics](two_dataset_comparison.csv) · [Paired contrasts](paired_method_contrasts.csv) · [Checks](verification.json) · [Protocol](protocol.json) · [W&B verification](online_verification.json)',
        '', 'Next: review these completed SimpleKT results and continue to RobustKT under the authorized sequential queue.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    protocol=guard();require_test_ready()
    assert protocol['methods']==METHODS
    assert [tuple(pair) for pair in protocol['contrasts']]==CONTRASTS
    assert base.read(OUT/'test_complete.json')['six_paired_prediction_files']
    base.dump(OUT/'summary_worker.json',{'pid':os.getpid(),'state':'running','started_utc':base.now(),'device':'cpu_statistics_only'})
    state='failed'
    try:
        aggregates,contrasts=[],[]
        for ds in DATASETS:
            a,c=summarize_dataset(protocol,ds);aggregates.append(a);contrasts.append(c)
        aggregate=pd.concat(aggregates,ignore_index=True);contrast=pd.concat(contrasts,ignore_index=True)
        aggregate.to_csv(OUT/'two_dataset_comparison.csv',index=False)
        contrast.to_csv(OUT/'paired_method_contrasts.csv',index=False)
        verification=verify(protocol)
        registry=base.read(OUT/'run_registry.json');registry['state']='local_results_complete_pending_wandb'
        base.dump(OUT/'run_registry.json',registry)
        sync_and_verify(protocol,aggregate)
        verification['online_final_verification']='passed';base.dump(OUT/'verification.json',verification)
        write_report(protocol,aggregate,contrast,verification)
        registry=base.read(OUT/'run_registry.json');registry.update(state='complete',completed_utc=base.now())
        base.dump(OUT/'run_registry.json',registry);state='complete'
        print(aggregate[['dataset','method','auc','delta_original_kt']].to_string(index=False),flush=True)
    finally:
        marker=base.read(OUT/'summary_worker.json');marker.update(state=state,finished_utc=base.now())
        base.dump(OUT/'summary_worker.json',marker)


if __name__=='__main__':
    with ExitStack() as stack:
        for path in [ROOT/'results/simplekt_sweeps_20261107/worker.lock',ROOT/'results/model_sequence_20261107/gpu_worker.lock']:
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
