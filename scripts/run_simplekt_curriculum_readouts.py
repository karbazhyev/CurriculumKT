"""Independent calibration CV/refits, then frozen SimpleKT test predictions."""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import torch

import run_simplekt_sweeps as base
from prepare_simplekt_curriculum import OUT, ROOT, DESIGNS, LAMBDAS, SEEDS, sha
from extract_simplekt_curriculum import guard, require_test_ready, load_features, save_npz, DATASETS
from simplekt_curriculum_features import mapping_layout
from run_pooling_batch_v2 import PooledLogistic, predict, folds, choose


def fit_id(ds, design, pair, lam=None, fold=None):
    suffix = f'cv{fold}_lambda{lam:g}' if fold is not None else 'refit'
    return f'{ds}/{design}/pair{pair}/{suffix}'


def fit_registry():
    entries = {}
    for ds in DATASETS:
        for design in DESIGNS:
            for pair in range(3):
                for lam in LAMBDAS:
                    for fold in range(3):
                        entries[fit_id(ds,design,pair,lam,fold)] = {'state':'prepared',
                            'dataset':ds,'design':design,'pair':pair,'lambda':lam,'fold':fold}
                entries[fit_id(ds,design,pair)] = {'state':'prepared','dataset':ds,
                    'design':design,'pair':pair,'lambda':'development_selected','stage':'refit'}
    assert len(entries) == 312
    return entries


def verify_feature_completion(split):
    completed = base.read(OUT/f'{split}_features_complete.json')
    assert completed['protocol_sha256'] == sha(OUT/'protocol.json')
    assert completed['all_twelve_complete'] and len(completed['manifests']) == 12
    for path, expected in completed['manifests'].items():
        assert sha(Path(path)) == expected


def design_inputs(protocol, ds, pair, split, design):
    original_seed, induced_seed, other_seed = SEEDS[pair], SEEDS[pair], SEEDS[(pair+1)%3]
    meta, orig = load_features(ds,'original',original_seed,split)
    c = meta['original_concept']
    scalar = np.asarray(orig['native_logit'],dtype=np.float32)
    native = np.asarray(orig['native_probability'],dtype=float)
    mapping_record = protocol['datasets'][ds]['mapping']
    assert sha(Path(mapping_record['path'])) == mapping_record['sha256']
    with np.load(mapping_record['path']) as mapping:
        layout = mapping_layout(mapping)
    nc = layout['original_concepts']
    if design in ['pooled_map','augment_induced']:
        other, induced = load_features(ds,'induced',induced_seed,split)
        assert all(np.array_equal(meta[k],other[k]) for k in meta)
        if design == 'pooled_map':
            anchor_weights = layout['weights']
            norm = np.sqrt(np.sum(anchor_weights**2,axis=1)+1e-12)
            x = np.asarray(induced['child_logits']).astype(np.float32)
            x *= layout['mask'][c]
            x *= norm[c,None].astype(np.float32)
            return meta,x,layout['mask'],anchor_weights/norm[:,None],np.array([1.]),native
        second = logit(np.clip(induced['practice_frequency'],1e-6,1-1e-6))
    elif design == 'augment_original':
        other, original2 = load_features(ds,'original',other_seed,split)
        assert all(np.array_equal(meta[k],other[k]) for k in meta)
        second = original2['native_logit']
    if design == 'pooled_original':
        x = scalar[:,None]
    elif design in ['augment_original','augment_induced']:
        x = np.column_stack([scalar,second]).astype(np.float32)
    else:
        raise ValueError(design)
    mask = np.ones((nc,x.shape[1]),dtype=bool)
    anchor = np.array([1.]) if x.shape[1] == 1 else np.array([1.,0.])
    return meta,x,mask,mask.astype(float),anchor,native


def record(key, **info):
    registry = base.read(OUT/'run_registry.json')
    registry['fits'][key].update(info)
    registry['updated_utc'] = base.now()
    base.dump(OUT/'run_registry.json',registry)


def fit_one(key,x,c,y,mask,basis,anchor,lam,student_count,run):
    destination = OUT/'fits'/key
    coefficient = Path(str(destination)+'.npz')
    marker = Path(str(destination)+'.json')
    if marker.exists():
        info = base.read(marker)
        assert info['converged'], f'Preserved failure requires a documented numerical amendment: {key}'
        assert sha(coefficient) == info['coefficient_sha256']
        assert info['protocol_sha256'] == sha(OUT/'protocol.json')
        record(key,state='complete',**info)
        with np.load(coefficient) as d:
            return dict(d),info
    assert not base.read(OUT/'run_registry.json')['heldout_started']
    current = base.read(OUT/'run_registry.json')['fits'][key]
    assert current['state'] != 'failed', f'Preserved failure requires a documented amendment: {key}'
    invocations = list(current.get('invocations',[]))
    if invocations and invocations[-1]['state'] == 'running':
        invocations[-1].update(state='interrupted',identified_utc=base.now())
    invocations.append({'state':'running','started_utc':base.now()})
    record(key,state='running',started_utc=base.now(),invocations=invocations)
    try:
        solver = PooledLogistic(x,c,y,mask,basis,anchor,lam,fixed=False)
        params,info = solver.solve()
        coefficient.parent.mkdir(parents=True,exist_ok=True)
        save_npz(coefficient,params)
        info.update(coefficient_file=str(coefficient),coefficient_sha256=sha(coefficient),
            protocol_sha256=sha(OUT/'protocol.json'),lambda_value=lam,
            fit_students=student_count,test_used=False,completed_utc=base.now())
        base.dump(marker,info)
        invocations[-1].update(state='complete' if info['converged'] else 'failed',finished_utc=base.now())
        record(key,state='complete' if info['converged'] else 'failed',invocations=invocations,**info)
        run.log({'readout/fit':key,'readout/lambda':lam,
            'readout/projected_gradient':info['projected_gradient'],
            'readout/fit_seconds':info['seconds'],'readout/converged':info['converged']})
        del solver
        gc.collect();torch.mps.empty_cache()
        assert info['converged'], f'Convergence check failed: {key}'
        return params,info
    except BaseException as exc:
        current = base.read(OUT/'run_registry.json')['fits'][key]
        if current['state'] != 'complete':
            state='interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed'
            invocations[-1].update(state=state,finished_utc=base.now(),error_type=type(exc).__name__)
            record(key,state=state,error_type=type(exc).__name__,failed_utc=base.now(),invocations=invocations)
        raise


def development(protocol,run):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.backends.mps.is_available()
    torch.set_num_threads(1)
    verify_feature_completion('calibration')
    registry = base.read(OUT/'run_registry.json')
    assert not registry['heldout_started'], 'Readout fitting is closed once held-out evaluation starts'
    expected = fit_registry()
    if not registry['fits']:
        registry['fits'] = expected
        base.dump(OUT/'run_registry.json',registry)
    else:
        assert set(registry['fits']) == set(expected)
    for ds in DATASETS:
        (OUT/ds).mkdir(exist_ok=True)
        selection_file = OUT/ds/'selection.json'
        if selection_file.exists():
            previous = base.read(selection_file)
            assert previous['protocol_sha256'] == sha(OUT/'protocol.json')
            for key,value in previous['full_fit_coefficient_hashes'].items():
                assert sha(Path(str(OUT/'fits'/key)+'.npz')) == value
            continue
        rows,selections,native_scores = [],{},[]
        for design in DESIGNS:
            family_rows=[]
            for pair in range(3):
                meta,x,mask,basis,anchor,native = design_inputs(protocol,ds,pair,'calibration',design)
                c,y = meta['original_concept'],meta['label']
                fold,groups = folds(meta)
                if design == DESIGNS[0] and pair == 0:
                    base.dump(OUT/ds/'development_folds.json',{'seed':20261108,
                        'students':[g.astype(int).tolist() for g in groups],
                        'events':[int(np.sum(fold==f)) for f in range(3)],
                        'calibration_events':len(y),'calibration_students':len(np.unique(meta['uid']))})
                if design == 'pooled_original':
                    native_scores.extend(float(roc_auc_score(y[fold==f],native[fold==f])) for f in range(3))
                for lam in LAMBDAS:
                    for f in range(3):
                        train,valid = fold!=f,fold==f
                        assert not set(meta['uid'][train]) & set(meta['uid'][valid])
                        key=fit_id(ds,design,pair,lam,f)
                        parameters,_=fit_one(key,x[train],c[train],y[train],mask,basis,anchor,lam,
                            len(np.unique(meta['uid'][train])),run)
                        p=predict(x[valid],c[valid],parameters)
                        row={'dataset':ds,'design':design,'pair':pair,'lambda':lam,'fold':f,
                            'auc':float(roc_auc_score(y[valid],p))}
                        rows.append(row);family_rows.append(row)
                        print(row,flush=True)
            selected,scores=choose(family_rows)
            selections[design]={'lambda':selected,'mean_fold_seed_auc':scores,'selected_auc':scores[selected]}
            pd.DataFrame(rows).to_csv(OUT/ds/'development_scores.csv',index=False)
            run.log({f'calibration/{ds}/{design}/selected_lambda':selected,
                     f'calibration/{ds}/{design}/mean_cv_auc':scores[selected]})
        native_auc=float(np.mean(native_scores))
        original='pooled_original' if selections['pooled_original']['selected_auc']>native_auc+1e-6 else 'original_kt'
        control=native_auc if original=='original_kt' else selections['pooled_original']['selected_auc']
        endpoints={'original_selected':original}
        for design in ['augment_induced','augment_original']:
            endpoints[design+'_selected']=design if selections[design]['selected_auc']>control+1e-6 else original
        hashes={}
        for design in DESIGNS:
            lam=selections[design]['lambda']
            for pair in range(3):
                meta,x,mask,basis,anchor,_=design_inputs(protocol,ds,pair,'calibration',design)
                key=fit_id(ds,design,pair)
                _,info=fit_one(key,x,meta['original_concept'],meta['label'],mask,basis,anchor,lam,
                    len(np.unique(meta['uid'])),run)
                hashes[key]=info['coefficient_sha256']
        base.dump(selection_file,{'dataset':ds,'families':selections,'native_cv_auc':native_auc,
            'retention':endpoints,'full_fit_coefficient_hashes':hashes,
            'protocol_sha256':sha(OUT/'protocol.json'),'test_used':False,'completed_utc':base.now()})
    entries=base.read(OUT/'run_registry.json')['fits']
    assert len(entries)==312 and all(e['state']=='complete' and e['converged'] for e in entries.values())
    base.dump(OUT/'development_complete.json',{'completed_utc':base.now(),
        'protocol_sha256':sha(OUT/'protocol.json'),'all_312_fits_complete':True,
        'calibration_feature_manifest_sha256':sha(OUT/'calibration_features_complete.json'),
        'selections':{ds:sha(OUT/ds/'selection.json') for ds in DATASETS},'test_used':False})
    run.summary['readout/completed_fits']=312
    run.summary['readout/all_convergence_checks_passed']=True
    for ds in DATASETS:
        selected=base.read(OUT/ds/'selection.json')
        for design,choice in selected['families'].items():
            run.summary[f'calibration/{ds}/{design}/selected_lambda']=choice['lambda']
            run.summary[f'calibration/{ds}/{design}/mean_cv_auc']=choice['selected_auc']


def evaluate(protocol,run):
    require_test_ready();verify_feature_completion('test')
    for ds in DATASETS:
        selection=base.read(OUT/ds/'selection.json')
        for pair in range(3):
            path=OUT/ds/f'test_pair{pair}.npz'
            if path.exists():
                marker=base.read(path.with_suffix('.json'))
                assert sha(path)==marker['sha256']
                assert marker['protocol_sha256']==sha(OUT/'protocol.json')
                assert marker['selection_sha256']==sha(OUT/ds/'selection.json')
                with np.load(path) as saved:
                    # Recover aggregate logging if a previous process stopped
                    # after saving predictions but before its W&B upload.
                    run.log({f'test/{ds}/seed{SEEDS[pair]}/{method}/auc':float(roc_auc_score(saved['label'],saved[method]))
                        for method in protocol['methods']})
                continue
            output=None
            for design in DESIGNS:
                meta,x,_,_,_,native=design_inputs(protocol,ds,pair,'test',design)
                if output is None:
                    output={**meta,'original_kt':native}
                else:
                    assert all(np.array_equal(meta[k],output[k]) for k in meta)
                key=fit_id(ds,design,pair);coef=Path(str(OUT/'fits'/key)+'.npz')
                assert sha(coef)==selection['full_fit_coefficient_hashes'][key]
                with np.load(coef) as d:
                    output[design]=predict(x,meta['original_concept'],dict(d))
            ind_meta,ind=load_features(ds,'induced',SEEDS[pair],'test')
            assert all(np.array_equal(ind_meta[k],output[k]) for k in ind_meta)
            output['practice_frequency']=np.asarray(ind['practice_frequency'])
            output['raw_induced_diagnostic']=np.asarray(ind['native_probability'])
            for endpoint,choice in selection['retention'].items():
                output[endpoint]=output[choice].copy()
            save_npz(path,output)
            marker={'sha256':sha(path),'selection_sha256':sha(OUT/ds/'selection.json'),
                'protocol_sha256':sha(OUT/'protocol.json'),'created_utc':base.now(),
                'fit_frozen_before_test':True,'all_target_events_and_labels_aligned':True}
            base.dump(path.with_suffix('.json'),marker)
            registry=base.read(OUT/'run_registry.json')
            registry['test_predictions'][f'{ds}/pair{pair}']=marker
            base.dump(OUT/'run_registry.json',registry)
            run.log({f'test/{ds}/seed{SEEDS[pair]}/{method}/auc':float(roc_auc_score(output['label'],output[method]))
                for method in protocol['methods']})
    require_test_ready()
    base.dump(OUT/'test_complete.json',{'completed_utc':base.now(),
        'protocol_sha256':sha(OUT/'protocol.json'),'six_paired_prediction_files':True})


def worker(stage):
    protocol=guard();base.OUT=OUT
    wandb=base.login();registry=base.read(OUT/'run_registry.json')
    ids=registry.setdefault('wandb_runs',{})
    entry=ids.setdefault(stage,{'run_id':wandb.util.generate_id()})
    base.dump(OUT/'run_registry.json',registry)
    run=wandb.init(entity=protocol['wandb']['entity'],project=protocol['wandb']['project'],
        id=entry['run_id'],resume='allow',settings=base.settings(),group=protocol['wandb']['group'],
        job_type='curriculum_'+stage,name='simplekt_curriculum_'+stage,
        config={'model':'simplekt','stage':stage,'protocol_sha256':sha(OUT/'protocol.json'),
                'planned_readout_fits':312,'datasets':DATASETS,'test_used_for_selection':False})
    registry=base.read(OUT/'run_registry.json');registry['wandb_runs'][stage].update(url=run.url)
    base.dump(OUT/'run_registry.json',registry)
    path=OUT/'readout_worker.json'
    base.dump(path,{'pid':os.getpid(),'stage':stage,'state':'running','started_utc':base.now()})
    state='failed'
    try:
        if stage=='development':development(protocol,run)
        else:evaluate(protocol,run)
        guard();state='complete';run.finish(exit_code=0)
    except BaseException:
        (OUT/f'{stage}_failure.log').write_text(traceback.format_exc())
        try:run.finish(exit_code=1)
        except Exception:pass
        raise
    finally:
        current=base.read(path);current.update(state=state,finished_utc=base.now());base.dump(path,current)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','evaluate'])
    stage=parser.parse_args().stage
    with ExitStack() as stack:
        for path in [ROOT/'results/simplekt_sweeps_20261107/worker.lock',
                     ROOT/'results/model_sequence_20261107/gpu_worker.lock']:
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker(stage)
