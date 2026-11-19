"""Identity-checked feature extraction, independent calibration and frozen tests."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from joint_tuning_common import ROOT, OUT, SEEDS, LAMBDAS, read, dump, now, sha, identity, verify_record
from joint_model_adapters import make, native, query, query_gate, inference_context
from joint_training import require_mps, tensors, start_log, audit_log, ensure_disk, cpu_state
from simplekt_curriculum_features import mapping_layout, event_metadata
from run_pooling_batch_v2 import PooledLogistic, predict, folds
from summarize_robustkt_curriculum import analyze_predictions, METHODS, ADJUSTED

DESIGNS=['pooled_map','pooled_original','augment_induced','augment_original']
META=['uid','row','position','event_id','label','original_concept']


def choose(rows):
    scores={lam:float(np.mean([r['auc'] for r in rows if r['lambda']==lam])) for lam in sorted({r['lambda'] for r in rows})}
    best=max(scores.values());return max(lam for lam,value in scores.items() if value>=best-1e-6),scores


def save_npz(path, arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def open_npz(path):
    with np.load(path) as f:return dict(f)


def release_feature_cache(dest):
    """Only our reproducible feature payloads; keep provenance and audit records."""
    dest=Path(dest).resolve();assert dest.is_relative_to(OUT.resolve())
    marker=dest/'complete.json'
    if not marker.exists() or (dest/'cache_released.json').exists():return
    saved=read(marker);files=[r for key,r in saved['files'].items() if key!='metadata']
    files+=saved.get('parts',[])
    released=[]
    for record in files:
        path=Path(record['path']).resolve();assert path.is_relative_to(dest)
        if path.exists():verify_record(record);path.unlink();released.append(record)
    dump(dest/'cache_released.json',{'released_utc':now(),'payloads':released,
        'reason':'Recomputable feature cache only. Retain backbones, coefficients, scores, metadata, gates and all payload identities.'})


def load_data(record, split, test_ready=None):
    assert split!='test' or test_ready is not None,'Held-out gate closed'
    arrays=open_npz(verify_record(record['files'][split]))
    if record.get('relabel',False):
        mapping=open_npz(verify_record(record['mapping']))
        q=arrays['questions'];valid=q>=0;labels=np.full(q.shape,-1,np.int64)
        labels[valid]=mapping['q_to_induced'][q[valid]];arrays['induced_concepts']=labels
    return arrays


def model_from(record, name, device='mps'):
    model=make(name,record['k'],record['candidate'],device,record['seed'])
    model.load_state_dict(torch.load(verify_record(record['checkpoint']),map_location='cpu',weights_only=True));model.eval()
    return model


@torch.inference_mode()
def extract_batch(model, arrays, tax, layout, offset):
    if model._joint_name=='akt':
        from akt_stable_features import extract_batch as akt_extract
        return akt_extract(model,arrays,tax,layout,row_offset=offset,query_batch_size=256)[:2]
    meta,rows,positions=event_metadata(arrays,offset)
    c,r=tensors(arrays,tax,slice(None),str(next(model.parameters()).device));p=native(model,c,r)[rows,positions-1].cpu().numpy()
    features={'native_probability':p,'native_logit':logit(np.clip(p.astype(float),1e-6,1-1e-6)).astype(np.float32)}
    if tax=='induced':
        oc=meta['original_concept'];events,slots=np.nonzero(layout['mask'][oc]);children=layout['children'][oc[events],slots]
        pchild=query(model,c,r,rows[events],positions[events],children,256 if model._joint_name=='sakt' else 32).cpu().numpy()
        logits=np.zeros((len(rows),layout['mask'].shape[1]),np.float32)
        logits[events,slots]=logit(np.clip(pchild.astype(float),1e-6,1-1e-6))
        features.update(child_logits=logits,practice_frequency=np.sum(expit(logits.astype(float))*layout['weights'][oc],axis=1))
    assert all(np.isfinite(v).all() for v in features.values())
    return meta,features


def extract(dest, name, ds_record, backbone, tax, split, protocol_hash, test_ready=None):
    require_mps();dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    contract={'checkpoint':backbone['checkpoint'],'data':ds_record['files'][split],'mapping':ds_record['mapping'],
              'taxonomy':tax,'split':split,'protocol_sha256':protocol_hash}
    marker=dest/'complete.json'
    if marker.exists() and all(Path(r['path']).exists() for r in read(marker)['files'].values()):
        saved=read(marker);assert saved['contract']==contract
        for record in saved['files'].values():verify_record(record)
        return load_features(dest)
    if marker.exists():assert (dest/'cache_released.json').exists(),'Missing retained feature payload'
    ensure_disk()
    arrays=load_data(ds_record,split,test_ready)
    layout=mapping_layout(open_npz(verify_record(ds_record['mapping'])))
    model=model_from(backbone,name)
    # Every trained checkpoint must pass the independent counterfactual gate.
    eligible=np.flatnonzero((arrays[tax+'_concepts']>=0).sum(1)>=20)[:2]
    if split=='test':
        # Gates must use development data, never adapt based on test behaviour.
        dev=load_data(ds_record,'train');eligible=np.flatnonzero((dev[tax+'_concepts']>=0).sum(1)>=20)[:2]
        c,r=tensors(dev,tax,eligible)
    else:c,r=tensors(arrays,tax,eligible)
    check=query_gate(model,c,r)
    if name!='akt':
        reference=make(name,backbone['k'],backbone['candidate'],'cpu',backbone['seed'])
        reference.load_state_dict(cpu_state(model));reference.eval()
        rows=[0,0,1,1];positions=[1,7,3,15];children=(c[rows,positions]+3)%backbone['k']
        p=query(model,c,r,rows,positions,children)
        expected=query(reference,c.cpu(),r.cpu(),rows,positions,children.cpu())
        delta=float((p.cpu()-expected).abs().max());assert delta<2e-5,delta
        check['selected_checkpoint_cpu_reference_delta']=delta
        del reference
    dump(dest/'query_gate.json',check)
    metadata=[];features={};part_records=[]
    for lo in range(0,len(arrays['uid']),16):
        ensure_disk(4)
        part=dest/'parts'/f'{lo:06d}.npz';mark=part.with_suffix('.json')
        if mark.exists() and part.exists():
            info=read(mark);assert info['contract']==contract;value=open_npz(verify_record(info['part']))
        else:
            chunk={key:value[lo:lo+16] for key,value in arrays.items()}
            if not ((chunk['original_concepts'][:,:-1]>=0)&(chunk['original_concepts'][:,1:]>=0)).any():continue
            meta,feature=extract_batch(model,chunk,tax,layout,lo)
            value={**{'meta__'+k:v for k,v in meta.items()},**{'feature__'+k:v for k,v in feature.items()}}
            save_npz(part,value);dump(mark,{'contract':contract,'part':identity(part)})
        metadata.append({key[6:]:value[key] for key in value if key.startswith('meta__')})
        for key in value:
            if key.startswith('feature__'):features.setdefault(key[9:],[]).append(value[key])
        part_records.append(identity(part))
        dump(dest/'progress.json',{'updated_utc':now(),'rows_complete':min(lo+16,len(arrays['uid'])),'total_rows':len(arrays['uid'])})
    meta={k:np.concatenate([m[k] for m in metadata]) for k in META}
    expected,_,_=event_metadata(arrays)
    assert all(np.array_equal(meta[k],expected[k]) for k in META),'Feature/source alignment failed'
    save_npz(dest/'metadata.npz',meta);files={'metadata':identity(dest/'metadata.npz')}
    for key,parts in features.items():
        path=dest/(key+'.npy');np.save(path,np.concatenate(parts));files[key]=identity(path)
    if tax=='induced':
        z=np.load(dest/'child_logits.npy',mmap_mode='r');expected=np.sum(expit(z.astype(float))*layout['weights'][meta['original_concept']],axis=1)
        assert np.allclose(expected,np.load(dest/'practice_frequency.npy'),atol=1e-10,rtol=0)
    dump(marker,{'contract':contract,'files':files,'events':len(meta['label']),'parts':part_records,'completed_utc':now()})
    del model,arrays,features;gc.collect();torch.mps.empty_cache()
    return load_features(dest)


def load_features(dest):
    dest=Path(dest);mark=read(dest/'complete.json')
    for record in mark['files'].values():verify_record(record)
    meta=open_npz(dest/'metadata.npz')
    feature={k:np.load(record['path'],mmap_mode='r') for k,record in mark['files'].items() if k!='metadata'}
    return meta,feature


def map_input(meta,feature,layout):
    c=meta['original_concept'];norm=np.sqrt((layout['weights']**2).sum(1)+1e-12)
    x=np.asarray(feature['child_logits'],np.float32)*layout['mask'][c]*norm[c,None].astype(np.float32)
    return x,layout['mask'],layout['weights']/norm[:,None],np.array([1.])


def fit_readout(path, meta, x, mask, basis, anchor, lam, protocol_hash):
    path=Path(path);marker=path.with_suffix('.json')
    if marker.exists():
        info=read(marker);assert info['protocol_sha256']==protocol_hash and info['lambda']==lam
        assert info['converged'],'Preserved readout convergence failure'
        return open_npz(verify_record(info['coefficients'])),info
    require_mps();solver=PooledLogistic(x,meta['original_concept'],meta['label'],mask,basis,anchor,lam,fixed=False)
    parameters,info=solver.solve();save_npz(path,parameters)
    info.update(coefficients=identity(path),protocol_sha256=protocol_hash,**{'lambda':lam},test_used=False,completed_utc=now())
    dump(marker,info);del solver;gc.collect();torch.mps.empty_cache()
    assert info['converged'],f'Readout failed independent KKT audit: {path}'
    return parameters,info


def select_induced(folder,name,ds_record,backbone,protocol_hash):
    folder=Path(folder);marker=folder/'mapped_selection.json'
    if marker.exists():
        saved=read(marker);assert saved['protocol_sha256']==protocol_hash
        for record in saved['coefficients']:verify_record(record)
        return saved
    mc,fc=extract(folder/'calibration',name,ds_record,backbone,'induced','calibration',protocol_hash)
    mv,fv=extract(folder/'validation',name,ds_record,backbone,'induced','validation',protocol_hash)
    assert not set(mc['uid'])&set(mv['uid'])
    layout=mapping_layout(open_npz(verify_record(ds_record['mapping'])))
    xc,mask,basis,anchor=map_input(mc,fc,layout);xv,*_=map_input(mv,fv,layout)
    rows=[];coefficients=[]
    for lam in LAMBDAS:
        params,info=fit_readout(folder/'readouts'/f'lambda{lam:g}.npz',mc,xc,mask,basis,anchor,lam,protocol_hash)
        score=float(roc_auc_score(mv['label'],predict(xv,mv['original_concept'],params)))
        rows.append({'lambda':lam,'auc':score});coefficients.append(info['coefficients'])
    best=max(row['auc'] for row in rows);selected=max((row for row in rows if row['auc']>=best-1e-6),key=lambda row:row['lambda'])
    saved={'protocol_sha256':protocol_hash,'rows':rows,'selected_lambda':selected['lambda'],'mapped_validation_auc':selected['auc'],
           'test_used':False,'coefficients':coefficients,'completed_utc':now()}
    dump(marker,saved)
    release_feature_cache(folder/'calibration');release_feature_cache(folder/'validation')
    return saved


def design_input(features,pair,design,layout):
    meta,orig=features['original'][pair];c=meta['original_concept'];scalar=np.asarray(orig['native_logit']);native_p=np.asarray(orig['native_probability'])
    if design in ['pooled_map','augment_induced']:
        other,ind=features['induced'][pair];assert all(np.array_equal(meta[k],other[k]) for k in META)
        if design=='pooled_map':return (meta,*map_input(meta,ind,layout),native_p)
        second=logit(np.clip(ind['practice_frequency'],1e-6,1-1e-6))
    elif design=='augment_original':
        other,orig2=features['original'][(pair+1)%3];assert all(np.array_equal(meta[k],other[k]) for k in META)
        second=orig2['native_logit']
    x=scalar[:,None] if design=='pooled_original' else np.column_stack([scalar,second]).astype(np.float32)
    mask=np.ones((layout['original_concepts'],x.shape[1]),bool)
    return meta,x,mask,mask.astype(float),np.array([1.]) if x.shape[1]==1 else np.array([1.,0.]),native_p


def calibrate(folder,name,records,backbones,protocol_hash,map_lambdas=None):
    """All datasets frozen before returning a capability for held-out access."""
    folder=Path(folder);complete=folder/'development_complete.json'
    if complete.exists():return test_gate(folder,protocol_hash)
    logs=[]
    for ds,record in records.items():
        selection_path=folder/ds/'readout_selection.json'
        if selection_path.exists():continue
        features={tax:[extract(folder/ds/'features'/f'{tax}_s{s}'/'calibration',name,record,backbones[ds][tax][str(s)],tax,'calibration',protocol_hash) for s in SEEDS] for tax in ['original','induced']}
        layout=mapping_layout(open_npz(verify_record(record['mapping'])))
        families={};native_scores=[];all_rows=[];coefficient_records={}
        for design in DESIGNS:
            rows=[]
            candidates=[map_lambdas[ds]] if design=='pooled_map' and map_lambdas else LAMBDAS
            for pair in range(3):
                meta,x,mask,basis,anchor,p=design_input(features,pair,design,layout)
                fold,groups=folds(meta)
                if design=='pooled_original':native_scores.extend(float(roc_auc_score(meta['label'][fold==f],p[fold==f])) for f in range(3))
                for lam in candidates:
                    for f in range(3):
                        tr=fold!=f;va=~tr;subset={k:v[tr] for k,v in meta.items()}
                        parameters,info=fit_readout(folder/ds/'fits'/design/f'pair{pair}_lambda{lam:g}_fold{f}.npz',subset,x[tr],mask,basis,anchor,lam,protocol_hash)
                        auc=float(roc_auc_score(meta['label'][va],predict(x[va],meta['original_concept'][va],parameters)))
                        rows.append({'lambda':lam,'auc':auc,'fold':f,'pair':pair})
                        all_rows.append({'design':design,**rows[-1]})
                        logs.append({'dataset':ds,'design':design,'pair':pair,'fold':f,'lambda':lam,'auc':auc,'projected_gradient':info['projected_gradient']})
            selected,scores=choose(rows);families[design]={'lambda':selected,'cv_auc':scores[selected]}
            for pair in range(3):
                meta,x,mask,basis,anchor,_=design_input(features,pair,design,layout)
                _,info=fit_readout(folder/ds/'fits'/design/f'pair{pair}_refit.npz',meta,x,mask,basis,anchor,selected,protocol_hash)
                coefficient_records[f'{design}/{pair}']=info['coefficients']
                logs.append({'dataset':ds,'design':design,'pair':pair,'fold':-1,'lambda':selected,'projected_gradient':info['projected_gradient']})
        native_cv=float(np.mean(native_scores));original='pooled_original' if families['pooled_original']['cv_auc']>native_cv+1e-6 else 'original_kt'
        control=native_cv if original=='original_kt' else families[original]['cv_auc']
        retention={'original_selected':original}
        for design in ['augment_induced','augment_original']:retention[design+'_selected']=design if families[design]['cv_auc']>control+1e-6 else original
        pd.DataFrame(all_rows).to_csv(folder/ds/'development_scores.csv',index=False)
        dump(selection_path,{'protocol_sha256':protocol_hash,'families':families,'native_cv_auc':native_cv,'retention':retention,
             'coefficients':coefficient_records,'test_used':False,'completed_utc':now()})
        dump(folder/ds/'readout_log_records.json',logs);logs=[]
    records_log=[row for ds in records for row in read(folder/ds/'readout_log_records.json')]
    logmarker=folder/'development_wandb.json'
    if not logmarker.exists():
        run=start_log(name+'-calibration',{'model':name,'protocol_sha256':protocol_hash,'stage':'calibration','test_used':False})
        for i,row in enumerate(records_log):run.log({'record_index':i,**row})
        run.summary['completed_readout_fits']=len(records_log);run.finish();dump(logmarker,{'id':run.id})
    audit_log(read(logmarker)['id'],records_log,folder/'development_online_audit.json')
    dump(complete,{'protocol_sha256':protocol_hash,'datasets':{ds:identity(folder/ds/'readout_selection.json') for ds in records},
         'online_audit':identity(folder/'development_online_audit.json'),'backbones':backbones,'completed_utc':now(),'test_used':False})
    return test_gate(folder,protocol_hash)


def test_gate(folder,protocol_hash):
    frozen=read(Path(folder)/'development_complete.json');assert frozen['protocol_sha256']==protocol_hash
    verify_record(frozen['online_audit'])
    for record in frozen['datasets'].values():
        selection=read(verify_record(record));assert selection['test_used'] is False
        for coefficients in selection['coefficients'].values():verify_record(coefficients)
    for ds in frozen['backbones'].values():
        for tax in ds.values():
            for model in tax.values():verify_record(model['checkpoint'])
    return frozen


def evaluate_comparison(folder,name,records,backbones,protocol_hash):
    folder=Path(folder);frozen=test_gate(folder,protocol_hash)
    assert set(frozen['datasets'])==set(records),'All datasets must be frozen before test access'
    aggregates=[];contrasts=[];logs=[]
    for ds,record in records.items():
        selection=read(folder/ds/'readout_selection.json')
        features={tax:[extract(folder/ds/'features'/f'{tax}_s{s}'/'test',name,record,backbones[ds][tax][str(s)],tax,'test',protocol_hash,frozen) for s in SEEDS] for tax in ['original','induced']}
        layout=mapping_layout(open_npz(verify_record(record['mapping'])))
        predictions=[]
        for pair in range(3):
            meta,orig=features['original'][pair];_,ind=features['induced'][pair]
            output={**meta,'original_kt':np.asarray(orig['native_probability']),
                    'raw_induced_diagnostic':np.asarray(ind['native_probability']),'practice_frequency':np.asarray(ind['practice_frequency'])}
            for design in DESIGNS:
                meta,x,_,_,_,_=design_input(features,pair,design,layout)
                params=open_npz(verify_record(selection['coefficients'][f'{design}/{pair}']))
                output[design]=predict(x,meta['original_concept'],params)
            for endpoint,target in selection['retention'].items():output[endpoint]=output[target].copy()
            save_npz(folder/ds/f'test_pair{pair}.npz',output);predictions.append(output)
            logs.extend({'dataset':ds,'seed':SEEDS[pair],'method':method,'auc':float(roc_auc_score(output['label'],output[method]))} for method in METHODS)
        statistics=analyze_predictions(predictions,selection['retention'],2000,20261109,lambda n:print(f'{name} {ds} bootstrap {n}/2000',flush=True))
        stats_contrasts=statistics['contrasts'].rename(columns={'designated_four_contrast_family':'designated_family'})
        for i,row in stats_contrasts.iterrows():
            if (row['method'],row['comparator']) in ADJUSTED:
                delta=statistics['bootstrap'][row['method']]-statistics['bootstrap'][row['comparator']]
                low,high=np.quantile(delta,[.05/(4*len(records)),1-.05/(4*len(records))])
                stats_contrasts.loc[i,['familywise_interval_low','familywise_interval_high']]=[low,high]
        for key in ['scores','per_concept','aggregate']:statistics[key].to_csv(folder/ds/(key+'.csv'),index=False)
        stats_contrasts.to_csv(folder/ds/'contrasts.csv',index=False);save_npz(folder/ds/'bootstrap.npz',statistics['bootstrap'])
        aggregates.append(statistics['aggregate'].assign(dataset=ds,model=name));contrasts.append(stats_contrasts.assign(dataset=ds,model=name))
    aggregate=pd.concat(aggregates,ignore_index=True);contrast=pd.concat(contrasts,ignore_index=True)
    aggregate.to_csv(folder/'comparison.csv',index=False);contrast.to_csv(folder/'contrasts.csv',index=False)
    marker=folder/'test_wandb.json'
    if not marker.exists():
        run=start_log(name+'-frozen-test',{'model':name,'protocol_sha256':protocol_hash,'settings_frozen_before_test':True})
        for i,row in enumerate(logs):run.log({'record_index':i,**row})
        run.finish();dump(marker,{'id':run.id})
    audit_log(read(marker)['id'],logs,folder/'test_online_audit.json')
    test_gate(folder,protocol_hash)
    lines=[f'# {name}: frozen comparison','', 'AUCs are means of seeds42/84/126. Induced is a native-target diagnostic; CurrKT is pure pooled positive map-back. Augmentation is reported separately.','',
           '| Dataset | Baseline | Induced | CurrKT | O+I | O+O |','|---|---:|---:|---:|---:|---:|']
    for ds in records:
        values=aggregate[aggregate.dataset==ds].set_index('method')['auc']
        lines.append('| '+ds+' | '+' | '.join(f'{values[m]:.6f}' for m in ['original_kt','raw_induced_diagnostic','pooled_map','augment_induced','augment_original'])+' |')
    lines+=['','All settings/readouts were selected before test access. Student-paired bootstrap uses2000 draws, conditional on frozen models and selections. Historical test exposure and campaign multiplicity make these exploratory results. Readout failures and capped/overfitting backbone diagnoses remain in the run records.','',
            'See comparison.csv for seed SD, within-concept AUC, Brier and log loss; contrasts.csv for paired uncertainty. No favorable result is required for completion.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(folder/'complete.json',{'state':'complete','protocol_sha256':protocol_hash,'report':identity(folder/'RESULTS.md'),
          'comparison':identity(folder/'comparison.csv'),'contrasts':identity(folder/'contrasts.csv'),'online_audit':identity(folder/'test_online_audit.json'),'completed_utc':now()})
    if folder.resolve().is_relative_to(OUT.resolve()):
        for ds in records:
            for marker in (folder/ds/'features').glob('*/*/complete.json'):release_feature_cache(marker.parent)
    return read(folder/'complete.json')
