"""Finite native-loss backbone fits; no held-out arrays are accepted here."""
import gc
import os
import random
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from joint_tuning_common import ROOT, SOURCE, OUT, read, dump, now, sha, identity, verify_record
from catchup_simplekt_adapters_20261118 import make, native, training_loss, query_gate
from run_akt_sweeps import diagnose


class SynchronizationError(RuntimeError):
    pass


class ResourceUnavailable(RuntimeError):
    pass


def ensure_disk(minimum_gb=6):
    free=shutil.disk_usage(OUT).free/1e9
    if free<minimum_gb:raise ResourceUnavailable(f'Free disk {free:.2f} GB below {minimum_gb} GB reserve; existing outputs retained.')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def require_mps():
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.backends.mps.is_available(), 'Actual MPS required; no CPU fallback'
    torch.set_num_threads(1)


def login():
    for key, folder in [('WANDB_DIR','wandb'),('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:
        path=OUT/folder; path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ.update(WANDB_DISABLE_CODE='true',WANDB_SAVE_CODE='false',WANDB_CONSOLE='off')
    os.environ['WANDB_API_KEY']=read(SOURCE/'configs/wandb.json')['api_key']
    import wandb
    assert wandb.Api(timeout=30).default_entity=='knowledgetracing42-ucla'
    return wandb


def start_log(name, config, run_id=None):
    try:
        wandb=login()
        return wandb.init(entity='knowledgetracing42-ucla',project='assist2012_publication_kia',
            group='currkt-joint-pca-k-20261109',name=name,id=run_id,resume='allow' if run_id else None,
            config=config,settings=wandb.Settings(mode='online',console='off',save_code=False,
            disable_code=True,disable_git=True,disable_job_creation=True,init_timeout=60))
    except Exception as exc:
        raise SynchronizationError('W&B initialization failed: '+type(exc).__name__) from exc


def audit_log(run_id, expected, destination):
    """Verify every aggregate history row against the finished online run."""
    last=None
    for attempt in range(6):
        try:
            wandb=login()
            remote=wandb.Api(timeout=30).run(f'knowledgetracing42-ucla/assist2012_publication_kia/{run_id}')
            assert remote.state=='finished', remote.state
            rows=list(remote.scan_history(page_size=1000))
            for i, wanted in enumerate(expected):
                matches=[row for row in rows if row.get('record_index')==i]
                assert len(matches)==1, (i,len(matches))
                for key,value in wanted.items():
                    actual=matches[0].get(key)
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        assert actual is not None and abs(float(actual)-value)<=1e-8*max(1,abs(value)),key
                    else: assert actual==value,key
            dump(destination,{'state':'verified','run_id':run_id,'rows':len(expected),'verified_utc':now()})
            return
        except Exception as exc:
            last=type(exc).__name__+': '+str(exc)
            if attempt<5: time.sleep(30)
    raise SynchronizationError('Aggregate online audit failed: '+str(last))


def tensors(arrays, tax, rows, device='mps'):
    return (torch.as_tensor(arrays[tax+'_concepts'][rows],device=device,dtype=torch.long),
            torch.as_tensor(arrays['responses'][rows],device=device,dtype=torch.long))


@torch.inference_mode()
def evaluate(model, arrays, tax):
    model.eval(); truth=[]; values=[]
    for lo in range(0,len(arrays['uid']),64):
        c,r=tensors(arrays,tax,slice(lo,lo+64));mask=(c[:,:-1]>=0)&(c[:,1:]>=0)
        if not mask.any():continue
        p=native(model,c,r)[mask]
        assert bool(p.isfinite().all()) and p.device.type=='mps'
        values.append(p.cpu().numpy());truth.append(r[:,1:][mask].cpu().numpy())
    p=np.concatenate(values); y=np.concatenate(truth)
    return {'auc':float(roc_auc_score(y,p)),
            'loss':float(log_loss(y,np.clip(p.astype(float),1e-7,1-1e-7),labels=[0,1])), 'events':len(y)}


def runtime_gate(name,k,config,arrays,tax,device='mps'):
    """Small training-data reference, including native timing and future leakage."""
    if device=='mps':require_mps()
    eligible=np.flatnonzero((arrays[tax+'_concepts']>=0).sum(1)>=20)[:2]
    assert len(eligible)==2
    c,r=tensors(arrays,tax,eligible,device)
    model=make(name,k,config,device)
    model.train();loss=training_loss(model,c,r);assert bool(loss.isfinite())
    loss.backward()
    assert all(p.grad.isfinite().all() and p.grad.device.type==device for p in model.parameters() if p.grad is not None)
    model.eval()
    check=query_gate(model,c,r)
    # Querying the actual current target must reproduce native prediction.
    from catchup_simplekt_adapters_20261118 import query
    with torch.inference_mode():
        p=native(model,c,r)
        rows=[0,0,1,1];positions=[1,7,3,15]
        queried=query(model,c,r,rows,positions,c[rows,positions])
        native_delta=float((queried-p[rows,np.array(positions)-1]).abs().max())
        assert native_delta<2e-5, native_delta
        # Current/future labels must not alter the actual native forecast.
        rr=r.clone();rr[:,15:]=1-rr[:,15:].clamp_min(0)
        assert float((p[:,:15]-native(model,c,rr)[:,:15]).abs().max())<2e-5
    if device=='mps':
        reference=make(name,k,config,'cpu');reference.load_state_dict(cpu_state(model));reference.eval()
        with torch.inference_mode():
            expected=native(reference,c.cpu(),r.cpu())
        mask=(c.cpu()[:,:-1]>=0)&(c.cpu()[:,1:]>=0)
        difference=float((expected[mask]-p.cpu()[mask]).abs().max())
        assert difference<2e-5, difference
        check['cpu_reference_max_probability_delta']=difference
    check.update(native_query_delta=native_delta,finite_gradients=True,k=k,width=config['width'],training_loss=float(loss.detach()),empirical_fits=0)
    del model;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return check


def fit(folder, name, ds, tax, candidate, seed, train, valid, k, protocol_hash):
    require_mps();ensure_disk();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    contract={'model':name,'dataset':ds,'taxonomy':tax,'candidate':candidate,'seed':seed,'k':k,'protocol_sha256':protocol_hash}
    if (folder/'contract.json').exists():assert read(folder/'contract.json')==contract
    else:dump(folder/'contract.json',contract)
    if (folder/'failure.json').exists():raise RuntimeError('Preserved trial failure: '+str(folder))
    if (folder/'result.json').exists():
        result=read(folder/'result.json');verify_record(result['checkpoint'])
        audit=folder/'online_audit.json'
        if not audit.exists():audit_log(result['wandb_run_id'],read(folder/'online_records.json'),audit)
        return result
    model=make(name,k,candidate,'mps',seed);optimizer=torch.optim.Adam(model.parameters(),lr=candidate['learning_rate'])
    parameter_count=sum(p.numel() for p in model.parameters())
    history=[];best_auc=-1.;best_epoch=0;patience_auc=-1.;patience_epoch=0;first=1;total=0.;stop='max_epochs'
    latest=folder/'latest.ckpt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['contract']==contract
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']);torch.mps.set_rng_state(saved['mps_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        history=saved['history'];best_auc=saved['best_auc'];best_epoch=saved['best_epoch'];patience_auc=saved['patience_auc'];patience_epoch=saved['patience_epoch'];total=saved['total'];first=saved['epoch']+1
        save(folder/'best.ckpt',saved['best_model'])
    run_id=read(folder/'wandb.json')['id'] if (folder/'wandb.json').exists() else None
    run=start_log(f'{name}-{ds}-{tax}-{candidate["id"]}-s{seed}',{**contract,'device':'mps','cpu_fallback':False,'parameter_count':parameter_count,'batch_size':64,'epochs_max':100,'patience':15,'test_used':False},run_id)
    dump(folder/'wandb.json',{'id':run.id,'url':run.url})
    try:
        for epoch in range(first,101):
            if history and history[-1]['epoch']-patience_epoch>=15:stop='early_stopping';break
            start=time.monotonic();model.train();objective=0.;events=0
            order=np.random.default_rng(seed*1000+epoch).permutation(len(train['uid']))
            for lo in range(0,len(order),64):
                c,r=tensors(train,tax,order[lo:lo+64]);n=int(((c[:,:-1]>=0)&(c[:,1:]>=0)).sum())
                if n==0:continue
                optimizer.zero_grad(set_to_none=True);loss=training_loss(model,c,r)
                assert bool(loss.isfinite()),'Nonfinite native objective'
                loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
                assert all(g.device.type=='mps' and bool(g.isfinite().all()) for g in grads),'Nonfinite or non-MPS gradients'
                if name=='dtransformer':torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();objective+=float(loss.detach())*n;events+=n
            tr=evaluate(model,train,tax);va=evaluate(model,valid,tax)
            if va['auc']>best_auc:best_auc=va['auc'];best_epoch=epoch;save(folder/'best.ckpt',cpu_state(model))
            if va['auc']>patience_auc+.0001:patience_auc=va['auc'];patience_epoch=epoch
            elapsed=time.monotonic()-start;total+=elapsed
            row={'epoch':epoch,'train/auc':tr['auc'],'train/loss':tr['loss'],'optimization/native_objective':objective/events,
                 'validation/auc':va['auc'],'validation/loss':va['loss'],'runtime/epoch_seconds':elapsed}
            history.append(row)
            run.log({'live_epoch':epoch,**row})
            save(latest,{'contract':contract,'model':cpu_state(model),'optimizer':optimizer.state_dict(),'history':history,
                 'best_model':torch.load(folder/'best.ckpt',map_location='cpu',weights_only=True),'best_auc':best_auc,'best_epoch':best_epoch,
                 'patience_auc':patience_auc,'patience_epoch':patience_epoch,'epoch':epoch,'total':total,
                 'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state().cpu(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
            dump(folder/'history.json',history)
            dump(folder/'progress.json',{'state':'running','updated_utc':now(),'epoch':epoch,'best_validation_auc':best_auc})
            print(f'{name} {ds} {tax} {candidate["id"]} seed{seed} epoch{epoch} val={va["auc"]:.6f}',flush=True)
        model.load_state_dict(torch.load(folder/'best.ckpt',map_location='cpu',weights_only=True));replay=evaluate(model,valid,tax)
        assert abs(replay['auc']-best_auc)<1e-6,'Checkpoint replay mismatch'
        if history[-1]['epoch']-patience_epoch>=15:stop='early_stopping'
        result={'state':'complete','candidate':candidate,'seed':seed,'k':k,'checkpoint':identity(folder/'best.ckpt'),
                'best_validation_auc':best_auc,'best_validation_loss':replay['loss'],'best_epoch':best_epoch,'epochs':len(history),
                'parameter_count':parameter_count,'diagnosis':diagnose(history,stop),'total_seconds':total,
                'validation_replay_passed':True,'wandb_run_id':run.id,'completed_utc':now()}
        # Publish the authoritative per-epoch history once. Local checkpoints
        # remain the resume source; interrupted attempts remain in W&B history.
        for i,row in enumerate(history):run.log({'record_index':i,**row})
        dump(folder/'online_records.json',history)
        run.summary.update({'best_validation_auc':best_auc,'best_epoch':best_epoch,'diagnosis':result['diagnosis']['status'],'test_used':False})
        run.finish();dump(folder/'result.json',result)
        audit_log(result['wandb_run_id'],history,folder/'online_audit.json')
        if latest.exists():
            dump(folder/'released_optimizer_cache.json',{'file':identity(latest),'released_utc':now(),
                 'reason':'Completed verified fit; best checkpoint and full history retained. No further training authorized for this trial.'})
            latest.unlink()
        return result
    except BaseException:
        run.finish(exit_code=1);raise
    finally:
        del model,optimizer;gc.collect();torch.mps.empty_cache()
