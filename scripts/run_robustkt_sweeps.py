"""Run/resume four bounded W&B grid sweeps, one MPS trial at a time."""
import argparse
import copy
import csv
import fcntl
import json
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.nn import functional as F

from check_attention_model_readiness import module, forward
from robustkt_mps_portability import robustkt_class
from evaluate_saved_curriculum_kt import SOURCE, ROOT, sha

OUT = ROOT / 'results/robustkt_sweeps_20261107'
DATA_CACHE = {}
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(p.read_text())


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def settings():
    import wandb
    return wandb.Settings(mode='online', console='off', save_code=False,
        disable_code=True, disable_git=True, disable_job_creation=True, init_timeout=60)


def login():
    for key, value in {'WANDB_DIR':str(OUT/'wandb'), 'WANDB_CACHE_DIR':str(OUT/'wandb_cache'),
                       'WANDB_DATA_DIR':str(OUT/'wandb_data'), 'WANDB_DISABLE_CODE':'true',
                       'WANDB_SAVE_CODE':'false', 'WANDB_CONSOLE':'off'}.items():
        os.environ[key] = value
    for sub in ['wandb','wandb_cache','wandb_data']:
        (OUT/sub).mkdir(exist_ok=True)
    os.environ['WANDB_API_KEY'] = read(SOURCE/'configs/wandb.json')['api_key']
    import wandb
    api = wandb.Api(timeout=30)
    assert api.default_entity == 'knowledgetracing42-ucla'
    return wandb


def verify(protocol, full=True):
    assert sha(OUT/'protocol.json') == read(OUT/'protocol_identity.json')['sha256']
    for path, expected in protocol['source_sha256'].items():
        p = Path(path)
        if full or p.suffix == '.py':
            assert sha(p) == expected, f'Frozen source changed: {path}'


def data(protocol, dataset, taxonomy, split):
    key = (dataset,taxonomy,split)
    if key not in DATA_CACHE:
        record = protocol['datasets'][dataset]['files'][split]
        p = Path(record['path'])
        assert sha(p) == record['sha256']
        d = np.load(p)
        c, r = d[taxonomy+'_concepts'], d['responses']
        valid = c >= 0
        mask = valid[:,1:] & valid[:,:-1]
        DATA_CACHE[key] = (np.maximum(c,0).astype(np.int64), np.maximum(r,0).astype(np.int64), mask)
    return DATA_CACHE[key]


def make_model(protocol, entry):
    cfg, fixed = entry['candidate'], protocol['fixed']
    torch.manual_seed(entry['seed']); torch.mps.manual_seed(entry['seed'])
    cls = robustkt_class('mps')
    return cls(n_question=protocol['datasets'][entry['dataset']][entry['taxonomy']+'_concepts'],
        n_pid=0,d_model=cfg['d_model'],dropout=cfg['dropout'],n_blocks=fixed['n_blocks'],
        d_ff=fixed['d_ff'],num_attn_heads=fixed['num_attn_heads'],
        final_fc_dim=fixed['final_fc_dim'],emb_type='qid',ks=fixed['ks'],
        kq_same=fixed['kq_same'],separate_qa=fixed['separate_qa'],l2=fixed['l2']).to('mps')


def tensors(values, ix):
    c,r,mask = values
    return torch.from_numpy(c[ix]).to('mps'),torch.from_numpy(r[ix]).to('mps'),torch.from_numpy(mask[ix]).to('mps')


def evaluate(model, values, batch_size=64):
    model.eval(); ps, ys = [], []
    with torch.inference_mode():
        for lo in range(0,len(values[0]),batch_size):
            c,r,mask = tensors(values,slice(lo,lo+batch_size))
            p = forward(model,c,r)[:,1:][mask].cpu().numpy()
            y = r[:,1:][mask].cpu().numpy()
            if len(y):
                ps.append(p); ys.append(y)
    p, y = np.concatenate(ps),np.concatenate(ys)
    assert len(y) == int(values[2].sum())
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return {'auc':float(roc_auc_score(y,p)),
            'loss':float(log_loss(y,np.clip(p.astype(np.float64),1e-7,1-1e-7),labels=[0,1])),
            'targets':len(y)}


def preflight(model, values):
    c,r,mask = tensors(values,slice(0,4))
    model.eval()
    with torch.no_grad():
        p = forward(model,c,r)
        assert p.device.type == 'mps' and p.isfinite().all()
        rr = r.clone(); rr[:,42:] = 1-rr[:,42:]
        delta = float((forward(model,c,rr)[:,:43]-p[:,:43]).abs().max())
        cc = c.clone(); cc[:,43:] = (cc[:,43:]+1)%model.n_question
        concept_delta = float((forward(model,cc,r)[:,:43]-p[:,:43]).abs().max())
    assert max(delta,concept_delta)<2e-5
    assert all(p.device.type=='mps' for p in model.parameters())
    return {'current_future_response_delta':delta,'future_concept_delta':concept_delta,'passed':True}


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def diagnose(history, stop_reason):
    best = max(history,key=lambda r:r['validation/auc']); last=history[-1]
    slope = float(np.polyfit([r['epoch'] for r in history[-10:]], [r['validation/auc'] for r in history[-10:]],1)[0]) if len(history)>1 else 0.
    drop = best['validation/auc']-last['validation/auc']
    status = 'validation_plateau'
    if stop_reason=='max_epochs' and (last['epoch']-best['epoch']<15 or slope>.0001):
        status='epoch_budget_limited'
    elif drop>.002 and last['train/loss']<best['train/loss']:
        status='validation_degradation_overfitting'
    return {'status':status,'stop_reason':stop_reason,'validation_drop_from_peak':drop,
            'last_train_validation_auc_gap':last['train/auc']-last['validation/auc'],
            'tail_10_epoch_auc_slope':slope,'interpretation':'Learning-curve diagnosis, not mathematical optimum certification.'}


def fit(run, protocol, key, entry):
    assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
    verify(protocol,full=False)
    folder=OUT/'trials'/key; folder.mkdir(parents=True,exist_ok=True)
    tr=data(protocol,entry['dataset'],entry['taxonomy'],'train')
    va=data(protocol,entry['dataset'],entry['taxonomy'],'validation')
    model=make_model(protocol,entry); opt=torch.optim.Adam(model.parameters(),lr=entry['candidate']['learning_rate'])
    parameter_count=sum(p.numel() for p in model.parameters())
    check=preflight(model,tr); dump(folder/'preflight.json',check)
    best_auc=-1.; best_loss=None; best_epoch=0; patience_auc=-1.; patience_epoch=0
    history=[]; best_state=None; first_epoch=1; total_seconds=0.
    latest=folder/'latest.ckpt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['protocol_sha256']==sha(OUT/'protocol.json')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng']);torch.mps.set_rng_state(saved['mps_rng'])
        best_auc=saved['best_auc'];best_epoch=saved['best_epoch'];best_loss=saved['best_loss']
        patience_auc=saved['patience_auc'];patience_epoch=saved['patience_epoch']
        history=saved['history'];first_epoch=saved['epoch']+1;best_state=saved['best_model']
        total_seconds=saved['total_seconds'];save(folder/'best.ckpt',best_state)
    run.name=key
    run.tags=['CurriculumKT','robustkt',entry['dataset'],entry['taxonomy'],'validation-only','mps']
    run.config.update({'candidate_id':entry['candidate']['id'],'device':'mps','parameter_count':parameter_count,
        'protocol_sha256':sha(OUT/'protocol.json'),'data_sha256':protocol['datasets'][entry['dataset']]['files']['train']['sha256'],
        'validation_data_sha256':protocol['datasets'][entry['dataset']]['files']['validation']['sha256'],
        'mapping_sha256':protocol['datasets'][entry['dataset']]['mapping']['sha256'],
        'source_sha256':protocol['source_sha256'][str(SOURCE/'pykt/models/robustkt.py')],
        'n_pid':0,'seq_len':200,'batch_size':64,'max_epochs':100,'patience':15,'patience_min_delta_auc':.0001,
        'optimizer':'Adam','gradient_clipping':'none','test_evaluated':False},allow_val_change=False)
    run.define_metric('epoch')
    for prefix in ['train/*','validation/*','optimization/*','runtime/*']:
        run.define_metric(prefix,step_metric='epoch')
    run.define_metric('validation/best_auc',summary='max')
    registry=read(OUT/'run_registry.json')
    registry['trials'][key].update({'state':'running','wandb_run_id':run.id,'wandb_url':run.url,'parameter_count':parameter_count,'started_utc':entry.get('started_utc',now())})
    registry.update({'state':'running','active_run':key,'updated_utc':now()});dump(OUT/'run_registry.json',registry)
    stop_reason='max_epochs'
    for epoch in range(first_epoch,101):
        if history and history[-1]['epoch']-patience_epoch>=15:
            stop_reason='early_stopping';break
        if STOP:
            raise InterruptedError('Stop requested; last complete epoch is resumable')
        torch.mps.synchronize();start=time.perf_counter();model.train()
        order=np.random.default_rng(entry['seed']*1000+epoch).permutation(len(tr[0]))
        loss_sum=0.;events=0;norms=[]
        for lo in range(0,len(order),64):
            ix=order[lo:lo+64]; n=int(tr[2][ix].sum())
            if not n:continue
            c,r,mask=tensors(tr,ix);opt.zero_grad(set_to_none=True)
            prediction,regularizer=model(c,r)
            p=prediction[:,1:][mask]
            loss=F.binary_cross_entropy(p,r[:,1:][mask].float())+regularizer
            assert bool(loss.isfinite()),'Nonfinite training loss'
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
            assert bool(norm.isfinite()),'Nonfinite gradient'
            assert all(g.device.type=='mps' for g in grads)
            opt.step();norms.append(float(norm));loss_sum+=float(loss.detach())*n;events+=n
        train_metrics=evaluate(model,tr);valid_metrics=evaluate(model,va)
        auc=valid_metrics['auc']
        if auc>best_auc:
            best_auc=auc;best_epoch=epoch;best_loss=valid_metrics['loss'];best_state=cpu_state(model)
            save(folder/'best.ckpt',best_state)
        if auc>patience_auc+.0001:
            patience_auc=auc;patience_epoch=epoch
        torch.mps.synchronize();elapsed=time.perf_counter()-start;total_seconds+=elapsed
        row={'epoch':epoch,'train/loss':train_metrics['loss'],'train/auc':train_metrics['auc'],
            'optimization/training_bce':loss_sum/events,'optimization/grad_norm_mean':float(np.mean(norms)),
            'optimization/grad_norm_max':float(np.max(norms)),'optimization/clipping_fraction':0.,
            'optimization/learning_rate':entry['candidate']['learning_rate'],
            'validation/loss':valid_metrics['loss'],'validation/auc':auc,'validation/best_auc':best_auc,
            'validation/best_epoch':best_epoch,'validation/patience_reference_epoch':patience_epoch,
            'runtime/epoch_seconds':elapsed,'runtime/total_epoch_seconds':total_seconds,
            'runtime/mps_memory_bytes':torch.mps.current_allocated_memory(),
            'runtime/mps_driver_memory_bytes':torch.mps.driver_allocated_memory()}
        history.append(row)
        save(latest,{'protocol_sha256':sha(OUT/'protocol.json'),'model':cpu_state(model),
            'optimizer':opt.state_dict(),'epoch':epoch,'best_model':best_state,'best_auc':best_auc,
            'best_loss':best_loss,'best_epoch':best_epoch,'patience_auc':patience_auc,'patience_epoch':patience_epoch,
            'history':history,'total_seconds':total_seconds,'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state().cpu()})
        dump(folder/'history.json',history)
        run.log(row,step=epoch)
        registry=read(OUT/'run_registry.json');e=registry['trials'][key]
        e.update({'epochs':epoch,'best_epoch':best_epoch,'best_validation_auc':best_auc,'best_validation_loss':best_loss,
                  'last_validation_auc':auc,'last_epoch_seconds':elapsed,'total_epoch_seconds':total_seconds,'updated_utc':now()})
        dump(OUT/'run_registry.json',registry)
        if epoch==1:
            dump(folder/'first_epoch_timing.json',{'epoch_seconds':elapsed,'train_chunks':len(tr[0]),'validation_chunks':len(va[0]),
                'includes':'Training plus full training/validation inference; initial kernel warm-up included.',
                'one_trial_100_epoch_linear_projection_seconds':elapsed*100})
        print(json.dumps({'trial':key,**row}),flush=True)
        if epoch-patience_epoch>=15:
            stop_reason='early_stopping';break
    model.load_state_dict(torch.load(folder/'best.ckpt',map_location='cpu',weights_only=True))
    reproduced=evaluate(model,va)
    assert abs(reproduced['auc']-best_auc)<1e-6,'Best-checkpoint validation replay mismatch'
    diagnosis=diagnose(history,stop_reason)
    result={'state':'complete','best_validation_auc':best_auc,'best_validation_loss':best_loss,'best_epoch':best_epoch,
        'epochs':history[-1]['epoch'],'diagnosis':diagnosis,'validation_reload_passed':True,
        'checkpoint_sha256':sha(folder/'best.ckpt'),'total_epoch_seconds':total_seconds,'finished_utc':now()}
    dump(folder/'result.json',result)
    for k,v in result.items():
        if k!='diagnosis':run.summary['result/'+k]=v
    for k,v in diagnosis.items():run.summary['convergence/'+k]=v
    registry=read(OUT/'run_registry.json');registry['trials'][key].update(result);registry['active_run']=None
    dump(OUT/'run_registry.json',registry)
    del model,opt
    torch.mps.empty_cache()


def callback(protocol, resume_key=None):
    import wandb
    run=None;key=resume_key
    try:
        if resume_key:
            entry=read(OUT/'run_registry.json')['trials'][resume_key]
            run=wandb.init(entity=protocol['wandb']['entity'],project=protocol['wandb']['project'],
                id=entry['wandb_run_id'],resume='must',settings=settings(),group=protocol['wandb']['group'])
        else:
            run=wandb.init(settings=settings(),group=protocol['wandb']['group'],job_type='hyperparameter_screening')
            cfg=dict(run.config)
            candidate=next(c for c in protocol['candidates'] if all(c[k]==cfg[k] for k in ['d_model','learning_rate','dropout']))
            key=f"{cfg['dataset']}_robustkt_{cfg['taxonomy']}_{candidate['id']}_seed42"
            entry=read(OUT/'run_registry.json')['trials'][key]
            assert entry['state'] in ['pending','running','interrupted'], 'Refuse duplicate terminal trial'
        fit(run,protocol,key,entry)
        run.finish(exit_code=0)
    except Exception as e:
        if key:
            registry=read(OUT/'run_registry.json')
            state='interrupted' if isinstance(e,InterruptedError) else 'failed'
            registry['trials'][key].update({'state':state,'error_type':type(e).__name__,'error':str(e),'updated_utc':now()})
            registry['active_run']=None;dump(OUT/'run_registry.json',registry)
            folder=OUT/'trials'/key;folder.mkdir(parents=True,exist_ok=True)
            (folder/'failure.log').write_text(traceback.format_exc())
        traceback.print_exc()
        if run:run.finish(exit_code=1)
        if STOP:raise KeyboardInterrupt
    finally:
        torch.mps.empty_cache()


def register(protocol):
    wandb=login();verify(protocol)
    registry=read(OUT/'run_registry.json')
    for cell,entry in registry['sweeps'].items():
        if entry.get('sweep_id'):continue
        config={'name':f'CurrKT {cell} 20261107','method':'grid','metric':{'name':'validation/best_auc','goal':'maximize'},
            'parameters':{'d_model':{'values':[64,128]},'learning_rate':{'values':[.0003,.001]},'dropout':{'values':[.1,.5]},
                'dataset':{'value':entry['dataset']},'taxonomy':{'value':entry['taxonomy']},'model':{'value':'robustkt'},'seed':{'value':42}}}
        sweep_id=wandb.sweep(config,entity=protocol['wandb']['entity'],project=protocol['wandb']['project'])
        entry.update({'state':'registered','sweep_id':sweep_id,
            'url':f"https://wandb.ai/{protocol['wandb']['entity']}/{protocol['wandb']['project']}/sweeps/{sweep_id}",
            'configuration':config,'registered_utc':now()})
        dump(OUT/'run_registry.json',registry)
        print(json.dumps({'cell':cell,'sweep_url':entry['url']}),flush=True)
    registry['state']='registered';dump(OUT/'run_registry.json',registry)


def summarize():
    registry=read(OUT/'run_registry.json');rows=[];selected={}
    for key,e in registry['trials'].items():
        rows.append({'trial':key,'state':e['state'],'dataset':e['dataset'],'taxonomy':e['taxonomy'],
            **e['candidate'],'epochs':e.get('epochs',0),'best_epoch':e.get('best_epoch'),
            'best_validation_auc':e.get('best_validation_auc'),'best_validation_loss':e.get('best_validation_loss'),
            'convergence':e.get('diagnosis',{}).get('status'),'wandb_url':e.get('wandb_url')})
    for cell in registry['sweeps']:
        entries=[(k,e) for k,e in registry['trials'].items() if e['cell']==cell]
        complete=[(k,e) for k,e in entries if e['state']=='complete']
        if complete:
            k,e=min(complete,key=lambda x:(-x[1]['best_validation_auc'],x[1]['best_validation_loss'],x[1]['parameter_count'],x[1]['candidate']['id']))
            selected[cell]={'trial':k,'candidate':e['candidate'],'validation_auc':e['best_validation_auc'],
                'checkpoint_sha256':e['checkpoint_sha256'],'all_eight_candidates_completed':len(complete)==8}
    with (OUT/'trial_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    dump(OUT/'selected_configurations.json',selected)
    counts={s:sum(e['state']==s for e in registry['trials'].values()) for s in ['pending','running','interrupted','complete','failed']}
    lines=['# RobustKT sweep results','',f'Run counts: {json.dumps(counts)}','',
           'Validation-only screening. No held-out or map-back AUC has been computed in this batch.','',
           '| Cell | Selected setting | Best validation AUC | All eight complete |','|---|---|---:|---|']
    for cell,s in selected.items():
        lines.append(f"| {cell} | {s['candidate']} | {s['validation_auc']:.6f} | {s['all_eight_candidates_completed']} |")
    lines+=['','See [all trials](trial_summary.csv) and [run registry with W&B links](run_registry.json).',
        '','Convergence labels describe learning curves, not proof of optimizer convergence. Failed and epoch-limited trials remain in the table.',
        '','Next: review RobustKT convergence and selected settings. Then run selected-setting seed confirmation and the frozen Curriculum KT comparison before starting another architecture.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return counts


def worker(protocol):
    global STOP
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0', 'CPU fallback must remain disabled'
    locks=[]
    for path in [OUT/'worker.lock',ROOT/'results/model_sequence_20261107/gpu_worker.lock',
                 ROOT/'results/simplekt_sweeps_20261107/worker.lock']:
        lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    assert torch.backends.mps.is_available(),'MPS required; no CPU fallback'
    wandb=login();verify(protocol)
    def stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop)
    dump(OUT/'worker.json',{'pid':os.getpid(),'started_utc':now(),'python':sys.executable,
        'python_version':platform.python_version(),'torch':torch.__version__,'wandb':wandb.__version__,
        'platform':platform.platform(),'mps_available':True,'device':'mps','cpu_fallback':False,'state':'running'})
    try:
        registry=read(OUT/'run_registry.json')
        for key,e in registry['trials'].items():
            if e['state'] in ['running','interrupted'] and e.get('wandb_run_id'):
                callback(protocol,resume_key=key)
        for cell,sweep in registry['sweeps'].items():
            if STOP:break
            current=read(OUT/'run_registry.json')
            pending=sum(e['cell']==cell and e['state']=='pending' for e in current['trials'].values())
            if not pending:continue
            current['sweeps'][cell]['state']='running';current['state']='running';dump(OUT/'run_registry.json',current)
            print(f'Starting {cell}, {pending} remaining trials',flush=True)
            wandb.agent(sweep['sweep_id'],function=lambda:callback(protocol),entity=protocol['wandb']['entity'],
                        project=protocol['wandb']['project'],count=pending)
            current=read(OUT/'run_registry.json')
            remaining=[e for e in current['trials'].values() if e['cell']==cell and e['state'] not in ['complete','failed']]
            current['sweeps'][cell]['state']='incomplete' if remaining else 'complete'
            dump(OUT/'run_registry.json',current);summarize()
        counts=summarize();registry=read(OUT/'run_registry.json')
        registry['state']='complete' if counts['complete']==32 else ('complete_with_failures' if counts['complete']+counts['failed']==32 else 'interrupted_or_incomplete')
        registry['active_run']=None;registry['updated_utc']=now();dump(OUT/'run_registry.json',registry)
        state=registry['state']
    except BaseException:
        traceback.print_exc();state='worker_error';raise
    finally:
        current=read(OUT/'worker.json');current.update({'state':locals().get('state','worker_error'),'finished_utc':now()});dump(OUT/'worker.json',current)
        summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','worker','summarize'])
    args=parser.parse_args();protocol=read(OUT/'protocol.json')
    if args.action=='register':register(protocol)
    elif args.action=='worker':worker(protocol)
    else:summarize()
