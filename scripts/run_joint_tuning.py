"""Finite local controller: finish AKT, then one joint-search model at a time.

No LLM/API-agent calls, cron job, cloud training or environment installation.
"""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from joint_tuning_common import ROOT,SOURCE,OUT,DATA,MODELS,DATASETS,SEEDS,read,dump,now,sha,identity,verify_record
from joint_training import fit, runtime_gate, require_mps, SynchronizationError, ResourceUnavailable
from joint_comparison import load_data, select_induced, calibrate, evaluate_comparison

QUEUE=ROOT/'results/model_sequence_20261107'
OLD_CONTROLLER=ROOT/'results/akt_unattended_20261109'


def guard():
    protocol=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'protocol_identity.json')['sha256']
    assert str(Path(sys.executable))==protocol['python'],'Use the established legacy interpreter'
    assert torch.__version__==protocol['environment']['torch'],'Frozen runtime changed'
    for record in protocol['identities']:verify_record(record)
    return protocol


def status(stage,state,**extra):
    dump(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'state':state,'updated_utc':now(),'codex_calls':0,**extra})


def update_queue(name,state,**extra):
    queue=read(OUT/'queue.json');queue['models'][name].update(state=state,**extra);queue['updated_utc']=now();dump(OUT/'queue.json',queue)
    main=read(QUEUE/'queue.json')
    # Preserve the prior controller's representation and unrelated histories.
    main['joint_tuning']={'queue':str(OUT/'queue.json'),'state':state,'active_model':name,'updated_utc':now()}
    main.update(active_model=name,active_stage=state,updated_utc=now())
    for row in main['models']:
        if row['model']==name:row.update(state=state,joint_queue=str(OUT/'queue.json'),datasets=DATASETS,**extra)
    dump(QUEUE/'queue.json',main)


def require_old_controller():
    deadline=time.monotonic()+72*3600
    status('akt_predecessor','waiting_for_existing_worker')
    while True:
        saved=read(OLD_CONTROLLER/'status.json')
        if saved['state']=='prepared_stages_complete' and saved['stage']=='comparison_implementation_pending':
            # Wait for the old process itself to release its stage locks.
            try:
                with (OLD_CONTROLLER/'controller.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    return
            except BlockingIOError:pass
        if saved['state'] in ['failed','blocked','interrupted','controller_error']:raise RuntimeError('Existing AKT controller needs attention: '+saved['state'])
        assert time.monotonic()<deadline,'AKT predecessor wait exceeded72h'
        time.sleep(30)


def acquire_gpu(stack):
    from inspect_model_workers import inspect
    deadline=time.monotonic()+72*3600
    paths=[QUEUE/'gpu_worker.lock',ROOT/'results/simplekt_sweeps_20261107/worker.lock',ROOT/'results/akt_sweeps_20261109/worker.lock']
    while True:
        trial=ExitStack()
        try:
            for path in paths:
                handle=trial.enter_context(path.open('a'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Only known GPU indicators from neutral inventory; an unrelated
            # generic Python process alone is not evidence of GPU occupancy.
            external=[p for p in inspect() if p['project']=='external' and
                      (p.get('mode') in ['generate','comet','metricx'] or (p.get('gpu_layers') or 0)>0)]
            if external:raise BlockingIOError('External GPU indicators active')
            stack.enter_context(trial.pop_all());return
        except BlockingIOError:trial.close()
        assert time.monotonic()<deadline,'GPU wait exceeded72h'
        status('gpu_availability','waiting');time.sleep(30)


def selected_checkpoint(result):
    return {key:result[key] for key in ['candidate','seed','k','checkpoint']}


def ds_record(ds,taxonomy):
    manifest=read(DATA/ds/'manifest.json')
    return {'files':manifest['files'],'mapping':manifest['taxonomies'][taxonomy]['mapping'],'relabel':True}


def choose_trials(entries,tax):
    complete=[entry for entry in entries if entry['state']=='complete']
    assert complete,'No eligible completed trial in '+tax
    if tax=='original':return min(complete,key=lambda e:(-e['best_validation_auc'],e['best_validation_loss'],e['parameter_count'],e['candidate']['id']))
    best=max(e['mapped_validation_auc'] for e in complete)
    tied=[e for e in complete if e['mapped_validation_auc']>=best-1e-6]
    return min(tied,key=lambda e:(-e['selected_lambda'],e['parameter_count'],e['candidate']['id']))


def execute_model(name,protocol):
    ph=sha(OUT/'protocol.json');folder=OUT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'comparison/complete.json').exists():return read(folder/'comparison/complete.json')
    if (folder/'failure.json').exists():return read(folder/'failure.json')
    design=read(OUT/'design.json');selected={};records={};backbones={};map_lambdas={}
    update_queue(name,'screening');status(name,'screening')
    for ds in DATASETS:
        manifest=read(DATA/ds/'manifest.json');selection=folder/ds/'backbone_selection.json'
        if selection.exists():
            selected[ds]=read(selection);assert selected[ds]['protocol_sha256']==ph
        else:
            outcomes={}
            for tax in ['original','induced']:
                entries=[]
                for candidate in design['trial_assignments'][name][ds][tax]:
                    guard();trial=folder/ds/'screening'/candidate['id'];trial.mkdir(parents=True,exist_ok=True)
                    taxonomy=candidate.get('taxonomy',next(iter(manifest['taxonomies'])))
                    record=ds_record(ds,taxonomy);k=manifest['original_concepts'] if tax=='original' else candidate['k']
                    if (trial/'failure.json').exists():entries.append(read(trial/'failure.json'));continue
                    try:
                        train=load_data(record,'train');valid=load_data(record,'validation')
                        assert not set(train['uid'])&set(valid['uid'])
                        gate=trial/'runtime_gate.json'
                        if not gate.exists():dump(gate,runtime_gate(name,k,candidate,train,tax))
                        result=fit(trial,name,ds,tax,candidate,42,train,valid,k,ph)
                        del train,valid;gc.collect();torch.mps.empty_cache()
                        if tax=='induced':
                            mapping=select_induced(trial/'mapping',name,record,selected_checkpoint(result),ph)
                            result={**result,'selected_lambda':mapping['selected_lambda'],'mapped_validation_auc':mapping['mapped_validation_auc']}
                        dump(trial/'screening_result.json',result);entries.append(result)
                    except (SynchronizationError,ResourceUnavailable):raise
                    except (RuntimeError,NotImplementedError,AssertionError,ValueError) as exc:
                        failure={'state':'failed','candidate':candidate,'seed':42,'error_type':type(exc).__name__,'error':str(exc),'failed_utc':now(),'test_used':False}
                        dump(trial/'failure.json',failure);entries.append(failure);gc.collect();torch.mps.empty_cache()
                        # Native compatibility errors should not burn the other
                        # configurations. They are architecture dispositions,
                        # distinct from numerical failures after fitting.
                        if not gate.exists():raise
                    dump(folder/ds/f'{tax}_screening_registry.json',{'trials':entries,'updated_utc':now()})
                outcomes[tax]=choose_trials(entries,tax)
            selected[ds]={'protocol_sha256':ph,'original':outcomes['original'],'induced':outcomes['induced'],'test_used':False,'frozen_utc':now()}
            dump(selection,selected[ds])
        records[ds]=ds_record(ds,selected[ds]['induced']['candidate']['taxonomy']);map_lambdas[ds]=selected[ds]['induced']['selected_lambda']
        backbones[ds]={}
        for tax in ['original','induced']:
            chosen=selected[ds][tax];backbones[ds][tax]={'42':selected_checkpoint(chosen)}
            for seed in [84,126]:
                status(name,'seed_confirmation',dataset=ds,taxonomy=tax,seed=seed)
                train=load_data(records[ds],'train');valid=load_data(records[ds],'validation')
                result=fit(folder/ds/'confirmation'/f'{tax}_seed{seed}',name,ds,tax,chosen['candidate'],seed,train,valid,chosen['k'],ph)
                backbones[ds][tax][str(seed)]=selected_checkpoint(result)
                del train,valid;gc.collect();torch.mps.empty_cache()
    dump(folder/'selected_models.json',{'models':backbones,'records':records,'test_used':False,'protocol_sha256':ph})
    status(name,'calibration');update_queue(name,'calibration')
    calibrate(folder/'comparison',name,records,backbones,ph,map_lambdas)
    guard();status(name,'frozen_test');update_queue(name,'frozen_test')
    return evaluate_comparison(folder/'comparison',name,records,backbones,ph)


def finish_akt():
    """New comparison implementation; all old training/controller sources fixed."""
    folder=OUT/'akt_fixed_comparison';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    from run_akt_seed_confirmation import verify_stage, OUT as SEED_ROOT
    parent=verify_stage();selected=read(SEED_ROOT/'selected_models.json');assert selected['all_twelve_available']
    for device in ['cpu','mps']:
        path=QUEUE/f'akt_stable_queries_{device}_checks.json';gate=read(path)
        assert gate['state']=='passed' and gate['all_twelve_selected_checkpoints'] and len(gate['checks'])==12
        assert all(c['passed'] for c in gate['checks'])
        for file,expected in gate['implementation_sha256'].items():assert sha(file)==expected
        assert gate['selected_models_sha256']==sha(SEED_ROOT/'selected_models.json')
    audit=read(QUEUE/'akt_online_history_checks.json');assert audit['all_currently_completed_runs_verified'] and len(audit['runs'])==40
    # Accept in a new artifact; the prior controller pins the proposal itself.
    accepted=folder/'accepted_inference_amendment.json'
    if not accepted.exists():dump(accepted,{'state':'accepted_after_mps_verification','training_changed':False,
        'proposal':identity(QUEUE/'akt_inference_amendment_01.json'),'cpu_gate':identity(QUEUE/'akt_stable_queries_cpu_checks.json'),
        'mps_gate':identity(QUEUE/'akt_stable_queries_mps_checks.json'),'accepted_utc':now()})
    records=parent['datasets'];backbones={}
    for ds,record in records.items():
        backbones[ds]={}
        for tax in ['original','induced']:
            backbones[ds][tax]={}
            for seed,value in selected['models'][f'{ds}_akt_{tax}'].items():
                from joint_comparison import open_npz
                layout=open_npz(verify_record(record['mapping']))['counts'].shape
                candidate=value['candidate'];backbones[ds][tax][seed]={'checkpoint':{'path':value['checkpoint'],'sha256':value['checkpoint_sha256']},
                    'candidate':{**candidate,'width':candidate['d_model']},'k':layout[0 if tax=='original' else 1],'seed':int(seed)}
    pfile=folder/'protocol.json'
    if not pfile.exists():dump(pfile,{'model':'akt','new_joint_protocol':identity(OUT/'protocol.json'),'seed_protocol':identity(SEED_ROOT/'protocol.json'),
        'selected_models':identity(SEED_ROOT/'selected_models.json'),'accepted_amendment':identity(accepted),'readout_selection':'3fold student calibration CV, four lambdas, three seeds, four designs;312 fits',
        'scope':'Original two frozen datasets/taxonomies; exact raw-logit features and unchanged pooling/augmentation. Complete both development selections before test.',
        'records':records,'backbones':backbones,'frozen_utc':now()})
    ph=sha(pfile);saved=read(pfile);assert saved['records']==records and saved['backbones']==backbones
    verify_record(saved['new_joint_protocol']);verify_record(saved['accepted_amendment'])
    status('akt','calibration');calibrate(folder,'akt',records,backbones,ph)
    status('akt','frozen_test');result=evaluate_comparison(folder,'akt',records,backbones,ph)
    main=read(QUEUE/'queue.json');main['akt_joint_handoff_completion']=result
    for row in main['models']:
        if row['model']=='akt':row.update(state='complete',comparison_report=str(folder/'RESULTS.md'),completion=result)
    dump(QUEUE/'queue.json',main)
    return result


def combined_report():
    import pandas as pd
    frames=[]
    for name in MODELS:
        path=OUT/'models'/name/'comparison/comparison.csv'
        if path.exists():frames.append(pd.read_csv(path))
    if frames:pd.concat(frames,ignore_index=True).to_csv(OUT/'combined_comparison.csv',index=False)
    queue=read(OUT/'queue.json');lines=['# Joint PCA/K and model tuning','',
        'All three datasets belong to each remaining architecture. The existing AKT two-dataset run is retained separately.','', '| Model | State | Report or failure |','|---|---|---|']
    for name,row in queue['models'].items():
        file=OUT/'models'/name/('comparison/RESULTS.md' if row['state']=='complete' else 'failure.json')
        lines.append(f'| {name} | {row["state"]} | [{file.name}]({file}) |' if file.exists() else f'| {name} | {row["state"]} | pending |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def execute():
    protocol=guard();require_old_controller()
    with ExitStack() as stack:
        acquire_gpu(stack);require_mps();guard();finish_akt()
        for name in MODELS:
            if read(OUT/'queue.json')['models'][name]['state'] in ['complete','failed','incompatible']:continue
            try:
                result=execute_model(name,protocol);update_queue(name,result['state'],result=result)
            except (SynchronizationError,ResourceUnavailable):raise
            except Exception as exc:
                failure={'state':'incompatible' if isinstance(exc,NotImplementedError) else 'failed','error_type':type(exc).__name__,
                         'error':str(exc),'traceback':traceback.format_exc(),'failed_utc':now(),'test_driven_reselection':False}
                dump(OUT/'models'/name/'failure.json',failure);update_queue(name,failure['state'],failure=failure)
            combined_report();guard()
            row=read(OUT/'queue.json')['models'][name]
            with (ROOT/'research/PROJECT_STATUS.md').open('a') as report:
                report.write(f'\n- Local joint controller {now()}: {name} {row["state"]}; all three datasets assigned. Details: [joint results](../results/joint_tuning_20261109/RESULTS.md). No Codex supervision or test-driven reselection.\n')
        states=[r['state'] for r in read(OUT/'queue.json')['models'].values()]
        status('finished','complete' if all(s=='complete' for s in states) else 'complete_with_failures')
        queue=read(OUT/'queue.json');queue['state']=read(OUT/'status.json')['state'];dump(OUT/'queue.json',queue)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','verify']);args=parser.parse_args()
    if args.action=='verify':guard();print('Frozen preparation identities verified; no GPU work.');return
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:execute()
        except BaseException as exc:
            status('attention_required','failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
