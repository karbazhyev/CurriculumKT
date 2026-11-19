"""CPU-only synthetic acceptance: actual adapters, isolated orchestration mocks."""
import argparse
from contextlib import ExitStack
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.special import expit
from joint_tuning_common import OUT,MODELS,DATASETS,read,dump,identity,now
import joint_model_adapters as adapters
import joint_comparison as comparison
import run_joint_tuning as runner


def adapters_check():
    torch.set_num_threads(1);checks=[]
    for name in MODELS:
        for width in [64,128]:
            try:
                model=adapters.make(name,31,{'width':width,'dropout':.1},'cpu')
                c=torch.randint(31,(2,200));r=torch.randint(2,(2,200));c[1,155:]=-1;r[1,155:]=-1
                loss=adapters.training_loss(model,c,r);loss.backward()
                assert loss.isfinite() and all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
                result=adapters.query_gate(model,c,r)
                # A native forecast must be causal as well as the query adapter.
                model.eval()
                with torch.inference_mode():
                    p=adapters.native(model,c,r);rr=r.clone();rr[:,15:]=1-rr[:,15:].clamp_min(0)
                    native_leak=float((p[:,:15]-adapters.native(model,c,rr)[:,:15]).abs().max())
                    assert native_leak<2e-5,native_leak
                    cc=c.clone();cc[:,16:]=(cc[:,16:].clamp_min(0)+9)%31
                    concept_leak=float((p[:,:15]-adapters.native(model,cc,r)[:,:15]).abs().max())
                    assert concept_leak<2e-5,concept_leak
                result.update(model=name,width=width,finite_gradients=True,native_response_leak=native_leak,native_future_concept_leak=concept_leak)
            except NotImplementedError as exc:
                assert name=='ukt';result={'model':name,'width':width,'state':'explicit_incompatibility','reason':str(exc)}
            checks.append(result);print(name,width,result.get('passed',result.get('state')),flush=True)
    dump(OUT/'adapter_final_cpu_checks.json',{'state':'passed_with_explicit_ukt_blocker','checks':checks,'MPS_calls':0,'empirical_fits':0,'completed_utc':now()})


def orchestration_check():
    calls=[];test_calls=[]
    with tempfile.TemporaryDirectory(prefix='currkt-joint-check-') as temp:
        root=Path(temp);mapping=root/'mapping.npz'
        comparison.save_npz(mapping,{'counts':np.ones((2,3),int),'practice_frequency':np.full((2,3),1/3)})
        checkpoint=root/'backbone';checkpoint.write_text('synthetic identity only')
        record={'mapping':identity(mapping)};records={ds:record for ds in DATASETS}
        backbones={ds:{tax:{str(seed):{'checkpoint':identity(checkpoint),'seed':seed} for seed in [42,84,126]} for tax in ['original','induced']} for ds in DATASETS}
        n=120;rng=np.random.default_rng(82);uid=np.repeat(np.arange(30),4)
        meta={'uid':uid,'row':np.arange(n)//4,'position':np.arange(n)%4+1,'event_id':np.arange(n),'label':np.tile([0,1,1,0],30),'original_concept':np.arange(n)%2}
        z=rng.normal(size=(n,3));layout=comparison.mapping_layout(comparison.open_npz(mapping))
        def extract(dest,name,ds_record,backbone,tax,split,protocol_hash,test_ready=None):
            if split=='test':
                assert test_ready and set(test_ready['datasets'])==set(DATASETS);test_calls.append(str(dest))
            p=expit(z[:,0]+.1*(backbone['seed']/42))
            features={'native_logit':np.log(p/(1-p)),'native_probability':p}
            if tax=='induced':features.update(child_logits=z,practice_frequency=expit(z).mean(1))
            return {k:v.copy() for k,v in meta.items()},features
        def fit(path,m,x,mask,basis,anchor,lam,ph):
            assert not test_calls,'Readout fit after test access';calls.append(str(path))
            q=np.asarray(basis,float).copy()
            if q.shape[1]>1 and len(anchor)>1:q*=anchor
            params={'q':q,'b':np.zeros(len(mask))}
            comparison.save_npz(path,params);info={'converged':True,'projected_gradient':0.,'coefficients':identity(path),'lambda':lam}
            return params,info
        class Log:
            id='synthetic-local-only';summary={}
            def log(self,*a,**k):pass
            def finish(self,*a,**k):pass
        def audit(run_id,rows,path):dump(path,{'state':'verified','synthetic':True,'rows':len(rows)})
        with ExitStack() as stack:
            for name,value in [('extract',extract),('fit_readout',fit),('start_log',lambda *a,**k:Log()),('audit_log',audit)]:stack.enter_context(patch.object(comparison,name,value))
            try:comparison.test_gate(root,'test-protocol');raise AssertionError('Premature test gate opened')
            except FileNotFoundError:pass
            comparison.calibrate(root,'synthetic',records,backbones,'test-protocol',{ds:.1 for ds in DATASETS})
            assert len(calls)==387,len(calls)
            count=len(calls);comparison.calibrate(root,'synthetic',records,backbones,'test-protocol',{ds:.1 for ds in DATASETS});assert len(calls)==count
            result=comparison.evaluate_comparison(root,'synthetic',records,backbones,'test-protocol')
            assert result['state']=='complete' and len(test_calls)==18
            coefficients=next(iter(read(root/DATASETS[0]/'readout_selection.json')['coefficients'].values()))
            Path(coefficients['path']).write_bytes(b'corrupted')
            try:comparison.test_gate(root,'test-protocol');raise RuntimeError('Corruption went undetected')
            except AssertionError:pass
        a={'state':'complete','candidate':{'id':'a'},'mapped_validation_auc':.7,'selected_lambda':.1,'parameter_count':100}
        b={**a,'candidate':{'id':'b'},'mapped_validation_auc':.7-.0000005,'selected_lambda':1.}
        assert runner.choose_trials([a,b],'induced')['candidate']['id']=='b'
        assert comparison.choose([{'lambda':.1,'auc':.7}])[0]==.1
        # A test request requires a capability before opening any source path.
        try:comparison.load_data({},'test');raise RuntimeError('Test gate bypass')
        except AssertionError:pass
    dump(OUT/'orchestration_cpu_checks.json',{'state':'passed','synthetic_readout_calls':387,'synthetic_test_feature_calls':18,
         'resumption_did_not_refit':True,'all_three_datasets_frozen_before_test':True,'coefficient_corruption_rejected':True,
         'mapped_auc_tie_rule_passed':True,'MPS_calls':0,'network_calls':0,'empirical_fits':0,'completed_utc':now()})


def queue_check():
    from joint_tuning_common import DATA
    import shutil
    from joint_training import SynchronizationError
    attempts=[];stages=[]
    with tempfile.TemporaryDirectory(prefix='currkt-joint-queue-') as tmp:
        folder=Path(tmp);shutil.copyfile(OUT/'design.json',folder/'design.json');(folder/'protocol.json').write_text('synthetic')
        def data(record,split):return {'uid':np.array([1,2]) if split=='train' else np.array([3,4])}
        def fit(dest,name,ds,tax,candidate,seed,*args):
            attempts.append((ds,tax,candidate['id'],seed));dest.mkdir(parents=True,exist_ok=True)
            if ds=='assist2012' and candidate['id']=='i17' and seed==42:raise RuntimeError('Synthetic numerical failure retained')
            ck=dest/'best.ckpt';ck.write_text('synthetic checkpoint')
            return {'state':'complete','checkpoint':identity(ck),'k':args[-2],'candidate':candidate,'seed':seed,
                    'best_validation_auc':.6+int(candidate['id'][1:])/1000,'best_validation_loss':.7,
                    'parameter_count':candidate['width']}
        def mapping(dest,name,record,backbone,ph):
            return {'selected_lambda':.1,'mapped_validation_auc':.6+backbone['k']/10000}
        def calibrate(dest,name,records,backbones,ph,lambdas):
            assert len(attempts)==120 and set(records)==set(DATASETS)
            assert all(set(values)=={'42','84','126'} for ds in backbones.values() for values in ds.values())
            stages.append('calibrate')
        def evaluate(*args):
            assert stages==['calibrate'];stages.append('test');return {'state':'complete'}
        replacements={'OUT':folder,'guard':lambda:{},'update_queue':lambda *a,**k:None,'status':lambda *a,**k:None,
                      'load_data':data,'fit':fit,'runtime_gate':lambda *a,**k:{'passed':True},'select_induced':mapping,
                      'calibrate':calibrate,'evaluate_comparison':evaluate}
        with ExitStack() as stack:
            for key,value in replacements.items():stack.enter_context(patch.object(runner,key,value))
            stack.enter_context(patch.object(torch.mps,'empty_cache',lambda:None))
            assert runner.execute_model('sakt',{})['state']=='complete'
        failure=folder/'models/sakt/assist2012/screening/i17/failure.json'
        assert failure.exists() and read(failure)['state']=='failed'
        for ds in DATASETS:
            selected=read(folder/f'models/sakt/{ds}/backbone_selection.json')
            assert selected['original']['candidate']['id']=='o17'
            assert selected['induced']['candidate']['k']==max(read(DATA/ds/'manifest.json')['taxonomies'][k]['k'] for k in read(DATA/ds/'manifest.json')['taxonomies'])
        assert stages==['calibrate','test']
    dump(OUT/'queue_cpu_checks.json',{'state':'passed','synthetic_backbone_attempts':120,'intentional_failure_retained':True,
        'three_dataset_order_and_seeds_passed':True,'native_original_and_mapped_induced_selection_passed':True,
        'MPS_calls':0,'network_calls':0,'empirical_fits':0,'completed_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--adapters',action='store_true');args=parser.parse_args()
    if args.adapters:adapters_check()
    else:orchestration_check();queue_check()
