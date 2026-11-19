"""Bounded synthetic KT benchmark; no dataset upload, tuning or checkpoint output.

Run only in an explicitly authorized CUDA session. Source payload is inserted by
prepare_colab_kt_microbenchmark_20261112.py. CPU verification uses two synthetic
sequences; the timed workload must use CUDA, never CPU fallback.
"""
import base64
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import types
from pathlib import Path

PAYLOAD = None  # inserted in the generated standalone file, not this template

def run(payload):
    import numpy as np
    import torch
    started=time.monotonic()
    assert torch.cuda.is_available(),'No CUDA GPU; stop without CPU benchmark fallback.'
    hardware=torch.cuda.get_device_name(0)
    assert any(x in hardware.upper() for x in ('T4','L4','A100')),'Unexpected accelerator; stop.'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source={}
    for name,entry in payload['source'].items():
        code=base64.b64decode(entry['base64']).decode()
        assert hashlib.sha256(code.encode()).hexdigest()==entry['sha256']
        source[name]=code
    package='_currkt_benchmark_pinned'
    pkg=types.ModuleType(package);pkg.__path__=[];sys.modules[package]=pkg
    def module(name,device):
        full=package+'.'+name
        if full not in sys.modules:
            mod=types.ModuleType(full);mod.__package__=package;sys.modules[full]=mod
            exec(compile(source[name],'<pinned-'+name+'>','exec'),mod.__dict__)
        mod=sys.modules[full];mod.device=torch.device(device);return mod
    namespace={'module':module,'__name__':'_currkt_pinned_adapter'}
    original='from check_attention_model_readiness import module'
    assert source['adapter'].count(original)==1
    exec(compile(source['adapter'].replace(original,'# Isolated, hash-checked loader supplied above.'),'<pinned-adapter>','exec'),namespace)
    make,native,lossfn,query,gate=(namespace[n] for n in ('make','native','training_loss','query','query_gate'))
    def sync():torch.cuda.synchronize()
    def elapsed(fn,reps=7):
        fn();fn();sync();times=[]
        for _ in range(reps):
            if time.monotonic()-started>240:raise TimeoutError('Four-minute benchmark time budget exceeded')
            sync();t=time.perf_counter();fn();sync();times.append(time.perf_counter()-t)
        return {'median_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),'repetitions':reps}
    reports=[]
    for name in ('dkvmn','sakt'):
        for width in (64,128):
            k=1000;cfg={'width':width,'dropout':0.3,'learning_rate':0.0003}
            # Independent synthetic identities, never released student data.
            rng=np.random.default_rng(42);cc=rng.integers(0,k,(64,200),dtype=np.int64);rr=rng.integers(0,2,(64,200),dtype=np.int64)
            lengths=rng.choice(payload['lengths']['values'],64,p=payload['lengths']['probabilities'])
            for i,length in enumerate(lengths):cc[i,length:]=-1;rr[i,length:]=-1
            # Two full synthetic histories make correctness gates independent of tail lengths.
            cc[:2]=rng.integers(0,k,(2,200));rr[:2]=rng.integers(0,2,(2,200))
            model=make(name,k,cfg,'cuda',42);c=torch.as_tensor(cc,device='cuda');r=torch.as_tensor(rr,device='cuda')
            before=torch.get_rng_state().clone()
            reference=make(name,k,cfg,'cpu',42);reference.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
            reference.eval();model.eval()
            with torch.inference_mode():
                expected=native(reference,c[:2].cpu(),r[:2].cpu()).numpy()
                actual=native(model,c[:2],r[:2]).cpu().numpy()
            delta=float(np.max(np.abs(expected-actual)));assert delta<2e-5,delta
            check=gate(model,c[:2],r[:2]);del reference
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
            def train_step():
                model.train();optimizer.zero_grad(set_to_none=True);loss=lossfn(model,c,r)
                assert bool(loss.isfinite());loss.backward()
                assert all(p.grad.device.type=='cuda' and bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
                optimizer.step();float(loss.detach())
            training=elapsed(train_step)
            @torch.inference_mode()
            def evaluate():
                model.eval();p=native(model,c,r);assert bool(p.isfinite().all());p.cpu().numpy()
            forward=elapsed(evaluate)
            rows=np.arange(32)%2;positions=1+np.arange(32)%150;children=(cc[rows,positions]+3)%k
            @torch.inference_mode()
            def mapped():
                model.eval();query(model,c[:2],r[:2],rows,positions,children,32).cpu().numpy()
            mapped_time=elapsed(mapped,5)
            proxy=payload['mps_reference']['training_batches']*training['median_seconds']+payload['mps_reference']['evaluation_batches']*forward['median_seconds']
            report={'model':name,'width':width,'k':k,'batch_size':64,'sequence_length':200,'dtype':'float32','tf32':False,
                'training_step':training,'evaluation_batch':forward,'query_32_targets':mapped_time,'cpu_reference_max_probability_delta':delta,
                'counterfactual_gate':check,'synthetic_targets_per_batch':int(((c[:,:-1]>=0)&(c[:,1:]>=0)).sum()),
                'estimated_compute_only_epoch_seconds':proxy,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'not_a_matched_mps_benchmark':True}
            if name=='dkvmn' and width==128:
                report['historical_mps_epoch_seconds']=payload['mps_reference']['median_epoch_seconds']
                report['historical_walltime_over_cuda_compute_proxy_ratio']=report['historical_mps_epoch_seconds']/proxy
            reports.append(report);print('KT_BENCH_CASE '+json.dumps(report),flush=True)
            del optimizer,model,c,r;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    result={'state':'complete','hardware':hardware,'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
        'elapsed_seconds':time.monotonic()-started,'source_sha256':{k:v['sha256'] for k,v in payload['source'].items()},
        'data':'Synthetic random concepts/responses, aggregate training length histogram only','test_data_used':False,'model_results':reports,
        'limitations':['No simultaneous local MPS benchmark while existing worker is active.','Historical MPS reference includes real-data metrics and other epoch overhead; CUDA synthetic proxy is not an end-to-end speedup measurement.','No accuracy or convergence conclusions from synthetic inputs.','Runtime/session startup and CU billing measured separately by operator.']}
    Path('/content/currkt_microbenchmark_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('KT_BENCH_RESULT '+json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    assert PAYLOAD is not None,'Use the prepared standalone benchmark or notebook.'
    run(PAYLOAD)
