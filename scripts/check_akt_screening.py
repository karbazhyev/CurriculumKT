"""Actual-MPS training-only correctness gates; no sweep or test evaluation."""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

import run_akt_sweeps as run
from akt_runtime import module, forward
from akt_target_queries import target_queries


def check(protocol, dataset, taxonomy, width, dropout):
    entry={'dataset':dataset,'taxonomy':taxonomy,'seed':42,
           'candidate':{'d_model':width,'dropout':dropout,'learning_rate':.001}}
    values=run.data(protocol,dataset,taxonomy,'train')
    valid_rows=np.flatnonzero(values[2].sum(1)>=64)[:2]
    assert len(valid_rows)==2
    c,r,mask=run.tensors(values,valid_rows)
    model=run.make_model(protocol,entry)
    model.eval()
    nc=model.n_question
    with torch.no_grad():
        p,reg=model(c,r)
        assert reg==0. and p.device.type=='mps'
        altered=r.clone();altered[:,42:]=1-altered[:,42:]
        response_delta=float((forward(model,c,altered)[:,:43]-p[:,:43]).abs().max())
        altered=c.clone();altered[:,43:]=(altered[:,43:]+1)%nc
        future_delta=float((forward(model,altered,r)[:,:43]-p[:,:43]).abs().max())
        prefix_delta=float((forward(model,c[:,:43],r[:,:43])-p[:,:43]).abs().max())
        # These are independent branches, including target-concept replacements.
        rows=torch.tensor([row for row in range(2) for t in [12,42] for k in range(3)],device='mps')
        positions=torch.tensor([t for row in range(2) for t in [12,42] for k in range(3)],device='mps')
        candidates=(c[rows,positions]+torch.tensor([0,1,7]*4,device='mps'))%nc
        batched,logits=target_queries(model,c,r,rows,positions,candidates)
        reference=[];reference_logits=[]
        for row,t,k in zip(rows.tolist(),positions.tolist(),candidates.tolist()):
            cc=c[row:row+1].clone();cc[0,t]=k
            pp,rr,hidden=model(cc,r[row:row+1],qtest=True)
            reference.append(pp[0,t]);reference_logits.append(model.out(hidden).squeeze(-1)[0,t])
        query_delta=float((batched-torch.stack(reference)).abs().max())
        query_logit_delta=float((logits-torch.stack(reference_logits)).abs().max())
        native=(c[rows,positions]==candidates)
        native_delta=float((batched[native]-p[rows[native],positions[native]]).abs().max())
    cpu=module('akt','cpu').AKT(n_question=nc,n_pid=0,d_model=width,n_blocks=1,
        dropout=dropout,d_ff=256,num_attn_heads=4,kq_same=1,separate_qa=False,l2=1e-5,emb_type='qid')
    cpu.load_state_dict(run.cpu_state(model));cpu.eval()
    with torch.no_grad():
        cpu_prediction=forward(cpu,c.cpu(),r.cpu())
        cpu_prefix=forward(cpu,c.cpu()[:,:43],r.cpu()[:,:43])
    cpu_delta=float((p.cpu()-cpu_prediction).abs().max())
    module('akt','mps')
    with torch.no_grad():
        prefix_cpu_delta=float((forward(model,c[:,:43],r[:,:43]).cpu()-cpu_prefix).abs().max())
        repeated_prefix_delta=max(float((forward(model,c[:,:43],r[:,:43])-p[:,:43]).abs().max()) for _ in range(8))
        native_model=module('akt','mps').AKT(n_question=nc,n_pid=0,d_model=width,n_blocks=1,
            dropout=dropout,d_ff=256,num_attn_heads=4,kq_same=1,separate_qa=False,l2=1e-5,emb_type='qid').to('mps')
        native_model.load_state_dict(model.state_dict());native_model.eval()
        native_200_delta=float((forward(native_model,c,r)-p).abs().max())
        del native_model
    # Compare numerical BCE and dL/dp to the upstream double-precision formula.
    pp=p[:,1:][mask].detach().clone().requires_grad_(True);yy=r[:,1:][mask].float()
    loss32=F.binary_cross_entropy(pp,yy);grad32=torch.autograd.grad(loss32,pp)[0].cpu()
    pp64=pp.detach().cpu().double().requires_grad_(True)
    loss64=F.binary_cross_entropy(pp64,yy.cpu().double());grad64=torch.autograd.grad(loss64,pp64)[0]
    loss_delta=abs(float(loss32.detach())-float(loss64.detach()))
    loss_gradient_delta=float((grad32.double()-grad64).abs().max())
    model.train();opt=torch.optim.Adam(model.parameters(),lr=.001)
    started=time.monotonic();prediction,regularizer=model(c,r)
    loss=F.binary_cross_entropy(prediction[:,1:][mask],yy)+regularizer
    assert regularizer==0. and bool(loss.isfinite())
    loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.device.type=='mps' and bool(g.isfinite().all()) for g in grads)
    opt.step();torch.mps.synchronize();seconds=time.monotonic()-started
    folder=run.OUT/'runtime_gate';folder.mkdir(exist_ok=True)
    checkpoint=folder/f'{dataset}_{taxonomy}_width{width}_dropout{dropout}.ckpt'
    run.save(checkpoint,run.cpu_state(model));model.eval()
    with torch.no_grad():expected=forward(model,c,r)
    restored=run.make_model(protocol,entry);restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));restored.eval()
    with torch.no_grad():reload_delta=float((expected-forward(restored,c,r)).abs().max())
    result={'dataset':dataset,'taxonomy':taxonomy,'width':width,'dropout':dropout,'device':'mps',
        'cpu_fallback':False,'targets':int(mask.sum()),'source_sha256':run.sha(run.SOURCE/'pykt/models/akt.py'),
        'input_sha256':protocol['datasets'][dataset]['files']['train']['sha256'],
        'current_future_response_delta':response_delta,'future_concept_delta':future_delta,
        'prefix_delta':prefix_delta,'cpu_reference_delta':cpu_delta,'checkpoint_reload_delta':reload_delta,
        'short_prefix_cpu_reference_delta':prefix_cpu_delta,'eight_repeated_prefix_max_delta':repeated_prefix_delta,
        'unchanged_native_200_forward_delta':native_200_delta,
        'independent_queries':len(rows),'query_probability_delta':query_delta,'query_logit_delta':query_logit_delta,
        'native_query_delta':native_delta,'bce_float32_vs_float64_delta':loss_delta,
        'bce_probability_gradient_max_delta':loss_gradient_delta,'returned_regularizer':float(regularizer),
        'loss':float(loss.detach()),'forward_backward_update_seconds':seconds,
        'all_model_gradients_on_mps':True,'checkpoint_sha256':run.sha(checkpoint),
        'passed':max(response_delta,future_delta,prefix_delta,cpu_delta,reload_delta,query_delta,native_delta)<2e-5
            and max(prefix_cpu_delta,repeated_prefix_delta)<2e-5 and native_200_delta==0.
            and query_logit_delta<2e-4 and loss_delta<5e-6 and loss_gradient_delta<1e-6}
    del model,cpu,restored,opt
    torch.mps.empty_cache()
    return result


def main():
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert torch.backends.mps.is_available()
    torch.set_num_threads(1)
    protocol=run.read(run.OUT/'protocol_draft.json')
    assert not (run.OUT/'protocol.json').exists()
    for path,expected in protocol['source_sha256'].items():assert run.sha(Path(path))==expected
    record={'started_utc':run.now(),'python':platform.python_version(),'torch':torch.__version__,
        'device_probe':str(torch.ones(1,device='mps').device),'cpu_fallback':False,
        'protocol_draft_sha256':run.sha(run.OUT/'protocol_draft.json'),
        'test_used':False,'calibration_used':False,'parameter_selection':False,'checks':[]}
    for ds in ['assist2017','assist2012']:
        for taxonomy in ['original','induced']:
            for width in [64,128]:
                for dropout in [.1,.5]:
                    result=check(protocol,ds,taxonomy,width,dropout)
                    record['checks'].append(result)
                    run.dump(run.OUT/'runtime_checks.json',record)
                    print(result,flush=True)
    record.update(completed_utc=run.now(),passed=all(c['passed'] for c in record['checks']))
    run.dump(run.OUT/'runtime_checks.json',record)
    assert record['passed'],'Review preserved runtime check failures'


if __name__=='__main__':
    with ExitStack() as stack:
        for path in [run.ROOT/'results/model_sequence_20261107/gpu_worker.lock',
                     run.ROOT/'results/simplekt_sweeps_20261107/worker.lock']:
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
