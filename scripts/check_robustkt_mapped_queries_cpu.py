"""CPU-only preparation for RobustKT map-back; no fitting or held-out loading."""
from pathlib import Path
import time

import numpy as np
import torch

import run_robustkt_sweeps as base
from robustkt_mps_portability import robustkt_class
from robustkt_cached_memory_queries import prepare_history, candidate_logits
from robustkt_target_queries import target_queries


def build(protocol, ds, taxonomy, candidate):
    torch.manual_seed(42)
    fixed=protocol['fixed']
    return robustkt_class('cpu')(n_question=protocol['datasets'][ds][taxonomy+'_concepts'],
        n_pid=0,d_model=candidate['d_model'],dropout=candidate['dropout'],n_blocks=fixed['n_blocks'],
        d_ff=fixed['d_ff'],num_attn_heads=fixed['num_attn_heads'],final_fc_dim=fixed['final_fc_dim'],
        emb_type='qid',ks=fixed['ks'],kq_same=fixed['kq_same'],separate_qa=fixed['separate_qa'],l2=fixed['l2']).eval()


def check(protocol, ds, taxonomy, candidate, checkpoint=None):
    model=build(protocol,ds,taxonomy,candidate)
    checkpoint_hash=None
    if checkpoint is not None:
        checkpoint_hash=base.sha(checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    data=base.data(protocol,ds,taxonomy,'train')
    indices=np.flatnonzero(data[2].sum(1)>=64)[:2]
    c,r=[torch.tensor(a[indices]) for a in data[:2]]
    rows=torch.tensor([row for row in range(2) for t in [0,4,42,199] for k in range(3)])
    positions=torch.tensor([t for row in range(2) for t in [0,4,42,199] for k in range(3)])
    candidates=(c[rows,positions]+torch.tensor([0,1,7]*8))%model.n_question
    start=time.monotonic();history=prepare_history(model,c,r)
    z=candidate_logits(model,history,rows,positions,candidates)
    batched_seconds=time.monotonic()-start
    direct,direct_z=target_queries(model,c,r,rows,positions,candidates,batch_size=1)
    delta=float((torch.sigmoid(z)-direct).abs().max())
    logit_delta=float((z-direct_z).abs().max())
    chunks=candidate_logits(model,history,rows,positions,candidates,batch_size=7)
    chunk_delta=float((z-chunks).abs().max())
    with torch.no_grad():native=model(c,r)[0]
    selected=c[rows,positions]==candidates
    native_delta=float((torch.sigmoid(z[selected])-native[rows[selected],positions[selected]]).abs().max())
    # Perturb actual current concept, current/future responses and future concepts.
    # Candidate concepts stay fixed; all queried past positions are unchanged.
    altered_c,altered_r=c.clone(),r.clone()
    altered_c[:,42:]=(altered_c[:,42:]+3)%model.n_question
    altered_r[:,42:]=1-altered_r[:,42:]
    altered=prepare_history(model,altered_c,altered_r)
    mask=positions==42
    leaked=candidate_logits(model,altered,rows[mask],positions[mask],candidates[mask])
    information_delta=float((leaked-z[mask]).abs().max())
    result={'dataset':ds,'taxonomy':taxonomy,'candidate':candidate,'checkpoint_sha256':checkpoint_hash,
        'reference':'Independent full length200 forward for each single-target replacement',
        'queries':len(rows),'probability_max_difference':delta,'raw_logit_max_difference':logit_delta,
        'native_query_probability_difference':native_delta,'chunk_size_logit_difference':chunk_delta,
        'forbidden_information_logit_difference':information_delta,'cache_and_query_seconds_cpu':batched_seconds,
        'passed':max(delta,native_delta)<2e-5 and max(logit_delta,chunk_delta,information_delta)<2e-4}
    assert all(parameter.device.type=='cpu' for parameter in model.parameters())
    return result


def main():
    torch.set_num_threads(1)
    protocol=base.read(base.OUT/'protocol.json');base.verify(protocol)
    registry=base.read(base.OUT/'run_registry.json')
    jobs=[(ds,tax,{'d_model':width,'dropout':.1},None) for ds in ['assist2017','assist2012']
          for tax in ['original','induced'] for width in [64,128]]
    for key,entry in registry['trials'].items():
        if entry['state']=='complete':
            checkpoint=base.OUT/'trials'/key/'best.ckpt'
            assert base.sha(checkpoint)==entry['checkpoint_sha256']
            jobs.append((entry['dataset'],entry['taxonomy'],entry['candidate'],checkpoint))
    record={'started_utc':base.now(),'device':'cpu','test_used':False,'calibration_used':False,
        'training_performed':False,'active_gpu_worker_untouched':True,'checks':[],
        'adapter':'cached response memory with native question attention and fixed query microbatch1',
        'source_sha256':{str(p):base.sha(p) for p in [Path(__file__),Path(__file__).with_name('robustkt_cached_memory_queries.py')]}}
    destination=base.ROOT/'results/model_sequence_20261107/robustkt_cached_memory_cpu_checks.json'
    for job in jobs:
        result=check(protocol,*job);record['checks'].append(result)
        base.dump(destination,record)
        print(result,flush=True)
    record.update(completed_utc=base.now(),passed=all(r['passed'] for r in record['checks']),
                  remaining_gate='Actual-MPS checks on all12 development-selected checkpoints after seed confirmation.')
    base.dump(destination,record);assert record['passed']


if __name__=='__main__':
    main()
