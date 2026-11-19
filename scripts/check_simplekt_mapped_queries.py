"""CPU inference reference checks using completed training checkpoints only."""
import json
from pathlib import Path

import numpy as np
import torch

from check_attention_model_readiness import module, forward
from evaluate_saved_curriculum_kt import ROOT, SOURCE, sha
from simplekt_mapped_queries import history_cache, query_logits


def main():
    torch.set_num_threads(1)
    root=ROOT/'results/simplekt_sweeps_20261107'
    out=ROOT/'results/model_sequence_20261107'
    registry=json.loads((root/'run_registry.json').read_text())
    d=np.load(ROOT/'results/curriculum_kt_all_methods_20261107/assist2017/train_sequences.npz')
    results=[]
    for taxonomy in ['original','induced']:
        for cfg in ['c00','c04']:
            key=f'assist2017_simplekt_{taxonomy}_{cfg}_seed42'
            entry=registry['trials'][key]
            assert entry['state']=='complete'
            width=entry['candidate']['d_model'];nc=68 if taxonomy=='original' else 100
            m=module('simplekt','cpu').simpleKT(n_question=nc,n_pid=0,d_model=width,n_blocks=1,
                dropout=.1,d_ff=256,num_attn_heads=4,emb_type='qid',seq_len=200)
            checkpoint=root/'trials'/key/'best.ckpt'
            assert sha(checkpoint)==entry['checkpoint_sha256']
            m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));m.eval()
            raw_c=d[taxonomy+'_concepts'][:3]
            c=torch.tensor(np.maximum(raw_c,0));r=torch.tensor(np.maximum(d['responses'][:3],0))
            rows=[];positions=[];candidates=[]
            for row in range(3):
                last=int((raw_c[row]>=0).sum())-1
                for position in sorted({1,min(7,last),min(31,last),min(71,last),last}):
                    if position<1:continue
                    for offset in [0,1,7]:
                        rows.append(row);positions.append(position);candidates.append((int(c[row,position])+offset)%nc)
            rows=torch.tensor(rows);positions=torch.tensor(positions);candidates=torch.tensor(candidates)
            cache=history_cache(m,c,r)
            actual=query_logits(m,cache,rows,positions,candidates,query_batch_size=7).sigmoid()
            expected=[]
            with torch.inference_mode():
                for row,position,k in zip(rows,positions,candidates):
                    changed=c[row:row+1].clone();changed[0,position]=k
                    expected.append(forward(m,changed,r[row:row+1])[0,position])
            expected=torch.stack(expected)
            delta=float((actual-expected).abs().max())
            rr=torch.tensor([0]);tt=torch.tensor([1]);kk=torch.tensor([3])
            before=query_logits(m,cache,rr,tt,kk)
            changed_c=c.clone();changed_c[:,1:]=(changed_c[:,1:]+5)%nc
            changed_r=r.clone();changed_r[:,1:]=1-changed_r[:,1:]
            after=query_logits(m,history_cache(m,changed_c,changed_r),rr,tt,kk)
            forbidden_delta=float((before-after).abs().max())
            assert delta<2e-5 and forbidden_delta==0
            results.append({'trial':key,'device':'cpu_inference_reference_only','queries':len(rows),
                'one_target_at_a_time_max_probability_difference':delta,
                'actual_target_and_future_information_max_logit_difference':forbidden_delta,
                'checkpoint_sha256':sha(checkpoint),'passed':True})
    report={'status':'CPU_reference_passed_MPS_pending_until_GPU_free',
        'scope':'Supported n_pid0,one-block,emb_type qid SimpleKT; each child queried independently with unchanged observed past.',
        'checks':results,'empirical_fits':0,'heldout_queries':0,
        'adapter_sha256':sha(Path(__file__).with_name('simplekt_mapped_queries.py')),
        'simplekt_source_sha256':sha(SOURCE/'pykt/models/simplekt.py')}
    (out/'simplekt_query_adapter_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
