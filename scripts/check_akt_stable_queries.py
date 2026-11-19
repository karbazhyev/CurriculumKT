"""Training-only reference gate for an inference arithmetic amendment.

CPU is a small reference diagnostic; --device mps is the required production
gate. No training, calibration/test access, tolerance changes or selection.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from akt_runtime import module, forward
from akt_selected_models import make_model, selected_models_ready
from evaluate_saved_curriculum_kt import ROOT, sha
from run_akt_seed_confirmation import OUT as SEED_ROOT
import run_akt_sweeps as base
from akt_stable_attention import stable_attention
from akt_stable_queries import prepare_history, candidate_logits
from akt_stable_features import extract_batch, mapping_layout


@torch.inference_mode()
def independent(model, c, r, rr, tt, kk, stable=False):
    captured = {}
    hook = model.out.register_forward_hook(lambda m, a, out: captured.__setitem__('z', out.detach()))
    result = []
    device = str(c.device)
    module('akt', device)
    try:
        for row, pos, child in zip(rr, tt, kk):
            cc = c[row:row+1].clone(); cc[0, pos] = child
            with ExitStack() as stack:
                if stable:
                    stack.enter_context(stable_attention(device))
                forward(model, cc, r[row:row+1])
            result.append(captured['z'][0, pos, 0].cpu())
    finally:
        hook.remove()
    return torch.stack(result)


def run(device):
    frozen_path=ROOT/'results/akt_curriculum_20261109/protocol.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        checked_path=ROOT/'results/model_sequence_20261107/akt_stable_queries_mps_checks.json'
        assert sha(checked_path)==frozen['selected_query_check_sha256']
        checked=json.loads(checked_path.read_text())
        for p,expected in checked['implementation_sha256'].items():assert sha(Path(p))==expected
        print('Frozen MPS verification preserved; no diagnostic overwritten.',flush=True)
        return
    protocol, selected = selected_models_ready()
    torch.set_num_threads(1)
    if device == 'mps':
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        assert torch.backends.mps.is_available()
    out = ROOT / f'results/model_sequence_20261107/akt_stable_queries_{device}_checks.json'
    assert not out.exists(), 'Preserve an existing diagnostic; inspect it before preparing a new attempt'
    proposal = base.read(ROOT/'results/model_sequence_20261107/akt_inference_amendment_01.json')
    assert proposal['training_protocol_sha256'] == sha(base.OUT/'protocol.json')
    for field in ['implementation_sha256', 'feature_preparation_implementation_sha256']:
        for path, expected in proposal[field].items():
            assert sha(Path(path)) == expected, f'Prepared inference source changed: {path}'
    for prefix in ['cpu_preparation_check', 'feature_preparation_check']:
        path = Path(proposal[prefix])
        assert sha(path) == proposal[prefix+'_sha256']
        assert base.read(path)['passed']
    report = {'state':'running','device':device,'cpu_fallback':False,'started_utc':base.now(),
        'empirical_fits':0,'heldout_queries':0,'calibration_queries':0,
        'reference':'Native pinned full-length200 single-target CPU float64 forward',
        'formula':'Exclusive reverse cumsum replaces total-minus-cumsum, avoiding cancellation before sqrt.',
        'tolerances':{'probability':2e-5,'logit':2e-4,'forbidden_logit':2e-5},
        'selected_models_sha256':sha(SEED_ROOT/'selected_models.json'),
        'implementation_sha256':{str(Path(__file__).with_name(p)):sha(Path(__file__).with_name(p))
            for p in ['check_akt_stable_queries.py','akt_stable_queries.py','akt_stable_attention.py',
                      'akt_stable_features.py','akt_selected_models.py','akt_runtime.py',
                      'simplekt_curriculum_features.py','check_attention_model_readiness.py']},
        'checks':[]}
    for ds in ['assist2017','assist2012']:
        record = protocol['datasets'][ds]['files']['train']
        assert sha(Path(record['path'])) == record['sha256']
        with np.load(record['path']) as d:
            ii = np.flatnonzero((d['original_concepts']>=0).sum(1)>=32)[:2]
            arrays = {k:d[k][ii] for k in d.files}
        mapping = protocol['datasets'][ds]['mapping']
        assert sha(Path(mapping['path'])) == mapping['sha256']
        with np.load(mapping['path']) as d: layout = mapping_layout(d)
        reference_meta = None
        for taxonomy in ['original','induced']:
            for seed in [42,84,126]:
                rec = selected['models'][f'{ds}_akt_{taxonomy}'][str(seed)]
                model = make_model(protocol,ds,taxonomy,rec,device)
                ref = make_model(protocol,ds,taxonomy,rec,'cpu').double()
                c = torch.tensor(np.maximum(arrays[taxonomy+'_concepts'],0),device=device)
                r = torch.tensor(np.maximum(arrays['responses'],0),device=device)
                rr,tt,kk = [],[],[]
                for row in range(2):
                    last = int((arrays[taxonomy+'_concepts'][row]>=0).sum())-1
                    for pos in sorted({1,7,31,last}):
                        for offset in [0,1,7]:
                            rr.append(row);tt.append(pos);kk.append((int(c[row,pos])+offset)%model.n_question)
                expected = independent(ref,c.cpu(),r.cpu(),rr,tt,kk)
                stable64 = independent(ref,c.cpu(),r.cpu(),rr,tt,kk,stable=True)
                native = independent(model,c,r,rr,tt,kk)
                full = independent(model,c,r,rr,tt,kk,stable=True)
                started = time.monotonic()
                with torch.inference_mode(), stable_attention(device):
                    cache=prepare_history(model,c,r)
                    actual=candidate_logits(model,cache,rr,tt,kk,31).cpu()
                    chunks=candidate_logits(model,cache,rr,tt,kk,7).cpu()
                    singleton=candidate_logits(model,cache,rr,tt,kk,1).cpu()
                    large=candidate_logits(model,cache,rr*11,tt*11,kk*11,256).cpu()
                    forbidden = 0.
                    for pos in sorted(set(tt)):
                        indices=[i for i,t in enumerate(tt) if t==pos]
                        a,b,e=[[v[i] for i in indices] for v in [rr,tt,kk]]
                        cc,rc=c.clone(),r.clone()
                        cc[:,pos:]=(cc[:,pos:]+5)%model.n_question
                        rc[:,pos:]=1-rc[:,pos:]
                        after=candidate_logits(model,prepare_history(model,cc,rc),a,b,e,31).cpu()
                        before=candidate_logits(model,cache,a,b,e,31).cpu()
                        forbidden=max(forbidden,float((after-before).abs().max()))
                result={'dataset':ds,'taxonomy':taxonomy,'seed':seed,'checkpoint_sha256':rec['checkpoint_sha256'],
                    'queries':len(rr),'native32_vs_native64_logit':float((native-expected).abs().max()),
                    'native32_vs_native64_probability':float((native.sigmoid()-expected.sigmoid()).abs().max()),
                    'stable64_vs_native64_logit':float((stable64-expected).abs().max()),
                    'stable32_vs_native64_logit':float((actual-expected).abs().max()),
                    'stable32_vs_native64_probability':float((actual.sigmoid()-expected.sigmoid()).abs().max()),
                    'adapter_vs_full_logit':float((actual-full).abs().max()),
                    'adapter_vs_full_probability':float((actual.sigmoid()-full.sigmoid()).abs().max()),
                    'chunk_logit':max(float((actual-chunks).abs().max()),float((actual-singleton).abs().max()),
                                      float((large-actual.repeat(11)).abs().max())),
                    'forbidden_information_logit':forbidden,'adapter_and_invariance_seconds':time.monotonic()-started}
                result['passed']=max(result['stable32_vs_native64_probability'],result['adapter_vs_full_probability'])<2e-5 and max(result['stable32_vs_native64_logit'],result['adapter_vs_full_logit'],result['chunk_logit'])<2e-4 and forbidden<2e-5 and result['stable64_vs_native64_logit']<2e-4
                report['checks'].append(result)
                if not result['passed']: report['state']='failed'
                out.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(result),flush=True)
                assert result['passed'], 'Stop on failed numerical reference gate'
                prefix={k:v.copy() for k,v in arrays.items()}
                for k in ['original_concepts','induced_concepts','responses']:prefix[k][:,5:]=-1
                meta,_,diagnostic=extract_batch(model,prefix,taxonomy,layout)
                if reference_meta is None:reference_meta=meta
                else:assert all(np.array_equal(meta[k],reference_meta[k]) for k in meta)
                result['feature_assembly']=diagnostic
                del model,ref
                if device=='mps': torch.mps.empty_cache()
    report.update(state='passed',status='passed',all_twelve_selected_checkpoints=True,completed_utc=base.now())
    selected_models_ready()
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],required=True)
    args=parser.parse_args()
    with ExitStack() as stack:
        if args.device=='mps':
            for path in [ROOT/'results/akt_sweeps_20261109/worker.lock',ROOT/'results/simplekt_sweeps_20261107/worker.lock',ROOT/'results/model_sequence_20261107/gpu_worker.lock']:
                lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.device)
