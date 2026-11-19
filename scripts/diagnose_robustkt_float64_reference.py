"""Verify every selected RobustKT checkpoint on training data, with exclusive MPS."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from check_attention_model_readiness import module, forward
from evaluate_saved_curriculum_kt import ROOT, SOURCE, sha
from run_robustkt_seed_confirmation import OUT as SEED_ROOT, verify_stage
from robustkt_cached_memory_queries import prepare_history as history_cache, candidate_logits as query_logits
from robustkt_mps_portability import robustkt_class
from robustkt_curriculum_features import extract_batch, mapping_layout

OUT = ROOT / 'results/model_sequence_20261107/robustkt_selected_query_gate/float64_reference_diagnostic.json'


def make_model(protocol, dataset, taxonomy, record, device):
    cfg, fixed = record['candidate'], protocol['fixed']
    cls = robustkt_class(device)
    model = cls(n_question=protocol['datasets'][dataset][taxonomy + '_concepts'],
        n_pid=0, d_model=cfg['d_model'], n_blocks=fixed['n_blocks'],
        dropout=cfg['dropout'], d_ff=fixed['d_ff'], num_attn_heads=fixed['num_attn_heads'],
        final_fc_dim=fixed['final_fc_dim'], ks=fixed['ks'], kq_same=fixed['kq_same'],
        separate_qa=fixed['separate_qa'], l2=fixed['l2'], emb_type='qid').to(device)
    assert sha(Path(record['checkpoint'])) == record['checkpoint_sha256']
    model.load_state_dict(torch.load(record['checkpoint'], map_location='cpu', weights_only=True))
    return model.eval()


def run():
    frozen_path=ROOT/'results/robustkt_curriculum_20261107/protocol.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        assert sha(OUT)==frozen['selected_query_check_sha256']
        checked=json.loads(OUT.read_text())
        for path,expected in checked['implementation_sha256'].items():assert sha(Path(path))==expected
        print('Existing frozen selected-checkpoint query verification remains valid; no files overwritten.',flush=True)
        return
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.backends.mps.is_available(), 'Actual MPS is required'
    torch.set_num_threads(1)
    protocol = verify_stage()
    selected = json.loads((SEED_ROOT / 'selected_models.json').read_text())
    assert selected['all_twelve_available']
    assert selected['protocol_sha256'] == sha(SEED_ROOT / 'protocol.json')
    assert json.loads((SEED_ROOT / 'run_registry.json').read_text())['state'] == 'complete'
    rows_out = []
    attempt_path=OUT.with_name(OUT.stem+'_attempt.json')
    for dataset in ['assist2017', 'assist2012']:
        ds = protocol['datasets'][dataset]
        data_path, mapping_path = Path(ds['files']['train']['path']), Path(ds['mapping']['path'])
        assert sha(data_path) == ds['files']['train']['sha256']
        assert sha(mapping_path) == ds['mapping']['sha256']
        with np.load(data_path) as d:
            long_rows=np.flatnonzero((d['original_concepts']>=0).sum(1)>=32)[:2]
            assert len(long_rows)==2
            arrays = {k: d[k][long_rows] for k in d.files}
        with np.load(mapping_path) as d:
            layout = mapping_layout(d)
        reference_metadata = None
        for taxonomy in ['original', 'induced']:
            cell = f'{dataset}_robustkt_{taxonomy}'
            for seed in [42, 84, 126]:
                record = selected['models'][cell][str(seed)]
                mps = make_model(protocol, dataset, taxonomy, record, 'mps')
                cpu = make_model(protocol, dataset, taxonomy, record, 'cpu').double()
                # Read-only import helper has a per-module device global.
                # Re-select it immediately before each ordinary forward call.
                raw_c = arrays[taxonomy + '_concepts']
                c_cpu = torch.tensor(np.maximum(raw_c, 0))
                r_cpu = torch.tensor(np.maximum(arrays['responses'], 0))
                c, r = c_cpu.to('mps'), r_cpu.to('mps')
                rr, tt, kk = [], [], []
                for row in range(len(c)):
                    last = int((raw_c[row] >= 0).sum()) - 1
                    for pos in sorted({1, min(7,last), min(31,last), last}):
                        if pos < 1 or pos > last:
                            continue
                        for offset in [0, 1, 7]:
                            rr.append(row); tt.append(pos)
                            kk.append((int(c_cpu[row, pos]) + offset) % mps.n_question)
                rows, positions, candidates = (torch.tensor(x, device='mps') for x in [rr, tt, kk])
                module('robustkt','mps')
                actual_logits = query_logits(mps, history_cache(mps, c, r), rows, positions, candidates, 7)
                actual = actual_logits.sigmoid()
                expected_mps, expected_cpu = [], []
                expected_mps_logits, expected_cpu_logits = [], []
                captured={}
                hm=mps.out.register_forward_hook(lambda _m,_a,output:captured.__setitem__('mps',output.detach()))
                hc=cpu.out.register_forward_hook(lambda _m,_a,output:captured.__setitem__('cpu',output.detach()))
                with torch.inference_mode():
                    for row, pos, child in zip(rr, tt, kk):
                        cm = c[row:row+1].clone(); cm[0,pos] = child
                        cp = c_cpu[row:row+1].clone(); cp[0,pos] = child
                        module('robustkt', 'mps')
                        expected_mps.append(forward(mps, cm, r[row:row+1])[0,pos])
                        expected_mps_logits.append(captured['mps'][0,pos,0])
                        module('robustkt', 'cpu')
                        expected_cpu.append(forward(cpu, cp, r_cpu[row:row+1])[0,pos])
                        expected_cpu_logits.append(captured['cpu'][0,pos,0])
                hm.remove();hc.remove()
                module('robustkt', 'mps')
                delta_mps = float((actual - torch.stack(expected_mps)).abs().max())
                delta_cpu = float((actual.cpu() - torch.stack(expected_cpu)).abs().max())
                delta_mps_logit=float((actual_logits-torch.stack(expected_mps_logits)).abs().max())
                delta_cpu_logit=float((actual_logits.cpu()-torch.stack(expected_cpu_logits)).abs().max())
                forbidden_delta = 0.
                for row, pos in sorted(set(zip(rr,tt))):
                    rows_t = torch.tensor([row], device='mps')
                    pos_t = torch.tensor([pos], device='mps')
                    child_t = torch.tensor([3 % mps.n_question], device='mps')
                    before = query_logits(mps, history_cache(mps,c,r), rows_t, pos_t, child_t)
                    cc, rc = c.clone(), r.clone()
                    cc[:,pos:] = (cc[:,pos:] + 5) % mps.n_question
                    rc[:,pos:] = 1 - rc[:,pos:]
                    after = query_logits(mps, history_cache(mps,cc,rc), rows_t,pos_t,child_t)
                    forbidden_delta = max(forbidden_delta, float((before-after).abs().max()))
                diagnostic={'dataset':dataset,'taxonomy':taxonomy,'seed':seed,'delta_mps':delta_mps,'delta_cpu':delta_cpu,'delta_mps_logit':delta_mps_logit,'delta_cpu_logit':delta_cpu_logit,'forbidden_delta':forbidden_delta}
                print(json.dumps(diagnostic),flush=True)
                (ROOT/'results/model_sequence_20261107/robustkt_selected_query_gate/float64_first_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                assert max(delta_mps, delta_cpu, forbidden_delta) < 2e-5
                assert max(delta_mps_logit,delta_cpu_logit)<2e-4
                feature_arrays={k:v.copy() for k,v in arrays.items()}
                for key in ['original_concepts','induced_concepts','responses']:
                    feature_arrays[key][:,5:]=-1
                meta, _, feature_diagnostic = extract_batch(mps, feature_arrays, taxonomy, layout, query_batch_size=31)
                if reference_metadata is None:
                    reference_metadata = meta
                else:
                    assert all(np.array_equal(meta[k], reference_metadata[k]) for k in meta)
                rows_out.append({'dataset':dataset, 'taxonomy':taxonomy, 'seed':seed,
                    'checkpoint_sha256': record['checkpoint_sha256'], 'queries':len(rr),
                    'mps_independent_forward_max_probability_difference':delta_mps,
                    'cpu_independent_forward_max_probability_difference':delta_cpu,
                    'mps_independent_forward_max_logit_difference':delta_mps_logit,
                    'cpu_independent_forward_max_logit_difference':delta_cpu_logit,
                    'current_future_information_max_logit_difference':forbidden_delta,
                    'feature_assembly':feature_diagnostic, 'passed':True})
                attempt_path.write_text(json.dumps({'status':'running','checks':rows_out},indent=2)+'\n')
                print(json.dumps(rows_out[-1]), flush=True)
                del mps, cpu
                torch.mps.empty_cache()
    assert len(rows_out) == 12
    scripts = ['check_robustkt_selected_queries.py', 'robustkt_cached_memory_queries.py', 'robustkt_mps_portability.py',
               'simplekt_curriculum_features.py',
               'robustkt_curriculum_features.py', 'check_attention_model_readiness.py']
    report = {'status':'passed', 'all_twelve_selected_checkpoints':True, 'checks':rows_out,
        'selected_models_sha256':sha(SEED_ROOT/'selected_models.json'),
        'seed_protocol_sha256':sha(SEED_ROOT/'protocol.json'),
        'implementation_sha256':{str(ROOT/'scripts'/s):sha(ROOT/'scripts'/s) for s in scripts},
        'source_sha256':sha(SOURCE/'pykt/models/robustkt.py'),
        'tolerances':{'probability':2e-5,'logit':2e-4,'forbidden_information_logit':2e-5},
        'empirical_fits':0, 'heldout_queries':0, 'device':'mps', 'cpu_fallback':False,
        'candidate_numerical_microbatch_size':1,'feature_checks_training_prefix_targets_per_row':4}
    temporary = OUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n');temporary.replace(OUT)


if __name__ == '__main__':
    with ExitStack() as stack:
        for path in [ROOT/'results/robustkt_sweeps_20261107/worker.lock',
                     ROOT/'results/simplekt_sweeps_20261107/worker.lock',
                     ROOT/'results/model_sequence_20261107/gpu_worker.lock']:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
