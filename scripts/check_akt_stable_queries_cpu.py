"""CPU preparation of inference arithmetic; selected-model MPS gate still required."""
from contextlib import nullcontext
from pathlib import Path
import json

import numpy as np
import torch

import run_akt_sweeps as base
from akt_runtime import akt_class
from akt_stable_attention import stable_attention
from akt_stable_queries import prepare_history, candidate_logits

OUT = base.ROOT / 'results/model_sequence_20261107/akt_stable_queries_cpu_preparation.json'


@torch.inference_mode()
def independent(model, c, r, rows, positions, candidates, amended=False):
    result = []
    with stable_attention('cpu') if amended else nullcontext():
        for row, pos, candidate in zip(rows, positions, candidates):
            branch = c[row:row+1].clone()
            branch[0, pos] = candidate
            _, reg, hidden = model(branch, r[row:row+1], qtest=True)
            assert reg == 0.
            result.append(model.out(hidden).squeeze(-1)[0, pos])
    return torch.stack(result)


def check(protocol, ds, taxonomy, width, checkpoint=None):
    torch.manual_seed(23)
    nc = protocol['datasets'][ds][taxonomy+'_concepts']
    args = dict(n_question=nc, n_pid=0, d_model=width, n_blocks=1, dropout=.1,
                d_ff=256, num_attn_heads=4, final_fc_dim=512, kq_same=1,
                separate_qa=False, l2=1e-5, emb_type='qid')
    model = akt_class('cpu')(**args).eval()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    reference = akt_class('cpu')(**args).double().eval()
    reference.load_state_dict(model.state_dict())
    assert all(p.device.type == 'cpu' for m in [model, reference] for p in m.parameters())
    data = protocol['datasets'][ds]['files']['train']
    assert base.sha(Path(data['path'])) == data['sha256']
    with np.load(data['path']) as source:
        concepts = source[taxonomy+'_concepts']
        selected = np.flatnonzero((concepts >= 0).sum(1) >= 64)[:2]
        c = torch.tensor(np.maximum(concepts[selected], 0), dtype=torch.long)
        r = torch.tensor(np.maximum(source['responses'][selected], 0), dtype=torch.long)
    rows = [row for row in range(2) for _ in range(15)]
    positions = [pos for _ in range(2) for pos in [0, 1, 12, 42, 199] for _ in range(3)]
    candidates = [(int(c[row, pos])+offset) % nc
                  for row in range(2) for pos in [0, 1, 12, 42, 199] for offset in [0, 1, 7]]
    expected = independent(reference, c, r, rows, positions, candidates)
    amended64 = independent(reference, c, r, rows, positions, candidates, True)
    native32 = independent(model, c, r, rows, positions, candidates)
    amended32 = independent(model, c, r, rows, positions, candidates, True)
    with torch.inference_mode(), stable_attention('cpu'):
        cache = prepare_history(model, c, r)
        actual = candidate_logits(model, cache, rows, positions, candidates, 31)
        chunk_delta = max(float((actual-candidate_logits(model, cache, rows, positions,
                          candidates, size)).abs().max()) for size in [1, 7, 256])
        large = candidate_logits(model, cache, rows*9, positions*9, candidates*9, 256)
        chunk_delta = max(chunk_delta, float((large-actual.repeat(9)).abs().max()))
        forbidden = 0.
        for pos in sorted(set(positions)):
            indexes = [i for i, value in enumerate(positions) if value == pos]
            rr, tt, kk = [[values[i] for i in indexes] for values in [rows, positions, candidates]]
            cc, rc = c.clone(), r.clone()
            cc[:, pos:] = (cc[:, pos:]+13) % nc
            rc[:, pos:] = 1-rc[:, pos:]
            after = candidate_logits(model, prepare_history(model, cc, rc), rr, tt, kk, 31)
            before = candidate_logits(model, cache, rr, tt, kk, 31)
            forbidden = max(forbidden, float((after-before).abs().max()))
    result = {'dataset': ds, 'taxonomy': taxonomy, 'width': width,
        'checkpoint': str(checkpoint) if checkpoint else 'untrained_seed23',
        'checkpoint_sha256': base.sha(checkpoint) if checkpoint else None,
        'queries': len(rows), 'large_chunk_queries': len(rows)*9,
        'native32_vs_native64_probability': float((native32.sigmoid()-expected.sigmoid()).abs().max()),
        'amended64_vs_native64_logit': float((amended64-expected).abs().max()),
        'adapter32_vs_native64_probability': float((actual.sigmoid()-expected.sigmoid()).abs().max()),
        'adapter32_vs_native64_logit': float((actual-expected).abs().max()),
        'adapter_vs_amended_full_probability': float((actual.sigmoid()-amended32.sigmoid()).abs().max()),
        'adapter_vs_amended_full_logit': float((actual-amended32).abs().max()),
        'chunk_logit_difference': chunk_delta, 'forbidden_information_logit_difference': forbidden}
    result['passed'] = max(result['adapter32_vs_native64_probability'], result['adapter_vs_amended_full_probability']) < 2e-5 and max(result['amended64_vs_native64_logit'], result['adapter32_vs_native64_logit'], result['adapter_vs_amended_full_logit'], chunk_delta) < 2e-4 and forbidden < 2e-5
    return result


def main():
    assert not OUT.exists(), 'Preserve completed diagnostics'
    torch.set_num_threads(1)
    protocol = base.read(base.OUT/'protocol.json')
    base.verify(protocol)
    registry = base.read(base.OUT/'run_registry.json')
    jobs = [(ds, tax, width, None) for ds in ['assist2017', 'assist2012']
            for tax in ['original', 'induced'] for width in [64, 128]]
    jobs += [(e['dataset'], e['taxonomy'], e['candidate']['d_model'], base.OUT/'trials'/key/'best.ckpt')
             for key, e in sorted(registry['trials'].items()) if e['state'] == 'complete']
    files = [Path(__file__), base.ROOT/'scripts/akt_stable_attention.py',
             base.ROOT/'scripts/akt_stable_queries.py', base.ROOT/'scripts/akt_runtime.py',
             base.SOURCE/'pykt/models/akt.py']
    report = {'started_utc': base.now(), 'device': 'cpu', 'training_fits': 0,
        'test_used': False, 'calibration_used': False, 'mps_acceptance': False,
        'proposal': 'Inference-only exclusive reverse cumulative tail; source/weights/loss unchanged.',
        'completed_screening_checkpoints_chosen_by': 'All complete at snapshot; no AUC-based selection.',
        'tolerances': {'probability': 2e-5, 'logit': 2e-4, 'forbidden_logit': 2e-5},
        'implementation_sha256': {str(p): base.sha(p) for p in files}, 'checks': []}
    for args in jobs:
        result = check(protocol, *args)
        report['checks'].append(result)
        base.dump(OUT, report)
        print(json.dumps(result), flush=True)
    base.verify(protocol)
    report.update(completed_utc=base.now(), passed=all(r['passed'] for r in report['checks']))
    base.dump(OUT, report)
    assert report['passed'], 'Review preserved numerical discrepancies'


if __name__ == '__main__':
    main()
