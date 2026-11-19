"""CPU-only development-data checks; selected-model MPS acceptance is separate."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from akt_runtime import akt_class
from akt_cached_queries import prepare_history, candidate_logits
import run_akt_sweeps as base


OUT = base.ROOT / 'results/model_sequence_20261107/akt_cached_query_cpu_checks.json'


def check(protocol, ds, taxonomy, width, dtype, checkpoint=None):
    torch.manual_seed(23)
    nc = protocol['datasets'][ds][taxonomy + '_concepts']
    model = akt_class('cpu')(n_question=nc, n_pid=0, d_model=width, n_blocks=1,
        dropout=.1, d_ff=256, num_attn_heads=4, final_fc_dim=512,
        kq_same=1, separate_qa=False, l2=1e-5, emb_type='qid').to(dtype=dtype).eval()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    assert all(p.device.type == 'cpu' for p in model.parameters())
    data = protocol['datasets'][ds]['files']['train']
    assert base.sha(Path(data['path'])) == data['sha256']
    with np.load(data['path']) as source:
        concepts, responses = source[taxonomy+'_concepts'], source['responses']
        eligible = np.flatnonzero((concepts >= 0).sum(1) >= 64)[:2]
        c = torch.tensor(np.maximum(concepts[eligible], 0), dtype=torch.long)
        r = torch.tensor(np.maximum(responses[eligible], 0), dtype=torch.long)
    rows = torch.tensor([row for row in range(2) for _ in range(15)])
    positions = torch.tensor([t for _ in range(2) for t in [0, 1, 12, 42, 199] for _ in range(3)])
    candidates = (c[rows, positions] + torch.tensor([0, 1, 7]*10)) % nc
    start = time.monotonic()
    with torch.inference_mode():
        cache = prepare_history(model, c, r)
        actual = candidate_logits(model, cache, rows, positions, candidates, batch_size=31)
        refs, probabilities = [], []
        for row, pos, candidate in zip(rows.tolist(), positions.tolist(), candidates.tolist()):
            branch = c[row:row+1].clone()
            branch[0, pos] = candidate
            pp, reg, hidden = model(branch, r[row:row+1], qtest=True)
            assert reg == 0.
            refs.append(model.out(hidden).squeeze(-1)[0, pos])
            probabilities.append(pp[0, pos])
        references = torch.stack(refs)
        probability_delta = float((actual.sigmoid()-torch.stack(probabilities)).abs().max())
        logit_delta = float((actual-references).abs().max())
        chunk_delta = max(float((actual-candidate_logits(model, cache, rows, positions, candidates,
                           batch_size=size)).abs().max()) for size in [1, 7, 256])
        # Same past and candidate, with the realized target and future changed.
        row, pos, candidate = 0, 42, (int(c[0, 42])+7) % nc
        expected = candidate_logits(model, cache, [row], [pos], [candidate])
        cc, rr = c.clone(), r.clone()
        cc[:, pos:] = (cc[:, pos:]+13) % nc
        rr[:, pos:] = 1-rr[:, pos:]
        alternate = candidate_logits(model, prepare_history(model, cc, rr), [row], [pos], [candidate])
        forbidden_delta = float((alternate-expected).abs().max())
        # Exercise chunk boundaries beyond the production chunk ceiling.
        large_c = torch.arange(263) % nc
        large_r = torch.zeros(263, dtype=torch.long)
        large_t = torch.full((263,), 42, dtype=torch.long)
        large_a = candidate_logits(model, cache, large_r, large_t, large_c, batch_size=256)
        large_b = candidate_logits(model, cache, large_r, large_t, large_c, batch_size=31)
        large_delta = float((large_a-large_b).abs().max())
    return {'dataset': ds, 'taxonomy': taxonomy, 'width': width, 'dtype': str(dtype),
        'device': 'cpu', 'checkpoint': str(checkpoint) if checkpoint else 'untrained_fixed_seed23',
        'checkpoint_sha256': base.sha(checkpoint) if checkpoint else None,
        'independent_queries': len(rows), 'large_chunk_queries': 263,
        'max_probability_difference': probability_delta, 'max_logit_difference': logit_delta,
        'max_chunk_logit_difference': max(chunk_delta, large_delta),
        'forbidden_information_logit_difference': forbidden_delta,
        'seconds': time.monotonic()-start,
        'passed': probability_delta < 2e-5 and max(logit_delta, chunk_delta, large_delta) < 2e-4
                  and forbidden_delta < 2e-5}


def main():
    assert not OUT.exists(), 'Preserve each completed diagnostic; use a new version for amendments'
    torch.set_num_threads(1)
    protocol = base.read(base.OUT/'protocol.json')
    base.verify(protocol)
    report = {'started_utc': base.now(), 'device': 'cpu', 'model_fits': 0,
        'test_used': False, 'calibration_used': False, 'mps_acceptance': False,
        'scope': 'Training examples only; native pinned arithmetic. Does not authorize feature extraction.',
        'source_sha256': {str(path): base.sha(path) for path in [Path(__file__),
            base.ROOT/'scripts/akt_cached_queries.py', base.ROOT/'scripts/akt_runtime.py',
            base.SOURCE/'pykt/models/akt.py']}, 'checks': []}
    for ds in ['assist2017', 'assist2012']:
        for taxonomy in ['original', 'induced']:
            for width in [64, 128]:
                for dtype in [torch.float32, torch.float64]:
                    result = check(protocol, ds, taxonomy, width, dtype)
                    report['checks'].append(result)
                    base.dump(OUT, report)
                    print(json.dumps(result), flush=True)
    # Choose already complete checkpoints by candidate ID, never score.
    registry = base.read(base.OUT/'run_registry.json')
    for width in [64, 128]:
        eligible = sorted((key, e) for key, e in registry['trials'].items()
                          if e['state'] == 'complete' and e['dataset'] == 'assist2017'
                          and e['taxonomy'] == 'original' and e['candidate']['d_model'] == width)
        if eligible:
            key, _ = eligible[0]
            checkpoint = base.OUT/'trials'/key/'best.ckpt'
            for dtype in [torch.float32, torch.float64]:
                result = check(protocol, 'assist2017', 'original', width, dtype, checkpoint)
                report['checks'].append(result)
                base.dump(OUT, report)
                print(json.dumps(result), flush=True)
    report.update(completed_utc=base.now(), passed=all(r['passed'] for r in report['checks']))
    base.dump(OUT, report)
    assert report['passed'], 'Preserve and review CPU adapter discrepancies before MPS checks'


if __name__ == '__main__':
    main()
