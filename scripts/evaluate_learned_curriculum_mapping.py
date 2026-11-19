"""Fit the prespecified monotone map-back on development data; freeze DKT.

This is an exploratory fitted readout, not a new KT training run. Both the
frozen DKT inference and the readout optimization use the existing MPS runtime.
"""
import json
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_saved_curriculum_kt import OUT, SOURCE, dump, sha, metrics


def children_table(mapping):
    support = mapping['counts'] > 0
    width = int(support.sum(1).max())
    children = np.zeros((len(support), width), np.int64)
    mask = np.zeros_like(children, dtype=np.float32)
    prior = np.zeros_like(mask)
    for c in range(len(support)):
        k = np.flatnonzero(support[c])
        children[c, :len(k)] = k
        mask[c, :len(k)] = 1
        prior[c, :len(k)] = mapping['practice_frequency'][c, k]
    return children, mask, prior


def features(run, split, children):
    destination = OUT / f"learned_seed{run['seed']}_{split}_features.npy"
    if destination.exists():
        return np.load(destination)
    d = dict(np.load(OUT / f'{split}_sequences.npz'))
    reference = np.load(OUT / f"induced_seed{run['seed']}_{split}_predictions.npz")
    model_class = runpy.run_path(str(SOURCE / 'pykt/models/dkt.py'))['DKT']
    model = model_class(1000, 256, 0.5)
    model.load_state_dict(torch.load(run['checkpoint'], map_location='cpu', weights_only=True))
    model.to('mps').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cs, rs = d['concepts'], d['responses']
    valid = (cs[:, :-1] >= 0) & (cs[:, 1:] >= 0)
    selected = d['selectmasks'][:, 1:] >= 0
    chunks = []
    cursor = 0
    with torch.inference_mode():
        for lo in range(0, len(cs), 64):
            hi = min(lo + 64, len(cs))
            c = torch.as_tensor(cs[lo:hi, :-1] * valid[lo:hi], device='mps')
            r = torch.as_tensor(rs[lo:hi, :-1] * valid[lo:hi], device='mps')
            prediction = model(c, r)
            smask = torch.as_tensor(selected[lo:hi], device='mps')
            queried = prediction[smask]
            size = len(queried)
            old = reference['raw_induced'][cursor:cursor + size]
            target = torch.as_tensor(cs[lo:hi, 1:][selected[lo:hi], None], device='mps')
            direct = queried.gather(1, target).squeeze(1).cpu().numpy()
            assert np.allclose(direct, old, atol=2e-6, rtol=1e-5)
            original = reference['original_concept'][cursor:cursor + size]
            indices = torch.as_tensor(children[original], device='mps')
            x = torch.logit(queried.gather(1, indices).clamp(1e-6, 1 - 1e-6))
            chunks.append(x.cpu().numpy())
            cursor += size
    assert cursor == len(reference['label'])
    x = np.concatenate(chunks)
    np.save(destination, x)
    return x


def fit(run, x, mask, prior):
    destination = OUT / f"learned_seed{run['seed']}_coefficients.npz"
    if destination.exists():
        return dict(np.load(destination))
    start = time.monotonic()
    d = np.load(OUT / f"induced_seed{run['seed']}_induced_valid_predictions.npz")
    counts = np.bincount(d['original_concept'], minlength=len(mask))
    positive = np.bincount(d['original_concept'], weights=d['label'], minlength=len(mask))
    eligible = (counts >= 50) & (positive >= 5) & (counts - positive >= 5)
    selected = eligible[d['original_concept']]
    data = torch.as_tensor(x[selected], device='mps')
    c = torch.as_tensor(d['original_concept'][selected], device='mps')
    y = torch.as_tensor(d['label'][selected].astype(np.float32), device='mps')
    count = torch.as_tensor(np.maximum(counts, 1).astype(np.float32), device='mps')
    elig = torch.as_tensor(eligible, device='mps')
    support = torch.as_tensor(mask, device='mps')
    anchor = torch.as_tensor(prior, device='mps')
    raw = torch.nn.Parameter(torch.log(torch.expm1(anchor.clamp_min(1e-4))))
    intercept = torch.nn.Parameter(torch.zeros(len(mask), device='mps'))
    optimizer = torch.optim.Adam([raw, intercept], lr=0.03)
    trace = []
    for iteration in range(1000):
        optimizer.zero_grad(set_to_none=True)
        coefficient = F.softplus(raw) * support
        logits = (data * coefficient[c]).sum(1) + intercept[c]
        event_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        bce = (event_loss / count[c]).sum() / elig.sum()
        ridge = 0.01 * ((coefficient - anchor)**2).sum(1) + 0.001 * intercept**2
        loss = bce + ridge[elig].mean()
        loss.backward()
        optimizer.step()
        if iteration == 0 or (iteration + 1) % 100 == 0:
            record = {'step': iteration + 1, 'objective': float(loss.detach().cpu()),
                      'bce_macro_concept': float(bce.detach().cpu()), 'seconds': time.monotonic() - start}
            trace.append(record)
            print(f"learned mapping seed={run['seed']} " + json.dumps(record), flush=True)
    with torch.no_grad():
        coefficients = (F.softplus(raw) * support).cpu().numpy()
        biases = intercept.cpu().numpy()
    assert np.isfinite(coefficients).all() and np.all(coefficients >= 0)
    assert np.all(coefficients[mask == 0] == 0)
    result = {'a': coefficients, 'b': biases, 'eligible': eligible, 'validation_counts': counts,
              'validation_positive_counts': positive}
    np.savez_compressed(destination, **result)
    dump(destination.with_suffix('.json'), {'seed': run['seed'], 'device': 'mps',
        'training_split': 'historical validation fold0, also used for DKT checkpoint selection',
        'test_used_for_fitting': False, 'dkt_frozen': True, 'eligible_concepts': int(eligible.sum()),
        'fallback_concepts': int((~eligible).sum()), 'fit_events': int(selected.sum()),
        'fit_students': len(np.unique(d['uid'][selected])), 'seconds': time.monotonic() - start,
        'coefficient_sha256_before_test': sha(destination), 'trace': trace,
        'convergence': 'Fixed 1000 steps; no assertion of exact optimization convergence'})
    return result


def evaluate(run, x, params):
    d = dict(np.load(OUT / f"induced_seed{run['seed']}_induced_test_predictions.npz"))
    c = torch.as_tensor(d['original_concept'], device='mps')
    with torch.no_grad():
        a = torch.as_tensor(params['a'], device='mps')
        b = torch.as_tensor(params['b'], device='mps')
        tensor = torch.as_tensor(x, device='mps')
        p = torch.sigmoid((tensor * a[c]).sum(1) + b[c]).cpu().numpy()
    fallback = ~params['eligible'][d['original_concept']]
    p[fallback] = d['practice_frequency'][fallback]
    target = OUT / f"learned_seed{run['seed']}_induced_test_predictions.npz"
    np.savez_compressed(target, **{k: d[k] for k in ['uid', 'row', 'position', 'label', 'original_concept']}, learned_monotone=p)
    result = {'seed': run['seed'], 'device': 'mps', 'fallback_test_events': int(fallback.sum()),
              'coefficient_sha256': sha(OUT / f"learned_seed{run['seed']}_coefficients.npz"),
              'prediction_sha256': sha(target), 'metrics': metrics(d['label'], p, d['original_concept'])}
    dump(target.with_suffix('.json'), result)
    print(json.dumps({'seed': run['seed'], 'learned_monotone_test_auc': result['metrics']['auc']}), flush=True)


def main():
    assert torch.backends.mps.is_available(), 'MPS required; no CPU fitting fallback'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    learned_protocol = json.loads((OUT / 'learned_mapping_protocol.json').read_text())
    assert learned_protocol['seeds'] == [42, 84, 126]
    children, mask, prior = children_table(np.load(OUT / 'mapping.npz'))
    for run in protocol['runs']:
        if run['kind'] != 'induced':
            continue
        assert sha(Path(run['checkpoint'])) == run['checkpoint_sha256']
        validation_features = features(run, 'induced_valid', children)
        params = fit(run, validation_features, mask, prior)
        del validation_features
        frozen_hash = sha(OUT / f"learned_seed{run['seed']}_coefficients.npz")
        test_features = features(run, 'induced_test', children)
        evaluate(run, test_features, params)
        assert sha(OUT / f"learned_seed{run['seed']}_coefficients.npz") == frozen_hash
        del test_features


if __name__ == '__main__':
    main()
