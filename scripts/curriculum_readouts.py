"""Prespecified frozen-DKT readouts for the requested exploratory method batch."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def table(mapping):
    support = mapping['counts'] > 0
    width = int(support.sum(1).max())
    children = np.zeros((len(support), width), np.int64)
    mask = np.zeros_like(children, np.float32)
    weights = {key: np.zeros_like(mask) for key in ['item_overlap', 'practice_frequency', 'equal_children']}
    lookup = np.full(support.shape, -1, np.int64)
    for c in range(len(support)):
        k = np.flatnonzero(support[c])
        children[c, :len(k)] = k
        mask[c, :len(k)] = 1
        lookup[c, k] = np.arange(len(k))
        for key in weights:
            weights[key][c, :len(k)] = mapping[key][c, k]
    return children, mask, weights, lookup


def fit_readouts(x, d, mapping, pairs, out, seed):
    """Fit only development targets. No test data is accepted by this function."""
    assert torch.backends.mps.is_available()
    children, support_np, weights, lookup = table(mapping)
    n_c, width = support_np.shape
    c_np, y_np, q_np, k_np = [d[key] for key in ['original_concept', 'label', 'question_index', 'induced_concept']]
    slot_np = lookup[c_np, k_np]
    assert np.all(slot_np >= 0)
    counts = np.bincount(c_np, minlength=n_c)
    positive = np.bincount(c_np, weights=y_np, minlength=n_c)
    eligible = (counts >= 50) & (positive >= 5) & (counts - positive >= 5)
    selected = eligible[c_np]
    assert selected.any()
    c = torch.tensor(c_np[selected], device='mps')
    y = torch.tensor(y_np[selected], dtype=torch.float32, device='mps')
    X = torch.tensor(x[selected], device='mps')
    slot = torch.tensor(slot_np[selected], device='mps')
    count = torch.tensor(np.maximum(counts, 1), dtype=torch.float32, device='mps')
    elig = torch.tensor(eligible, device='mps')
    support = torch.tensor(support_np, device='mps')
    anchor = torch.tensor(weights['practice_frequency'], device='mps')
    parent_x = torch.logit((X.sigmoid() * anchor[c]).sum(1).clamp(1e-6, 1 - 1e-6))
    coef = {}
    traces = {}
    # Learned monotone multivariate readout, scalar monotone calibration, and
    # child-wise response calibration fit distinct, documented objectives.
    edge_n = np.zeros_like(support_np)
    edge_p = np.zeros_like(support_np)
    np.add.at(edge_n, (c_np, slot_np), 1)
    np.add.at(edge_p, (c_np, slot_np), y_np)
    edge_eligible = (edge_n >= 20) & (edge_p >= 2) & (edge_n - edge_p >= 2)
    edge_eligible &= eligible[:, None]
    edge_mask = torch.tensor(edge_eligible, device='mps')
    for kind in ['learned_monotone', 'parent_calibration', 'intersection']:
        start = time.monotonic()
        shape = (n_c,) if kind == 'parent_calibration' else (n_c, width)
        initial = anchor.clamp_min(1e-4) if kind == 'learned_monotone' else torch.ones(shape, device='mps')
        raw = torch.nn.Parameter(torch.log(torch.expm1(initial)))
        b = torch.nn.Parameter(torch.zeros(n_c, device='mps'))
        delta = torch.nn.Parameter(torch.zeros((n_c, width), device='mps')) if kind == 'intersection' else None
        parameters = [raw, b] + ([delta] if delta is not None else [])
        optimizer = torch.optim.Adam(parameters, lr=0.03)
        trace = []
        for step in range(1000):
            optimizer.zero_grad(set_to_none=True)
            a = F.softplus(raw)
            if kind == 'learned_monotone':
                a = a * support
                logits = (X * a[c]).sum(1) + b[c]
                penalty = 0.01 * ((a - anchor)**2).sum(1) + 0.001 * b**2
            elif kind == 'parent_calibration':
                logits = a[c] * parent_x + b[c]
                penalty = 0.01 * (a - 1)**2 + 0.001 * b**2
            else:
                a = torch.where(edge_mask, a, torch.ones_like(a))
                offset = delta * edge_mask
                logits = a[c, slot] * X.gather(1, slot[:, None]).squeeze(1) + b[c] + offset[c, slot]
                denom = edge_mask.sum(1).clamp_min(1)
                penalty = 0.01 * (((a - 1)**2 + offset**2) * edge_mask).sum(1) / denom + 0.001 * b**2
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') / count[c]).sum() / elig.sum() + penalty[elig].mean()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 250 == 0:
                trace.append({'step': step + 1, 'loss': float(loss.detach().cpu()), 'seconds': time.monotonic() - start})
                print(f'seed={seed} {kind}: {trace[-1]}', flush=True)
        with torch.no_grad():
            a = F.softplus(raw)
            if kind == 'learned_monotone':
                a = a * support
            elif kind == 'intersection':
                a = torch.where(edge_mask, a, torch.ones_like(a))
            coef[kind + '_a'] = a.cpu().numpy()
            coef[kind + '_b'] = b.cpu().numpy()
            if kind == 'intersection':
                coef['intersection_delta'] = (delta * edge_mask).cpu().numpy()
        traces[kind] = trace
    # Item-aware extension: a shrunk question-within-original-concept intercept
    # added to the fitted intersection response curve. Unseen pairs stay zero.
    pair_lookup = {(int(c0), int(q0)): i for i, (c0, q0, _) in enumerate(pairs)}
    pair_np = np.array([pair_lookup[int(c0), int(q0)] for c0, q0 in zip(c_np, q_np)])
    pn = np.bincount(pair_np, minlength=len(pairs))
    pp = np.bincount(pair_np, weights=y_np, minlength=len(pairs))
    pair_ok = (pn >= 5) & (pp >= 1) & (pn - pp >= 1)
    pair_ok &= eligible[pairs[:, 0]]
    pi = torch.tensor(pair_np[selected], device='mps')
    pmask = torch.tensor(pair_ok, device='mps')
    pc = torch.tensor(pairs[:, 0], device='mps')
    with torch.no_grad():
        a = torch.tensor(coef['intersection_a'], device='mps')
        b = torch.tensor(coef['intersection_b'], device='mps')
        delta = torch.tensor(coef['intersection_delta'], device='mps')
        base = a[c, slot] * X.gather(1, slot[:, None]).squeeze(1) + b[c] + delta[c, slot]
    effect = torch.nn.Parameter(torch.zeros(len(pairs), device='mps'))
    optimizer = torch.optim.Adam([effect], lr=0.03)
    start = time.monotonic()
    trace = []
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        applied = effect * pmask
        logits = base + applied[pi]
        # Weight each item's ridge by its development frequency, keeping the
        # shrinkage strength comparable to that item's mean likelihood.
        ridge = 0.1 * (applied[pi]**2 / count[c]).sum() / elig.sum()
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') / count[c]).sum() / elig.sum() + ridge
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach().cpu()), 'seconds': time.monotonic() - start})
    coef['item_effect'] = (effect * pmask).detach().cpu().numpy()
    coef['eligible'] = eligible
    coef['children'] = children
    coef['support'] = support_np
    coef['pairs'] = pairs
    np.savez_compressed(out / f'seed{seed}_readout_coefficients.npz', **coef)
    traces['item_aware'] = trace
    dump(out / f'seed{seed}_readout_fit.json', {'device': 'mps', 'seed': seed,
        'development_events': len(y_np), 'eligible_concepts': int(eligible.sum()),
        'eligible_intersections': int(edge_eligible.sum()), 'eligible_item_pairs': int(pair_ok.sum()),
        'test_used': False, 'fixed_dkt': True, 'trace': traces})
    return coef


def predict_readouts(x, d, mapping, coef):
    """Apply frozen maps; score product rules on a stable logarithmic scale."""
    children, mask, weights, lookup = table(mapping)
    c_np = d['original_concept']
    c = torch.tensor(c_np, device='mps')
    X = torch.tensor(x, device='mps')
    result = {}
    with torch.inference_mode():
        p = X.sigmoid()
        for key, value in weights.items():
            w = torch.tensor(value, device='mps')
            result[key] = (p * w[c]).sum(1).cpu().numpy()
        for kind in ['learned_monotone', 'parent_calibration']:
            a = torch.tensor(coef[kind + '_a'], device='mps')
            b = torch.tensor(coef[kind + '_b'], device='mps')
            if kind == 'learned_monotone':
                z = (X * a[c]).sum(1) + b[c]
            else:
                scalar = torch.tensor(result['practice_frequency'], device='mps').clamp(1e-6, 1 - 1e-6)
                z = a[c] * torch.logit(scalar) + b[c]
            result[kind] = z.sigmoid().cpu().numpy()
        a = torch.tensor(coef['intersection_a'], device='mps')
        b = torch.tensor(coef['intersection_b'], device='mps')
        offset = torch.tensor(coef['intersection_delta'], device='mps')
        z = X * a[c] + b[c, None] + offset[c]
        for weight_kind in ['item_overlap', 'practice_frequency']:
            w = torch.tensor(weights[weight_kind], device='mps')
            result['intersection_' + weight_kind] = (z.sigmoid() * w[c]).sum(1).cpu().numpy()
        # Explicitly average item-specific response curves over each original
        # concept's reference bank; the target item's identity is not queried.
        item_p = np.empty(len(c_np), np.float32)
        pairs = coef['pairs']
        for original in np.unique(c_np):
            event_ix = np.flatnonzero(c_np == original)
            pair_ix = np.flatnonzero(pairs[:, 0] == original)
            child_slot = torch.tensor(lookup[original, pairs[pair_ix, 2]], device='mps')
            effects = torch.tensor(coef['item_effect'][pair_ix], device='mps')
            for lo in range(0, len(event_ix), 256):
                ix = event_ix[lo:lo + 256]
                ix_tensor = torch.tensor(ix, device='mps')
                values = torch.sigmoid(z[ix_tensor][:, child_slot] + effects).mean(1)
                item_p[ix] = values.cpu().numpy()
        result['item_aware_reference_average'] = item_p
        # AND and noisy-OR are diagnostic aggregation rules, not calibrated
        # event correctness models or validated latent mastery estimates.
        support = torch.tensor(mask, device='mps')[c]
        result['and_log_score'] = (F.logsigmoid(X) * support).sum(1).cpu().numpy().astype(float)
        result['noisy_or_log_score'] = (-F.logsigmoid(-X) * support).sum(1).cpu().numpy().astype(float)
    fallback = ~coef['eligible'][c_np]
    for key in ['learned_monotone', 'parent_calibration', 'intersection_item_overlap',
                'intersection_practice_frequency', 'item_aware_reference_average']:
        result[key][fallback] = result['practice_frequency'][fallback]
    result['and_product'] = np.exp(result['and_log_score'])
    result['noisy_or'] = -np.expm1(-result['noisy_or_log_score'])
    result['direct_item_average'] = result['item_overlap'].copy()
    return result
