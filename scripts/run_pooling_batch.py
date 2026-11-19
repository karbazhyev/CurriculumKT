"""One finite, development-selected frozen-DKT pooling/augmentation batch.

No source/legacy writes. Stage boundaries prevent test evaluation before both
datasets' selections and full-calibration coefficient hashes are committed.
"""
import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from evaluate_saved_curriculum_kt import ROOT, sha
from run_readout_followup import OUT as PRIOR, runs, load_mapping

OUT = ROOT / 'results/curriculum_kt_pooling_batch_20261107'
DATASETS = ['assist2012', 'assist2017']
DESIGNS = ['pooled_map', 'fixed_map', 'pooled_original', 'augment_induced', 'augment_original']
LAMBDAS = [0.001, 0.01, 0.1, 1.0]
SEEDS = {'assist2012': [42, 84, 255], 'assist2017': [42, 84, 126]}
INDUCED_SEEDS = [42, 84, 126]
FOLD_SEED = 20261108


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def record(key, **fields):
    path = OUT / 'run_registry.json'
    obj = read(path)
    obj['runs'][key].update(fields)
    save(path, obj)


def source_dir(ds, kind, seed):
    return PRIOR / ds / f'{kind}_seed{seed}'


def guard():
    manifest = read(OUT/'frozen_identities.json')
    assert sha(OUT/'protocol.json') == manifest['protocol_sha256']
    assert read(OUT/'protocol.json')['frozen']
    for path, expected in manifest['scripts'].items():
        assert sha(Path(path)) == expected, ('Frozen script changed', path)
    assert sha(OUT/'protected_source_hashes.json') == manifest['input_manifest_sha256']
    for path, expected in read(OUT/'protected_source_hashes.json').items():
        assert sha(Path(path)) == expected, ('Protected source changed', path)


def load_source(ds, kind, seed, split):
    path = source_dir(ds, kind, seed)
    identity = read(path / f'{split}_complete.json')
    for name in [f'{split}_metadata.npz', f'{split}_logits.npy']:
        assert sha(path / name) == identity['files'][name], (path, name)
    return dict(np.load(path / f'{split}_metadata.npz')), np.load(path / f'{split}_logits.npy', mmap_mode='r')


def design_inputs(ds, pair, split, design):
    """Only history logits and original target c enter the predictor design.

    Target question/induced label/response fields never enter X. Responses are
    returned separately for fit/metrics. No learned calibration feature is reused.
    """
    oc = SEEDS[ds][pair]
    oi = SEEDS[ds][(pair + 1) % 3]
    ic = INDUCED_SEEDS[pair]
    od, oz = load_source(ds, 'original', oc, split)
    c = od['original_concept']
    o = oz[np.arange(len(c)), c].astype(np.float32)
    mapping = load_mapping(ds)
    w = mapping['practice_frequency'].astype(float)
    support = mapping['counts'] > 0
    assert np.isfinite(w).all() and np.all(w >= 0)
    assert np.allclose(w.sum(1), 1., rtol=0, atol=1e-6)
    assert np.all(w[~support] == 0) and np.all(support.any(1))
    assert len(w) == oz.shape[1] and np.all((c >= 0) & (c < len(w)))
    nc = len(w)
    meta = {k: od[k] for k in ['uid', 'row', 'position', 'event_id', 'label', 'original_concept']}
    native = od['original_dkt'].astype(float)
    if design in ['pooled_map', 'fixed_map', 'augment_induced']:
        ind, iz = load_source(ds, 'induced', ic, split)
        assert iz.shape[1] == w.shape[1]
        assert all(np.array_equal(od[k], ind[k]) for k in meta)
        if design in ['pooled_map', 'fixed_map']:
            support = mapping['counts'] > 0
            width = int(support.sum(1).max())
            children = np.zeros((nc, width), dtype=np.int64)
            mask = np.zeros((nc, width), dtype=bool)
            anchor = np.zeros((nc, width))
            for j in range(nc):
                kk = np.flatnonzero(support[j])
                children[j, :len(kk)] = kk
                mask[j, :len(kk)] = True
                anchor[j, :len(kk)] = w[j, kk]
            norm = np.sqrt(np.sum(anchor ** 2, axis=1) + 1e-12)
            x = np.take_along_axis(iz, children[c], axis=1).astype(np.float32)
            x *= mask[c]
            x *= norm[c, None].astype(np.float32)
            basis = anchor / norm[:, None]
            return meta, x, mask, basis, np.array([1.]), native
        # Probability averaging is frozen. No other aggregate is searched.
        q = np.empty(len(c), dtype=np.float64)
        for lo in range(0, len(c), 2048):
            sl = slice(lo, lo + 2048)
            q[sl] = np.sum(expit(iz[sl].astype(float)) * w[c[sl]], axis=1)
        second = logit(np.clip(q, 1e-6, 1 - 1e-6))
    elif design == 'augment_original':
        other, zother = load_source(ds, 'original', oi, split)
        assert all(np.array_equal(od[k], other[k]) for k in meta)
        second = zother[np.arange(len(c)), c]
    if design == 'pooled_original':
        x = o[:, None]
    elif design.startswith('augment_'):
        x = np.column_stack([o, second]).astype(np.float32)
    else:
        raise ValueError(design)
    mask = np.ones((nc, x.shape[1]), dtype=bool)
    basis = np.ones_like(mask, dtype=float)
    global_anchor = np.array([1.]) if x.shape[1] == 1 else np.array([1., 0.])
    return meta, x, mask, basis, global_anchor, native


class PooledLogistic:
    """Convex direct-coordinate fit, MPS data products, independent CPU audit.

    Map q=a/||w|| and B=w/||w||. Scalar/stack q=a, B=1.
    Optimize C times the protocol's mean-event objective for numerical scaling.
    Unseen concepts are analytically set to their pooling center.
    """
    def __init__(self, x, c, y, mask, basis, global_anchor, lam, fixed=False):
        assert torch.backends.mps.is_available(), 'MPS required; no CPU fitting fallback'
        self.xnp = np.asarray(x, np.float32)
        self.cnp = np.asarray(c, np.int64)
        self.ynp = np.asarray(y, float)
        self.mask = mask
        self.basis = basis
        self.nc, self.nd = mask.shape
        self.anchor = global_anchor
        self.ng = len(global_anchor)
        self.lam = lam
        self.fixed = fixed
        self.counts = np.bincount(c, minlength=self.nc)
        self.observed = self.counts > 0
        self.free = mask & self.observed[:, None]
        self.ix = np.flatnonzero(self.free.ravel())
        self.bi = np.flatnonzero(self.observed)
        self.x = torch.tensor(self.xnp, device='mps')
        self.c = torch.tensor(self.cnp, device='mps')
        self.calls = 0
        self.theta = np.r_[self.center(self.anchor).ravel()[self.ix], np.zeros(len(self.bi))]
        self.bounds = [(0, None)] * len(self.ix) + [(None, None)] * len(self.bi)
        if not fixed:
            self.theta = np.r_[self.theta, self.anchor, 0.]
            self.bounds += [(0, None)] * self.ng + [(None, None)]

    def center(self, s):
        return self.basis * s.reshape(1, -1) if self.ng == self.nd else self.basis * s[0]

    def unpack(self, t):
        n = len(self.ix)
        s, b0 = (self.anchor, 0.) if self.fixed else (t[-self.ng-1:-1], t[-1])
        q = self.center(s).copy()
        q.ravel()[self.ix] = t[:n]
        b = np.full(self.nc, b0)
        b[self.bi] = t[n:n+len(self.bi)]
        return q, b, s, b0

    def penalty(self, q, b, s, b0, gq, gb):
        delta = (q - self.center(s)) * self.mask
        db = b - b0
        loss = self.lam * (np.sum(delta ** 2) + .1 * np.sum(db ** 2))
        gq += 2 * self.lam * delta
        gb += .2 * self.lam * db
        grad = [gq.ravel()[self.ix], gb[self.bi]]
        if not self.fixed:
            if self.ng == self.nd:
                gs = -2 * self.lam * np.sum(delta * self.basis, axis=0)
            else:
                gs = np.array([-2 * self.lam * np.sum(delta * self.basis)])
            weak = self.nc * 1e-6
            loss += weak * (np.sum((s - self.anchor) ** 2) + b0 ** 2)
            gs += 2 * weak * (s - self.anchor)
            gb0 = -.2 * self.lam * np.sum(db) + 2 * weak * b0
            grad.extend([gs, np.array([gb0])])
        return loss, np.concatenate(grad)

    def objective(self, t):
        q, b, s, b0 = self.unpack(t)
        with torch.inference_mode():
            tq = torch.tensor(q, dtype=torch.float32, device='mps')
            tb = torch.tensor(b, dtype=torch.float32, device='mps')
            z = ((self.x * tq[self.c]).sum(1) + tb[self.c]).cpu().numpy().astype(float)
            err = (expit(z) - self.ynp) * (self.nc / len(z))
            te = torch.tensor(err, dtype=torch.float32, device='mps')
            gq = torch.zeros(q.shape, device='mps').index_add_(0, self.c, self.x * te[:, None]).cpu().numpy().astype(float)
        gb = np.bincount(self.cnp, weights=err, minlength=self.nc)
        penalty, grad = self.penalty(q, b, s, b0, gq, gb)
        loss = self.nc * np.mean(np.logaddexp(0, z) - self.ynp * z) + penalty
        self.calls += 1
        return float(loss), grad

    def audit64(self, t):
        # Fixed-coefficient verification only: never a CPU optimization fallback.
        q, b, s, b0 = self.unpack(t)
        gq = np.zeros_like(q)
        gb = np.zeros_like(b)
        loss = 0.
        for c in self.bi:
            take = self.cnp == c
            x = self.xnp[take].astype(float)
            y = self.ynp[take]
            z = x @ q[c] + b[c]
            e = (expit(z) - y) * self.nc / len(self.ynp)
            gq[c] = x.T @ e
            gb[c] = e.sum()
            loss += np.sum(np.logaddexp(0, z) - y * z) * self.nc / len(self.ynp)
        penalty, grad = self.penalty(q, b, s, b0, gq, gb)
        return loss + penalty, grad

    def pg(self, t, grad):
        g = grad.copy()
        for j, bound in enumerate(self.bounds):
            if bound[0] == 0 and t[j] <= 1e-8 and g[j] > 0:
                g[j] = 0
        return float(np.max(np.abs(g)))

    def solve(self):
        start = time.monotonic()
        t = self.theta
        initial = self.audit64(t)[0]
        attempts = []
        for attempt in range(3):
            res = minimize(self.objective, t, jac=True, method='L-BFGS-B', bounds=self.bounds,
                           options={'maxiter': 1500, 'maxfun': 3000, 'ftol': 1e-12,
                                    'gtol': 2e-5, 'maxcor': 20, 'maxls': 40})
            t = res.x
            value, grad = self.audit64(t)
            pg = self.pg(t, grad)
            attempts.append({'attempt': attempt+1, 'iterations': int(res.nit),
                             'message': str(res.message), 'objective_scaled': float(value),
                             'float64_projected_gradient': pg})
            if pg <= 2e-4:
                break
        q, b, s, b0 = self.unpack(t)
        assert value <= initial + 1e-5
        assert np.min(q[self.mask]) >= 0 and np.min(s) >= 0
        info = {'device': str(self.x.device), 'mps_available_in_fit': True,
                'optimizer': 'CPU L-BFGS-B control; MPS data products and gradients',
                'audit': 'CPU float64 fixed-coefficient KKT check',
                'objective_scale': self.nc, 'initial_objective_scaled': float(initial),
                'final_objective_scaled': float(value), 'projected_gradient': pg,
                'converged': pg <= 2e-4, 'attempts': attempts, 'function_calls': self.calls,
                'fit_events': len(self.ynp), 'observed_concepts': int(self.observed.sum()),
                'unseen_concepts': int((~self.observed).sum()), 'seconds': time.monotonic()-start}
        return {'q': q, 'b': b, 'global_s': s, 'global_b': np.array(b0), 'counts': self.counts}, info


def predict(x, c, params):
    # Applying saved coefficients is CPU metric preprocessing, not training.
    result = np.empty(len(c))
    for lo in range(0, len(c), 2048):
        sl = slice(lo, lo+2048)
        result[sl] = expit(np.sum(x[sl].astype(float) * params['q'][c[sl]], axis=1) + params['b'][c[sl]])
    assert np.isfinite(result).all()
    return result


def fit_id(ds, design, pair, lam, fold=None):
    stage = f'cv{fold}_lambda{lam:g}' if fold is not None else 'refit'
    return f'{ds}/{design}/pair{pair}/{stage}'


def fit_one(key, x, c, y, mask, basis, anchor, lam, fixed, selected_students):
    dest = OUT / 'fits' / key
    # Decimal lambda text is part of the directory name, so avoid with_suffix
    # accidentally collapsing distinct lambdas: use explicit full-name suffix.
    mark = Path(str(dest) + '.json')
    coef = Path(str(dest) + '.npz')
    if mark.exists():
        info = read(mark)
        assert info['converged'] and sha(coef) == info['coefficient_sha256']
        assert info['protocol_sha256'] == sha(OUT/'protocol.json')
        return dict(np.load(coef)), info
    record(key, state='running', started_utc=now())
    solver = PooledLogistic(x, c, y, mask, basis, anchor, lam, fixed)
    params, info = solver.solve()
    coef.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coef, **params)
    info.update({'coefficient_sha256': sha(coef), 'lambda': lam, 'fixed_center': fixed,
                 'protocol_sha256': sha(OUT/'protocol.json'), 'test_used': False,
                 'fit_students': len(selected_students), 'completed_utc': now()})
    save(mark, info)
    record(key, state='complete' if info['converged'] else 'failed', **info)
    del solver
    gc.collect()
    torch.mps.empty_cache()
    if not info['converged']:
        raise RuntimeError(f'Convergence check failed: {key}')
    return params, info


def folds(meta):
    ids = np.unique(meta['uid'])
    shuffled = np.random.default_rng(FOLD_SEED).permutation(ids)
    groups = np.array_split(shuffled, 3)
    assignment = np.empty(len(meta['uid']), np.int64)
    for j, group in enumerate(groups):
        assignment[np.isin(meta['uid'], group)] = j
    return assignment, groups


def choose(rows):
    # One choice per dataset/family; seeds and folds are averaged, not selected.
    scores = {lam: float(np.mean([r['auc'] for r in rows if r['lambda'] == lam])) for lam in LAMBDAS}
    best = max(scores.values())
    selected = max(lam for lam, value in scores.items() if best-value <= 1e-6)
    return selected, scores


def development():
    guard()
    assert torch.backends.mps.is_available(), 'MPS required'
    torch.set_num_threads(1)
    save(OUT/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
         'mps_available': True, 'device_probe': str(torch.ones(1, device='mps').device),
         'script_sha256': sha(Path(__file__)), 'started_utc': now()})
    for ds in DATASETS:
        final = OUT/ds/'selection.json'
        if final.exists():
            old_selection = read(final)
            assert old_selection['protocol_sha256'] == sha(OUT/'protocol.json')
            for key, expected in old_selection['full_fit_coefficient_hashes'].items():
                assert sha(Path(str(OUT/'fits'/key)+'.npz')) == expected
            continue
        rows = []
        selections = {}
        native_scores = []
        for design in DESIGNS:
            family_rows = []
            for pair in range(3):
                meta, x, mask, basis, anchor, native = design_inputs(ds, pair, 'calibration', design)
                c, y = meta['original_concept'], meta['label']
                fold, groups = folds(meta)
                if design == DESIGNS[0] and pair == 0:
                    save(OUT/ds/'development_folds.json', {'seed': FOLD_SEED,
                         'students': [g.astype(int).tolist() for g in groups],
                         'events': [int(np.sum(fold == f)) for f in range(3)],
                         'calibration_events': len(y), 'calibration_students': len(np.unique(meta['uid']))})
                if design == 'pooled_original':
                    native_scores.extend(float(roc_auc_score(y[fold == f], native[fold == f])) for f in range(3))
                for lam in LAMBDAS:
                    for f in range(3):
                        train, valid = fold != f, fold == f
                        assert not set(meta['uid'][train]) & set(meta['uid'][valid])
                        key = fit_id(ds, design, pair, lam, f)
                        params, _ = fit_one(key, x[train], c[train], y[train], mask, basis, anchor,
                                            lam, design == 'fixed_map', np.unique(meta['uid'][train]))
                        p = predict(x[valid], c[valid], params)
                        row = {'dataset': ds, 'design': design, 'pair': pair, 'lambda': lam,
                               'fold': f, 'auc': float(roc_auc_score(y[valid], p))}
                        family_rows.append(row)
                        rows.append(row)
                        print(json.dumps({'stage': 'cv', **row}), flush=True)
            selected, scores = choose(family_rows)
            selections[design] = {'lambda': selected, 'mean_fold_seed_auc': scores,
                                  'selected_auc': scores[selected]}
            pd.DataFrame(rows).to_csv(OUT/ds/'development_scores.csv', index=False)
        native_auc = float(np.mean(native_scores))
        original = 'pooled_original' if selections['pooled_original']['selected_auc'] > native_auc + 1e-6 else 'original_dkt'
        control_auc = native_auc if original == 'original_dkt' else selections['pooled_original']['selected_auc']
        endpoints = {'original_selected': original}
        for design in ['augment_induced', 'augment_original']:
            endpoints[design+'_selected'] = design if selections[design]['selected_auc'] > control_auc + 1e-6 else original
        # Refit every selected family, even if a retention endpoint wins.
        hashes = {}
        for design in DESIGNS:
            lam = selections[design]['lambda']
            for pair in range(3):
                meta, x, mask, basis, anchor, _ = design_inputs(ds, pair, 'calibration', design)
                key = fit_id(ds, design, pair, lam)
                _, info = fit_one(key, x, meta['original_concept'], meta['label'], mask, basis, anchor,
                                  lam, design == 'fixed_map', np.unique(meta['uid']))
                hashes[key] = info['coefficient_sha256']
        save(final, {'dataset': ds, 'families': selections, 'native_cv_auc': native_auc,
                     'retention': endpoints, 'full_fit_coefficient_hashes': hashes,
                     'protocol_sha256': sha(OUT/'protocol.json'),
                     'test_used': False, 'completed_utc': now()})
    selections = {ds: sha(OUT/ds/'selection.json') for ds in DATASETS}
    save(OUT/'development_complete.json', {'completed_utc': now(), 'selections': selections,
         'protocol_sha256': sha(OUT/'protocol.json'), 'test_not_evaluated_by_this_batch': True})


def test():
    guard()
    frozen = read(OUT/'development_complete.json')
    assert sha(OUT/'protocol.json') == frozen['protocol_sha256']
    assert all(sha(OUT/ds/'selection.json') == h for ds, h in frozen['selections'].items())
    for ds in DATASETS:
        selection = read(OUT/ds/'selection.json')
        for pair in range(3):
            file = OUT/ds/f'test_pair{pair}.npz'
            if file.exists():
                old_prediction = read(file.with_suffix('.json'))
                assert sha(file) == old_prediction['sha256']
                assert old_prediction['protocol_sha256'] == sha(OUT/'protocol.json')
                assert old_prediction['selection_sha256'] == sha(OUT/ds/'selection.json')
                continue
            output = None
            for design in DESIGNS:
                meta, x, _, _, _, native = design_inputs(ds, pair, 'test', design)
                if output is None:
                    output = dict(meta)
                    output['original_dkt'] = native
                else:
                    assert all(np.array_equal(output[k], meta[k]) for k in meta)
                key = fit_id(ds, design, pair, selection['families'][design]['lambda'])
                coef = Path(str(OUT/'fits'/key)+'.npz')
                assert sha(coef) == selection['full_fit_coefficient_hashes'][key]
                params = dict(np.load(coef))
                output[design] = predict(x, meta['original_concept'], params)
                assert sha(coef) == selection['full_fit_coefficient_hashes'][key]
            for name, chosen in selection['retention'].items():
                output[name] = output[chosen].copy()
            # Frozen no-fit reference is retained for context only.
            ind, _ = load_source(ds, 'induced', INDUCED_SEEDS[pair], 'test')
            output['practice_frequency'] = ind['practice_frequency'].astype(float)
            np.savez_compressed(file, **output)
            save(file.with_suffix('.json'), {'sha256': sha(file), 'created_utc': now(),
                 'selection_sha256': sha(OUT/ds/'selection.json'), 'protocol_sha256': sha(OUT/'protocol.json'),
            'fit_frozen_before_test': True, 'all_target_events_and_labels_aligned': True})
            print(json.dumps({'stage': 'test_prediction_saved', 'dataset': ds, 'pair': pair}), flush=True)
    save(OUT/'test_complete.json', {'completed_utc': now(), 'protocol_sha256': sha(OUT/'protocol.json')})


def selfcheck():
    """Small synthetic diagnostic, no empirical responses or test arrays."""
    assert torch.backends.mps.is_available(), 'MPS required'
    rng = np.random.default_rng(1931)
    checks = []
    for design in ['pooled_map', 'fixed_map', 'augmentation']:
        nc, nd = 4, (3 if design != 'augmentation' else 2)
        x = rng.normal(size=(120, nd)).astype(np.float32)
        c = rng.integers(3, size=120)  # Fourth concept deliberately unseen.
        y = rng.integers(2, size=120)
        mask = np.ones((nc, nd), bool)
        basis = np.ones((nc, nd)) if design == 'augmentation' else rng.uniform(.1, 1, (nc, nd))
        if design != 'augmentation':
            basis /= np.linalg.norm(basis, axis=1, keepdims=True)
        anchor = np.array([1., 0.]) if design == 'augmentation' else np.array([1.])
        solver = PooledLogistic(x, c, y, mask, basis, anchor, .1, design == 'fixed_map')
        t = solver.theta.copy() + .15
        loss, grad = solver.audit64(t)
        mpsloss, mpsgrad = solver.objective(t)
        fd = np.zeros_like(t)
        for j in range(len(t)):
            right, left = t.copy(), t.copy()
            right[j] += 1e-5
            left[j] -= 1e-5
            fd[j] = (solver.audit64(right)[0] - solver.audit64(left)[0]) / 2e-5
        assert np.max(np.abs(fd-grad)) < 1e-7
        assert np.max(np.abs(mpsgrad-grad)) < 2e-6
        assert abs(mpsloss-loss) < 2e-6
        params, info = solver.solve()
        assert info['converged']
        assert np.allclose(params['q'][3], solver.center(params['global_s'])[3])
        assert params['b'][3] == params['global_b']
        checks.append({'design': design, 'gradient_error': float(np.max(np.abs(fd-grad))),
                       'mps_gradient_error': float(np.max(np.abs(mpsgrad-grad))),
                       'unseen_center_verified': True, **info})
    save(OUT/'solver_diagnostic.json', {'scope': 'Three small synthetic numerical diagnostics, not empirical experiments',
                                      'checks': checks, 'completed_utc': now()})
    print('Synthetic MPS gradient, optimizer and unseen-concept checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['development', 'test', 'selfcheck'])
    args = parser.parse_args()
    try:
        {'development': development, 'test': test, 'selfcheck': selfcheck}[args.stage]()
    except Exception as exc:
        if args.stage != 'selfcheck' and (OUT/'run_registry.json').exists():
            registry = read(OUT/'run_registry.json')
            registry.update(state='failed', failed_stage=args.stage, failure=repr(exc), failed_utc=now())
            save(OUT/'run_registry.json', registry)
        raise
