"""Fixed-coefficient development diagnostics; no empirical optimization/AUC."""
import numpy as np
import torch
from run_pooling_batch import PooledLogistic as Original
from run_pooling_batch_v2 import PooledLogistic as Precise, design_inputs, folds, OUT, save, now

assert torch.backends.mps.is_available()
meta, x, mask, basis, anchor, _ = design_inputs('assist2012', 0, 'calibration', 'pooled_map')
fold, _ = folds(meta)
take = fold != 0
args = (x[take], meta['original_concept'][take], meta['label'][take], mask, basis, anchor, 1., False)
old, new = Original(*args), Precise(*args)
p = np.load(OUT/'numerical_failure_01/failed_fit.npz')
t = np.r_[p['q'].ravel()[old.ix], p['b'][old.bi], p['global_s'], p['global_b']]
rng = np.random.default_rng(404)
direction = rng.normal(size=len(t))
direction /= np.linalg.norm(direction)
rows = []
for step in [0., 1e-4, 1e-6, 1e-8]:
    point = t + step * direction
    exact, eg = old.audit64(point)
    before, bg = old.objective(point)
    after, ag = new.objective(point)
    assert abs(after-exact) < 1e-9
    assert np.max(np.abs(ag-eg)) < 1e-5
    rows.append({'fixed_perturbation': step, 'exact_objective': float(exact),
                 'old_objective_error': float(before-exact), 'precise_objective_error': float(after-exact),
                 'old_gradient_max_error': float(np.max(np.abs(bg-eg))),
                 'precise_gradient_max_error': float(np.max(np.abs(ag-eg)))})
save(OUT/'precision_diagnostic.json', {'scope': 'Fixed coefficients and fixed small perturbations on failed development fit; no optimization or new AUC',
     'checks': rows, 'all_precise_objective_checks_passed': True,
     'all_precise_gradient_checks_passed': True, 'completed_utc': now()})
print(rows)
