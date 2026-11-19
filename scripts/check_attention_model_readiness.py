"""Small MPS correctness checks only; no hyperparameter campaign or test scoring."""
import hashlib
import importlib.util
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
OUT = ROOT / 'results/model_readiness_20261107'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def module(name, device):
    package = '_currkt_readonly_pykt'
    if package not in sys.modules:
        pkg = types.ModuleType(package)
        pkg.__path__ = [str(SOURCE / 'pykt/models')]
        sys.modules[package] = pkg
    fullname = package + '.' + name
    if fullname not in sys.modules:
        spec = importlib.util.spec_from_file_location(fullname, SOURCE / 'pykt/models' / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[fullname]
    mod.device = torch.device(device)
    return mod


def model(name, concepts, device):
    torch.manual_seed(42)
    if device == 'mps':
        torch.mps.manual_seed(42)
    mod = module(name, device)
    cls = mod.simpleKT if name == 'simplekt' else mod.Robustkt
    return cls(n_question=concepts, n_pid=0, d_model=64, n_blocks=1,
               dropout=.1, d_ff=256, num_attn_heads=4, emb_type='qid').to(device)


def forward(m, c, r):
    if m.model_name == 'robustkt':
        return m(c, r)[0]
    z = torch.zeros_like(c)
    d = {'cseqs':c[:, :-1], 'rseqs':r[:, :-1], 'qseqs':z[:, :-1],
         'shft_cseqs':c[:, 1:], 'shft_rseqs':r[:, 1:], 'shft_qseqs':z[:, 1:]}
    return m(d)


def batch(dataset, kind):
    if dataset == 'assist2017':
        d = np.load(ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017/train_sequences.npz')
        c, r = d[kind + '_concepts'][:8], d['responses'][:8]
        n = 68 if kind == 'original' else 100
        path = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017/train_sequences.npz'
    else:
        path = ROOT / 'results/curriculum_kt_assist2012_k1000_20261107/induced_valid_sequences.npz'
        d = np.load(path)
        r = d['responses'][:8]
        c = d['concepts'][:8].copy()
        n = 162 if kind == 'original' else 1000
        if kind == 'original':
            mp = np.load(ROOT / 'results/curriculum_kt_assist2012_k1000_20261107/mapping.npz')
            qc = mp['q_to_original']
            q = d['questions'][:8]
            c[q >= 0] = qc[q[q >= 0]]
    valid = c >= 0
    return np.maximum(c, 0), np.maximum(r, 0), valid, n, path


def check(name, dataset, kind):
    ca, ra, valid, n, path = batch(dataset, kind)
    c, r = torch.tensor(ca, device='mps'), torch.tensor(ra, device='mps')
    mask = torch.tensor(valid[:, 1:] & valid[:, :-1], device='mps')
    m = model(name, n, 'mps')
    assert all(p.device.type == 'mps' for p in m.parameters())
    m.eval()
    with torch.no_grad():
        baseline = forward(m, c, r)
        rr = r.clone(); rr[:, 42:] = 1 - rr[:, 42:]
        response_delta = float((forward(m, c, rr)[:, :43] - baseline[:, :43]).abs().max())
        cc = c.clone(); cc[:, 43:] = (cc[:, 43:] + 1) % n
        future_delta = float((forward(m, cc, r)[:, :43] - baseline[:, :43]).abs().max())
        prefix_delta = float((forward(m, c[:, :43], r[:, :43]) - baseline[:, :43]).abs().max())
        padded_c, padded_r = c.clone(), r.clone()
        pad = torch.tensor(~valid, device='mps')
        padded_c[pad] = (padded_c[pad] + 1) % n
        padded_r[pad] = 1 - padded_r[pad]
        padding_delta = float((forward(m, padded_c, padded_r)[:, 1:][mask] - baseline[:, 1:][mask]).abs().max())
    # Diagnostic CPU inference reference; empirical update remains MPS-only.
    cpu = model(name, n, 'cpu')
    cpu.load_state_dict({k:v.detach().cpu() for k,v in m.state_dict().items()})
    cpu.eval()
    with torch.no_grad():
        reference = forward(cpu, torch.tensor(ca), torch.tensor(ra))
    cpu_delta = float((reference - baseline.cpu()).abs().max())
    module(name, 'mps')
    opt = torch.optim.Adam(m.parameters(), lr=.001)
    m.train()
    torch.mps.synchronize(); start = time.perf_counter()
    p = forward(m, c, r)
    loss = F.binary_cross_entropy(p[:, 1:][mask], r[:, 1:][mask].float())
    assert loss.isfinite()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(g.device.type == 'mps' and g.isfinite().all() for g in grads)
    opt.step()
    torch.mps.synchronize(); elapsed = time.perf_counter() - start
    checkpoint = OUT / f'smoke_{name}_{dataset}_{kind}.ckpt'
    torch.save(m.state_dict(), checkpoint)
    m.eval()
    with torch.no_grad():
        expected = forward(m, c, r)
    restored = model(name, n, 'mps')
    restored.load_state_dict(torch.load(checkpoint, map_location='mps', weights_only=True))
    restored.eval()
    with torch.no_grad():
        reload_delta = float((expected - forward(restored, c, r)).abs().max())
    result = {'model':name, 'dataset':dataset, 'taxonomy':kind, 'concepts':n,
        'device':'mps', 'batch_size':8, 'sequence_length':200, 'targets':int(mask.sum()),
        'configuration':{'d_model':64, 'n_blocks':1, 'dropout':.1, 'd_ff':256, 'num_attn_heads':4, 'n_pid':0},
        'variant':'Supported concept-only setting; target item identity is not supplied.',
        'loss':float(loss), 'forward_backward_update_seconds':elapsed,
        'current_and_future_response_delta':response_delta, 'future_concept_delta':future_delta,
        'prefix_delta':prefix_delta, 'padding_delta':padding_delta,
        'cpu_reference_max_abs_delta':cpu_delta, 'checkpoint_reload_max_abs_delta':reload_delta,
        'all_parameters_and_gradients_on_mps':True, 'parameters':sum(p.numel() for p in m.parameters()),
        'data_path':str(path), 'data_sha256':sha(path), 'checkpoint_sha256':sha(checkpoint),
        'passed':max(response_delta, future_delta, prefix_delta, padding_delta, cpu_delta, reload_delta) < 2e-5}
    print(json.dumps(result), flush=True)
    del m, cpu, restored, opt
    torch.mps.empty_cache()
    return result


def main():
    assert torch.backends.mps.is_available(), 'MPS required; no CPU training fallback'
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'python':platform.python_version(), 'torch':torch.__version__, 'platform':platform.platform(),
        'mps_available':True, 'test_evaluation':False, 'hyperparameter_tuning':False,
        'device_adapter':'Read-only source imports; override module.device to MPS. No source modification.',
        'source_sha256':{name:sha(SOURCE / 'pykt/models' / f'{name}.py') for name in ['simplekt','robustkt','utils']}, 'checks':[]}
    for name in ['simplekt', 'robustkt']:
        for dataset in ['assist2017', 'assist2012']:
            for kind in ['original','induced']:
                try:
                    report['checks'].append(check(name, dataset, kind))
                except Exception as e:
                    report['checks'].append({'model':name, 'dataset':dataset, 'taxonomy':kind, 'passed':False,
                                             'error_type':type(e).__name__, 'error':str(e)})
                (OUT / 'mps_smoke_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(r['passed'] for r in report['checks']), 'Review recorded smoke failures'


if __name__ == '__main__':
    main()
