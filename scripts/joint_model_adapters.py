"""Pinned native models, explicit losses and independent target-query semantics."""
from contextlib import nullcontext
import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from check_attention_model_readiness import module


def make(name, k, config, device='mps', seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)
    module('utils', device)
    source = {'dkt+': 'dkt_plus', 'atktfix': 'atkt'}.get(name, name)
    mod = module(source, device); w, drop = config['width'], config['dropout']
    if name == 'sakt':
        m = mod.SAKT(k, 200, w, 4, drop, num_en=1)
    elif name in ['dkt', 'dkt+']:
        cls = mod.DKT if name == 'dkt' else mod.DKTPlus
        extra = {} if name == 'dkt' else dict(lambda_r=.2, lambda_w1=1., lambda_w2=10.)
        m = cls(k, w, dropout=drop, **extra)
    elif name in ['dkvmn', 'deep_irt', 'skvmn']:
        cls = getattr(mod, {'dkvmn': 'DKVMN', 'deep_irt': 'DeepIRT', 'skvmn': 'SKVMN'}[name])
        m = cls(k, w, 50, dropout=drop)
        if name == 'skvmn':
            # Native hop assignment aliases a stored earlier hidden state and
            # rewrites past predictions. Clone before the in-place hop update;
            # cell state already has this protection in the pinned source.
            import inspect
            import textwrap
            import types
            source_text = textwrap.dedent(inspect.getsource(cls.forward))
            old = 'hx[j,:] = hidden_state[idx_values[0][2]][j]'
            assert source_text.count(old) == 1
            source_text = source_text.replace(old, 'hx = hx.clone()\n                '+old)
            namespace = {}
            exec(compile(source_text, '<CurriculumKT-SKVMN-causal-clone>', 'exec'), mod.__dict__, namespace)
            m.forward = types.MethodType(namespace['forward'], m)
    elif name == 'kqn':
        m = mod.KQN(k, w, w, w, dropout=drop)
        m.two_eye = m.two_eye.to(device); m.eye = m.eye.to(device)
    elif name == 'atktfix':
        m = mod.ATKT(k, w, w, w, dropout=drop, fix=True, epsilon=10., beta=.2)
    elif name == 'saint':
        m = mod.SAINT(0, k, 200, w, 4, drop, n_blocks=2)
    elif name in ['sparsekt', 'cskt', 'akt']:
        cls = getattr(mod, {'sparsekt': 'sparseKT', 'cskt': 'CSKT', 'akt': 'AKT'}[name])
        emb_type = 'qid_sparseattn' if name == 'sparsekt' else 'qid'
        m = cls(k, 0, w, 1, drop, d_ff=256, num_attn_heads=4, emb_type=emb_type)
    elif name == 'dtransformer':
        m = mod.DTransformer(k, 0, d_model=w, d_ff=256, num_attn_heads=4,
              n_blocks=3, n_know=16, dropout=drop, lambda_cl=.1, window=1, emb_type='qid_cl')
        # Preserve -1 for native sequence lengths/augmentation, but never send
        # padding sentinels into embedding tables. Padded losses stay masked.
        import types
        native_base = mod.DTransformer.base_emb
        def padded_base(self, q, r):
            return native_base(self, q.clamp_min(0), r.clamp_min(0))
        m.base_emb = types.MethodType(padded_base, m)
    elif name == 'ukt':
        raise NotImplementedError('Native UKT requires r_aug/shft_r_aug, but the pinned dataset code contains no uncertainty-augmentation producer. Preserve CL/augmentation; do not substitute BCE or invent its rule.')
    else:
        raise ValueError(name)
    m = m.to(device); m._joint_name = name; m._joint_k = k
    return m


def dictionary(c, r):
    cc, rr = c.clamp_min(0), r.clamp_min(0); z = torch.zeros_like(cc)
    return {'cseqs': cc[:, :-1], 'shft_cseqs': cc[:, 1:],
            'rseqs': rr[:, :-1], 'shft_rseqs': rr[:, 1:],
            'qseqs': z[:, :-1], 'shft_qseqs': z[:, 1:],
            'masks': (c[:, :-1] >= 0) & (c[:, 1:] >= 0)}


def activate(model):
    device = str(next(model.parameters()).device)
    module('utils', device)
    module({'dkt+': 'dkt_plus', 'atktfix': 'atkt'}.get(model._joint_name, model._joint_name), device)


def native(model, c, r):
    """One probability per target position1..199, never current-response input."""
    activate(model)
    name = model._joint_name; cc, rr = c.clamp_min(0), r.clamp_min(0)
    if name in ['dkt', 'dkt+']:
        return model(cc[:, :-1], rr[:, :-1]).gather(2, cc[:, 1:, None]).squeeze(-1)
    if name == 'atktfix':
        return model(cc[:, :-1], rr[:, :-1])[0].gather(2, cc[:, 1:, None]).squeeze(-1)
    if name in ['sakt', 'kqn']:
        return model(cc[:, :-1], rr[:, :-1], cc[:, 1:])
    if name in ['dkvmn', 'deep_irt', 'skvmn']:
        return model(cc, rr)[:, 1:]
    if name == 'saint':
        return model(torch.zeros_like(cc), cc, rr[:, :-1])[:, 1:]
    if name == 'sparsekt':
        return model(dictionary(c, r))[0][:, 1:]
    if name == 'cskt':
        return model(dictionary(c, r))[:, 1:]
    if name == 'dtransformer':
        return model.predict(c, r, None)[0][:, 1:].sigmoid()
    if name == 'akt':
        return model(cc, rr)[0][:, 1:]
    raise ValueError(name)


def training_loss(model, c, r):
    activate(model)
    mask = (c[:, :-1] >= 0) & (c[:, 1:] >= 0)
    cc, rr = c.clamp_min(0), r.clamp_min(0)
    name = model._joint_name
    bce = lambda p: F.binary_cross_entropy(p[mask], rr[:, 1:][mask].float())
    if name == 'dkt+':
        all_p = model(cc[:, :-1], rr[:, :-1])
        next_p = all_p.gather(2, cc[:, 1:, None]).squeeze(-1)
        current = all_p.gather(2, cc[:, :-1, None]).squeeze(-1)
        reconstruct = F.binary_cross_entropy(current[mask], rr[:, :-1][mask].float())
        change = all_p[:, 1:] - all_p[:, :-1]
        wmask = mask[:, 1:]
        assert wmask.any(), 'DKT+ waviness requires valid adjacent predictions'
        w1 = change.abs().sum(-1)[wmask].mean() / model.num_c
        w2 = change.square().sum(-1)[wmask].mean() / model.num_c
        return bce(next_p) + model.lambda_r*reconstruct + model.lambda_w1*w1 + model.lambda_w2*w2
    if name == 'atktfix':
        all_p, features = model(cc[:, :-1], rr[:, :-1])
        first = bce(all_p.gather(2, cc[:, 1:, None]).squeeze(-1))
        grad = torch.autograd.grad(first, features, retain_graph=True)[0].detach()
        # Same per-sequence L2 normalization as the pinned CPU helper, on MPS.
        perturb = model.epsilon * grad / (grad.square().sum((1, 2), keepdim=True).sqrt() + 1e-16)
        adverse = model(cc[:, :-1], rr[:, :-1], perturb)[0]
        return first + model.beta*bce(adverse.gather(2, cc[:, 1:, None]).squeeze(-1))
    if name == 'dtransformer':
        if int((r >= 0).sum(1).min()) < 5:
            # Repair the pinned short-sequence fallback's undefined pid/logits
            # names. Its intended fallback is ordinary prediction+regularizer.
            logits, _, _, _, reg, _ = model.predict(c, r, None)
            return bce(logits[:, 1:].sigmoid()) + reg
        p, regularizer = model.get_cl_loss(c, r, c.new_empty((len(c), 0)))
        return bce(p[:, 1:]) + regularizer
    if name == 'akt':
        p, reg = model(cc, rr)
        return bce(p[:, 1:]) + reg
    return bce(native(model, c, r))


def inference_context(model):
    if model._joint_name == 'akt':
        from akt_stable_attention import stable_attention
        return stable_attention(str(next(model.parameters()).device))
    return nullcontext()


@torch.inference_mode()
def query(model, c, r, rows, positions, candidates, batch_size=32):
    """Candidate-only replacement, preserving every query's observed past."""
    activate(model)
    rows = torch.as_tensor(rows, device=c.device, dtype=torch.long)
    positions = torch.as_tensor(positions, device=c.device, dtype=torch.long)
    candidates = torch.as_tensor(candidates, device=c.device, dtype=torch.long)
    name = model._joint_name
    if name in ['dkt', 'dkt+', 'atktfix']:
        output = model(c.clamp_min(0)[:, :-1], r.clamp_min(0)[:, :-1])
        if name == 'atktfix': output = output[0]
        return output[rows, positions-1, candidates]
    if name == 'akt':
        from akt_stable_queries import prepare_history, candidate_logits
        with inference_context(model):
            cache = prepare_history(model, c.clamp_min(0), r.clamp_min(0))
            return candidate_logits(model, cache, rows, positions, candidates, batch_size).sigmoid()
    if name == 'sakt':
        assert model.num_en == 1
        # Native one-block SAKT has independent current queries over fixed
        # history keys/values. Cache projected memory, not target responses.
        cc, rr = c.clamp_min(0), r.clamp_min(0)
        x = model.interaction_emb(cc[:, :-1]+model.num_c*rr[:, :-1])
        x = x + model.position_emb(torch.arange(x.shape[1], device=c.device))
        block = model.blocks[0]; attn = block.attn; width = x.shape[-1]
        weight, bias = attn.in_proj_weight, attn.in_proj_bias
        key = F.linear(x, weight[width:2*width], bias[width:2*width])
        value = F.linear(x, weight[2*width:], bias[2*width:])
        heads = model.num_attn_heads; dim = width//heads
        key = key.view(len(x), -1, heads, dim).transpose(1, 2)
        value = value.view(len(x), -1, heads, dim).transpose(1, 2)
        result = []
        for lo in range(0, len(rows), batch_size):
            sl = slice(lo, lo+batch_size); row, pos, child = rows[sl], positions[sl], candidates[sl]
            emb = model.exercise_emb(child)
            q = F.linear(emb, weight[:width], bias[:width]).view(-1, heads, dim)
            score = (q[:, :, None]*key[row]).sum(-1)/math.sqrt(dim)
            score.masked_fill_(torch.arange(x.shape[1], device=c.device)[None, None, :] >= pos[:, None, None], -float('inf'))
            v = (score.softmax(-1)[..., None]*value[row]).sum(-2).reshape(-1, width)
            v = F.linear(v, attn.out_proj.weight, attn.out_proj.bias)
            v = block.attn_layer_norm(emb+v)
            v = block.FFN_layer_norm(v+block.FFN(v))
            result.append(model.pred(v).squeeze(-1).sigmoid())
        return torch.cat(result)
    results = []
    for lo in range(0, len(rows), batch_size):
        sl = slice(lo, lo+batch_size); row, pos, child = rows[sl], positions[sl], candidates[sl]
        cc, rr = c[row].clone(), r[row].clone()
        cc[torch.arange(len(row), device=c.device), pos] = child
        results.append(native(model, cc, rr)[torch.arange(len(row), device=c.device), pos-1])
    return torch.cat(results)


@torch.inference_mode()
def query_gate(model, c, r):
    model.eval(); device = c.device
    rows = [0, 0, 1, 1]; positions = [1, 7, 3, 15]
    candidates = [(int(c[row, pos])+3) % model._joint_k for row, pos in zip(rows, positions)]
    with inference_context(model):
        actual = query(model, c, r, rows, positions, candidates)
        reference = []
        forbidden = []
        for row, pos, child in zip(rows, positions, candidates):
            cc = c[row:row+1].clone(); rr = r[row:row+1].clone(); cc[0, pos] = child
            reference.append(native(model, cc, rr)[0, pos-1])
            cc[:, pos:] = (cc[:, pos:].clamp_min(0)+5) % model._joint_k
            rr[:, pos:] = 1-rr[:, pos:].clamp_min(0)
            forbidden.append(query(model, cc, rr, [0], [pos], [child])[0])
        reference, forbidden = torch.stack(reference), torch.stack(forbidden)
    delta = float((actual-reference).abs().max()); leakage = float((actual-forbidden).abs().max())
    assert max(delta, leakage) < 2e-5, (delta, leakage)
    return {'device': str(device), 'independent_queries': 4, 'probability_delta': delta,
            'forbidden_information_delta': leakage, 'passed': True}
