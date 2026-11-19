"""Cached independent target-concept queries for the one-block concept-only AKT.

Only the current concept changes. Earlier concept representations, inclusive
self-attention states, and response-memory states are reused without relabeling.
No realized target item or response is an input to the candidate query.
"""
import math

import torch
from torch.nn import functional as F


def _project(layer, value, kind):
    attention = layer.masked_attn_head
    projection = getattr(attention, kind+'_linear')
    return projection(value).reshape(*value.shape[:-1], attention.h, attention.d_k).transpose(-3, -2)


def _prepare_history(model, concepts, responses):
    assert model.model_name == 'akt'
    assert not model.training and model.n_pid == 0 and model.emb_type == 'qid'
    assert not model.separate_qa
    assert concepts.ndim == 2 and concepts.shape[1] == 200
    architecture = model.model
    assert len(architecture.blocks_1) == 1 and len(architecture.blocks_2) == 2
    assert all(block.masked_attn_head.kq_same for block in [*architecture.blocks_1, *architecture.blocks_2])
    with torch.inference_mode():
        raw = model.q_embed(concepts)
        x0 = raw
        y = model.qa_embed(responses)+raw
        y = architecture.blocks_1[0](mask=1, query=y, key=y, values=y, pdiff=None)
        self_block, cross_block = architecture.blocks_2
        x1 = self_block(mask=1, query=x0, key=x0, values=x0, apply_pos=False, pdiff=None)
        return {'raw':raw,
                'self_k':_project(self_block, x0, 'k'),
                'self_v':_project(self_block, x0, 'v'),
                'cross_k':_project(cross_block, x1, 'k'),
                'cross_v':_project(cross_block, y, 'v')}


def prepare_history(model, concepts, responses):
    # Process each native history independently; verify this shape on MPS later.
    parts=[_prepare_history(model, concepts[i:i+1], responses[i:i+1]) for i in range(len(concepts))]
    return {key:torch.cat([part[key] for part in parts],dim=0) for key in parts[0]}


def _one_position_attention(layer, query, keys, values, positions, strict):
    attention = layer.masked_attn_head
    n, heads, length, depth = keys.shape
    q = attention.k_linear(query).reshape(n, heads, 1, depth)
    scores = torch.matmul(q, keys.transpose(-2, -1))/math.sqrt(depth)
    key_positions = torch.arange(length, device=query.device)
    allowed = key_positions[None, :] < positions[:, None] if strict else key_positions[None, :] <= positions[:, None]
    allowed = allowed[:, None, None, :]
    with torch.no_grad():
        base = F.softmax(scores.masked_fill(~allowed, -1e32), dim=-1)*allowed.float()
        distance = (positions[:,None]-key_positions[None,:]).abs()[:,None,None,:]
        # Preserve the pinned AKT arithmetic, including its detached distance.
        tail = base.sum(dim=-1, keepdim=True) - base.cumsum(dim=-1)
        distance = torch.clamp(tail*distance.float(), min=0.).sqrt().detach()
    gamma = -F.softplus(attention.gammas).unsqueeze(0)
    effect = torch.clamp(torch.exp(distance*gamma), min=1e-5, max=1e5)
    weights = F.softmax((scores*effect).masked_fill(~allowed, -1e32), dim=-1)
    if strict:
        weights = weights.masked_fill((positions==0)[:,None,None,None], 0.)
    weights = attention.dropout(weights)
    output = torch.matmul(weights, values).transpose(1,2).reshape(n, attention.d_model)
    return attention.out_proj(output)


def candidate_logits(model, history, rows, positions, candidates, batch_size=256):
    assert not model.training and model.n_pid == 0
    assert len(rows) == len(positions) == len(candidates)
    device = history['raw'].device
    rows, positions, candidates = [torch.as_tensor(v, dtype=torch.long, device=device) for v in (rows, positions, candidates)]
    assert batch_size > 0
    assert bool(((rows >= 0) & (rows < history['raw'].shape[0])).all())
    assert bool(((positions>=0)&(positions<200)).all())
    assert bool(((candidates>=0)&(candidates<model.n_question)).all())
    if not len(rows):
        return torch.empty(0,device=device)
    result = []
    architecture = model.model
    self_block, cross_block = architecture.blocks_2
    with torch.inference_mode():
        for lo in range(0, len(rows), batch_size):
            rr, tt, cc = rows[lo:lo+batch_size], positions[lo:lo+batch_size], candidates[lo:lo+batch_size]
            ix = torch.arange(len(rr), device=device)
            current = model.q_embed(cc)
            # The inclusive question self-attention sees the candidate at t.
            # Earlier keys/values stay fixed; the later retrieval excludes t.
            k, v = history['self_k'][rr].clone(), history['self_v'][rr].clone()
            attention = self_block.masked_attn_head
            k[ix,:,tt,:] = attention.k_linear(current).reshape(len(rr),attention.h,attention.d_k)
            v[ix,:,tt,:] = attention.v_linear(current).reshape(len(rr),attention.h,attention.d_k)
            attended = _one_position_attention(self_block,current,k,v,tt,strict=False)
            x = self_block.layer_norm1(current+self_block.dropout1(attended))
            attended = _one_position_attention(cross_block,x,history['cross_k'][rr],history['cross_v'][rr],tt,strict=True)
            x = cross_block.layer_norm1(x+cross_block.dropout1(attended))
            residual = cross_block.linear2(cross_block.dropout(cross_block.activation(cross_block.linear1(x))))
            x = cross_block.layer_norm2(x+cross_block.dropout2(residual))
            result.append(model.out(torch.cat([x,current],dim=-1)).squeeze(-1))
    return torch.cat(result)
