"""Inference-only, cancellation-free evaluation of RobustKT's distance tail.

sum(p)-cumsum(p) equals the exclusive reverse cumulative sum of p. Computing
the latter directly avoids subtracting two near-equal float32 values before a
square root. Parameters, masks, attention formula and training stay unchanged.
Use explicitly within a context; the pinned source file is never modified.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F
from check_attention_model_readiness import module


def exclusive_tail(probabilities):
    inclusive = probabilities.flip(-1).cumsum(-1).flip(-1)
    return F.pad(inclusive[..., 1:], (0, 1))


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    assert not torch.is_grad_enabled(), 'This amendment is inference-only'
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    length = scores.shape[-1]
    positions = torch.arange(length, device=q.device)
    distance = (positions[None, :] - positions[:, None]).abs()
    base = F.softmax(scores.masked_fill(mask == 0, -1e32), dim=-1)
    base = base * mask.to(dtype=base.dtype, device=q.device)
    dist = (exclusive_tail(base) * distance).clamp(min=0).sqrt()
    gamma = -F.softplus(gamma).unsqueeze(0)
    if pdiff is not None:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist.shape[1], pdiff.shape[1], pdiff.shape[2])
        gamma = gamma * diff.sigmoid().exp()
    effect = (dist * gamma).exp().clamp(min=1e-5, max=1e5)
    weights = F.softmax((scores * effect).masked_fill(mask == 0, -1e32), dim=-1)
    if zero_pad:
        weights = torch.cat([torch.zeros_like(weights[:, :, :1]), weights[:, :, 1:]], dim=2)
    return torch.matmul(dropout(weights), v)


@contextmanager
def stable_attention(device):
    mod = module('robustkt', device)
    previous = mod.attention
    mod.attention = attention
    try:
        yield
    finally:
        mod.attention = previous
