"""Exact reference queries: change only one target concept per copied history."""
import torch


def target_queries(model, concepts, responses, rows, positions, candidates, batch_size=16):
    """Return probabilities/logits for independent (row, position, child) queries.

    Each batch entry is its own observed sequence. All other concept and response
    positions stay fixed; no jointly relabeled sequence is ever evaluated.
    This deliberately simple reference supports later efficient-adapter checks.
    """
    assert not model.training
    assert len(rows) == len(positions) == len(candidates)
    probabilities, logits = [], []
    with torch.inference_mode():
        for lo in range(0, len(rows), batch_size):
            rr = rows[lo:lo+batch_size]
            tt = positions[lo:lo+batch_size]
            kk = candidates[lo:lo+batch_size]
            c, r = concepts[rr].clone(), responses[rr].clone()
            ix = torch.arange(len(rr), device=c.device)
            c[ix, tt] = kk
            p, regularizer, hidden = model(c, r, qtest=True)
            assert model.n_pid == 0 and regularizer == 0.
            z = model.out(hidden[ix, tt]).squeeze(-1)
            probabilities.append(p[ix, tt])
            logits.append(z)
    return torch.cat(probabilities), torch.cat(logits)
