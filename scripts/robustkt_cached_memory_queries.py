"""Conservative independent RobustKT queries with reusable response memory.

Each candidate keeps both native length200 question-attention calls. Only the
response-memory stream, which depends solely on the observed past at prediction
time, is cached. Query microbatches stay at one to preserve reference arithmetic.
This is a slower fallback for the still-unverified partial-attention prototype.
"""
import torch


def prepare_history(model, concepts, responses):
    assert not model.training and model.n_pid == 0 and model.emb_type == 'qid'
    assert not model.separate_qa and concepts.shape[1] == 200
    architecture = model.model
    assert len(architecture.blocks_1) == 1 and len(architecture.blocks_2) == 2
    raw, memory = [], []
    with torch.inference_mode():
        # Keep the same native batch shape as independent single-query inference.
        for row in range(len(concepts)):
            q = model.q_embed(concepts[row:row+1])
            y = architecture.smooth(model.qa_embed(responses[row:row+1])+q)
            y = architecture.blocks_1[0](mask=1, query=y, key=y, values=y, pdiff=None)
            raw.append(q)
            memory.append(y)
    return {'raw':torch.cat(raw), 'memory':torch.cat(memory)}


def candidate_logits(model, history, rows, positions, candidates, batch_size=256):
    assert not model.training and model.n_pid == 0
    assert len(rows) == len(positions) == len(candidates) and batch_size > 0
    device = history['raw'].device
    rr, tt, cc = [torch.as_tensor(v, dtype=torch.long, device=device)
                  for v in (rows, positions, candidates)]
    assert bool(((tt>=0)&(tt<200)).all())
    assert bool(((cc>=0)&(cc<model.n_question)).all())
    result = []
    with torch.inference_mode():
        for lo in range(0, len(rr), batch_size):
            # Outer chunks bound caller bookkeeping; every numerical query uses
            # the identical batch1/length200 native question-stream computation.
            for ix in range(lo, min(lo+batch_size, len(rr))):
                raw = history['raw'][rr[ix:ix+1]].clone()
                raw[0,tt[ix]] = model.q_embed(cc[ix])
                x = model.model.smooth(raw)
                x = model.model.blocks_2[0](mask=1,query=x,key=x,values=x,apply_pos=False,pdiff=None)
                x = model.model.blocks_2[1](mask=0,query=x,key=x,
                    values=history['memory'][rr[ix:ix+1]],apply_pos=True,pdiff=None)
                logits = model.out(torch.cat([x,raw],dim=-1)).squeeze(-1)
                result.append(logits[0,tt[ix]].reshape(1))
    return torch.cat(result) if result else torch.empty(0,device=device)
