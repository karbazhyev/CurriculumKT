"""Exact target-concept queries for supported one-block concept-only SimpleKT.

This adapter changes one queried target concept while preserving observed past
interactions. It does not relabel the complete history with a hypothetical child.
MPS validation must run when the active training worker has released the GPU.
"""
import torch


@torch.inference_mode()
def history_cache(model, concepts, responses):
    assert model.model_name == 'simplekt' and not model.training
    assert model.n_pid == 0 and model.emb_type == 'qid'
    assert len(model.model.blocks_2) == 1, 'Multi-block query dependencies need a different verified adapter'
    layer = model.model.blocks_2[0]
    head = layer.masked_attn_head
    q_embed, qa_embed = model.base_emb(concepts, responses)
    position = model.model.position_emb(q_embed)
    q_position = q_embed + position
    qa_position = qa_embed + position
    b,l,_ = q_embed.shape
    keys = head.k_linear(q_position).reshape(b,l,head.h,head.d_k).transpose(1,2)
    values = head.v_linear(qa_position).reshape(b,l,head.h,head.d_k).transpose(1,2)
    return {'keys':keys,'values':values,'position':position,'length':l}


@torch.inference_mode()
def query_logits(model, cache, rows, positions, candidate_concepts, query_batch_size=256):
    """Return logits for independent (history row, target position, candidate KC)."""
    assert not model.training and model.n_pid == 0 and len(model.model.blocks_2) == 1
    assert rows.shape == positions.shape == candidate_concepts.shape
    if len(rows) == 0:
        return torch.empty(0,device=cache['keys'].device)
    assert bool(((positions>=1)&(positions<cache['length'])).all())
    layer = model.model.blocks_2[0]
    head = layer.masked_attn_head
    result = []
    indices = torch.arange(cache['length'],device=rows.device)
    for lo in range(0,len(rows),query_batch_size):
        sl=slice(lo,lo+query_batch_size);rr=rows[sl];tt=positions[sl];kk=candidate_concepts[sl]
        raw_q=model.q_embed(kk)
        query=raw_q+cache['position'][0,tt]
        projection=head.k_linear(query) if head.kq_same else head.q_linear(query)
        query_heads=projection.reshape(len(rr),head.h,1,head.d_k)
        scores=torch.matmul(query_heads,cache['keys'][rr].transpose(-2,-1))/(head.d_k**.5)
        past=indices[None,:]<tt[:,None]
        scores=scores.masked_fill(~past[:,None,None,:],-1e32)
        weights=head.dropout(torch.softmax(scores,dim=-1))
        context=torch.matmul(weights,cache['values'][rr]).transpose(1,2).reshape(len(rr),head.d_model)
        attended=head.out_proj(context)
        output=layer.layer_norm1(query+layer.dropout1(attended))
        feedforward=layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
        output=layer.layer_norm2(output+layer.dropout2(feedforward))
        result.append(model.out(torch.cat([output,raw_q],dim=-1)).squeeze(-1))
    return torch.cat(result)
