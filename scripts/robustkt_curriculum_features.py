"""Compact Curriculum KT features with independently queried target concepts.

Only original target concepts determine mapped children/weights. The realized
induced target is used solely for the explicitly separate native diagnostic.
Observed past concepts/responses remain unchanged for every child query.
"""
import numpy as np
import torch
from scipy.special import expit

from check_attention_model_readiness import forward
from robustkt_cached_memory_queries import prepare_history, candidate_logits
from simplekt_curriculum_features import mapping_layout, event_metadata



@torch.inference_mode()
def extract_batch(model, arrays, taxonomy, layout, row_offset=0, query_batch_size=256):
    """Return event-aligned scalar original or sparse induced-child features.

    This is inference only. A caller must enforce the frozen stage protocol,
    checkpoint/data identities, GPU exclusivity and calibration/test boundary.
    CPU execution is reserved for small training-data reference diagnostics.
    """
    assert taxonomy in ['original', 'induced']
    assert not model.training
    assert model.n_question == layout[taxonomy + '_concepts']
    device = next(model.parameters()).device
    meta, rows, positions = event_metadata(arrays, row_offset)
    assert len(rows) > 0, 'Pass a nonempty batch of predictable events'
    cc = np.maximum(arrays[taxonomy + '_concepts'], 0).astype(np.int64)
    rr = np.maximum(arrays['responses'], 0).astype(np.int64)
    c, r = torch.as_tensor(cc, device=device), torch.as_tensor(rr, device=device)
    rows_t = torch.as_tensor(rows, device=device)
    positions_t = torch.as_tensor(positions, device=device)
    cache = prepare_history(model, c, r)
    # Numerical batch1 matches the conservative independent-query reference.
    # The full length200 native model is unchanged; only inference batching is fixed.
    native_matrix = torch.cat([forward(model,c[ix:ix+1],r[ix:ix+1])
                               for ix in range(len(c))],dim=0)
    native = native_matrix[rows_t,positions_t].cpu().numpy()
    native_target = c[rows_t, positions_t]
    native_logits = candidate_logits(model, cache, rows_t, positions_t, native_target,
                                 batch_size=query_batch_size).cpu().numpy()
    native_delta = float(np.max(np.abs(expit(native_logits) - native)))
    assert native_delta < 2e-5, f'Native logit query differs from full forward: {native_delta}'
    features = {'native_probability': native, 'native_logit': native_logits}
    if taxonomy == 'induced':
        original = meta['original_concept']
        active = layout['mask'][original]
        event_ix, slot_ix = np.nonzero(active)
        candidate = layout['children'][original[event_ix], slot_ix]
        child_logits = np.zeros(active.shape, dtype=np.float32)
        values = candidate_logits(model, cache,
            torch.as_tensor(rows[event_ix], device=device),
            torch.as_tensor(positions[event_ix], device=device),
            torch.as_tensor(candidate, device=device), batch_size=query_batch_size)
        child_logits[event_ix, slot_ix] = values.cpu().numpy()
        weights = layout['weights'][original]
        mapped_probability = np.sum(expit(child_logits.astype(np.float64)) * weights, axis=1)
        assert np.isfinite(mapped_probability).all()
        assert ((mapped_probability >= 0) & (mapped_probability <= 1)).all()
        features.update(child_logits=child_logits, practice_frequency=mapped_probability)
        queries = len(event_ix)
    else:
        queries = 0
    assert all(np.isfinite(v).all() for v in features.values())
    return meta, features, {'events': len(rows), 'independent_child_queries': queries,
                           'native_query_max_probability_difference': native_delta,
                           'device': str(device), 'native_inference_batch_size': 1,
                           'candidate_numerical_microbatch_size': 1}
