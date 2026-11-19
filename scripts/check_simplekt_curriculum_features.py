"""Small training-data checks for compact feature assembly; CPU or exclusive MPS."""
import argparse
import fcntl
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from check_attention_model_readiness import module, forward
from evaluate_saved_curriculum_kt import ROOT, sha
from simplekt_curriculum_features import extract_batch, mapping_layout


def check(device):
    torch.set_num_threads(1)
    if device == 'mps':
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        assert torch.backends.mps.is_available()
    root = ROOT / 'results/simplekt_sweeps_20261107'
    protocol = json.loads((root / 'protocol.json').read_text())
    registry = json.loads((root / 'run_registry.json').read_text())
    dataset = protocol['datasets']['assist2017']
    record = dataset['files']['train']
    assert sha(Path(record['path'])) == record['sha256']
    with np.load(record['path']) as d:
        arrays = {k: d[k][:2] for k in d.files}
    assert sha(Path(dataset['mapping']['path'])) == dataset['mapping']['sha256']
    with np.load(dataset['mapping']['path']) as d:
        layout = mapping_layout(d)
    results = []
    meta_original = None
    for taxonomy, cfg in [('original', 'c04'), ('induced', 'c05')]:
        key = f'assist2017_simplekt_{taxonomy}_{cfg}_seed42'
        entry = registry['trials'][key]
        assert entry['state'] == 'complete'
        checkpoint = root / 'trials' / key / 'best.ckpt'
        assert sha(checkpoint) == entry['checkpoint_sha256']
        model = module('simplekt', device).simpleKT(
            n_question=dataset[taxonomy + '_concepts'], n_pid=0,
            d_model=entry['candidate']['d_model'], n_blocks=1,
            dropout=entry['candidate']['dropout'], d_ff=256, num_attn_heads=4,
            emb_type='qid', seq_len=200).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        meta, features, diagnostics = extract_batch(model, arrays, taxonomy, layout, query_batch_size=31)
        if meta_original is None:
            meta_original = meta
        else:
            assert all(np.array_equal(meta[k], meta_original[k]) for k in meta)
        reference_checks, largest = 0, 0.
        if taxonomy == 'induced':
            # Recompute complete weighted sums at selected targets through the
            # ordinary model, changing only ONE target concept per forward call.
            for ix in np.unique(np.linspace(0, len(meta['label'])-1, 7).astype(int)):
                row, t, oc = (int(meta[k][ix]) for k in ['row', 'position', 'original_concept'])
                c = torch.tensor(np.maximum(arrays['induced_concepts'][row:row+1], 0), device=device)
                r = torch.tensor(np.maximum(arrays['responses'][row:row+1], 0), device=device)
                total = 0.
                for slot in np.flatnonzero(layout['mask'][oc]):
                    changed = c.clone()
                    changed[0, t] = int(layout['children'][oc, slot])
                    with torch.inference_mode():
                        probability = float(forward(model, changed, r)[0, t])
                    largest = max(largest, abs(probability - expit(float(features['child_logits'][ix, slot]))))
                    total += layout['weights'][oc, slot] * probability
                    reference_checks += 1
                largest = max(largest, abs(total - features['practice_frequency'][ix]))
            assert largest < 2e-5
            assert (features['child_logits'][~layout['mask'][meta['original_concept']]] == 0).all()
        results.append({'trial': key, 'checkpoint_sha256': sha(checkpoint), **diagnostics,
            'ordinary_one_target_reference_checks': reference_checks,
            'sparse_child_and_weighted_map_max_difference': largest, 'passed': True})
        del model
        if device == 'mps':
            torch.mps.empty_cache()
    report = {'device': device, 'status': 'passed', 'checks': results,
        'paired_metadata_identical': True, 'training_rows': 2,
        'empirical_fits': 0, 'heldout_queries': 0,
        'data_sha256': record['sha256'], 'mapping_sha256': dataset['mapping']['sha256'],
        'implementation_sha256': {str(p): sha(p) for p in [
            Path(__file__).resolve(), Path(__file__).with_name('simplekt_curriculum_features.py').resolve(),
            Path(__file__).with_name('simplekt_mapped_queries.py').resolve()]},
        'limitation': 'Small ASSIST2017 reference diagnostic; final selected-checkpoint MPS checks for both datasets still required.'}
    path = ROOT / f'results/model_sequence_20261107/simplekt_feature_assembly_{device}_checks.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    device = parser.parse_args().device
    if device == 'cpu':
        check(device)
    else:
        with ExitStack() as stack:
            for path in [ROOT / 'results/simplekt_sweeps_20261107/worker.lock',
                         ROOT / 'results/model_sequence_20261107/gpu_worker.lock']:
                lock = stack.enter_context(path.open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check(device)
