"""Read-only checkpoint/mapping diagnostics; no inference or fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SAVED = ROOT / 'results/curriculum_kt_assist2012_k1000_20261107'
EXPANDED = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2012'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    protocol = json.loads((SAVED / 'protocol.json').read_text())
    mapping = np.load(SAVED / 'mapping.npz')
    support = mapping['counts'] > 0
    weight = mapping['item_overlap'].astype(float)
    children = support.sum(axis=1)
    cross = support.sum(axis=0) > 1
    result = {
        'scope': 'Read-only CPU linear algebra and existing-score summaries; no fitting or new predictive evaluation',
        'source_protocol_sha256': digest(SAVED / 'protocol.json'),
        'mapping_sha256': digest(SAVED / 'mapping.npz'),
        'scores_sha256': digest(EXPANDED / 'scores_by_seed.csv'),
        'mapping': {
            'original_concepts': int(support.shape[0]),
            'induced_concepts': int(support.shape[1]),
            'edges': int(support.sum()),
            'child_count_min_median_max': [int(children.min()), float(np.median(children)), int(children.max())],
            'crossing_clusters': int(cross.sum()),
            'crossing_cluster_reference_item_fraction': float(np.mean(cross[mapping['q_to_induced']])),
            'item_weight_rank': int(np.linalg.matrix_rank(weight)),
            'rows_minus_unique_rows': int(len(weight) - len(np.unique(weight, axis=0))),
            'item_vs_frequency_total_variation_median': float(np.median(np.abs(weight-mapping['practice_frequency']).sum(axis=1)/2)),
        },
        'checkpoints': [],
        'existing_metric_means': {},
    }
    for run in protocol['runs']:
        if run['kind'] != 'induced':
            continue
        actual_hash = digest(run['checkpoint'])
        assert actual_hash == run['checkpoint_sha256']
        state = torch.load(run['checkpoint'], map_location='cpu', weights_only=True)
        key = next(k for k in state if k.endswith('out_layer.weight'))
        output = state[key].numpy().astype(np.float64)
        singular = np.linalg.svd(output, compute_uv=False)
        local_ranks = [int(np.linalg.matrix_rank(output[s])) for s in support]
        result['checkpoints'].append({
            'seed': run['seed'], 'checkpoint': run['checkpoint'],
            'sha256': actual_hash, 'matches_saved_hash': True,
            'output_key': key, 'shape': list(output.shape),
            'rank': int(np.linalg.matrix_rank(output)),
            'condition_number': float(singular[0]/singular[-1]),
            'all_mapped_submatrices_full_row_rank': bool(np.array_equal(local_ranks, children)),
        })
    with (EXPANDED / 'scores_by_seed.csv').open() as f:
        rows = list(csv.DictReader(f))
    fields = ['auc', 'within_original_macro_auc', 'within_original_weighted_auc', 'brier', 'log_loss']
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method and r['population'] == 'comparison_test']
        if selected:
            result['existing_metric_means'][method] = {
                f: float(np.mean([float(r[f]) for r in selected])) for f in fields
            }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
