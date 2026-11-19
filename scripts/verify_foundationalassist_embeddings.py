"""Verify API journals, saved matrices, PCA parameters and cluster labels offline."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'

import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from threadpoolctl import threadpool_limits

from foundationalassist_preparation import DATA, PROTOCOL, RESULTS, canonical_hash, dump, sha, utc, verify_manifest


def main():
    started = time.monotonic()
    root = DATA / 'openai_embeddings_v1'
    cluster = DATA / 'large1536_pca100_k177_v1'
    manifest = verify_manifest(root / 'manifest.json')
    cluster_report = verify_manifest(cluster / 'manifest.json')
    protocol = json.loads(PROTOCOL.read_text())
    with (DATA / 'text_audit_v1/questions.csv').open() as f:
        questions = sorted([r for r in csv.DictReader(f) if r['verified_text_only_candidate'] == 'True'],
                           key=lambda r: int(r['problem_id']))
    checks, reports = {}, []
    accepted_tokens = 0
    request_ids = set()
    for config in protocol['embedding']['configurations']:
        model, dimensions = config['model'], config['dimensions']
        folder = root / f'{model}_{dimensions}'
        matrix = np.load(folder / 'embeddings.npy', allow_pickle=False)
        ids = json.loads((folder / 'items.json').read_text())
        assert ids == [{'problem_id': r['problem_id'], 'text_sha256': r['text_sha256']} for r in questions]
        assert matrix.shape == (len(questions), dimensions)
        assert np.isfinite(matrix).all() and (np.linalg.norm(matrix, axis=1) > 0).all()
        batches = sorted(folder.glob('batch_*.npz'))
        assert len(batches) == 21
        offset = 0
        model_tokens = 0
        for batch_path in batches:
            with np.load(batch_path, allow_pickle=False) as loaded:
                values, metadata = loaded['values'], json.loads(str(loaded['metadata']))
            journal_path = batch_path.with_suffix('.response.json.gz')
            with gzip.open(journal_path, 'rt') as f:
                journal = json.load(f)
            raw = journal['response']
            records = raw['data']
            assert raw['model'] == model
            assert sorted(x['index'] for x in records) == list(range(len(records)))
            reference = np.asarray([x['embedding'] for x in sorted(records, key=lambda r: r['index'])], dtype=np.float32)
            assert np.array_equal(values, reference), 'Raw API output was altered.'
            assert np.array_equal(values, matrix[offset:offset + len(values)])
            assert values.shape[1] == dimensions
            expected = questions[offset:offset + len(values)]
            assert metadata['items'] == [{'problem_id': r['problem_id'], 'text_sha256': r['text_sha256']} for r in expected]
            assert metadata['input_sha256'] == canonical_hash([r['problem_text'] for r in expected])
            assert metadata['response_journal_sha256'] == sha(journal_path)
            assert metadata['values_sha256'] == hashlib.sha256(values.tobytes()).hexdigest()
            assert metadata['request_id'] == journal['request_id']
            assert journal['request_id'] not in request_ids
            request_ids.add(journal['request_id'])
            assert raw['usage'] == metadata['usage']
            model_tokens += raw['usage']['total_tokens']
            offset += len(values)
        assert offset == len(questions)
        accepted_tokens += model_tokens
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        reports.append({'model': model, 'shape': list(matrix.shape), 'validated_batches': len(batches),
                        'accepted_usage_tokens': model_tokens, 'minimum_norm': float(norms.min()),
                        'maximum_norm': float(norms.max()), 'maximum_unit_norm_deviation': float(abs(norms - 1).max())})
    assert accepted_tokens == manifest['api_usage_tokens_total']
    checks.update(all_saved_matrices_match_raw_API_values=True,all_model_dimensions_indices_and_text_hashes_match=True,
                  complete_item_order_alignment=True,all_vectors_finite_and_nonzero=True,
                  all_API_batch_hashes_valid=True,accepted_token_usage_matches_journals=True)
    with (cluster / 'item_clusters.csv').open() as f:
        items = list(csv.DictReader(f))
    folder = root / 'text-embedding-3-large_1536'
    ids = json.loads((folder / 'items.json').read_text())
    lookup = {r['problem_id']: i for i, r in enumerate(ids)}
    all_values = np.load(folder / 'embeddings.npy', allow_pickle=False)
    values = all_values[[lookup[r['problem_id']] for r in items]]
    labels = np.load(cluster / 'labels.npy', allow_pickle=False)
    scores = np.load(cluster / 'pca_scores.npy', allow_pickle=False)
    assert len(items) == len({r['problem_id'] for r in items}) == 2317
    assert scores.shape == (2317, 100) and len(np.unique(labels)) == 177
    assert np.array_equal(labels, [int(r['induced_concept']) for r in items])
    with np.load(cluster / 'pca_kmeans_parameters.npz', allow_pickle=False) as loaded:
        parameters = dict(loaded)
    with threadpool_limits(limits=1):
        reconstructed = (values - parameters['mean']) @ parameters['components'].T
        nearest, distances = pairwise_distances_argmin_min(scores, parameters['centers'])
    assert np.allclose(reconstructed, scores, atol=1e-5, rtol=1e-4)
    assert np.allclose(parameters['components'] @ parameters['components'].T, np.eye(100), atol=1e-5)
    assert np.array_equal(nearest, labels)
    assert np.isclose(np.sum(distances ** 2), cluster_report['inertia'], atol=.01, rtol=1e-5)
    assert cluster_report['deterministic_refit_labels_equal'] and cluster_report['deterministic_refit_PCA_equal']
    assert cluster_report['config'] == protocol['clustering']
    checks.update(PCA_parameters_reconstruct_scores=True,PCA_components_orthonormal=True,
                  saved_cluster_labels_equal_nearest_centers=True,cluster_mapping_complete=True,
                  frozen_cluster_configuration_preserved=True,deterministic_refit_passed=True)
    report = {'created_utc': utc(), 'passed': all(checks.values()), 'checks': checks, 'models': reports,
              'accepted_usage_tokens': accepted_tokens, 'successful_retained_API_requests': len(request_ids),
              'discarded_attempt': 'One earlier authenticated128-item response failed an implementation-only norm check before persistence; usage was not retained and is not included in accepted_usage_tokens.',
              'cluster_items': len(items), 'cluster_count': 177,
              'max_PCA_reconstruction_error': float(abs(reconstructed - scores).max()),
              'elapsed_seconds': time.monotonic() - started, 'source_sha256': sha(Path(__file__)),
              'embedding_manifest_sha256': sha(root / 'manifest.json'),
              'cluster_manifest_sha256': sha(cluster / 'manifest.json'), 'protocol_sha256': sha(PROTOCOL)}
    destination = RESULTS / 'embedding_cluster_verification_20261109.json'
    if destination.exists():
        destination = destination.with_name(destination.stem + '.recheck.' + str(time.time_ns()) + '.json')
    dump(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
