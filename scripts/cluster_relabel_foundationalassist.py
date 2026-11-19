"""CPU-only event exports, PCA/K-means and paired label exports. No API or torch."""
import os
for _variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_variable] = '1'

import argparse
import csv
import gzip
import itertools
import json
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from foundationalassist_preparation import (
    DATA, EVENTS, PROTOCOL, ROLES, SELECTION, dump, load_contract,
    selected_items, sha, utc, verify_manifest, versions, write_csv,
)

SHARED = DATA / 'matched_events_v1'
CLUSTERS = DATA / 'large1536_pca100_k177_v1'
EXPORTS = DATA / 'paired_exports_v1'
FIELDS = ['questions', 'original_concepts', 'responses', 'timestamps', 'event_id', 'source_row']


def artifacts(folder):
    return {str(p.relative_to(folder)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'manifest.json'}


def staged(destination):
    assert not destination.exists(), 'Output exists; verify/reuse rather than overwrite.'
    return Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=destination.parent))


def sequence_counts(arrays):
    valid = arrays['questions'] >= 0
    mask = np.zeros_like(valid)
    mask[:, 1:] = valid[:, 1:] & valid[:, :-1]
    assert np.array_equal(arrays['target_mask'], mask)
    assert (np.diff(valid.astype(int), axis=1) <= 0).all()
    for name in FIELDS:
        assert np.array_equal(arrays[name] >= 0, valid), name
    assert np.isin(arrays['responses'][valid], [0, 1]).all()
    assert len(np.unique(arrays['event_id'][valid])) == int(valid.sum())
    return {'students': len(np.unique(arrays['uid'])), 'chunks': len(arrays['uid']),
            'events': int(valid.sum()), 'predictable_targets': int(mask.sum())}


def shared_events():
    protocol = load_contract()
    if SHARED.exists():
        return verify_manifest(SHARED / 'manifest.json')
    start = time.monotonic()
    items = {r['problem_id']: r for r in selected_items()}
    with (DATA / 'canonical_v1/student_split.csv').open() as f:
        assignments = {r['student_id']: r['split'] for r in csv.DictReader(f)}
    students = {student: i for i, student in enumerate(sorted(assignments))}
    stage = staged(SHARED)
    chunks = {role: [] for role in ROLES}
    role_counts, excluded = Counter(), Counter()
    current_student, current_split, part = None, None, []

    def flush():
        if not part:
            return
        row = {'uid': students[current_student]}
        for index, name in enumerate(FIELDS):
            values = np.full(200, -1, dtype=np.int64)
            values[:len(part)] = [p[index] for p in part]
            row[name] = values
        mask = np.zeros(200, bool)
        mask[1:len(part)] = True
        row['target_mask'] = mask
        chunks[current_split].append(row)
        part.clear()

    columns = ['event_id', 'source_event_id', 'source_row', 'student_id', 'uid', 'problem_id',
               'question_index', 'original_skill_ids', 'original_concept', 'response',
               'end_time_utc', 'timestamp_ms', 'split', 'source_duplicate_count']
    try:
        with gzip.open(EVENTS, 'rt', newline='') as source, gzip.open(stage / 'events.csv.gz', 'wt', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for event in csv.DictReader(source):
                item = items.get(event['problem_id'])
                if item is None:
                    excluded['incomplete_text' if event['verified_text_only_candidate'] != 'True' else 'multiple_original_skills'] += 1
                    continue
                assert event['original_skill_count'] == '1'
                assert json.loads(event['original_skill_ids']) == [item['original_skill_id']]
                assert assignments[event['student_id']] == event['split']
                if current_student != event['student_id']:
                    flush()
                    current_student, current_split = event['student_id'], event['split']
                timestamp = int(datetime.fromisoformat(event['end_time_utc']).timestamp() * 1000)
                part.append((int(item['question_index']), int(item['original_concept']), int(event['response']),
                             timestamp, int(event['source_row']), int(event['source_row'])))
                writer.writerow({'event_id': event['event_id'], 'source_event_id': event['source_event_id'],
                                 'source_row': event['source_row'], 'student_id': event['student_id'],
                                 'uid': students[event['student_id']], 'problem_id': event['problem_id'],
                                 'question_index': item['question_index'], 'original_skill_ids': event['original_skill_ids'],
                                 'original_concept': item['original_concept'], 'response': event['response'],
                                 'end_time_utc': event['end_time_utc'], 'timestamp_ms': timestamp,
                                 'split': event['split'], 'source_duplicate_count': event['source_duplicate_count']})
                role_counts[current_split] += 1
                if len(part) == 200:
                    flush()
            flush()
        counts, memberships = {}, {}
        (stage / 'original').mkdir()
        for role in ROLES:
            arrays = {k: np.stack([r[k] for r in chunks[role]]) for k in chunks[role][0]}
            arrays['sequence_id'] = np.arange(len(chunks[role]), dtype=np.int64)
            counts[role] = sequence_counts(arrays)
            assert counts[role]['events'] == protocol['role_counts'][role]['events'] == role_counts[role]
            assert counts[role]['students'] == protocol['role_counts'][role]['students']
            memberships[role] = set(arrays['uid'])
            assert arrays['questions'].max() < protocol['items']
            assert arrays['original_concepts'].max() < protocol['original_concepts']
            np.savez_compressed(stage / f'{role}_sequences.npz', **arrays)
            baseline = {k: v for k, v in arrays.items() if k != 'original_concepts'}
            baseline['concepts'] = arrays['original_concepts']
            np.savez_compressed(stage / 'original' / f'{role}_sequences.npz', **baseline)
        assert all(not memberships[a] & memberships[b] for a, b in itertools.combinations(ROLES, 2))
        assert sum(role_counts.values()) == protocol['counts']['retained_events']
        assert excluded['incomplete_text'] == protocol['counts']['excluded_incomplete_text_events']
        assert excluded['multiple_original_skills'] == protocol['counts']['excluded_multi_skill_text_events']
        dump(stage / 'student_vocabulary.json', students)
        for name in ['items.csv', 'original_vocabulary.json', 'item_eligibility.csv']:
            shutil.copyfile(SELECTION / name, stage / name)
        report = {'created_utc': utc(), 'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(Path(__file__)),
                  'device': 'cpu', 'counts': counts, 'excluded_events': dict(excluded),
                  'elapsed_seconds': time.monotonic() - start,
                  'checks': {'exact_common_population': True, 'disjoint_student_roles': True,
                             'events_unique_within_role': True, 'same_fixed_student_assignments': True,
                             'original_labels_preserved': True, 'target_masks_verified': True},
                  'files': artifacts(stage)}
        dump(stage / 'manifest.json', report)
        stage.rename(SHARED)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def fit_labels(vectors, config):
    with threadpool_limits(limits=1):
        pca = PCA(n_components=config['pca_components'], svd_solver=config['pca_svd_solver'],
                  whiten=config['pca_whiten'], random_state=config['random_state'])
        reduced = pca.fit_transform(vectors)
        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=config['random_state'],
                        n_init=config['n_init'], algorithm=config['algorithm'],
                        max_iter=config['max_iter'], tol=config['tol'])
        labels = kmeans.fit_predict(reduced)
    return labels, reduced, pca, kmeans


def cluster():
    protocol = load_contract()
    if CLUSTERS.exists():
        return verify_manifest(CLUSTERS / 'manifest.json')
    start = time.monotonic()
    embedding_root = DATA / 'openai_embeddings_v1'
    verify_manifest(embedding_root / 'manifest.json')
    config = protocol['clustering']
    folder = embedding_root / f"{config['model']}_{config['dimensions']}"
    catalog = json.loads((folder / 'items.json').read_text())
    lookup = {row['problem_id']: i for i, row in enumerate(catalog)}
    items = selected_items()
    for item in items:
        assert catalog[lookup[item['problem_id']]]['text_sha256'] == item['text_sha256']
    full = np.load(folder / 'embeddings.npy', allow_pickle=False)
    values = full[[lookup[r['problem_id']] for r in items]]
    assert values.shape == (protocol['items'], 1536) and np.isfinite(values).all()
    labels, reduced, pca, km = fit_labels(values, config)
    labels2, reduced2, _, km2 = fit_labels(values.copy(), config)
    assert np.array_equal(labels, labels2), 'Deterministic rerun failed.'
    assert np.array_equal(reduced, reduced2)
    assert len(np.unique(labels)) == config['n_clusters']
    assert np.isfinite(reduced).all() and np.isfinite(km.cluster_centers_).all()
    stage = staged(CLUSTERS)
    try:
        mapping = [{**r, 'induced_concept': int(k)} for r, k in zip(items, labels)]
        write_csv(stage / 'item_clusters.csv', list(mapping[0]), mapping)
        np.save(stage / 'labels.npy', labels, allow_pickle=False)
        np.save(stage / 'pca_scores.npy', reduced, allow_pickle=False)
        np.savez_compressed(stage / 'pca_kmeans_parameters.npz', mean=pca.mean_, components=pca.components_,
                            explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_,
                            centers=km.cluster_centers_)
        sizes = np.bincount(labels, minlength=config['n_clusters'])
        write_csv(stage / 'cluster_sizes.csv', ['induced_concept', 'items'],
                  [{'induced_concept': i, 'items': int(n)} for i, n in enumerate(sizes)])
        report = {'created_utc': utc(), 'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(Path(__file__)),
                  'embedding_manifest_sha256': sha(embedding_root / 'manifest.json'), 'config': config,
                  'versions': versions(), 'device': 'cpu', 'threads': 1, 'input_dtype': str(values.dtype),
                  'items': len(items), 'dimensions': values.shape[1], 'realized_clusters': len(sizes),
                  'pca_explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
                  'inertia': float(km.inertia_), 'iterations': int(km.n_iter_),
                  'cluster_size_min': int(sizes.min()), 'cluster_size_max': int(sizes.max()),
                  'singleton_clusters': int((sizes == 1).sum()),
                  'duplicate_text_item_count': len(items) - len({r['text_sha256'] for r in items}),
                  'deterministic_refit_labels_equal': True, 'deterministic_refit_PCA_equal': True,
                  'refit_inertia_difference': float(abs(km.inertia_ - km2.inertia_)),
                  'elapsed_seconds_including_refit': time.monotonic() - start, 'files': artifacts(stage)}
        dump(stage / 'manifest.json', report)
        stage.rename(CLUSTERS)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def induced_arrays(shared, q_to_induced):
    result = {k: v.copy() for k, v in shared.items()}
    valid = result['questions'] >= 0
    induced = np.full_like(result['questions'], -1)
    induced[valid] = q_to_induced[result['questions'][valid]]
    result['induced_concepts'] = induced
    assert np.array_equal(induced >= 0, valid)
    return result


def export():
    protocol = load_contract()
    if EXPORTS.exists():
        return verify_manifest(EXPORTS / 'manifest.json')
    shared_manifest = verify_manifest(SHARED / 'manifest.json')
    verify_manifest(CLUSTERS / 'manifest.json')
    start = time.monotonic()
    items = selected_items()
    q_to_original = np.asarray([int(r['original_concept']) for r in items], dtype=np.int64)
    q_to_induced = np.load(CLUSTERS / 'labels.npy', allow_pickle=False).astype(np.int64)
    assert len(q_to_original) == len(q_to_induced) == protocol['items']
    stage = staged(EXPORTS)
    try:
        for name in ['paired', 'original', 'induced']:
            (stage / name).mkdir()
        counts, membership = {}, {}
        practices = np.zeros((protocol['original_concepts'], protocol['clustering']['n_clusters']), np.int64)
        for role in ROLES:
            with np.load(SHARED / f'{role}_sequences.npz', allow_pickle=False) as loaded:
                shared = dict(loaded)
            paired = induced_arrays(shared, q_to_induced)
            valid = paired['questions'] >= 0
            assert np.array_equal(paired['original_concepts'][valid], q_to_original[paired['questions'][valid]])
            counts[role] = sequence_counts(paired)
            assert counts[role] == shared_manifest['counts'][role]
            membership[role] = set(paired['uid'])
            original = {k: v for k, v in paired.items() if k not in ['original_concepts', 'induced_concepts']}
            original['concepts'] = paired['original_concepts']
            induced = {k: v for k, v in original.items() if k != 'concepts'}
            induced['concepts'] = paired['induced_concepts']
            for key in set(original) - {'concepts'}:
                assert np.array_equal(original[key], induced[key]), (role, key)
            np.savez_compressed(stage / 'paired' / f'{role}_sequences.npz', **paired)
            np.savez_compressed(stage / 'original' / f'{role}_sequences.npz', **original)
            np.savez_compressed(stage / 'induced' / f'{role}_sequences.npz', **induced)
            if role == 'train':
                np.add.at(practices, (paired['original_concepts'][valid], paired['induced_concepts'][valid]), 1)
        assert all(not membership[a] & membership[b] for a, b in itertools.combinations(ROLES, 2))
        overlap = np.zeros_like(practices)
        np.add.at(overlap, (q_to_original, q_to_induced), 1)
        assert (overlap.sum(1) > 0).all()
        effective = practices.copy()
        missing = effective.sum(1) == 0
        effective[missing] = overlap[missing]
        np.savez_compressed(stage / 'mapping.npz', counts=overlap, raw_training_event_counts=practices,
                            practice_counts=effective, item_overlap=overlap / overlap.sum(1, keepdims=True),
                            practice_frequency=effective / effective.sum(1, keepdims=True),
                            q_to_original=q_to_original, q_to_induced=q_to_induced,
                            raw_question_ids=np.asarray([r['problem_id'] for r in items]))
        shutil.copyfile(CLUSTERS / 'item_clusters.csv', stage / 'item_mapping.csv')
        report = {'created_utc': utc(), 'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(Path(__file__)),
                  'shared_manifest_sha256': sha(SHARED / 'manifest.json'),
                  'cluster_manifest_sha256': sha(CLUSTERS / 'manifest.json'), 'device': 'cpu',
                  'counts': counts, 'items': len(items), 'original_concepts': len(overlap), 'induced_concepts': overlap.shape[1],
                  'mapping_fit_population': 'Training-event frequencies only; for original concepts absent from training, use content-only item-overlap fallback as in the current campaign.',
                  'original_concepts_without_training_events': int(missing.sum()),
                  'induced_clusters_without_training_events': int((practices.sum(0) == 0).sum()),
                  'checks': {'same_events': True, 'same_responses': True, 'same_order': True,
                             'same_items': True, 'same_students_and_splits': True, 'same_target_masks': True,
                             'only_taxonomy_fields_differ': True, 'no_pseudo_event_expansion': True,
                             'no_heldout_response_mapping_fit': True},
                  'elapsed_seconds': time.monotonic() - start, 'files': artifacts(stage)}
        dump(stage / 'manifest.json', report)
        stage.rename(EXPORTS)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['events', 'cluster', 'export'])
    args = parser.parse_args()
    result = {'events': shared_events, 'cluster': cluster, 'export': export}[args.stage]()
    print(json.dumps({k: v for k, v in result.items() if k not in ['files', 'config']}, indent=2), flush=True)
