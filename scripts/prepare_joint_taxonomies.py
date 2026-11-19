"""CPU content-only PCA/K cache, preserving all frozen events and student splits."""
import argparse
from decimal import Decimal
import itertools
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from joint_tuning_common import ROOT, SOURCE, DATA, OUT, DATASETS, read, dump, identity, sha, now

PCA_DIMS = [10, 50, 100]
KS = {'assist2012': [162, 300, 1000], 'assist2017': [68, 100, 200],
      'foundationalassist': [177, 300, 600]}


def inputs(ds):
    old = read(ROOT / 'results/simplekt_sweeps_20261107/protocol.json')
    if ds != 'foundationalassist':
        record = old['datasets'][ds]
        with np.load(record['mapping']['path']) as a:
            mapping = dict(a)
        qpath = SOURCE / 'clustering/data_subsets' / ds / 'questions.csv'
        epath = SOURCE / 'clustering/embeddings' / ds / 'openai_text-embedding-3-large_1536.pt'
        try:
            questions = pd.read_csv(qpath, dtype=str, encoding='utf-8-sig')
        except UnicodeDecodeError:
            questions = pd.read_csv(qpath, dtype=str, encoding='ISO-8859-1')
        col = 'problem_id' if ds == 'assist2012' else 'problemId'
        questions = questions.dropna(subset=[col, 'problem_body']).reset_index(drop=True)
        embedded = torch.load(epath, map_location='cpu', weights_only=True).numpy()
        assert len(questions) == len(embedded) and embedded.shape[1] == 1536
        # Match the existing known-item protocol: average repeated question rows.
        numeric_ids = questions[col].map(Decimal)
        assert all(x.is_finite() and x == x.to_integral_value() for x in numeric_ids)
        labels = numeric_ids.map(lambda x: str(int(x))).to_numpy()
        unique, inverse = np.unique(labels, return_inverse=True)
        means = np.zeros((len(unique), 1536), np.float64)
        np.add.at(means, inverse, embedded)
        means /= np.bincount(inverse)[:, None]
        if ds == 'assist2012':
            raw_ids = mapping['raw_question_ids'].astype(str)
            pairs = np.column_stack([mapping['q_to_original'], np.arange(len(raw_ids))])
            extra = []
        else:
            vpath = SOURCE / 'data/assist2017/keyid2idx.json'
            vocab = read(vpath)['questions']
            raw_ids = np.asarray([k for k, v in sorted(vocab.items(), key=lambda kv: int(kv[1]))])
            assert sorted(map(int, vocab.values())) == list(range(len(raw_ids)))
            pairs = mapping['pairs'][:, :2]
            extra = [vpath]
        lookup = {key: i for i, key in enumerate(unique)}
        missing = set(raw_ids) - set(lookup)
        assert not missing, f'{ds}: {len(missing)} frozen questions have no aligned embedding'
        vectors = means[[lookup[key] for key in raw_ids]].astype(np.float32)
        sources = [qpath, epath, ROOT / 'results/simplekt_sweeps_20261107/protocol.json', *extra]
    else:
        handoff = read(ROOT / 'results/foundationalassist_20261108/training_data_handoff_v1.json')
        for name, h in handoff['artifacts'].items():
            assert sha(ROOT / name) == h
        parent = ROOT / 'data/foundationalassist/derived/paired_exports_v1'
        with np.load(parent / 'mapping.npz') as a:
            mapping = dict(a)
        raw_ids = mapping['raw_question_ids'].astype(str)
        pairs = np.column_stack([mapping['q_to_original'], np.arange(len(raw_ids))])
        emb = ROOT / 'data/foundationalassist/derived/openai_embeddings_v1/text-embedding-3-large_1536'
        items = read(emb / 'items.json'); lookup = {str(r['problem_id']): i for i, r in enumerate(items)}
        vectors = np.load(emb / 'embeddings.npy')[[lookup[key] for key in raw_ids]]
        record = {'original_concepts': 177, 'questions': len(raw_ids),
            'files': {s: identity(parent / 'paired' / f'{s}_sequences.npz')
                      for s in ['train', 'validation', 'calibration', 'test']},
            'mapping': identity(parent / 'mapping.npz')}
        sources = [emb / 'items.json', emb / 'embeddings.npy', ROOT / 'results/foundationalassist_20261108/training_data_handoff_v1.json']
    assert vectors.shape == (record['questions'], 1536) and np.isfinite(vectors).all()
    assert len(raw_ids) == len(set(raw_ids))
    assert sha(record['mapping']['path']) == record['mapping']['sha256']
    for file in record['files'].values():
        assert sha(file['path']) == file['sha256']
    return record, vectors, raw_ids, pairs.astype(np.int64), sources


def prepare(ds):
    folder = DATA / ds; folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        for item in manifest['taxonomies'].values():
            assert sha(item['mapping']['path']) == item['mapping']['sha256']
        return
    record, vectors, raw_ids, pairs, sources = inputs(ds)
    # Store one aligned matrix in this workspace; no external encoder runtime.
    np.save(folder / 'embeddings.npy', vectors)
    dump(folder / 'question_ids.json', raw_ids.tolist())
    np.save(folder / 'original_question_pairs.npy', pairs)
    arrays = {}
    for split in ['train', 'validation', 'calibration']:
        with np.load(record['files'][split]['path']) as a:
            arrays[split] = dict(a)
    groups = {s: set(a['uid']) for s, a in arrays.items()}
    assert all(not groups[a] & groups[b] for a, b in itertools.combinations(groups, 2))
    train = arrays['train']; valid = train['questions'] >= 0
    trq, trc = train['questions'][valid], train['original_concepts'][valid]
    reducer = PCA(n_components=100, svd_solver='full', whiten=False)
    reduced = reducer.fit_transform(vectors.astype(np.float64))
    np.savez_compressed(folder / 'pca.npz', scores=reduced, components=reducer.components_,
                        mean=reducer.mean_, variance_ratio=reducer.explained_variance_ratio_)
    manifest = {'dataset': ds, 'created_utc': now(), 'original_concepts': record['original_concepts'],
        'questions': len(raw_ids), 'files': record['files'], 'taxonomies': {},
        'embedding': 'text-embedding-3-large/1536; unique-question mean; no added normalization',
        'embedding_alignment': 'Exact row count of filtered question table; explicit item-ID join; no missing frozen items',
        'known_item_transductive': True, 'heldout_responses_loaded': False,
        'population_events_order_splits_unchanged': True,
        'source_sha256': {str(p): sha(p) for p in sources},
        'aligned_embeddings': identity(folder / 'embeddings.npy'), 'pca': identity(folder / 'pca.npz')}
    for dim, k in itertools.product(PCA_DIMS, KS[ds]):
        key = f'pca{dim}_k{k}'; path = folder / key / 'mapping.npz'; path.parent.mkdir(exist_ok=True)
        if not path.exists():
            estimator = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300,
                               tol=.0001, algorithm='lloyd')
            clusters = estimator.fit_predict(reduced[:, :dim])
            assert len(np.unique(clusters)) == k
            nc = record['original_concepts']; counts = np.zeros((nc, k), np.int64)
            np.add.at(counts, (pairs[:, 0], clusters[pairs[:, 1]]), 1)
            practice = np.zeros_like(counts); np.add.at(practice, (trc, clusters[trq]), 1)
            empty = practice.sum(1) == 0; practice[empty] = counts[empty]
            assert (counts.sum(1) > 0).all() and (practice.sum(1) > 0).all()
            frequency = practice / practice.sum(1, keepdims=True)
            assert not np.any(frequency[counts == 0])
            np.savez_compressed(path, q_to_induced=clusters, raw_question_ids=raw_ids,
                counts=counts, practice_counts=practice, practice_frequency=frequency,
                item_overlap=counts/counts.sum(1, keepdims=True), pairs=pairs,
                centers=estimator.cluster_centers_)
        with np.load(path) as a:
            clusters = a['q_to_induced']
            assert len(clusters) == len(raw_ids) and len(np.unique(clusters)) == k
            assert np.allclose(a['practice_frequency'].sum(1), 1.)
            for split, data in arrays.items():
                v = data['questions'] >= 0
                # Relabeling is a view; response and event arrays are reused verbatim.
                assert (a['counts'][data['original_concepts'][v], clusters[data['questions'][v]]] > 0).all()
        manifest['taxonomies'][key] = {'pca_dim': dim, 'k': k, 'mapping': identity(path),
            'pca_variance_retained': float(reducer.explained_variance_ratio_[:dim].sum()),
            'cluster_min': int(np.bincount(clusters).min()), 'cluster_max': int(np.bincount(clusters).max())}
        dump(folder / 'progress.json', manifest)
        print(json.dumps({'dataset': ds, 'taxonomy': key, 'state': 'verified'}), flush=True)
    dump(manifest_path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args(); torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for dataset in ([args.dataset] if args.dataset else DATASETS):
            prepare(dataset)
    dump(OUT / 'taxonomy_preparation.json', {'state': 'complete', 'completed_utc': now(),
         'manifests': {ds: identity(DATA / ds / 'manifest.json') for ds in ([args.dataset] if args.dataset else DATASETS)}})
