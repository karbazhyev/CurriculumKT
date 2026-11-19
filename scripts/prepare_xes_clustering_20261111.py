"""Versioned, response-blind XES question audit and CPU clustering preparation.

This produces item taxonomies, not a training-ready event/split handoff.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from foundationalassist_preparation import canonical_hash, dump, sha, utc, versions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/non_assistments_preparation_20261111'
OUT = DATA / 'xes3g5m'
RESULTS = ROOT / 'results/non_assistments_preparation_20261111'
PROTOCOL = RESULTS / 'xes_clustering_protocol_v1.json'
VISUAL = r'如图|下图|图中|上图|右图|左图|下表|上表|如下图|下面的图|图形|图所示|表格|看图'
IMAGE = r'(?:question|analysis)_\d+-image|<img\b|data:image|!\['


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    import re
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = DATA / 'release/XES3G5M/metadata/questions.json'
    routes = DATA / 'release/XES3G5M/metadata/kc_routes_map.json'
    image_path = DATA / 'raw/xes_question_image_ids.json'
    images = set(read(image_path))
    questions, names = read(source), read(routes)
    assert len(set(names.values())) == len(names), 'Ambiguous leaf-name to original-ID mapping'
    skill_ids = {name: sid for sid, name in names.items()}
    eligible, excluded = [], []
    for qid, value in sorted(questions.items(), key=lambda x: int(x[0])):
        assert isinstance(value['content'], str) and isinstance(value['options'], dict)
        assert all(isinstance(x, str) for x in value['options'].values())
        # Only prompt and uniformly included option text enter the embedding input.
        text = value['content'].strip()
        if value['options']:
            text += '\n' + '\n'.join(f'{key}. {value["options"][key]}' for key in sorted(value['options']))
        reasons = []
        if not text: reasons.append('empty_text')
        if qid in images or re.search(IMAGE, text, flags=re.I): reasons.append('question_image')
        if re.search(VISUAL, text): reasons.append('conservative_visual_reference')
        if re.search(r'<(?:img|table|svg|math)\b', text, flags=re.I): reasons.append('unrendered_markup')
        if len(text.encode()) > 8191: reasons.append('exceeds_conservative_embedding_limit')
        skills = sorted({skill_ids[r.split('----')[-1]] for r in value['kc_routes']}, key=int)
        if not skills: reasons.append('missing_original_skills')
        row = {'problem_id': qid, 'problem_text': text,
               'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'original_skill_ids': skills}
        if reasons: excluded.append({'problem_id': qid, 'reasons': reasons})
        else: eligible.append(row)
    assert len(eligible) + len(excluded) == len(questions)
    tokens_upper = sum(len(x['problem_text'].encode()) for x in eligible)
    audit_record = {
        'created_utc': utc(), 'total_questions': len(questions), 'eligible_questions': len(eligible),
        'excluded_questions': len(excluded),
        'exclusion_reason_counts_overlapping': dict(collections.Counter(r for x in excluded for r in x['reasons'])),
        'eligible_original_skills': len({s for x in eligible for s in x['original_skill_ids']}),
        'multi_skill_questions': sum(len(x['original_skill_ids']) > 1 for x in eligible),
        'distinct_eligible_texts': len({x['text_sha256'] for x in eligible}),
        'max_utf8_bytes': max(len(x['problem_text'].encode()) for x in eligible),
        'tokens_upper_bound_utf8_bytes': tokens_upper,
        'estimated_api_cost_upper_bound_usd': tokens_upper * .13 / 1000000,
        'price_per_million_tokens_usd': .13,
        'price_source': 'https://developers.openai.com/api/docs/models/text-embedding-3-large',
        'price_checked_date': '2026-11-11',
        'text_rule': 'Exclude every question with an archived prompt image, embedded image marker, or listed visual-reference phrase. Conservative exclusions can remove otherwise usable text-only items. Preserve Chinese and LaTeX; no OCR/translation/truncation.',
        'human_verified_every_item': False,
        'skill_join': 'Unique exact terminal KC name to original released KC ID; full multiskill sets preserved separately from embedding inputs.',
        'response_files_opened': False, 'known_item_transductive': True,
        'event_population_or_student_splits_ready': False,
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in [source, routes, image_path, DATA/'raw/xes_license_response.txt']},
    }
    protocol = {'dataset': 'xes3g5m', 'authorization': 'User requested preparing clustering for the new datasets now; existing OpenAI credential reuse authorization retained.',
        'eligible_items_sha256': canonical_hash(eligible), 'audit': audit_record,
        'embedding_model': 'text-embedding-3-large', 'embedding_dimensions': 1536,
        'encoder_reason': 'Same encoder/dimensions as the existing main joint tuning campaign. No encoder selection or supplied RoBERTa substitution.',
        'pca_dimensions': [10, 50, 100], 'pca': {'svd_solver': 'full', 'whiten': False, 'fit_once_components': 100},
        'k_candidates': [300, 600, 1000],
        'k_rationale': 'Bounded coarse/intermediate/fine grid fixed from text-only population size before fitting or response access; minimum mean item support at K=1000 must be at least five.',
        'kmeans': {'random_state': 42, 'n_init': 10, 'max_iter': 300, 'tol': .0001, 'algorithm': 'lloyd'},
        'no_test_or_development_response_selection': True, 'training_ready': False,
        'source_sha256': sha(Path(__file__))}
    assert len(eligible) >= 5000
    if PROTOCOL.exists():
        old = read(PROTOCOL)
        assert old['eligible_items_sha256'] == protocol['eligible_items_sha256']
        assert old['source_sha256'] == protocol['source_sha256']
        print(json.dumps({'state': 'existing_audit_verified', 'eligible': len(eligible)})); return
    dump(OUT/'eligible_questions.json', eligible)
    dump(OUT/'excluded_questions.json', excluded)
    dump(RESULTS/'xes_text_audit.json', audit_record)
    dump(PROTOCOL, protocol)
    print(json.dumps(audit_record), flush=True)


def load_contract():
    protocol = read(PROTOCOL)
    assert sha(Path(__file__)) == protocol['source_sha256']
    for name, digest in protocol['audit']['sources'].items(): assert sha(ROOT/name) == digest
    assert canonical_hash(read(OUT/'eligible_questions.json')) == protocol['eligible_items_sha256']
    return protocol


def cluster():
    protocol = load_contract()
    embedding = OUT/'openai_embeddings_v1'
    manifest = read(embedding/'manifest.json')
    for name, info in manifest['files'].items(): assert sha(embedding/name) == info['sha256']
    model = embedding/'text-embedding-3-large_1536'
    rows, items = read(OUT/'eligible_questions.json'), read(model/'items.json')
    assert [r['problem_id'] for r in rows] == [r['problem_id'] for r in items]
    vectors = np.load(model/'embeddings.npy', allow_pickle=False)
    assert vectors.shape == (len(rows), 1536) and np.isfinite(vectors).all()
    target = OUT/'taxonomies_v1'; target.mkdir(exist_ok=True)
    if (target/'manifest.json').exists():
        old = read(target/'manifest.json')
        assert old['protocol_sha256'] == sha(PROTOCOL)
        for name, info in old['files'].items(): assert sha(target/name) == info['sha256']
        print(json.dumps({'state': 'existing_taxonomies_verified', 'count': len(old['taxonomies'])})); return
    raw_ids = np.asarray([r['problem_id'] for r in rows])
    original_ids = sorted({s for r in rows for s in r['original_skill_ids']}, key=int)
    original_lookup = {s: i for i,s in enumerate(original_ids)}
    pairs = np.asarray([(original_lookup[s], i) for i,r in enumerate(rows) for s in r['original_skill_ids']], dtype=np.int64)
    np.save(target/'original_question_pairs.npy', pairs)
    dump(target/'original_skill_ids.json', original_ids)
    reducer = PCA(n_components=100, svd_solver='full', whiten=False)
    with threadpool_limits(limits=1):
        reduced = reducer.fit_transform(vectors.astype(np.float64))
        assert np.isfinite(reduced).all() and np.linalg.matrix_rank(reduced) >= 100
        np.savez_compressed(target/'pca.npz', scores=reduced, components=reducer.components_, mean=reducer.mean_, variance_ratio=reducer.explained_variance_ratio_)
        reports = []
        for dim in protocol['pca_dimensions']:
            for k in protocol['k_candidates']:
                folder = target/f'pca{dim}_k{k}'; folder.mkdir(exist_ok=True)
                km = KMeans(n_clusters=k, **protocol['kmeans'])
                clusters = km.fit_predict(reduced[:, :dim])
                assert len(np.unique(clusters)) == k
                counts = np.zeros((len(original_ids), k), dtype=np.int64)
                np.add.at(counts, (pairs[:,0], clusters[pairs[:,1]]), 1)
                assert (counts.sum(1) > 0).all()
                np.savez_compressed(folder/'mapping.npz', raw_question_ids=raw_ids, q_to_induced=clusters,
                    pairs=pairs, counts=counts, item_overlap=counts/counts.sum(1,keepdims=True), centers=km.cluster_centers_)
                with (folder/'question_clusters.csv').open('w', newline='') as f:
                    writer = csv.writer(f); writer.writerow(['question_id','induced_concept_id','original_skill_ids'])
                    writer.writerows((r['problem_id'], int(c), '|'.join(r['original_skill_ids'])) for r,c in zip(rows,clusters))
                sizes = np.bincount(clusters, minlength=k)
                report = {'pca_dimension': dim, 'k': k, 'inertia': float(km.inertia_), 'iterations': int(km.n_iter_),
                    'smallest_cluster': int(sizes.min()), 'largest_cluster': int(sizes.max()),
                    'singleton_clusters': int((sizes==1).sum()), 'mean_items_per_cluster': float(sizes.mean())}
                dump(folder/'fit.json', report); reports.append(report)
                print(json.dumps(report), flush=True)
    dump(target/'manifest.json', {'created_utc': utc(), 'protocol_sha256': sha(PROTOCOL),
        'embedding_manifest_sha256': sha(embedding/'manifest.json'), 'versions': versions(), 'device': 'CPU single thread',
        'taxonomies': reports, 'questions': len(rows), 'original_skills': len(original_ids),
        'response_files_opened': False, 'known_item_transductive': True,
        'training_ready': False, 'remaining': 'Event/split audit, training-only practice weights and model handoff must be prepared separately. Item taxonomies are frozen for reuse.',
        'files': {str(p.relative_to(target)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(target.rglob('*')) if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'cluster'])
    args = parser.parse_args()
    os.umask(0o077)
    {'audit': audit, 'cluster': cluster}[args.stage]()
