"""Shared, CPU-only contracts for the authorized FoundationalASSIST preparation."""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/foundationalassist/derived'
RESULTS = ROOT / 'results/foundationalassist_20261108'
QUESTIONS = DATA / 'text_audit_v1/questions.csv'
EVENTS = DATA / 'canonical_v1/canonical_events.csv.gz'
SELECTION = DATA / 'selection_v1'
PROTOCOL = RESULTS / 'preparation_protocol_v1.json'
MODELS = [('text-embedding-3-small', 768), ('text-embedding-3-large', 1536)]
ROLES = ['train', 'validation', 'calibration', 'test']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, columns, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def versions():
    return {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn', 'openai']}


def verify_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    for name, entry in manifest['files'].items():
        assert sha(path.parent / name) == entry['sha256'], name
    return manifest


def questions():
    with QUESTIONS.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len({r['problem_id'] for r in rows}) == len(rows)
    for r in rows:
        assert hashlib.sha256(r['problem_text'].encode()).hexdigest() == r['text_sha256']
        assert len(json.loads(r['original_skill_ids'])) == int(r['original_skill_count'])
    return sorted(rows, key=lambda r: int(r['problem_id']))


def freeze():
    if PROTOCOL.exists():
        return load_contract()
    assert not SELECTION.exists(), 'Incomplete selection already exists; inspect before resume.'
    verify_manifest(DATA / 'text_audit_v1/audit.json')
    verify_manifest(DATA / 'canonical_v1/manifest.json')
    rows = questions()
    item_lookup = {r['problem_id']: r for r in rows}
    observed, eligible = set(), set()
    counts = Counter()
    role_events = Counter()
    role_students, train_items = defaultdict(set), set()
    with gzip.open(EVENTS, 'rt', newline='') as f:
        for event in csv.DictReader(f):
            item = item_lookup[event['problem_id']]
            observed.add(event['problem_id'])
            if item['verified_text_only_candidate'] != 'True':
                counts['excluded_incomplete_text_events'] += 1
            elif item['original_skill_count'] != '1':
                counts['excluded_multi_skill_text_events'] += 1
            else:
                eligible.add(event['problem_id'])
                counts['retained_events'] += 1
                role_events[event['split']] += 1
                role_students[event['split']].add(event['student_id'])
                if event['split'] == 'train':
                    train_items.add(event['problem_id'])
    kept = [r for r in rows if r['problem_id'] in eligible]
    skills = sorted({json.loads(r['original_skill_ids'])[0] for r in kept}, key=int)
    original = {s: i for i, s in enumerate(skills)}
    SELECTION.mkdir(mode=0o700)
    items = [{'problem_id': r['problem_id'], 'question_index': i,
              'original_skill_id': json.loads(r['original_skill_ids'])[0],
              'original_concept': original[json.loads(r['original_skill_ids'])[0]],
              'text_sha256': r['text_sha256'], 'training_observed': r['problem_id'] in train_items}
             for i, r in enumerate(kept)]
    write_csv(SELECTION / 'items.csv', list(items[0]), items)
    ledger = []
    for r in rows:
        reason = ('incomplete_text' if r['verified_text_only_candidate'] != 'True'
                  else 'multiple_original_skills' if r['original_skill_count'] != '1'
                  else 'no_valid_canonical_events' if r['problem_id'] not in observed
                  else 'retained')
        ledger.append({'problem_id': r['problem_id'], 'original_skill_ids': r['original_skill_ids'],
                       'observed_in_canonical': r['problem_id'] in observed, 'disposition': reason})
    write_csv(SELECTION / 'item_eligibility.csv', list(ledger[0]), ledger)
    dump(SELECTION / 'original_vocabulary.json', original)
    assert sum(counts.values()) == 1709601
    assert counts['retained_events'] == 1294486 and len(kept) == 2317 and len(original) == 177
    protocol = {
        'created_utc': utc(), 'schema_version': 1,
        'authorization': 'User authorized API embeddings, CPU clustering and relabeling; no GPU needed. Reuse the existing original-project API key in place.',
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in [QUESTIONS, EVENTS,
                    DATA / 'text_audit_v1/audit.json', DATA / 'canonical_v1/manifest.json',
                    DATA / 'canonical_v1/student_split.csv']},
        'selection_files': {p.name: sha(p) for p in sorted(SELECTION.iterdir())},
        'shared_population': 'Valid unique canonical events on text-complete items with exactly one released original skill; preserve the original skill, without choosing a primary skill or expanding responses into pseudo-events.',
        'exclusions': 'Identical for original and induced conditions. Full source/KC sets remain in canonical_v1. Item eligibility is determined without response outcomes.',
        'counts': dict(counts), 'items': len(kept), 'original_concepts': len(original),
        'role_counts': {r: {'events': role_events[r], 'students': len(role_students[r])} for r in ROLES},
        'items_without_training_events': len(eligible - train_items),
        'split': 'Reuse canonical_v1 assignments without reshuffling: train/validation/calibration/test = 48/16/16/20 percent, seed42.',
        'embedding': {'configurations': [{'model': m, 'dimensions': n} for m, n in MODELS],
                      'scope': 'All 2594 text-complete audited item texts, with one output row per item; primary clustering filters to the shared eligible item population.',
                      'batch_items': 128, 'batch_utf8_bytes_limit': 100000, 'item_utf8_bytes_limit': 8191,
                      'length_policy': 'No truncation. UTF-8 bytes conservatively upper-bound byte-BPE tokens; these short texts need no extra tokenizer installation. Record actual aggregate API token usage.',
                      'transmitted_fields': ['problem_text'], 'endpoint': 'https://api.openai.com/v1/embeddings',
                      'reference': 'https://developers.openai.com/api/reference/resources/embeddings/methods/create'},
        'clustering': {'model': 'text-embedding-3-large', 'dimensions': 1536, 'pca_components': 100,
                       'pca_svd_solver': 'full', 'pca_whiten': False, 'n_clusters': len(original),
                       'random_state': 42, 'n_init': 10, 'algorithm': 'lloyd', 'max_iter': 300, 'tol': 0.0001,
                       'input_normalization': 'None beyond API-returned vectors, matching the original pipeline.',
                       'fit_population': 'All 2317 eligible known items; content-only transductive fitting, not unseen-item generalization.',
                       'selection': 'One declared condition; no response-based taxonomy selection.'},
        'export': {'sequence_length': 200, 'padding': -1,
                   'history': 'Filter both taxonomies first, preserve per-student chronological/source-row order; non-overlapping chunks of at most200 events.',
                   'targets': 'A valid event is predictable only when the immediately previous position in the same chunk is valid; first position masked.',
                   'event_array_identity': 'source_row is the numeric event_id in NPZ arrays; shared events.csv.gz maps it to canonical event_id and source_event_id.',
                   'durations': 'Not present in the source; do not fabricate.'},
        'compute': 'OpenAI embedding API plus local CPU preprocessing/clustering. No GPU or training.',
        'created_by_source_sha256': sha(Path(__file__)), 'versions': versions(),
    }
    dump(PROTOCOL, protocol)
    return protocol


def load_contract():
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in protocol['selection_files'].items():
        assert sha(SELECTION / name) == digest, name
    return protocol


def selected_items():
    with (SELECTION / 'items.csv').open(newline='') as f:
        return list(csv.DictReader(f))


if __name__ == '__main__':
    os.umask(0o077)
    p = freeze()
    print(json.dumps({'protocol': str(PROTOCOL), 'counts': p['counts'], 'items': p['items'],
                      'original_concepts': p['original_concepts'], 'role_counts': p['role_counts']}, indent=2))
