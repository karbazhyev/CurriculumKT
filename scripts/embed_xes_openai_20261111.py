"""Resumable original OpenAI embedding configurations; no local GPU or torch."""
import argparse
import fcntl
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI, APIError
from openai.types import CreateEmbeddingResponse

from prepare_xes_clustering_20261111 import (
    OUT as DATA, PROTOCOL, ROOT, canonical_hash, dump, load_contract, read, sha, utc, versions,
)

MODELS = [("text-embedding-3-large", 1536)]

def questions():
    return read(DATA / "eligible_questions.json")

OUT = DATA / 'openai_embeddings_v1'


def batches(rows, max_items=128, max_bytes=100000):
    batch, count = [], 0
    for row in rows:
        size = len(row['problem_text'].encode('utf-8'))
        if not row['problem_text'].strip() or not 0 < size <= 8191:
            raise ValueError('Input is empty or exceeds the conservative byte/token limit; no truncation allowed.')
        if batch and (len(batch) == max_items or count + size > max_bytes):
            yield batch
            batch, count = [], 0
        batch.append(row)
        count += size
    if batch:
        yield batch


def response_vectors(response, count, dimensions, model):
    if response.model != model:
        raise ValueError('Response model does not match requested model.')
    indices = [r.index for r in response.data]
    if sorted(indices) != list(range(count)):
        raise ValueError('Response indices are missing, duplicated or outside the request.')
    values = np.asarray([r.embedding for r in sorted(response.data, key=lambda r: r.index)], dtype=np.float32)
    if values.shape != (count, dimensions) or not np.isfinite(values).all():
        raise ValueError('Invalid response vector shape or nonfinite values.')
    # Preserve the original pipeline's raw API values. Near-unit norms can
    # differ at API output precision; report norms instead of normalizing or
    # imposing an extra unit-norm criterion absent from the frozen protocol.
    if (np.linalg.norm(values, axis=1) <= 0).any():
        raise ValueError('Embedding response contains a zero vector.')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-config', type=Path, help='Existing JSON configuration; key read in memory only.')
    parser.add_argument('--resume', action='store_true', help='Reuse validated completed batches.')
    parser.add_argument('--capture-one-response', action='store_true', help='Persist one raw response for offline validation and stop.')
    args = parser.parse_args()
    os.umask(0o077)
    protocol = load_contract()
    rows = questions()
    assert len(rows) == protocol['audit']['eligible_questions']
    planned = list(batches(rows))
    items = [{'problem_id': r['problem_id'], 'text_sha256': r['text_sha256']} for r in rows]
    identity = {'protocol_sha256': sha(PROTOCOL), 'items': items, 'models': MODELS,
                'batch_input_hashes': [canonical_hash([r['problem_text'] for r in batch]) for batch in planned]}
    run_id = canonical_hash(identity)
    OUT.mkdir(mode=0o700, exist_ok=True)
    with (OUT / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = OUT / 'request_identity.json'
        if identity_path.exists():
            assert json.loads(identity_path.read_text())['identity_sha256'] == run_id
            assert args.resume, 'Existing work found; explicitly resume to avoid duplicate requests.'
        else:
            dump(identity_path, {'identity_sha256': run_id, **identity})
        if (OUT / 'manifest.json').exists():
            manifest = json.loads((OUT / 'manifest.json').read_text())
            for name, info in manifest['files'].items():
                assert sha(OUT / name) == info['sha256']
            print(json.dumps({'state': 'already_complete_verified', 'new_requests': 0}))
            return
        key = os.environ.get('OPENAI_API_KEY', '').strip()
        if args.key_config:
            key = json.loads(args.key_config.read_text())['OPENAI_API_KEY'].strip()
        if not key:
            raise RuntimeError('Existing OpenAI API key is required.')
        # Explicit official endpoint; no inherited proxy/base-URL routing.
        client = OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=90)
        started = time.monotonic()
        model_reports = []
        for model, dimensions in MODELS:
            folder = OUT / f'{model}_{dimensions}'
            folder.mkdir(exist_ok=True)
            saved, metadata = [], []
            for i, batch in enumerate(planned):
                batch_id = canonical_hash({'identity': run_id, 'model': model, 'dimensions': dimensions, 'batch': i})
                target = folder / f'batch_{i:03d}.npz'
                if target.exists():
                    with np.load(target, allow_pickle=False) as cache:
                        info = json.loads(str(cache['metadata']))
                        assert info['batch_id'] == batch_id
                        values = cache['values']
                        assert values.shape == (len(batch), dimensions) and np.isfinite(values).all()
                        assert info['values_sha256'] == __import__('hashlib').sha256(values.tobytes()).hexdigest()
                else:
                    begin = time.monotonic()
                    raw_path = folder / f'batch_{i:03d}.response.json.gz'
                    try:
                        if raw_path.exists():
                            with gzip.open(raw_path, 'rt') as f:
                                journal = json.load(f)
                            assert journal['batch_id'] == batch_id
                            response = CreateEmbeddingResponse.model_validate(journal['response'])
                        else:
                            response = client.embeddings.create(
                                input=[r['problem_text'] for r in batch], model=model,
                                dimensions=dimensions, encoding_format='float',
                                extra_headers={'X-Client-Request-Id': batch_id})
                            journal = {'batch_id': batch_id, 'created_utc': utc(),
                                       'request_id': response._request_id,
                                       'elapsed_seconds': time.monotonic() - begin,
                                       'source_sha256': sha(Path(__file__)), 'response': response.model_dump()}
                            raw_temp = raw_path.with_suffix('.tmp')
                            with gzip.open(raw_temp, 'wt') as f:
                                json.dump(journal, f, allow_nan=False)
                            raw_temp.replace(raw_path)
                    except APIError as exc:
                        # Never print the SDK exception body; authentication errors can contain key fragments.
                        raw_code = getattr(exc, 'code', None)
                        code = raw_code if isinstance(raw_code, str) and re.fullmatch(r'[a-z_]{1,80}', raw_code) else None
                        safe = {'created_utc': utc(), 'model': model, 'batch': i,
                                'exception_type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None),
                                'code': code, 'request_id': getattr(exc, 'request_id', None),
                                'automatic_retry': False, 'completed_batches_retained': True}
                        dump(OUT / ('failure_' + str(time.time_ns()) + '.json'), safe)
                        print(json.dumps({'state': 'API_request_failed', **safe}), flush=True)
                        return 2
                    if args.capture_one_response:
                        print(json.dumps({'state': 'raw_response_saved_for_validation', 'model': model,
                                          'items': len(batch), 'request_id': journal['request_id']}), flush=True)
                        client.close()
                        return 0
                    values = response_vectors(response, len(batch), dimensions, model)
                    info = {'batch_id': batch_id, 'batch': i, 'model': model, 'dimensions': dimensions,
                            'items': [{'problem_id': r['problem_id'], 'text_sha256': r['text_sha256']} for r in batch],
                            'input_sha256': canonical_hash([r['problem_text'] for r in batch]),
                            'created_utc': journal['created_utc'], 'request_id': journal['request_id'],
                            'usage': response.usage.model_dump(), 'elapsed_seconds': journal['elapsed_seconds'],
                            'response_journal_sha256': sha(raw_path),
                            'source_sha256': sha(Path(__file__)),
                            'values_sha256': __import__('hashlib').sha256(values.tobytes()).hexdigest()}
                    temp = target.with_suffix('.tmp')
                    with temp.open('wb') as f:
                        np.savez_compressed(f, values=values, metadata=json.dumps(info))
                    temp.replace(target)
                    print(json.dumps({'model': model, 'batch': i + 1, 'batches': len(planned),
                                      'items': len(batch), 'tokens': response.usage.total_tokens}), flush=True)
                saved.append(values)
                metadata.append(info)
            matrix = np.concatenate(saved)
            assert matrix.shape == (len(rows), dimensions)
            np.save(folder / 'embeddings.npy', matrix, allow_pickle=False)
            dump(folder / 'items.json', items)
            report = {'model': model, 'dimensions': dimensions, 'shape': list(matrix.shape),
                      'batches': len(metadata), 'usage_tokens': sum(x['usage']['total_tokens'] for x in metadata),
                      'requests': metadata, 'minimum_norm': float(np.linalg.norm(matrix, axis=1).min()),
                      'maximum_norm': float(np.linalg.norm(matrix, axis=1).max())}
            dump(folder / 'manifest.json', report)
            model_reports.append({k: v for k, v in report.items() if k != 'requests'})
        client.close()
        manifest = {'created_utc': utc(), 'protocol_sha256': sha(PROTOCOL), 'identity_sha256': run_id,
                    'source_sha256': sha(Path(__file__)), 'versions': versions(), 'interpreter': sys.executable,
                    'device': 'OpenAI API; local CPU storage/validation only', 'models': model_reports,
                    'length_audit': {'items': len(rows), 'maximum_utf8_bytes': max(len(r['problem_text'].encode()) for r in rows),
                                     'total_utf8_bytes': sum(len(r['problem_text'].encode()) for r in rows),
                                     'byte_counts_are_token_upper_bounds_not_exact_counts': True, 'truncated_items': 0},
                    'api_usage_tokens_total': sum(m['usage_tokens'] for m in model_reports),
                    'elapsed_seconds_this_invocation': time.monotonic() - started,
                    'files': {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                              for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != '.worker.lock' and not p.name.startswith('failure_')}}
        dump(OUT / 'manifest.json', manifest)
        print(json.dumps({'state': 'complete', 'models': model_reports, 'tokens': manifest['api_usage_tokens_total']}), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
