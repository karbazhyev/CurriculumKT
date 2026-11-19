"""Versioned, local-only joint taxonomy/model search contracts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
OUT = ROOT / 'results/joint_tuning_20261109'
DATA = ROOT / 'data/joint_tuning_20261109'
MODELS = ['sakt', 'dkvmn', 'dkt', 'sparsekt', 'cskt', 'dtransformer', 'ukt',
          'kqn', 'deep_irt', 'skvmn', 'saint', 'atktfix', 'dkt+']
DATASETS = ['assist2017', 'assist2012', 'foundationalassist']
SEEDS = [42, 84, 126]
LAMBDAS = [.001, .01, .1, 1.]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def identity(path):
    return {'path': str(path), 'sha256': sha(path)}


def verify_record(record):
    path = Path(record['path'])
    assert sha(path) == record['sha256'], path
    return path


def load_arrays(ds, split, condition=None, test_allowed=False):
    import numpy as np
    assert split != 'test' or test_allowed, 'Held-out data boundary'
    manifest = read(DATA / ds / 'manifest.json')
    path = verify_record(manifest['files'][split])
    with np.load(path) as a:
        arrays = {k: a[k] for k in a.files}
    if condition is not None:
        record = manifest['taxonomies'][condition]
        path = verify_record(record['mapping'])
        with np.load(path) as mapping:
            labels = mapping['q_to_induced']
        q = arrays['questions']; valid = q >= 0
        induced = np.full(q.shape, -1, np.int64)
        induced[valid] = labels[q[valid]]
        arrays['induced_concepts'] = induced
    return arrays
