"""Verify the user-authorized reduction without changing frozen scientific inputs."""
import hashlib
import json
from pathlib import Path
ROOT = Path('/Users/kiakarbasi/CurriculumKT')
OUT = ROOT / 'results/scope_amendment_20261111'
def verify_scope():
    marker = json.loads((OUT / 'amendment_identity.json').read_text())
    path = Path(marker['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == marker['sha256']
    scope = json.loads(path.read_text())
    for record in scope['identities']:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256'], record['path']
    assert scope['execution_models'] == ['dkvmn', 'dkt', 'sparsekt', 'dtransformer', 'kqn', 'atktfix']
    assert set(scope['deferred_models']) == {'ukt', 'skvmn', 'deep_irt', 'cskt', 'saint', 'dkt+'}
    return scope
