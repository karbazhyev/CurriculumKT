"""Read-only handoff adapter; never imports or alters the running ASSIST controller."""
from pathlib import Path
import json
import numpy as np
from foundationalassist_preparation import sha
ROOT=Path(__file__).resolve().parents[1]/'data/non_assistments_primary_20261111'

def training_record(dataset,condition):
    """Produce an identity-pinned record only after independent data verification.

    The future controller still owns GPU locks, MPS/model gates, candidate freeze,
    development selection and held-out evaluation authorization.
    """
    if dataset not in ('dbekt22','xes3g5m'):raise ValueError('Unknown dataset')
    path=ROOT/dataset/'manifest.json'
    manifest=json.loads(path.read_text())
    report_path=ROOT.parents[1]/'results/non_assistments_preparation_20261111'/f'{dataset}_primary_verification.json'
    report=json.loads(report_path.read_text())
    if not report['passed'] or report['manifest_sha256']!=sha(path):raise ValueError('Independent verification missing or stale')
    selected=manifest['taxonomies'][condition]
    for record in [manifest['protocol'],selected['mapping'],*manifest['files'].values()]:verified(record)
    return {'dataset':dataset,'questions':manifest['questions'],'original_concepts':manifest['original_concepts'],
        'k':selected['k'],'pca_dim':selected['pca_dim'],'mapping':selected['mapping'],'files':manifest['files'],
        'relabel':True,'condition':condition,'data_manifest':{'path':str(path),'sha256':sha(path)},
        'verification':{'path':str(report_path),'sha256':sha(report_path)}}

def verified(record):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('Artifact identity mismatch: '+str(path))
    return path

def load_arrays(dataset,split,condition=None,test_allowed=False):
    if dataset not in ('dbekt22','xes3g5m'):raise ValueError('Unknown dataset')
    if split not in ('train','validation','calibration','test'):raise ValueError('Unknown partition')
    if split=='test' and not test_allowed:raise PermissionError('Held-out data requires explicit frozen evaluation authorization')
    manifest=json.loads((ROOT/dataset/'manifest.json').read_text())
    verified(manifest['protocol'])
    with np.load(verified(manifest['files'][split]),allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    if condition is not None:
        with np.load(verified(manifest['taxonomies'][condition]['mapping']),allow_pickle=False) as mapping:labels=mapping['q_to_induced']
        q=arrays['questions'];valid=q>=0;induced=np.full(q.shape,-1,np.int64);induced[valid]=labels[q[valid]]
        arrays['induced_concepts']=induced
    return arrays
