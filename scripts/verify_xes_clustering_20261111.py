"""Independent CPU verification of saved XES embeddings, PCA and taxonomy exports."""
import csv
import gzip
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from foundationalassist_preparation import dump, sha, utc

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/non_assistments_preparation_20261111/xes3g5m'
RESULTS = ROOT/'results/non_assistments_preparation_20261111'


def read(path):
    return json.loads(path.read_text())


def main():
    checks = []
    def check(name, value):
        if not value: raise AssertionError(name)
        checks.append(name)
    embedding = BASE/'openai_embeddings_v1'
    em = read(embedding/'manifest.json')
    for name, info in em['files'].items():
        check('embedding hash: '+name, sha(embedding/name) == info['sha256'])
    model = embedding/'text-embedding-3-large_1536'
    rows, items = read(BASE/'eligible_questions.json'), read(model/'items.json')
    matrix = np.load(model/'embeddings.npy', allow_pickle=False)
    check('aligned item IDs and text hashes', [(x['problem_id'],x['text_sha256']) for x in rows] == [(x['problem_id'],x['text_sha256']) for x in items])
    check('matrix finite and shape', matrix.shape == (5363,1536) and np.isfinite(matrix).all())
    start = 0
    for batch in sorted(model.glob('batch_*.npz')):
        with np.load(batch, allow_pickle=False) as b:
            values = b['values']; info = json.loads(str(b['metadata']))
            with gzip.open(batch.with_suffix('.response.json.gz'), 'rt') as f: journal = json.load(f)
            response = journal['response']
            ordered = sorted(response['data'], key=lambda x:x['index'])
            check('response identity: '+batch.name, response['model']=='text-embedding-3-large' and [x['index'] for x in ordered]==list(range(len(values))))
            check('raw API vectors: '+batch.name, np.array_equal(values,np.asarray([x['embedding'] for x in ordered],dtype=np.float32)))
            check('concatenation: '+batch.name,np.array_equal(matrix[start:start+len(values)],values))
            start += len(values)
    check('all rows accounted for', start==len(rows))
    folder = BASE/'taxonomies_v1'; manifest = read(folder/'manifest.json')
    check('protocol identity', manifest['protocol_sha256']==sha(RESULTS/'xes_clustering_protocol_v1.json'))
    for name, info in manifest['files'].items():
        check('taxonomy hash: '+name, sha(folder/name)==info['sha256'])
    with np.load(folder/'pca.npz',allow_pickle=False) as p:
        reconstructed = (matrix.astype(np.float64)-p['mean']) @ p['components'].T
        check('PCA score reconstruction', np.allclose(reconstructed,p['scores'],atol=1e-9,rtol=1e-9))
    original_ids=read(folder/'original_skill_ids.json'); lookup={s:i for i,s in enumerate(original_ids)}
    expected_pairs=np.asarray([(lookup[s],i) for i,r in enumerate(rows) for s in r['original_skill_ids']],dtype=np.int64)
    check('multiskill pairs', np.array_equal(expected_pairs,np.load(folder/'original_question_pairs.npy')))
    with threadpool_limits(limits=1):
        for dim in (10,50,100):
            for k in (300,600,1000):
                name=f'pca{dim}_k{k}';path=folder/name
                with np.load(path/'mapping.npz',allow_pickle=False) as m:
                    labels=m['q_to_induced'];counts=np.zeros((len(original_ids),k),dtype=np.int64)
                    check(name+' item order',m['raw_question_ids'].tolist()==[r['problem_id'] for r in rows])
                    check(name+' all clusters represented',np.array_equal(np.unique(labels),np.arange(k)))
                    predicted=pairwise_distances_argmin(reconstructed[:,:dim],m['centers'])
                    check(name+' independent nearest-center inference',np.array_equal(labels,predicted))
                    np.add.at(counts,(expected_pairs[:,0],labels[expected_pairs[:,1]]),1)
                    check(name+' multiskill counts',np.array_equal(counts,m['counts']))
                    check(name+' map-back item overlaps',np.allclose(counts/counts.sum(1,keepdims=True),m['item_overlap']))
                    with (path/'question_clusters.csv').open(newline='') as f:export=list(csv.DictReader(f))
                    check(name+' CSV export',[(x['question_id'],int(x['induced_concept_id']),x['original_skill_ids']) for x in export]==[(r['problem_id'],int(c),'|'.join(r['original_skill_ids'])) for r,c in zip(rows,labels)])
    result={'created_utc':utc(),'state':'passed','checks':checks,'check_count':len(checks),'questions':len(rows),'taxonomies':9,'API_tokens':em['api_usage_tokens_total'],'response_files_opened':False,'training_ready':False,'source_sha256':sha(Path(__file__))}
    dump(RESULTS/'xes_clustering_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':
    main()
