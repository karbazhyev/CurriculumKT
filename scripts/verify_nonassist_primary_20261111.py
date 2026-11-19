"""Independent source/event, split, label, PCA and map-weight verification."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[n]='1'
import argparse
import csv
import hashlib
import itertools
import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from foundationalassist_preparation import sha,dump,utc
from nonassist_data_20261111 import load_arrays,verified
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/non_assistments_preparation_20261111'
DATA=ROOT/'data/non_assistments_primary_20261111'
ROLES=('train','validation','calibration','test')
def read(p):return json.loads(Path(p).read_text())

def verify(ds):
    folder=DATA/ds;manifest=read(folder/'manifest.json');protocol=read(folder/'protocol.json');checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    for name,record in manifest['artifacts'].items():check('hash:'+name,sha(record['path'])==record['sha256'])
    for record in protocol['sources']:check('source_hash:'+record['path'],sha(record['path'])==record['sha256'])
    rows=read(folder/'question_table.json');skills=read(folder/'original_skill_ids.json');si={s:i for i,s in enumerate(skills)}
    ql={r['problem_id']:i for i,r in enumerate(rows)};original=np.array([si[r['original_skill_ids'][0]] for r in rows])
    check('all_primary_single_skill',all(len(r['original_skill_ids'])==1 for r in rows))
    check('question_text_hashes',all(hashlib.sha256(r['problem_text'].encode()).hexdigest()==r['text_sha256'] for r in rows))
    assignments=read(folder/'student_split.json');db=sqlite3.connect('file:'+str(folder/'events.sqlite')+'?mode=ro',uri=True)
    # Reconstruct every retained event directly from immutable released sources.
    order='source_row' if ds=='dbekt22' else 'event_id'
    cursor=iter(db.execute('SELECT * FROM events ORDER BY '+order));observed=next(cursor,None);matched=0;source_uids={}
    def match(expected):
        nonlocal observed,matched
        if tuple(expected)!=observed:raise AssertionError('Source/event mismatch at retained event '+str(matched))
        observed=next(cursor,None);matched+=1
    if ds=='dbekt22':
        path=BASE/'release/DBE-KT22/Transaction.csv'
        with path.open(newline='') as f:all_uids={int(r['student_id']) for r in csv.DictReader(f)}
        ranked=sorted(all_uids,key=lambda u:hashlib.sha256(f'dbekt22:42:{u}'.encode()).hexdigest())
        bounds=[int(len(ranked)*p) for p in (.48,.64,.8)]
        expected_split={str(u):ROLES[sum(i>=b for b in bounds)] for i,u in enumerate(ranked)}
        check('independent_student_split',expected_split==assignments)
        links={}
        with (BASE/'release/DBE-KT22/Question_KC_Relationships.csv').open() as f:
            for r in csv.DictReader(f):links.setdefault(r['question_id'],set()).add(r['knowledgecomponent_id'])
        check('original_skill_join',all(links[r['problem_id']]==set(r['original_skill_ids']) for r in rows))
        with path.open(newline='') as f:
            for line,r in enumerate(csv.DictReader(f),2):
                if r['is_hidden']=='true' or r['question_id'] not in ql:continue
                q=ql[r['question_id']];u=int(r['student_id'])
                raw=r['end_time'];fmt='%Y-%m-%d %H:%M:%S.%f %z' if '.' in raw else '%Y-%m-%d %H:%M:%S %z'
                t=int(datetime.strptime(raw,fmt).timestamp()*1000)
                match((int(r['id']),u,assignments[str(u)],q,int(original[q]),int(r['answer_state']=='true'),t,line,0))
    else:
        expected_rows=[r for r in read(BASE/'xes3g5m/eligible_questions.json') if len(r['original_skill_ids'])==1]
        check('independent_single_skill_selection',rows==expected_rows)
        for fi,name in enumerate(('train_valid_sequences_quelevel.csv','test_quelevel.csv')):
            with (BASE/'release/XES3G5M/question_level'/name).open(newline='') as f:
                for line,r in enumerate(csv.DictReader(f),2):
                    u=int(r['uid']);fold=int(r['fold']);role='test' if fi else {0:'validation',1:'calibration',2:'train',3:'train',4:'train'}[fold]
                    assert source_uids.setdefault(str(u),role)==role
                    for pos,(q,c,y,t) in enumerate(zip(*(r[n].split(',') for n in ('questions','concepts','responses','timestamps')))):
                        if q not in ql:continue
                        qi=ql[q];assert c==skills[original[qi]]
                        match((fi*10**12+line*1000000+pos,u,role,qi,int(original[qi]),int(y),int(t),line,pos))
        check('official_holdout_and_folds_preserved',source_uids==assignments)
    check('all_source_events_match_export_database',observed is None and matched==manifest['event_audit']['counts']['retained_events'])
    groups={};arrays={};all_event_ids=[];diagnostics={}
    for role in ROLES:
        a=load_arrays(ds,role,test_allowed=(role=='test'));arrays[role]=a;v=a['questions']>=0
        check(role+':left_aligned_padding',np.all(np.diff(v.astype(int),axis=1)<=0))
        for name in ('original_concepts','responses','timestamps','event_id','source_row','source_position'):check(role+':padding:'+name,np.array_equal(a[name]>=0,v))
        expected_mask=np.zeros_like(v);expected_mask[:,1:]=v[:,1:]&v[:,:-1]
        check(role+':context_mask',np.array_equal(expected_mask,a['target_mask']))
        check(role+':original_concepts',np.array_equal(a['original_concepts'][v],original[a['questions'][v]]))
        check(role+':binary_responses',np.isin(a['responses'][v],[0,1]).all())
        groups[role]=set(a['uid'].tolist());all_event_ids.append(a['event_id'][v])
        check(role+':split_assignments',all(assignments[str(u)]==role for u in groups[role]))
        stored=np.column_stack([a['event_id'][v],np.broadcast_to(a['uid'][:,None],v.shape)[v],a['questions'][v],a['original_concepts'][v],a['responses'][v],a['timestamps'][v],a['source_row'][v],a['source_position'][v]])
        expected=np.asarray(db.execute('SELECT event_id,uid,q,c,response,timestamp,source_row,source_position FROM events WHERE role=? ORDER BY uid,timestamp,source_row,source_position',(role,)).fetchall(),dtype=np.int64)
        check(role+':ordered_export_matches_database',np.array_equal(stored,expected))
        check(role+':counts',int(v.sum())==manifest['counts'][role]['events'])
        diagnostics[role]={'single_event_students':int(db.execute('SELECT COUNT(*) FROM (SELECT uid FROM events WHERE role=? GROUP BY uid HAVING COUNT(*)=1)',(role,)).fetchone()[0]),'observed_original_skills':len(np.unique(a['original_concepts'][v]))}
        del stored,expected
    check('student_disjoint_roles',all(not groups[a]&groups[b] for a,b in itertools.combinations(ROLES,2)))
    events=np.concatenate(all_event_ids);check('unique_event_identity_across_all_roles',len(events)==len(np.unique(events)))
    try:load_arrays(ds,'test')
    except PermissionError:check('heldout_loader_guard',True)
    else:raise AssertionError('Missing heldout guard')
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'data';p.write_text('before');record={'path':str(p),'sha256':sha(p)};p.write_text('after')
        try:verified(record)
        except ValueError:check('tamper_rejected',True)
        else:raise AssertionError('No identity validation')
    vectors=np.load(folder/'embeddings.npy',allow_pickle=False)
    check('valid_embedding_matrix',vectors.shape==(len(rows),1536) and np.isfinite(vectors).all())
    if ds=='xes3g5m':
        embfolder=BASE/'xes3g5m/openai_embeddings_v1/text-embedding-3-large_1536'
        items=read(embfolder/'items.json');idx={r['problem_id']:i for i,r in enumerate(items)};prior=np.load(embfolder/'embeddings.npy')
        check('exact_reuse_of_xes_vectors',np.array_equal(vectors,prior[[idx[r['problem_id']] for r in rows]]))
    else:
        import gzip
        prior_items=read(BASE/'dbekt22/eligible_questions.json');prior=np.load(BASE/'dbekt22/openai_embeddings_v1/text-embedding-3-large_1536/embeddings.npy');lookup={r['text_sha256']:v for r,v in zip(prior_items,prior)}
        changed=[r for r in rows if r['text_sha256'] not in lookup]
        with gzip.open(BASE/'dbekt22/text_v2/changed_tables.response.json.gz','rt') as f:response=json.load(f)
        for r,v in zip(changed,sorted(response['data'],key=lambda x:x['index'])):lookup[r['text_sha256']]=np.asarray(v['embedding'],np.float32)
        check('exact_reuse_or_api_response_vectors_dbe',np.array_equal(vectors,np.stack([lookup[r['text_sha256']] for r in rows])))
        check('known_missing_context_excluded',not set(ql)&{'24','25','26','27','68','69','70','71','72','78','82','153'})
        check('table_rows_preserved',all('| $T_{1}$ | $T_{2}$ |' in next(r['problem_text'] for r in rows if r['problem_id']==q) for q in ('216','217','218','219')))
    with np.load(folder/'pca.npz') as p:
        reduced=(vectors.astype(float)-p['mean'])@p['components'].T
        check('pca_transform_independent',np.allclose(reduced,p['scores'],atol=1e-9,rtol=1e-8))
        check('pca_orthonormal',np.allclose(p['components']@p['components'].T,np.eye(len(p['components'])),atol=1e-9))
    train=arrays['train'];v=train['questions']>=0;trq=train['questions'][v];trc=train['original_concepts'][v]
    coldq=sorted(set(range(len(rows)))-set(trq));coldskills=sorted(set(range(len(skills)))-set(trc))
    taxonomy_diagnostics={}
    with threadpool_limits(limits=1):
        for key,record in manifest['taxonomies'].items():
            with np.load(record['mapping']['path'],allow_pickle=False) as m:
                labels=m['q_to_induced'];k=record['k'];dim=record['pca_dim']
                check(key+':all_clusters_present',len(np.unique(labels))==k)
                check(key+':nearest_centers',np.array_equal(pairwise_distances_argmin(reduced[:,:dim],m['centers']),labels))
                counts=np.zeros((len(skills),k),np.int64)
                for ci,li in zip(original,labels):counts[ci,li]+=1
                practice=np.bincount(trc*k+labels[trq],minlength=len(skills)*k).reshape(len(skills),k)
                effective=practice.copy();fallback=practice.sum(1)==0;effective[fallback]=counts[fallback]
                check(key+':item_counts',np.array_equal(counts,m['counts']))
                check(key+':raw_train_only_practice_counts',np.array_equal(practice,m['raw_training_practice_counts']))
                check(key+':fallback',np.array_equal(fallback,m['practice_fallback_mask']) and np.array_equal(effective,m['practice_counts']))
                check(key+':normalized_weights',np.allclose(m['practice_frequency'],effective/effective.sum(1,keepdims=True)) and np.allclose(m['item_overlap'],counts/counts.sum(1,keepdims=True)))
                check(key+':ids_and_pairs',np.array_equal(m['raw_question_ids'],np.asarray([r['problem_id'] for r in rows])) and np.array_equal(m['pairs'],np.column_stack([original,np.arange(len(rows))])))
                with open(record['question_clusters']['path']) as f:csvrows=list(csv.DictReader(f))
                check(key+':csv_labels',[(r['question_id'],int(r['induced_concept_id']),r['original_skill_id']) for r in csvrows]==[(r['problem_id'],int(c),r['original_skill_ids'][0]) for r,c in zip(rows,labels)])
                # Loader relabeling must leave every other field bit-identical.
                view=load_arrays(ds,'validation',key)
                check(key+':paired_loader_unchanged',all(np.array_equal(view[n],a) for n,a in arrays['validation'].items()))
                q=view['questions'];valid=q>=0
                check(key+':paired_loader_labels',np.array_equal(view['induced_concepts'][valid],labels[q[valid]]) and (view['induced_concepts'][~valid]==-1).all())
                taxonomy_diagnostics[key]={'original_skills_split_across_multiple_clusters':int(((counts>0).sum(1)>1).sum()),'singleton_clusters':record['singleton_clusters'],'untrained_clusters':record['no_training_clusters']}
    db.close()
    report={'created_utc':utc(),'dataset':ds,'passed':True,'checks_passed':len(checks),'checks':checks,
        'all_retained_source_events_verified':matched,'questions':len(rows),'original_skills':len(skills),'counts':manifest['counts'],
        'cold_start_question_ids':[rows[i]['problem_id'] for i in coldq],'cold_start_original_skills':[skills[i] for i in coldskills],
        'unused_primary_questions':manifest['event_audit']['unused_primary_questions'],'diagnostics':diagnostics,'taxonomies':taxonomy_diagnostics,
        'manifest_sha256':sha(folder/'manifest.json'),'verifier_sha256':sha(Path(__file__)),
        'test_metrics_computed':False,'gpu_used':False,'new_training_launched':False,
        'model_specific_mps_counterfactual_preflight':'Required at future training launch; not performed during CPU data preparation.'}
    dump(ROOT/f'results/non_assistments_preparation_20261111/{ds}_primary_verification.json',report)
    print(json.dumps({k:report[k] for k in ('dataset','passed','checks_passed','all_retained_source_events_verified','questions','original_skills','cold_start_question_ids','cold_start_original_skills')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['dbekt22','xes3g5m'],required=True);args=parser.parse_args();verify(args.dataset)
