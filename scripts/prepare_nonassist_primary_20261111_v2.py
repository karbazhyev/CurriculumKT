"""Versioned CPU preparation for the authorized single-skill non-ASSIST cohorts.

No torch, training, development selection or test scoring. Source releases stay read-only.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from foundationalassist_preparation import sha, dump, utc, versions, canonical_hash

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/non_assistments_preparation_20261111'
OUT=ROOT/'data/non_assistments_primary_20261111'
RESULTS=ROOT/'results/non_assistments_preparation_20261111'
ROLES=('train','validation','calibration','test')
FIELDS=('questions','original_concepts','responses','timestamps','event_id','source_row','source_position')
def read(p):return json.loads(Path(p).read_text())
def identity(p):return {'path':str(p),'sha256':sha(p)}
def artifacts(folder):return {str(p.relative_to(folder)):identity(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}

def inputs(ds):
    if ds=='dbekt22':
        folder=BASE/ds/'text_v2';rows=read(folder/'eligible_questions.json')
        assert sha(folder/'embeddings.npy')==read(folder/'embedding_manifest.json')['embeddings_sha256']
        vectors=np.load(folder/'embeddings.npy',allow_pickle=False)
        sources=[folder/'eligible_questions.json',folder/'embeddings.npy',folder/'protocol.json',folder/'embedding_manifest.json',BASE/'release/DBE-KT22/Transaction.csv']
        sources+=list((BASE/'release/DBE-KT22').glob('Question*.csv'))
        return rows,vectors,sources,[10,25,50],[10,20,40]
    folder=BASE/ds
    original=read(folder/'eligible_questions.json');model=folder/'openai_embeddings_v1/text-embedding-3-large_1536'
    embedding_manifest=read(folder/'openai_embeddings_v1/manifest.json')
    assert sha(model/'embeddings.npy')==embedding_manifest['files']['text-embedding-3-large_1536/embeddings.npy']['sha256']
    items=read(model/'items.json')
    assert [(r['problem_id'],r['text_sha256']) for r in items]==[(r['problem_id'],r['text_sha256']) for r in original]
    chosen=[i for i,r in enumerate(original) if len(r['original_skill_ids'])==1]
    vectors=np.load(model/'embeddings.npy',allow_pickle=False)[chosen]
    sources=[folder/'eligible_questions.json',model/'embeddings.npy',model/'items.json',folder/'openai_embeddings_v1/manifest.json']
    release=BASE/'release/XES3G5M'
    sources += [release/'metadata/questions.json',release/'metadata/kc_routes_map.json',release/'question_level/train_valid_sequences_quelevel.csv',release/'question_level/test_quelevel.csv']
    return [original[i] for i in chosen],vectors,sources,[10,50,100],[300,600,900]

def freeze(ds):
    rows,vectors,sources,dims,ks=inputs(ds)
    folder=OUT/ds;folder.mkdir(parents=True,exist_ok=True)
    contract={'dataset':ds,'created_utc':utc(),'source_sha256':sha(Path(__file__)),'sources':[identity(p) for p in sources],
        'question_table_sha256':canonical_hash(rows),'questions':len(rows),'population':'single-skill and text-sufficient questions only',
        'authorization':'User chose single-skill questions for direct compatibility with existing runs, 2026-11-11.',
        'pca_dimensions':dims,'k_candidates':ks,'embedding_model':'text-embedding-3-large','embedding_dimensions':1536,
        'pca':{'svd_solver':'full','whiten':False,'fit_components':max(dims)},
        'kmeans':{'random_state':42,'n_init':10,'max_iter':300,'tol':0.0001,'algorithm':'lloyd'},
        'grid_amendment':('DBE ranks/K retained after prompt-only exclusions; 63 questions.' if ds=='dbekt22' else 'XES single-skill cohort 4672: upper K reduced from 1000 to 900 to retain original minimum mean item support of five. Old all-text mappings are preserved and are not primary.'),
        'split_policy':('All released students sorted by SHA256(dbekt22:42:<student_id>); first floor(48%) train, next through floor(64%) validation, next through floor(80%) calibration, remainder test. No response stratification.' if ds=='dbekt22' else 'Official released test students remain test; train_valid fold0 validation, fold1 calibration, folds2/3/4 training. Student identities are disjoint.'),
        'event_policy':('Unique Transaction IDs; exclude hidden attempts; binary answer_state; order by end_time then original source row. Repeated attempts retained.' if ds=='dbekt22' else 'Question-level non-window files only; respect source chunk/position order per student; padding removed; all selectmasks must be1. Retain repeated attempts. IDs identify released positions; upstream original attempt IDs are unavailable.'),
        'sequence_policy':'Filter to primary questions, concatenate accepted history within student, non-overlapping length200 chunks; first event of each chunk is context only; retain tails and one-event students.',
        'timestamp_units':'milliseconds UTC','known_item_transductive':True,'taxonomy_fit_uses':'All eligible unique question texts, including held-out-only and unused items; no responses.',
        'mapping_weight_policy':'Training events only, including context events. For original skills without training events use item-overlap counts; save raw counts and fallback flags.',
        'selection':'Future PCA/K/network/readout selection on development only; held-out outcomes used here only for export/integrity checks, never performance.',
        'training_started':False,'gpu_used':False}
    path=folder/'protocol.json'
    if path.exists():
        existing=read(path)
        for key in ('source_sha256','sources','question_table_sha256','pca_dimensions','k_candidates'):assert existing[key]==contract[key],key
    else:dump(path,contract)
    dump(folder/'question_table.json',rows)
    np.save(folder/'embeddings.npy',vectors,allow_pickle=False)
    return folder,rows,vectors,read(path)

def ingest(ds,folder,rows):
    db=folder/'events.sqlite'
    if (folder/'event_ingest.json').exists():
        report=read(folder/'event_ingest.json');assert sha(db)==report['database_sha256'];return report
    assert not db.exists(),'Partial ingest found: inspect and document before recovery.'
    connection=sqlite3.connect(db)
    connection.execute('CREATE TABLE events(event_id INTEGER PRIMARY KEY, uid INTEGER, role TEXT, q INTEGER, c INTEGER, response INTEGER, timestamp INTEGER, source_row INTEGER, source_position INTEGER)')
    ids=sorted({r['original_skill_ids'][0] for r in rows},key=int);sl={v:i for i,v in enumerate(ids)}
    lookup={r['problem_id']:(i,sl[r['original_skill_ids'][0]]) for i,r in enumerate(rows)}
    counts=collections.Counter();students={};pending=[];qid_seen=set();seenids=set();order={}
    def put(event):
        pending.append(event)
        if len(pending)>=10000:
            connection.executemany('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',pending);pending.clear()
    if ds=='dbekt22':
        path=BASE/'release/DBE-KT22/Transaction.csv'
        with path.open(newline='') as f:uids={int(r['student_id']) for r in csv.DictReader(f)}
        ranked=sorted(uids,key=lambda u:hashlib.sha256(f'dbekt22:42:{u}'.encode()).hexdigest())
        bounds=[int(len(ranked)*p) for p in (.48,.64,.8)]
        students={u:ROLES[sum(i>=b for b in bounds)] for i,u in enumerate(ranked)}
        with path.open(newline='') as f:
            for line,r in enumerate(csv.DictReader(f),2):
                counts['source_events']+=1;eid=int(r['id']);assert eid not in seenids;seenids.add(eid)
                assert r['answer_state'] in ('true','false') and r['is_hidden'] in ('true','false')
                if r['is_hidden']=='true':counts['excluded_hidden']+=1;continue
                if r['question_id'] not in lookup:counts['excluded_nonprimary_question']+=1;continue
                q,c=lookup[r['question_id']];u=int(r['student_id'])
                t=int(datetime.fromisoformat(r['end_time'].replace(' -','-').replace(' +','+')).timestamp()*1000)
                put((eid,u,students[u],q,c,int(r['answer_state']=='true'),t,line,0));qid_seen.add(r['question_id'])
    else:
        release=BASE/'release/XES3G5M'
        metadata=read(release/'metadata/questions.json');routes=read(release/'metadata/kc_routes_map.json');sid={v:k for k,v in routes.items()}
        assert len(sid)==len(routes)
        original={q:tuple(sorted({sid[p.split('----')[-1]] for p in r['kc_routes']},key=int)) for q,r in metadata.items()}
        for fi,name in enumerate(('train_valid_sequences_quelevel.csv','test_quelevel.csv')):
            with (release/'question_level'/name).open(newline='') as f:
                for line,r in enumerate(csv.DictReader(f),2):
                    u=int(r['uid']);fold=int(r['fold']);assert fold in (range(5) if fi==0 else [-1])
                    role='test' if fi else {0:'validation',1:'calibration',2:'train',3:'train',4:'train'}[fold]
                    assert students.setdefault(u,role)==role,'Student role leak'
                    fields=[r[n].split(',') for n in ('questions','concepts','responses','timestamps')]
                    assert len({len(v) for v in fields})==1 and len(fields[0])<1000000
                    masks=r.get('selectmasks');masks=masks.split(',') if masks else None
                    padded=False
                    for position,(q,c,y,t) in enumerate(zip(*fields)):
                        if q=='-1':
                            padded=True;assert c==y==t=='-1';continue
                        assert not padded
                        if masks:assert masks[position]=='1'
                        counts['source_events']+=1;assert y in ('0','1')
                        assert tuple(sorted(c.split('_'),key=int))==original[q],f'KC mismatch {q}'
                        t=int(t);assert t>=order.get(u,-1),'Released timestamps out of order';order[u]=t
                        if q not in lookup:counts['excluded_nonprimary_question']+=1;continue
                        qi,ci=lookup[q];eid=fi*10**12+line*1000000+position
                        put((eid,u,role,qi,ci,int(y),t,line,position));qid_seen.add(q)
    if pending:connection.executemany('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',pending)
    connection.execute('CREATE INDEX student_order ON events(uid,timestamp,source_row,source_position)');connection.commit()
    kept=connection.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    assert kept==counts['source_events']-sum(v for k,v in counts.items() if k.startswith('excluded_'))
    counts['retained_events']=kept
    connection.close()
    dump(folder/'student_split.json',{str(k):v for k,v in sorted(students.items())})
    dump(folder/'original_skill_ids.json',ids)
    report={'created_utc':utc(),'counts':dict(counts),'assigned_students':len(students),'unused_primary_questions':[q for q in lookup if q not in qid_seen],
        'database_sha256':sha(db),'protocol_sha256':sha(folder/'protocol.json'),'source_event_id_limitation':ds=='xes3g5m'}
    dump(folder/'event_ingest.json',report);print(json.dumps({'dataset':ds,'stage':'ingested',**report}),flush=True)
    return report

def export(folder):
    if (folder/'exports.json').exists():
        record=read(folder/'exports.json')
        for item in record['files'].values():assert sha(item['path'])==item['sha256']
        return record
    connection=sqlite3.connect('file:'+str(folder/'events.sqlite')+'?mode=ro',uri=True)
    counts={};files={};members={};ledger=folder/'events.csv.gz'
    with gzip.open(ledger,'wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(['event_id','uid','split','question_index','original_concept','response','timestamp_ms','source_row','source_position'])
        for role in ROLES:
            seq=[];part=[];last=None
            def flush():
                if not part:return
                arr=np.full((len(FIELDS),200),-1,np.int64);arr[:,:len(part)]=np.asarray(part,np.int64).T
                seq.append((last,arr));part.clear()
            for eid,uid,split,q,c,y,t,sr,sp in connection.execute('SELECT * FROM events WHERE role=? ORDER BY uid,timestamp,source_row,source_position',(role,)):
                if last!=uid:flush();last=uid
                part.append((q,c,y,t,eid,sr,sp));writer.writerow((eid,uid,split,q,c,y,t,sr,sp))
                if len(part)==200:flush()
            flush();assert seq
            matrix=np.stack([s[1] for s in seq]);arrays={name:matrix[:,i,:] for i,name in enumerate(FIELDS)}
            arrays['uid']=np.asarray([s[0] for s in seq],np.int64);arrays['sequence_id']=np.arange(len(seq),dtype=np.int64)
            valid=arrays['questions']>=0;mask=np.zeros_like(valid);mask[:,1:]=valid[:,1:]&valid[:,:-1];arrays['target_mask']=mask
            members[role]=set(arrays['uid'].tolist())
            assert np.isin(arrays['responses'][valid],[0,1]).all()
            assert len(np.unique(arrays['event_id'][valid]))==valid.sum()
            path=folder/f'{role}_sequences.npz';np.savez_compressed(path,**arrays);files[role]=identity(path)
            counts[role]={'students':len(members[role]),'chunks':len(seq),'events':int(valid.sum()),'predictable_targets':int(mask.sum())}
    connection.close()
    assert all(not members[a]&members[b] for a,b in itertools.combinations(ROLES,2))
    record={'created_utc':utc(),'files':files,'counts':counts,'ledger':identity(ledger)};dump(folder/'exports.json',record)
    return record

def cluster(folder,rows,vectors,protocol,exports):
    original_ids=read(folder/'original_skill_ids.json');si={s:i for i,s in enumerate(original_ids)}
    original=np.array([si[r['original_skill_ids'][0]] for r in rows],np.int64);raw=np.asarray([r['problem_id'] for r in rows]);pairs=np.column_stack([original,np.arange(len(rows))])
    with np.load(exports['files']['train']['path'],allow_pickle=False) as train:
        v=train['questions']>=0;trq=train['questions'][v];trc=train['original_concepts'][v]
    pca_file=folder/'pca.npz'
    with threadpool_limits(limits=1):
        if pca_file.exists():
            with np.load(pca_file) as a:reduced=a['scores'];variance=a['variance_ratio']
        else:
            reducer=PCA(n_components=max(protocol['pca_dimensions']),svd_solver='full',whiten=False)
            reduced=reducer.fit_transform(vectors.astype(np.float64));variance=reducer.explained_variance_ratio_
            np.savez_compressed(pca_file,scores=reduced,components=reducer.components_,mean=reducer.mean_,variance_ratio=variance)
        reports={}
        for dim,k in itertools.product(protocol['pca_dimensions'],protocol['k_candidates']):
            key=f'pca{dim}_k{k}';target=folder/key;target.mkdir(exist_ok=True);path=target/'mapping.npz'
            if not path.exists():
                km=KMeans(n_clusters=k,**protocol['kmeans']);labels=km.fit_predict(reduced[:,:dim]);assert len(np.unique(labels))==k
                counts=np.zeros((len(original_ids),k),np.int64);np.add.at(counts,(original,labels),1)
                practice=np.zeros_like(counts);np.add.at(practice,(trc,labels[trq]),1)
                effective=practice.copy();fallback=practice.sum(1)==0;effective[fallback]=counts[fallback]
                np.savez_compressed(path,q_to_original=original,q_to_induced=labels,raw_question_ids=raw,pairs=pairs,counts=counts,
                    raw_training_practice_counts=practice,practice_counts=effective,practice_frequency=effective/effective.sum(1,keepdims=True),
                    practice_fallback_mask=fallback,item_overlap=counts/counts.sum(1,keepdims=True),centers=km.cluster_centers_)
                dump(target/'fit.json',{'inertia':float(km.inertia_),'iterations':int(km.n_iter_)})
            with np.load(path,allow_pickle=False) as a:
                labels=a['q_to_induced'];sizes=np.bincount(labels,minlength=k)
                with (target/'question_clusters.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['question_id','induced_concept_id','original_skill_id'])
                    w.writerows((r['problem_id'],int(c),r['original_skill_ids'][0]) for r,c in zip(rows,labels))
                reports[key]={'pca_dim':dim,'k':k,'mapping':identity(path),'question_clusters':identity(target/'question_clusters.csv'),
                    'pca_variance_retained':float(variance[:dim].sum()),'cluster_min':int(sizes.min()),'cluster_max':int(sizes.max()),
                    'singleton_clusters':int((sizes==1).sum()),'no_training_skill_fallbacks':int(a['practice_fallback_mask'].sum()),
                    'no_training_clusters':int((np.bincount(labels[trq],minlength=k)==0).sum())}
            print(json.dumps({'dataset':protocol['dataset'],'taxonomy':key,'state':'complete'}),flush=True)
    return reports

def prepare(ds):
    started=time.monotonic();folder=OUT/ds
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json')
        for r in m['artifacts'].values():assert sha(r['path'])==r['sha256']
        print(json.dumps({'dataset':ds,'state':'already_complete_verified'}));return
    folder,rows,vectors,protocol=freeze(ds)
    ingest_report=ingest(ds,folder,rows);exports=export(folder)
    taxonomies=cluster(folder,rows,vectors,protocol,exports)
    m={'dataset':ds,'created_utc':utc(),'questions':len(rows),'original_concepts':len(read(folder/'original_skill_ids.json')),
        'files':exports['files'],'counts':exports['counts'],'taxonomies':taxonomies,'protocol':identity(folder/'protocol.json'),
        'embedding':'text-embedding-3-large/1536; raw vectors; primary question subset; no normalization',
        'known_item_transductive':True,'training_launched':False,'data_export_complete':True,'independent_verification_pending':True,
        'active_controller_modified':False,'versions':versions(),'elapsed_seconds':time.monotonic()-started,'event_audit':ingest_report,
        'artifacts':artifacts(folder)}
    dump(folder/'manifest.json',m);print(json.dumps({'dataset':ds,'state':'prepared','counts':m['counts'],'elapsed_seconds':m['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['dbekt22','xes3g5m'],required=True);args=parser.parse_args()
    os.umask(0o077);prepare(args.dataset)
