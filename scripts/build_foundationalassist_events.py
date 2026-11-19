#!/usr/bin/env python3
"""Canonical unique response events and a fixed student split; retain KC sets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def build(source,output):
    if output.exists():raise ValueError('Output exists; verify/reuse or choose a new version.')
    os.umask(0o077)
    audit=json.loads((source/'audit.json').read_text())
    for name,info in audit['files'].items():assert sha(source/name)==info['sha256']
    ledger=pd.read_csv(source/'events.csv.gz',dtype=str,keep_default_na=False)
    repeated=ledger[ledger.source_event_id.duplicated(keep=False)]
    fields=['student_id','problem_id','response','source_end_time','original_skill_ids']
    conflicting=repeated.groupby('source_event_id')[fields].nunique().max(axis=1).gt(1)
    assert not conflicting.any(),'Conflicting official event IDs need a separate disposition.'
    eligible=ledger[ledger.basic_fields_valid.eq('True')].copy()
    invalid=len(ledger)-len(eligible)
    counts=eligible.source_event_id.value_counts()
    # Equal modeled fields for repeated official IDs: one canonical event.
    # Full source-row ledger is retained separately for every excluded copy.
    canonical=eligible.sort_values('source_row',key=lambda x:x.astype(np.int64),kind='stable').drop_duplicates('source_event_id',keep='first').copy()
    copies=len(eligible)-len(canonical)
    canonical['source_duplicate_count']=canonical.source_event_id.map(counts)
    canonical['event_id']=audit['source_revision']+':'+canonical.source_event_id
    canonical['response']=canonical.response.astype(float).astype(np.int8)
    canonical['source_row']=canonical.source_row.astype(np.int64)
    canonical['end_time_utc']=pd.to_datetime(canonical.end_time_utc,utc=True,format='mixed')
    canonical=canonical.sort_values(['student_id','end_time_utc','source_row'],kind='stable').reset_index(drop=True)
    students=np.array(sorted(canonical.student_id.unique()))
    np.random.default_rng(42).shuffle(students)
    n=len(students);boundaries=[0,int(n*.48),int(n*.64),int(n*.80),n]
    roles=['train','validation','calibration','test']
    split={student:role for role,a,b in zip(roles,boundaries[:-1],boundaries[1:]) for student in students[a:b]}
    canonical['split']=canonical.student_id.map(split)
    assert canonical.event_id.is_unique and canonical.source_event_id.is_unique
    assert canonical.response.isin([0,1]).all() and canonical.end_time_utc.notna().all()
    assert canonical.groupby('student_id').split.nunique().eq(1).all()
    assert len(ledger)==invalid+copies+len(canonical)
    item_ids=sorted(canonical.problem_id.unique())
    item_vocab={item:i for i,item in enumerate(item_ids)}
    canonical['item_index']=canonical.problem_id.map(item_vocab)
    candidate=(canonical.verified_text_only_candidate.eq('True'))
    single=candidate & canonical.original_skill_count.astype(int).eq(1)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'source_revision':audit['source_revision'],
            'source_audit_sha256':sha(source/'audit.json'),'script_sha256':sha(Path(__file__)),
            'counts':{'source_rows':len(ledger),'excluded_invalid_fields':invalid,'excluded_redundant_valid_id_copies':copies,
                      'canonical_events':len(canonical),'students':n,'items':len(item_vocab),
                      'verified_text_only_candidate_events':int(candidate.sum()),'single_skill_text_candidate_events':int(single.sum())},
            'splits':{role:{'students':sum(v==role for v in split.values()),'events':int(canonical.split.eq(role).sum())} for role in roles},
            'split_policy':'seed42, sorted user IDs permuted with NumPy default_rng; 48/16/16/20 percent; preserve these assignments under later common eligibility filtering',
            'event_deduplication':'Identical modeled fields under the same official event ID; first source row retained, multiplicity recorded; distinct IDs are not merged solely for matching time/item/response',
            'KC_policy':'Full original sets retained, no scalarization or multi-KC pseudo-events; final paired training export is pending.',
            'checks':{'source_hashes_pass':True,'no_conflicting_modeled_duplicate_ids':True,'exact_source_row_accounting':True,
                      'unique_canonical_events':True,'binary_response':True,'valid_timestamps':True,'disjoint_student_roles':True}}
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    try:
        canonical.to_csv(stage/'canonical_events.csv.gz',index=False,compression={'method':'gzip','compresslevel':1,'mtime':0})
        pd.DataFrame(sorted(split.items()),columns=['student_id','split']).to_csv(stage/'student_split.csv',index=False)
        (stage/'item_vocabulary.json').write_text(json.dumps(item_vocab,sort_keys=True)+'\n')
        report['files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in stage.iterdir()}
        (stage/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        os.rename(stage,output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(json.dumps({'counts':report['counts'],'splits':report['splits']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/foundationalassist/derived/text_audit_v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/foundationalassist/derived/canonical_v1')
    a=p.parse_args();build(a.source,a.output)
