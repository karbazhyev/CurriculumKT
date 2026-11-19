"""Amend DBE text audit before training; reuse unchanged API vectors."""
import gzip
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from prepare_dbe_text_20261111 import DATA as OLD, SOURCE, RESULTS, TextRenderer, table, read, sha, dump, utc

OUT=OLD/'text_v2'
MISSING={'68','69','70','71','72','78','82','153'}

class Renderer(TextRenderer):
    def handle_data(self,data):
        # HTML source newlines are layout whitespace, not table row boundaries.
        if not self.skip:self.out.append(re.sub(r'\s+',' ',data))

def prepare():
    OUT.mkdir(exist_ok=True)
    source={r['id']:r for r in table('Questions.csv')}
    prior=read(OLD/'eligible_questions.json');rows=[]
    for r in prior:
        if r['problem_id'] in MISSING:continue
        item=dict(r)
        if '<table' in source[r['problem_id']]['question_rich_text'].lower():
            text=Renderer().render(source[r['problem_id']]['question_rich_text'])
            opts=sorted([c for c in table('Question_Choices.csv') if c['question_id']==r['problem_id']],key=lambda x:int(x['id']))
            text+='\n'+'\n'.join(f'{i}. '+Renderer().render(c['choice_text']) for i,c in enumerate(opts,1))
            item.update(problem_text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest())
        rows.append(item)
    assert len(rows)==63
    dump(OUT/'eligible_questions.json',rows)
    return rows

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--embed',action='store_true');args=parser.parse_args()
    rows=prepare()
    prior=read(OLD/'eligible_questions.json');oldmodel=OLD/'openai_embeddings_v1/text-embedding-3-large_1536'
    oldvectors=np.load(oldmodel/'embeddings.npy',allow_pickle=False)
    lookup={r['text_sha256']:v for r,v in zip(prior,oldvectors)}
    changed=[r for r in rows if r['text_sha256'] not in lookup]
    contract={'created_utc':utc(),'source_sha256':sha(Path(__file__)),
        'eligible_items_sha256':sha(OUT/'eligible_questions.json'),'prior_protocol_sha256':sha(RESULTS/'dbe_clustering_protocol_v1.json'),
        'excluded_missing_table_contents':sorted(MISSING,key=int),'eligible_questions':len(rows),
        'changed_table_renderings':[r['problem_id'] for r in changed],
        'reason':'Manual prompt-only review: eight row-count questions omit the required Study/Course/Enrol tables. Four table questions preserve HTML row boundaries with normalized source whitespace. No responses used.',
        'pca_dimensions':[10,25,50],'k_candidates':[10,20,40],
        'prior_embeddings_retained':True,'unchanged_vectors_reused':len(rows)-len(changed)}
    p=OUT/'protocol.json'
    if p.exists():
        old=read(p);assert old['source_sha256']==contract['source_sha256'];assert old['eligible_items_sha256']==contract['eligible_items_sha256']
    else:dump(p,contract)
    if not args.embed:
        print(json.dumps(contract));return
    raw=OUT/'changed_tables.response.json.gz'
    if not raw.exists():
        from openai import OpenAI, APIError
        key=read(Path(__file__).resolve().parents[1]/'.credentials/openai.json')['OPENAI_API_KEY']
        with OpenAI(api_key=key,base_url='https://api.openai.com/v1',max_retries=0,timeout=90) as client:
            try:response=client.embeddings.create(model='text-embedding-3-large',dimensions=1536,input=[r['problem_text'] for r in changed],encoding_format='float')
            except APIError as e:
                dump(OUT/'failure.json',{'type':type(e).__name__,'status':e.status_code});raise SystemExit('API request failed; safe failure record saved.')
        with gzip.open(raw,'wt') as f:json.dump(response.model_dump(),f)
    with gzip.open(raw,'rt') as f:response=json.load(f)
    assert response['model']=='text-embedding-3-large'
    values=sorted(response['data'],key=lambda r:r['index']);assert [r['index'] for r in values]==list(range(len(changed)))
    for item,value in zip(changed,values):lookup[item['text_sha256']]=np.asarray(value['embedding'],np.float32)
    vectors=np.stack([lookup[r['text_sha256']] for r in rows]);assert vectors.shape==(63,1536) and np.isfinite(vectors).all()
    np.save(OUT/'embeddings.npy',vectors,allow_pickle=False)
    dump(OUT/'embedding_manifest.json',{'created_utc':utc(),'protocol_sha256':sha(p),'embeddings_sha256':sha(OUT/'embeddings.npy'),
        'prior_embedding_sha256':sha(oldmodel/'embeddings.npy'),'response_sha256':sha(raw),'new_items':len(changed),
        'reused_items':len(rows)-len(changed),'usage':response['usage'],'model':'text-embedding-3-large','dimensions':1536})
    print(json.dumps({'state':'complete','items':len(rows),'new_items':len(changed),'usage':response['usage']}))

if __name__=='__main__':main()
