"""Response-blind DBE rendering and single-skill primary embedding contract."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
from foundationalassist_preparation import canonical_hash, dump, sha, utc, versions

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/non_assistments_preparation_20261111/dbekt22'
SOURCE=ROOT/'data/non_assistments_preparation_20261111/release/DBE-KT22'
RESULTS=ROOT/'results/non_assistments_preparation_20261111'
PROTOCOL=RESULTS/'dbe_clustering_protocol_v1.json'


class TextRenderer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out=[]; self.errors=[]; self.skip=0; self.formulas=0; self.tables=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ('style','script','head'):self.skip+=1;return
        if self.skip:return
        if tag=='img':
            u=urlsplit(a.get('src',''))
            if u.hostname in ('latex.codecogs.com','latex.codecogs.com.') and u.query:
                self.out.append('$'+unquote(u.query)+'$');self.formulas+=1
            else:self.errors.append('unresolved_image')
        elif tag in ('table','tr','br','p','div','h2'):
            self.out.append('\n')
            if tag=='table': self.tables+=1;self.out.append('[table]\n')
        elif tag in ('td','th'):
            if a.get('colspan','1')!='1' or a.get('rowspan','1')!='1':self.errors.append('merged_table_cells')
            self.out.append('| ')
        elif tag in ('svg','object','iframe','math'):self.errors.append('unsupported_visual_markup')
    def handle_endtag(self,tag):
        if tag in ('style','script','head'):
            self.skip=max(0,self.skip-1);return
        if self.skip:return
        if tag=='tr':self.out.append('|\n')
        elif tag=='table':self.out.append('[/table]\n')
        elif tag in ('br','p','div','h2'):self.out.append('\n')
    def handle_data(self,data):
        if not self.skip:self.out.append(data)
    def render(self,s):
        self.feed(s.replace('\\r\\n','\n'));self.close()
        return '\n'.join(re.sub(r'[ \t]+',' ',l).strip() for l in ''.join(self.out).splitlines() if l.strip())


def read(path):return json.loads(Path(path).read_text())
def table(name):return list(csv.DictReader((SOURCE/name).open(encoding='utf-8-sig',newline='')))
def questions():return read(DATA/'eligible_questions.json')


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    q=table('Questions.csv'); choices=defaultdict(list);skills=defaultdict(set)
    for r in table('Question_Choices.csv'):choices[r['question_id']].append(r)
    for r in table('Question_KC_Relationships.csv'):skills[r['question_id']].add(r['knowledgecomponent_id'])
    skill_vocab={r['id'] for r in table('KCs.csv')}
    assert len({r['id'] for r in q})==len(q)==212
    assert set(choices)<=set(r['id'] for r in q) and set(skills)<=set(r['id'] for r in q)
    eligible=[];audit=[]
    for r in sorted(q,key=lambda x:int(x['id'])):
        renderer=TextRenderer();text=renderer.render(r['question_rich_text']);errors=list(renderer.errors)
        # These four records explicitly reference a missing shaded figure.
        if re.search(r'shaded area of the figure',text,re.I):errors.append('missing_referenced_figure')
        options=[]
        for index,c in enumerate(sorted(choices[r['id']],key=lambda x:int(x['id'])),1):
            cr=TextRenderer();options.append(f'{index}. '+cr.render(c['choice_text']));errors+=cr.errors
        if not options:errors.append('missing_options')
        text+='\n'+'\n'.join(options)
        ids=sorted(skills[r['id']],key=int);assert set(ids)<=skill_vocab
        if not ids:errors.append('missing_skills')
        if len(ids)!=1:errors.append('multiple_original_skills')
        if not text.strip() or len(text.encode())>8191:errors.append('embedding_input_length')
        item={'problem_id':r['id'],'problem_text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'original_skill_ids':ids}
        audit.append({**item,'excluded_reasons':sorted(set(errors)),'restored_formula_images':renderer.formulas,'rendered_tables':renderer.tables})
        if not errors:eligible.append(item)
    report={'created_utc':utc(),'total_questions':len(q),'eligible_questions':len(eligible),
        'eligible_original_skills':len({s for r in eligible for s in r['original_skill_ids']}),
        'exclusion_reasons_overlapping':dict(Counter(e for r in audit for e in r['excluded_reasons'])),
        'restored_formula_images_all_questions':sum(r['restored_formula_images'] for r in audit),
        'rendered_tables_all_questions':sum(r['rendered_tables'] for r in audit),
        'input_utf8_bytes':sum(len(r['problem_text'].encode()) for r in eligible),
        'response_outcomes_used':False,'rule':'Render rich prompt and options only; recover URL-encoded Codecogs LaTeX locally without fetching images; preserve table rows, columns and empty cells; exclude missing figures and multi-skill questions. No title/ID, answer markers, hints, explanations, difficulty or KC text in embeddings.',
        'multi_skill_annotations_preserved':True,'user_single_skill_decision':'2026-11-11: Use single-skill questions for direct compatibility with existing runs.'}
    assert len(eligible)>50
    protocol={'dataset':'dbekt22','audit':report,'eligible_items_sha256':canonical_hash(eligible),
        'source_sha256':sha(Path(__file__)),'sources':{str(SOURCE/n):sha(SOURCE/n) for n in ['Questions.csv','Question_Choices.csv','Question_KC_Relationships.csv','KCs.csv']},
        'embedding_model':'text-embedding-3-large','embedding_dimensions':1536,
        'pca_dimensions':[10,25,50],'k_candidates':[10,20,40],
        'grid_reason':'Three feasible ranks and coarse/intermediate/fine K levels for the audited single-skill question population; chosen without outcome access.',
        'known_item_transductive':True,'training_ready':False}
    if PROTOCOL.exists():
        old=read(PROTOCOL);assert old['eligible_items_sha256']==protocol['eligible_items_sha256'];assert old['source_sha256']==protocol['source_sha256'];return
    dump(DATA/'eligible_questions.json',eligible);dump(DATA/'question_audit.json',audit)
    dump(RESULTS/'dbe_text_audit.json',report);dump(PROTOCOL,protocol)
    print(json.dumps(report),flush=True)


def load_contract():
    p=read(PROTOCOL);assert sha(Path(__file__))==p['source_sha256']
    for name,h in p['sources'].items():assert sha(name)==h
    assert canonical_hash(questions())==p['eligible_items_sha256']
    return p


if __name__=='__main__':main()
