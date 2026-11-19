#!/usr/bin/env python3
"""Audit a pinned official snapshot and extract source-traceable text/events.

CPU preprocessing only. No training, embeddings, scalar projection of skill
sets, or response-based selection is performed here.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

from bs4 import BeautifulSoup, NavigableString
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / 'data/foundationalassist/raw/release_20261108'
DEFAULT_OUTPUT = ROOT / 'data/foundationalassist/derived/text_audit_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def render_math(node, unsupported):
    if isinstance(node, NavigableString):
        return str(node).strip()
    name = node.name
    children = [c for c in node.children if not isinstance(c, NavigableString) or str(c).strip()]
    parts = [render_math(c, unsupported) for c in children]
    text = ' '.join(p for p in parts if p)
    if name in {'mn', 'mi', 'mo', 'mtext', 'ms'}:
        return node.get_text().strip()
    if name in {'math', 'mrow', 'mstyle', 'mpadded', 'mtd'}:
        return text
    if name in {'mspace', 'none'}:
        return ''
    if name == 'mfrac' and len(parts) == 2:
        return f'({parts[0]}) / ({parts[1]})'
    if name in {'msup', 'msub'} and len(parts) == 2:
        return f'({parts[0]}){"^" if name == "msup" else "_"}({parts[1]})'
    if name == 'msqrt':
        return f'sqrt({text})'
    if name == 'mroot' and len(parts) == 2:
        return f'root(index={parts[1]}, radicand={parts[0]})'
    if name == 'mfenced':
        separators = node.get('separators', ',').replace(' ', '') or ' '
        joined = ''.join(p + (separators[min(i, len(separators)-1)] + ' ' if i < len(parts)-1 else '') for i, p in enumerate(parts))
        return node.get('open', '(') + joined + node.get('close', ')')
    if name == 'mtr':
        return '[' + ' | '.join(parts) + ']'
    if name == 'mtable':
        return 'matrix(' + '; '.join(parts) + ')'
    # Preserve the tag name and content, but flag unsupported visual notation.
    # Do not pretend a flattened long division/stack/overbar is faithful text.
    unsupported.add(name)
    return f'{name}({text})'


def clean_markup(value):
    soup = BeautifulSoup(str(value), 'html.parser')
    unsupported = set()
    unresolved_images = 0
    converted_math_images = 0
    for tag in soup(['script', 'style']):
        tag.decompose()
    for math in list(soup.find_all('math')):
        if math.parent is not None:
            math.replace_with(' ' + render_math(math, unsupported) + ' ')
    for img in soup.find_all('img'):
        encoded = img.get('data-mathml', '')
        if encoded:
            decoded = html.unescape(encoded.replace('«', '<').replace('»', '>').replace('¨', '"').replace('§#', '&#'))
            math = BeautifulSoup(decoded, 'html.parser').find('math')
            if math:
                img.replace_with(' ' + render_math(math, unsupported) + ' ')
                converted_math_images += 1
                continue
        # Even a nonempty alt may be a filename or incomplete figure caption.
        # Preserve it for review, but do not count it as verified text coverage.
        alt = img.get('alt', '').strip()
        img.replace_with(' [Image: ' + alt + '] ' if alt else ' [Image unavailable in text] ')
        unresolved_images += 1
    for response in soup.find_all('ast-r'):
        response.replace_with(' [response blank] ')
    for table in list(soup.find_all('table')):
        rows = [' | '.join(cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'], recursive=False)) for row in table.find_all('tr')]
        table.replace_with(' [Table: ' + ' ; '.join(rows) + '] ')
    text = ' '.join(html.unescape(soup.get_text(' ')).split())
    return text, unresolved_images, converted_math_images, sorted(unsupported)


def item_input(row):
    fields = [('Problem', row['Problem Body'])]
    if row['Problem Type'].startswith('Multiple Choice'):
        fields.append(('Presented choices', row['Multiple Choice Options']))
    # Numeric/order/etc. Fill-in Options contains answer keys in this release.
    if row['Answer Types'] == 'Drop Down':
        fields.append(('Presented dropdown choices', row['Fill-in Options']))
    return fields


def prepare(raw, output):
    if output.exists():
        raise ValueError('Output already exists; verify/reuse it or choose a new version.')
    os.umask(0o077)
    manifest = json.loads((raw / 'download_manifest.json').read_text())
    for filename, record in manifest['files'].items():
        assert sha(raw / filename) == record['sha256'], filename
    p = pd.read_csv(raw/'Data/Problems.csv', dtype=str, keep_default_na=False)
    if 'Answer Types' not in p:
        p = p.rename(columns={'Answer Type': 'Answer Types'})
    s = pd.read_csv(raw/'Data/Skills.csv', dtype=str, keep_default_na=False)
    assert not s[['problem_id', 'skill_id']].eq('').any().any()
    skill_sets = s.groupby('problem_id')['skill_id'].agg(lambda x: sorted(set(x), key=lambda v: (len(v), v))).to_dict()
    items = []
    for problem_id, rows in p.groupby('problem_id', sort=True):
        inputs = {json.dumps(item_input(row), ensure_ascii=False) for _, row in rows.iterrows()}
        conflict = len(inputs) != 1
        text, images, math_images, tags = '', 0, 0, set()
        if not conflict:
            parts = []
            for label, value in item_input(rows.iloc[0]):
                cleaned, bad, converted, unknown = clean_markup(value)
                parts.append(label + ': ' + cleaned)
                images += bad; math_images += converted; tags.update(unknown)
            text = '\n'.join(parts)
        skills = skill_sets.get(problem_id, [])
        items.append({'problem_id': problem_id, 'problem_source_rows': json.dumps(rows.index.tolist()),
                      'source_row_count': len(rows), 'conflicting_input_variants': conflict,
                      'problem_text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'original_skill_ids': json.dumps(skills), 'original_skill_count': len(skills),
                      'unresolved_image_count': images, 'converted_math_image_count': math_images,
                      'unsupported_math_tags': json.dumps(sorted(tags)),
                      'verified_text_only_candidate': bool(text and not conflict and images == 0 and not tags),
                      'problem_type': rows.iloc[0]['Problem Type'], 'answer_type': rows.iloc[0]['Answer Types']})
    items = pd.DataFrame(items)
    assert items.problem_id.is_unique
    columns = pd.read_csv(raw/'Data/Interactions.csv', nrows=0).columns
    user = 'user_id' if 'user_id' in columns else 'user_xid'
    events = pd.read_csv(raw/'Data/Interactions.csv', usecols=['id', user, 'problem_id', 'discrete_score', 'end_time'], dtype=str, keep_default_na=False)
    events = events.rename(columns={'id':'source_event_id', user:'student_id', 'discrete_score':'response', 'end_time':'source_end_time'})
    events.insert(0, 'source_row', np.arange(len(events), dtype=np.int64))
    times = pd.to_datetime(events.source_end_time, format='mixed', utc=True, errors='coerce')
    events['end_time_utc'] = times
    events['response'] = pd.to_numeric(events.response, errors='coerce')
    # Duplicate IDs are flagged; source row remains the unique lineage key.
    events['duplicate_source_event_id'] = events.source_event_id.duplicated(keep=False)
    mapping = items.set_index('problem_id')
    events['known_item'] = events.problem_id.isin(mapping.index)
    events['original_skill_ids'] = events.problem_id.map(mapping.original_skill_ids).fillna('[]')
    events['original_skill_count'] = events.problem_id.map(mapping.original_skill_count).fillna(0).astype(int)
    events['verified_text_only_candidate'] = events.problem_id.map(mapping.verified_text_only_candidate).fillna(False).astype(bool)
    events['single_skill_text_candidate'] = events.verified_text_only_candidate & events.original_skill_count.eq(1)
    events['basic_fields_valid'] = (events.student_id.ne('') & events.source_event_id.ne('') & events.response.isin([0, 1]) & times.notna() & events.known_item & events.original_skill_count.gt(0))
    assert len(events) == manifest['validation']['observed_interaction_rows']
    assert events.source_row.is_unique
    duplicate_keys = events.duplicated(['student_id', 'problem_id', 'source_end_time', 'response'], keep=False)
    events['repeated_student_item_time_response'] = duplicate_keys
    events = events.sort_values(['student_id', 'end_time_utc', 'source_row'], kind='stable').reset_index(drop=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': manifest['source']['resolved_revision'],
              'source_manifest_sha256':sha(raw/'download_manifest.json'), 'preprocessor_sha256':sha(Path(__file__)),
              'counts': {'source_interactions':len(events), 'students':events.student_id.nunique(),
                         'source_problem_rows':len(p), 'unique_problem_ids':len(items), 'source_skill_rows':len(s),
                         'unique_skill_ids':s.skill_id.nunique(), 'unique_skill_codes':s.node_code.nunique(),
                         'duplicate_problem_id_groups':int((items.source_row_count>1).sum()),
                         'conflicting_problem_input_groups':int(items.conflicting_input_variants.sum()),
                         'items_with_multiple_skills':int(items.original_skill_count.gt(1).sum()),
                         'items_with_unresolved_images':int(items.unresolved_image_count.gt(0).sum()),
                         'items_with_unsupported_math':int(items.unsupported_math_tags.ne('[]').sum()),
                         'verified_text_only_items':int(items.verified_text_only_candidate.sum()),
                         'basic_valid_events':int(events.basic_fields_valid.sum()),
                         'events_on_verified_text_only_items':int((events.basic_fields_valid & events.verified_text_only_candidate).sum()),
                         'single_skill_text_candidate_events':int((events.basic_fields_valid & events.single_skill_text_candidate).sum()),
                         'duplicate_source_event_id_rows':int(events.duplicate_source_event_id.sum()),
                         'repeated_student_item_time_response_rows':int(duplicate_keys.sum()),
                         'invalid_timestamps':int(times.isna().sum())},
              'skill_count_by_item':{str(k):int(v) for k,v in items.original_skill_count.value_counts().sort_index().items()},
              'scope':'Source-traceable full event ledger and item text audit, not final training population or paired exports.',
              'policies':{'source_event_identity':'release revision + source_row, preserving official id separately',
                          'order':'student_id, parsed UTC end_time, source_row stable tie-break',
                          'student_answer_text_ingested':False,'answer_keys_embedded':False,
                          'multi_KC':'full original sets retained; no scalar projection',
                          'ambiguous_item_inputs':'flagged, cleaned text empty; never keep an arbitrary variant',
                          'visual_content':'math-image MathML rendered; other images flagged, alt retained for review',
                          'split':'not yet frozen; final eligibility and multi-KC policy pending'},
              'checks':{'source_hashes_match':True,'one_item_row_per_id':True,'event_count_preserved':True,
                        'source_row_identity_unique':True,'no_many_to_many_join':True}}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.', dir=output.parent))
    try:
        items.to_csv(staging/'questions.csv', index=False)
        s.to_csv(staging/'original_skill_sets.csv', index=False)
        events.to_csv(staging/'events.csv.gz', index=False, compression={'method':'gzip','compresslevel':1,'mtime':0})
        report['files']={f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in staging.iterdir() if f.is_file()}
        (staging/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        os.rename(staging,output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(report['counts'],indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=DEFAULT_RAW)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    prepare(args.raw,args.output)
