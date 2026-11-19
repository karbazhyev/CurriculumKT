"""Independent streaming checks against canonical events; no API or training."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import argparse
import csv
import gzip
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from foundationalassist_preparation import DATA, EVENTS, PROTOCOL, RESULTS, ROLES, dump, sha, utc, verify_manifest


def sequence_events(arrays):
    for i, uid in enumerate(arrays['uid']):
        n = int((arrays['questions'][i] >= 0).sum())
        for j in range(n):
            yield int(uid), {k: int(arrays[k][i, j]) for k in
                            ['questions', 'original_concepts', 'responses', 'timestamps', 'event_id', 'source_row']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-paired', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    shared = DATA / 'matched_events_v1'
    exports = DATA / 'paired_exports_v1'
    protocol = json.loads(PROTOCOL.read_text())
    verify_manifest(shared / 'manifest.json')
    with (DATA / 'text_audit_v1/questions.csv').open() as f:
        texts = {r['problem_id']: r for r in csv.DictReader(f)}
    with (shared / 'items.csv').open() as f:
        items = {r['problem_id']: r for r in csv.DictReader(f)}
    with (DATA / 'canonical_v1/student_split.csv').open() as f:
        expected_roles = {r['student_id']: r['split'] for r in csv.DictReader(f)}
    students = json.loads((shared / 'student_vocabulary.json').read_text())
    expected_items, seen, actual_role_counts, student_events = set(), set(), Counter(), Counter()
    arrays = {}
    checks = {}
    for role in ROLES:
        with np.load(shared / f'{role}_sequences.npz', allow_pickle=False) as a:
            arrays[role] = dict(a)
        with np.load(shared / 'original' / f'{role}_sequences.npz', allow_pickle=False) as b:
            baseline = dict(b)
        for key, value in arrays[role].items():
            assert np.array_equal(value, baseline['concepts' if key == 'original_concepts' else key])
        valid = arrays[role]['questions'] >= 0
        assert (np.diff(valid.astype(int), axis=1) <= 0).all()
        expected_mask = np.zeros_like(valid)
        expected_mask[:, 1:] = valid[:, 1:] & valid[:, :-1]
        assert np.array_equal(arrays[role]['target_mask'], expected_mask)
        for name in ['original_concepts', 'responses', 'event_id', 'source_row', 'timestamps']:
            assert np.array_equal(arrays[role][name] >= 0, valid)
    iterators = {role: iter(sequence_events(a)) for role, a in arrays.items()}
    excluded = Counter()
    previous = {}
    with gzip.open(EVENTS, 'rt', newline='') as source, gzip.open(shared / 'events.csv.gz', 'rt', newline='') as saved:
        output = iter(csv.DictReader(saved))
        for original in csv.DictReader(source):
            text = texts[original['problem_id']]
            if text['verified_text_only_candidate'] != 'True':
                excluded['incomplete_text'] += 1
                continue
            if len(json.loads(text['original_skill_ids'])) != 1:
                excluded['multiple_original_skills'] += 1
                continue
            row = next(output)
            for key in ['event_id', 'source_event_id', 'source_row', 'student_id', 'problem_id',
                        'original_skill_ids', 'response', 'end_time_utc', 'split', 'source_duplicate_count']:
                assert row[key] == original[key], key
            item = items[row['problem_id']]
            assert int(row['original_concept']) == int(item['original_concept'])
            assert json.loads(row['original_skill_ids']) == [item['original_skill_id']]
            assert expected_roles[row['student_id']] == row['split']
            assert int(row['uid']) == students[row['student_id']]
            assert int(row['question_index']) == int(item['question_index'])
            instant_ns = pd.Timestamp(original['end_time_utc']).value
            expected_ms = instant_ns // 1000000
            assert int(row['timestamp_ms']) == expected_ms
            ordered_key = (instant_ns, int(row['source_row']))
            assert ordered_key >= previous.get(row['student_id'], ordered_key)
            previous[row['student_id']] = ordered_key
            uid, event = next(iterators[row['split']])
            assert uid == int(row['uid'])
            assert event == {'questions': int(item['question_index']), 'original_concepts': int(item['original_concept']),
                             'responses': int(original['response']), 'timestamps': expected_ms,
                             'event_id': int(original['source_row']), 'source_row': int(original['source_row'])}
            assert original['event_id'] not in seen
            seen.add(original['event_id'])
            expected_items.add(row['problem_id'])
            actual_role_counts[row['split']] += 1
            student_events[(row['split'], uid)] += 1
        assert next(output, None) is None
    assert all(next(it, None) is None for it in iterators.values())
    assert expected_items == set(items)
    for role in ROLES:
        assert actual_role_counts[role] == protocol['role_counts'][role]['events']
        chunk_counts = Counter(int(x) for x in arrays[role]['uid'])
        assert all(chunk_counts[uid] == (n + 199) // 200 for (r, uid), n in student_events.items() if r == role)
    assert all(not set(arrays[a]['uid']) & set(arrays[b]['uid']) for a, b in itertools.combinations(ROLES, 2))
    assert len(seen) + sum(excluded.values()) == 1709601
    checks.update(every_event_matches_canonical_source=True,exact_eligibility_population=True,
                  original_skill_sets_and_labels_preserved=True,exact_student_roles=True,
                  timestamps_and_order_preserved=True,arrays_trace_to_every_source_event=True,
                  no_duplicate_events=True,no_missing_or_extra_events=True,no_overlapping_students=True,
                  masks_and_padding_correct=True,chunk_boundaries_match_protocol=True,
                  original_baseline_equals_shared_arrays=True)
    paired_complete = (exports / 'manifest.json').exists()
    if args.require_paired:
        assert paired_complete, 'Paired exports are not complete.'
    if paired_complete:
        verify_manifest(exports / 'manifest.json')
        with np.load(exports / 'mapping.npz', allow_pickle=False) as saved:
            mapping = dict(saved)
        frequency = np.zeros_like(mapping['raw_training_event_counts'])
        for role in ROLES:
            with np.load(exports / 'paired' / f'{role}_sequences.npz', allow_pickle=False) as saved:
                paired = dict(saved)
            with np.load(exports / 'original' / f'{role}_sequences.npz', allow_pickle=False) as saved:
                original = dict(saved)
            with np.load(exports / 'induced' / f'{role}_sequences.npz', allow_pickle=False) as saved:
                induced = dict(saved)
            for key in arrays[role]:
                assert np.array_equal(arrays[role][key], paired[key])
                assert np.array_equal(arrays[role][key], original['concepts' if key == 'original_concepts' else key])
            for key in original:
                if key != 'concepts':
                    assert np.array_equal(original[key], induced[key])
            valid = arrays[role]['questions'] >= 0
            assert np.array_equal(induced['concepts'][valid], mapping['q_to_induced'][arrays[role]['questions'][valid]])
            assert np.array_equal(induced['concepts'], paired['induced_concepts'])
            if role == 'train':
                np.add.at(frequency, (original['concepts'][valid], induced['concepts'][valid]), 1)
        assert np.array_equal(frequency, mapping['raw_training_event_counts'])
        assert np.allclose(mapping['practice_frequency'].sum(1), 1)
        checks.update(paired_same_population_order_responses_masks=True,
                      complete_induced_item_alignment=True,only_taxonomy_fields_differ=True,
                      mapping_frequencies_use_training_only=True)
    report = {'created_utc': utc(), 'checks': checks, 'passed': all(checks.values()),
              'scope': 'Complete paired exports' if paired_complete else 'Shared population and original baseline only; induced labels await API embeddings.',
              'canonical_events_independently_checked': len(seen), 'role_events': dict(actual_role_counts),
              'excluded_events': dict(excluded), 'paired_exports_complete': paired_complete,
              'elapsed_seconds': time.monotonic() - started, 'source_sha256': sha(Path(__file__)),
              'protocol_sha256': sha(PROTOCOL), 'shared_manifest_sha256': sha(shared / 'manifest.json')}
    destination = RESULTS / ('paired_export_verification_20261109.json' if paired_complete else 'shared_export_verification_20261109.json')
    if destination.exists():
        destination = destination.with_name(destination.stem + '.recheck.' + str(time.time_ns()) + '.json')
    dump(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
