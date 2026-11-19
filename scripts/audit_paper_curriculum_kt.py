"""Read legacy artifacts and record whether paper map-back AUC can be computed.

Run with the existing thesis environment and -B. This script does not evaluate
or train models, modify source repositories, or access online experiment logs.
"""

import collections
import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')
PAPER = Path('/Users/kiakarbasi/mathnlp_paper')
OUTPUT = PROJECT / 'results' / 'paper_curriculum_kt_audit_20261107'
TARGETS = {
    'assist2012': {'pca': 100, 'clusters': 300, 'paper_baseline_auc': 0.7367,
                   'paper_induced_auc': 0.7841},
    'assist2017': {'pca': 10, 'clusters': 100, 'paper_baseline_auc': 0.7131,
                   'paper_induced_auc': 0.7403},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_group(group):
    configs = []
    counts = collections.Counter()
    representatives = {}
    predictions = 0
    for path in sorted(group.rglob('config.json')):
        if not (path.parent / 'qid_model.ckpt').is_file():
            continue
        config = json.loads(path.read_text())
        dimension = config['data_config']['num_c']
        counts[str(dimension)] += 1
        representatives.setdefault(str(dimension), path)
        configs.append({'path': str(path), 'sha256': sha256(path),
                        'num_concepts': dimension})
        predictions += (path.parent / 'qid_test_question_predictions.txt').is_file()
    inspected = []
    # CPU loading here inspects tensor dimensions only; it performs no inference.
    import torch
    for dim, path in representatives.items():
        checkpoint = path.parent / 'qid_model.ckpt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        shapes = {key: list(value.shape) for key, value in state.items()
                  if key.startswith(('out_layer.', 'interaction_emb.'))}
        inspected.append({'config_path': str(path), 'checkpoint_path': str(checkpoint),
                          'checkpoint_sha256': sha256(checkpoint),
                          'configured_concepts': int(dim), 'tensor_shapes': shapes})
    return {'checkpoint_count': len(configs), 'configured_concept_counts': dict(counts),
            'question_prediction_file_count': predictions,
            'representative_tensor_checks': inspected, 'configs': configs}


def archived_condition(dataset, target):
    relative = f'wandb_exports/{dataset}_final_publication/results.csv'
    ref = f'eab57c6:{relative}'
    run = subprocess.run(['git', 'show', ref], cwd=SOURCE, capture_output=True,
                         text=True, check=True)
    rows = list(csv.DictReader(io.StringIO(run.stdout)))
    selected = [row for row in rows if row['condition_type'] == 'clustered'
                and int(float(row['pca_dim'])) == target['pca']
                and int(float(row['n_clusters'])) == target['clusters']]
    baseline = [row for row in rows if row['condition_type'] == 'baseline']
    return {'source_git_object': ref,
            'source_content_sha256': hashlib.sha256(run.stdout.encode()).hexdigest(),
            'target_rows': selected,
            'target_row_count': len(selected),
            'target_mean_logged_testauc': sum(float(r['testauc']) for r in selected) / len(selected),
            'target_test_equals_validation_count': sum(float(r['testauc']) == float(r['validauc'])
                                                       for r in selected),
            'baseline_row_count': len(baseline),
            'baseline_mean_logged_testauc': (sum(float(r['testauc']) for r in baseline) / len(baseline)
                                             if baseline else None),
            'has_run_id_column': 'run_id' in rows[0],
            'has_embedding_model_column': 'embedding_model' in rows[0]}


def main():
    # This is the only write destination.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = {'method_name': 'Curriculum KT', 'generated_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'awaiting_historical_artifacts',
             'source_root': str(SOURCE), 'training_runs': 0, 'inference_runs': 0,
             'mapped_auc_computed': False, 'datasets': {},
             'paper_source': {'path': str(PAPER / 'sections/04_results.tex'),
                              'sha256': sha256(PAPER / 'sections/04_results.tex')},
             'required_recovery': [
                 'Historical induced and original DKT checkpoints or aligned prediction exports',
                 'Exact cluster-to-output-index and question-to-cluster mappings',
                 'Original concept assignments, including policy for item-label conflicts',
                 'Matching held-out events, pre-response histories, masks, and student splits',
                 'Run/embedding/seed identifiers and selection/aggregation provenance',
                 'Separate development/calibration predictions for learned alternatives'],
             'search_scope': ['legacy examples/models', 'legacy studies', 'legacy examples/wandb',
                              'dkt_archive', 'pykt', 'pykt_emb_chef', 'pykt_temp_git', 'dktt_light', 'thesis'],
             'search_limit': 'Local search only; no assertion about remote artifacts or unsearched locations.'}
    for dataset, target in TARGETS.items():
        group = SOURCE / 'examples/models' / f'dkt_tiaocan_{dataset}'
        inventory = inspect_group(group)
        audit['datasets'][dataset] = {
            **target, 'mapped_auc': None, 'mapped_minus_baseline_auc': None,
            'legacy_model_inventory': inventory,
            'archived_condition': archived_condition(dataset, target),
            'matching_induced_dimension_found_in_legacy_models':
                str(target['clusters']) in inventory['configured_concept_counts'],
        }
    (OUTPUT / 'artifact_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    summary = {dataset: {
        'paper_baseline_auc': data['paper_baseline_auc'],
        'paper_induced_auc_before_mapping': data['paper_induced_auc'],
        'mapped_auc': data['mapped_auc'],
        'saved_concept_counts': data['legacy_model_inventory']['configured_concept_counts'],
        'archived_target_rows': data['archived_condition']['target_row_count'],
        'archived_target_mean_logged_auc': data['archived_condition']['target_mean_logged_testauc'],
    } for dataset, data in audit['datasets'].items()}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
