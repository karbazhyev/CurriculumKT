"""Final identity/alignment checks and project reporting for the completed batch."""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_readout_followup import OUT, OLD, A17, ROOT, runs, folder, save_json
from evaluate_saved_curriculum_kt import sha


def main():
    reg=json.loads((OUT/'run_registry.json').read_text())
    assert reg['state']=='complete' and reg['earlier_2017_methods']=='complete'
    ver=json.loads((OUT/'verification.json').read_text())
    prepared=json.loads((A17/'protocol.json').read_text())
    for path,digest in prepared['source_sha256'].items():assert sha(Path(path))==digest
    for filename,digest in prepared['derived_sha256'].items():assert sha(A17/filename)==digest
    for ds in ['assist2012','assist2017']:
        for run in runs(ds):
            assert sha(Path(run['checkpoint']))==run['checkpoint_sha256']
            fit=json.loads((folder(ds,run)/'fit_complete.json').read_text())
            for method,digest in fit['coefficient_hashes'].items():
                assert sha(folder(ds,run)/f'{method}_coefficients.npz')==digest
    for run in runs('assist2017'):
        key=f"{run['kind']}_seed{run['seed']}"
        old=dict(np.load(A17/f'{key}_test_predictions.npz'));new=dict(np.load(folder('assist2017',run)/'test_predictions.npz'))
        for field in ['uid','row','position','label','original_concept']:
            assert np.array_equal(old[field],new[field]),(key,field)
        fields=['original_dkt'] if run['kind']=='original' else ['raw_induced','item_overlap','practice_frequency','equal_children']
        for field in fields:assert np.allclose(old[field],new[field],atol=3e-6,rtol=1e-5),(key,field)
    audit=json.loads((ROOT/'results/paper_curriculum_kt_audit_20261107/artifact_audit.json').read_text())
    assert sha(Path(audit['paper_source']['path']))==audit['paper_source']['sha256']
    for seed in [42,84,126]:
        a=json.loads((OUT/'assist2012'/f'old_objective_audit_seed{seed}'/'audit.json').read_text())
        assert sha(OLD/f'learned_seed{seed}_coefficients.npz')==a['old_coefficients_sha256_unchanged']
    scripts=['prepare_readout_followup.py','run_readout_followup.py','summarize_readout_followup.py',
             'execute_readout_followup.py','complete_prior_2017_methods.py','finalize_readout_followup.py']
    for script in scripts:ast.parse((ROOT/'scripts'/script).read_text())
    report=OUT/'RESULTS.md';text=report.read_text()
    marker='## Outcome of this batch\n'
    conclusion=('## Outcome of this batch\n\n'
      'None of the new map-back readouts exceeds native original-taxonomy DKT on either dataset in the common-calibration comparison. '
      'The strongest new mapped readout is positive mapped weights in 2012 (0.730657 versus native 0.735370) and the induced-state head in 2017 (0.703966 versus 0.717069). '
      'The development-selected original control is scalar calibration in all three 2012 runs (mean 0.742552) and the native model in all three 2017 runs. '
      'Thus this batch does not establish transfer of the induced models\' higher raw AUC into better original-taxonomy forecasting.\n\n'
      'The 2017 state head improves over its positive mapped readout by +0.013416 (paired interval [+0.008677, +0.017710]), '
      'but remains below native original DKT by -0.013104 ([-0.017782, -0.008603]). '
      'Against an equally fitted original state head, its difference is -0.003015 ([-0.007395, +0.000986]). '
      'These are conditional, exploratory intervals. [Additional paired method contrasts](paired_method_contrasts.csv).\n\n'
      'The poor held-out performance of the wider 2012 readouts despite lower fitting loss is consistent with overfitting in this limited calibration set. '
      'It does not prove that broader readouts can never help. No settings were revised to chase these test outcomes.\n\n')
    if marker not in text:
        parts=text.split('\n\n',2);text=parts[0]+'\n\n'+parts[1]+'\n\n'+conclusion+parts[2]
    if '## Compute and model sizes' not in text:
        b=json.loads((OUT/'backbone_summary.json').read_text())
        text+='\n## Compute and model sizes\n\n'
        text+=f"Six new 2017 DKT fits took {b['2017_recorded_training_seconds']:.1f} seconds of recorded training time, excluding preprocessing, mapping fits and reporting. Six 2012 checkpoints were reused. All training/inference and readout data gradients used MPS; optimizer control, metrics and fixed-coefficient verification used CPU.\n\n"
        text+='| Dataset | Original DKT parameters | Induced DKT parameters |\n|---|---:|---:|\n'
        for ds in ['assist2012','assist2017']:
            vals={r['kind']:r['parameter_count'] for r in b['backbones'] if r['dataset']==ds and r['seed']==42}
            text+=f"| {ds} | {vals['original']:,} | {vals['induced']:,} |\n"
        text+='\nThe 2017 hidden widths and optimization settings match; different taxonomy cardinalities still change embedding/output parameter counts. Historical 2012 widths do not match. [Backbone identities and counts](backbone_summary.json).\n'
    report.write_text(text)
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target:assert (report.parent/target).resolve().exists(),target
    ver.update({'source_2017_hashes_unchanged':True,'prepared_2017_data_hashes_unchanged':True,
        'old_and_new_2017_predictions_aligned':True,'inspected_paper_source_hash_unchanged':True,
        'old_2012_readout_coefficients_preserved':True,'all_local_report_links_resolve':True,
        'scripts_parse':True,'script_identities':{s:sha(ROOT/'scripts'/s) for s in scripts}})
    ver['outputs'].update({p.name:sha(p) for p in [report,OUT/'paired_method_contrasts.csv',OUT/'backbone_summary.json']})
    save_json(OUT/'verification.json',ver)
    # Update the earlier report's status without changing its recorded 2012 rows.
    previous=ROOT/'results/curriculum_kt_all_methods_20261107/RESULTS.md'
    old=previous.read_text()
    note=('> **Follow-up completed:** The user subsequently authorized the six matched 2017 fits and the new readout comparison. '
          'Both datasets and all earlier mapping families now have results in the '
          '[completed follow-up report](../curriculum_kt_readout_followup_20261107/RESULTS.md). '
          'The preparation-only section below records the earlier state.\n\n')
    if '> **Follow-up completed:**' not in old:
        old=old.replace('## Executed 2012 results\n',note+'## Executed 2012 results\n')
        previous.write_text(old)
    status=ROOT/'research/PROJECT_STATUS.md';s=status.read_text()
    begin=s.index('Current state:');end=s.index('\n\nNext milestone:',begin)
    s=s[:begin]+('Current state: the authorized 2012/2017 map-back follow-up is complete. All six saved 2012 DKT checkpoints were reused; six necessary matched 2017 fits and all requested readout comparisons finished on MPS. None of the new mapped methods exceeded native original DKT in the common-calibration comparison. Main execution milestones have not started. Exact historical paper-condition checkpoints remain unrecovered.')+s[end:]
    a=s.index('### Active authorized follow-up (2026-11-07)');b=s.index('\nThe established sibling environment',a)
    section='''### Completed authorized follow-up (2026-11-07)

The user explicitly authorized execution and 2017 training, using W&B logs to train only what was necessary. The earlier training clarification is resolved.

- Inspected 1,033 local 2017 W&B summaries with model paths; six matched the fixed archived width256/dropout0.5/lr0.0005 settings. All 1,042 main saved 2017 configurations had 68 outputs. Saved original models included the fold now reserved for independent calibration, so three original and three induced models were needed for the matched design. No sweep or embedding generation.
- Completed six 2017 MPS fits in 184.9 seconds of recorded training time, with validation reload and causality checks. Best epochs: original39/39/40, induced38/38/41. Reused all six 2012 checkpoints.
- Audited the exact earlier 2012 monotone objective without modifying its features or coefficients. All three constrained fits passed independent gradient checks. Mean AUC changed only from 0.737899 to 0.737920; lack of optimizer convergence did not explain the earlier gap.
- New 2012 readouts use 596 common validation students and identical re-encoded histories; test remains 63,869 positions/1,464 students. Native original0.735370; development-selected scalar-calibrated original0.742552; positive mapped0.730657; signed mapped0.727787; soft mapping0.709076; full logits0.665893; induced state0.685190. Historical backbone training/widths remain unmatched, previous test inspection and repeated positions remain limitations, and earlier full-calibration rows are preserved separately.
- New matched 2017 test: 48,389 predictable targets/340 students. Native and development-selected original0.717069; positive mapped0.690550; signed mapped0.693060; soft mapping0.701023; full logits0.699803; induced state0.703966. Raw induced0.731652 is a target-induced-label diagnostic, not map-back.
- The 2017 state head improves over positive mapping by +0.013416, but remains below native original by -0.013104 (paired student 95% interval [-0.017782,-0.008603]). Primary induced-state versus selected-original differences are negative in both datasets. No test-driven retuning was performed; null/negative outcomes are retained.
- Completed the earlier intersection/item-aware/calibration/AND/OR mapping families for 2017 on the same backbones. Their original fixed-step fitting protocol is retained and distinguished from the convergence-checked new fits. Two-dataset tables include both batches with explicit calibration-population labels.
- Checks passed: all 42 main readout fits meet the declared projected-gradient tolerance; calibration-only selection; frozen checkpoint/coefficient hashes; identical paired test events and labels; independent 2017 student splits; unchanged source and prepared-data hashes; earlier/new 2017 prediction alignment; preserved earlier 2012 coefficients; inspected paper hash unchanged; scripts parse and report links resolve.
- Outputs: [complete results](../results/curriculum_kt_readout_followup_20261107/RESULTS.md), [new-method table](../results/curriculum_kt_readout_followup_20261107/two_dataset_comparison.csv), [earlier-method table](../results/curriculum_kt_readout_followup_20261107/earlier_method_families_two_datasets.csv), [paired contrasts](../results/curriculum_kt_readout_followup_20261107/paired_method_contrasts.csv), [verification](../results/curriculum_kt_readout_followup_20261107/verification.json), [W&B reuse audit](../results/curriculum_kt_readout_followup_20261107/wandb_reuse_audit.json), [run registry](../results/curriculum_kt_readout_followup_20261107/run_registry.json).
- This bounded batch is complete. No main milestone was advanced; M02 remains next. No paper/thesis or legacy-source edits and no Git pushes occurred.
'''
    s=s[:a]+section+s[b:]
    s=s.replace('M05 still needs to verify the additional planned model paths; no fresh DKT or campaign model was trained.','M05 still needs to verify the additional planned model paths. Six fresh 2017 DKT models were trained only for the authorized follow-up; the main campaign has not started.')
    s=s.replace('Only the specifically authorized saved-model evaluation and learned-map fitting ran. No campaign, new embeddings, paper submissions, or Git pushes were launched.','Only the specifically authorized saved-model evaluations, six-fit 2017 batch, and mapping-layer comparisons ran. No main campaign, new embeddings, paper submissions, or Git pushes were launched.')
    status.write_text(s)
    print(json.dumps({'batch':'complete','verification':str(OUT/'verification.json'),'main_readout_fits':ver['main_readout_fits'],'new_dkt_fits':6,'report':str(report)}),flush=True)


if __name__=='__main__':main()
