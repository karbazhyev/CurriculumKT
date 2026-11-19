"""Produce the requested convergence/readiness report; never fit a model."""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_wandb_convergence import OUT, ROOT, SOURCE, dump


def main():
    audit = json.loads((OUT / 'selected_run_convergence.json').read_text())
    online = json.loads((OUT / 'online_history_verification.json').read_text())
    smoke = json.loads((OUT / 'mps_smoke_checks.json').read_text())
    assert len(smoke['checks']) == 8 and all(r['passed'] for r in smoke['checks'])
    assert len(online['runs']) == 12 and all(r.get('local_online_max_auc_difference') == 0 for r in online['runs'])
    for name, digest in smoke['source_sha256'].items():
        assert hashlib.sha256((SOURCE / 'pykt/models' / f'{name}.py').read_bytes()).hexdigest() == digest
    groups = []
    for dataset in ['assist2012', 'assist2017']:
        for taxonomy in ['original', 'induced']:
            if dataset == 'assist2012':
                rows = [r for r in audit['historical_wandb_runs'] if r['dataset']==dataset and r['taxonomy']==taxonomy]
            else:
                rows = [r for r in audit['matched_2017_local_logs'] if Path(r['path']).parent.name.startswith(taxonomy)]
            groups.append((dataset, taxonomy, rows))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    colors = ['#176B87', '#B85C38', '#6A4C93']
    for ax, (dataset, taxonomy, rows) in zip(axes.flat, groups):
        twin = ax.twinx()
        for i, row in enumerate(rows):
            history = [r for r in row['history'] if not r.get('checkpoint_summary_merged_into_epoch')]
            epoch = [r['epoch'] for r in history]
            auc = [r.get('validauc', r.get('validation_auc')) for r in history]
            seed = row.get('seed', Path(row.get('path','')).parent.name.split('seed')[-1])
            ax.plot(epoch, auc, color=colors[i], lw=1.8, label=f'Seed {seed}')
            peak = max(range(len(auc)), key=auc.__getitem__)
            ax.scatter(epoch[peak], auc[peak], color=colors[i], marker='o', s=25, zorder=4)
            twin.plot(epoch, [r['train_loss'] for r in history], color=colors[i], alpha=.35, ls='--', lw=1.2)
        ax.set_title(f"{dataset.replace('assist','ASSIST')} · {taxonomy} taxonomy", loc='left', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Validation AUC')
        twin.set_ylabel('Training loss (dashed)', color='#777777'); twin.tick_params(axis='y', colors='#777777')
        ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Validation performance peaks while training loss keeps falling', x=.06, ha='left', fontsize=15, fontweight='bold')
    fig.text(.06, .025, '2012: historical W&B curves. 2017: recent matched DKT fits. Dots mark validation peaks.\nFinal historical rows that mix checkpoint summaries with epoch metrics are excluded.', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0,.075,1,.95]); fig.savefig(OUT/'learning_curves.png', dpi=180); plt.close(fig)

    configs = [{'id':f'c{i:02d}', 'd_model':w, 'learning_rate':lr, 'dropout':drop}
               for i,(w,lr,drop) in enumerate(itertools.product([64,128],[.0003,.001],[.1,.5]))]
    proposal = {
        'state':'proposed_not_launched', 'scope':'First two-model validation-only tuning batch, not the full milestone campaign',
        'models':['simplekt','robustkt'], 'datasets':['assist2012','assist2017'], 'taxonomies':['original','induced'],
        'model_scope':'Recommended default pending the user scope reply',
        'wandb':{'entity':'knowledgetracing42-ucla', 'project':None, 'state':'awaiting destination clarification',
                 'recommended_existing_project':'assist2012_publication_kia',
                 'group':'currkt-model-pilot-20261107',
                 'write_policy':'New runs only; preserve existing runs. No raw student records, predictions, credentials, source snapshots or checkpoints uploaded automatically.'},
        'input_variant':'Supported n_pid=0 concept-only setting: past concepts/responses and queried target concept; preserve available question IDs for alignment, do not supply actual target item to map-back.',
        'candidates':configs, 'screening_seed':42, 'screening_fits':64,
        'proposed_followup_seeds':[84,126], 'additional_selected_configuration_fits':16,
        'total_if_screening_and_seed_confirmation_both_executed':80,
        'fixed':{'n_blocks':1, 'd_ff':256, 'num_attn_heads':4, 'n_pid':0, 'sequence_length':200,
                 'robustkt_kernel_size':5, 'optimizer':'Adam', 'batch_size':64, 'max_epochs':100,
                 'patience':15, 'patience_min_delta_auc':.0001,
                 'checkpoint':'Save every strictly better finite validation AUC; patience threshold is separate from checkpoint saving.'},
        'data_gate':{
            'assist2017':'Reuse verified matched train/validation/calibration/test sequence exports and mapping without re-embedding or reclustering.',
            'assist2012':'Create a paired original-label view of the existing induced sequence positions using the saved question-to-original mapping; both conditions must use the same positions, histories, item IDs, responses, resets and student memberships. Freeze hashes and counts before fitting.',
            'split':'Training folds2/3/4; checkpoint/configuration validation fold0; independent mapping calibration fold1; held-out fold-1 reserved.',
            'assist2012_limit':'Retains historical duplicate sequence positions and K1000 taxonomy. This is an exploratory legacy-export pilot, not a deduplicated benchmark or the paper K300 condition. New matched original models are required; old original checkpoint scores are contextual only.'},
        'selection':{
            'backbone':'For each dataset/model/taxonomy, select candidate by highest validation AUC; break exact ties by lower validation log loss, then fewer parameters, then configuration ID. Same search space and trial budget for original and induced.',
            'map_back':'Keep established fixed frequency map and pooled original+induced method. Fit/select mapping coefficients using calibration students only. The backbone screening criterion is native validation AUC, not a claim to optimize mapped AUC.',
            'reporting':'Compare original-model baseline, induced raw diagnostic, pure map-back, original+induced, and original+original control on identical original-taxonomy targets in the subsequent frozen evaluation batch. No test-driven model selection.',
            'boundary':'Screening and seed confirmation are separately resumable. No held-out scoring in the tuning pilot.'},
        'convergence_diagnostics':['finite losses and gradients','per-epoch training/validation loss and AUC','generalization gap','validation peak and subsequent decline','last-10-epoch validation slope','early stopping versus epoch cap','gradient norm and clipping fraction','learning rate','MPS device and parameter count','elapsed time and checkpoint reload'],
        'epoch_cap_policy':'Mark budget-limited runs explicitly; do not call them converged or silently extend/discard them.',
        'runtime_gate':'Time full epochs at actual batch size before launching the grid. The small correctness checks are not campaign runtime estimates.',
        'wandb_metrics':['epoch','train/loss','train/auc','validation/loss','validation/auc','validation/best_auc','optimization/grad_norm','optimization/learning_rate','runtime/epoch_seconds','runtime/mps_memory_bytes'],
        'wandb_config':['dataset','model','taxonomy','seed','configuration_id','source_sha256','data_sha256','split_sha256','mapping_sha256','device','parameter_count','protocol_sha256']}
    dump(OUT/'tuning_proposal.json', proposal)
    report = '''# Model readiness and W&B convergence audit

Date: 2026-11-07. Requested readiness/logging/tuning follow-up; no main milestone advancement.

**SimpleKT and RobustKT passed the initial MPS correctness checks. A tuning campaign has not been launched.** W&B authentication and historical online access work under `knowledgetracing42-ucla`; the destination for new runs needs the user's project clarification because the relevant historical runs span several projects.

## What ran

- Inventoried 3,544 local W&B journals without writing to the legacy repository.
- Read the epoch histories of 12 relevant historical W&B runs: six saved 2012 backbones and six 2017 records matching the earlier fixed settings. Verified all 12 online; their AUC histories match the local copies exactly and W&B marks all finished. Repeated 2017 histories are not independent replications.
- Audited the six recent matched 2017 DKT fits from their local histories. These six fits were local-only; they do not yet have W&B run IDs. No historical metrics were uploaded or relabeled.
- Ran eight one-update MPS checks: two models × two datasets × original/induced labels. These are correctness diagnostics using development data, not trained-model AUC results or convergence experiments.

## Convergence findings

| Models inspected | Peak validation epoch | Interpretation |
|---|---:|---|
| Historical 2012 original DKT, three seeds | 7–8 | Validation AUC peaked near 0.7345–0.7346, then fell to 0.6721–0.6785 in the last uncontaminated epoch while training loss improved. |
| Historical 2012 induced DKT, three seeds | 9–10 | Validation AUC peaked near 0.7841–0.7843, then fell to 0.7587–0.7630 while training loss improved. |
| Historical 2017 original DKT, six run records | 32–38 | Validation peaked and later declined; some records repeat the same training history. |
| Recent matched 2017 original DKT, three seeds | 39–40 | Best checkpoints were retained; validation subsequently declined during the ten-epoch patience window. |
| Recent matched 2017 induced DKT, three seeds | 38–41 | Best checkpoints were retained; validation subsequently declined during the ten-epoch patience window. |

These curves show learning followed by overfitting. They do not establish mathematical convergence to an optimum. Merely increasing the epoch limit is not supported by these runs. Tuning regularization, capacity and learning rate is more appropriate; new architectures still need their own measured curves.

![Learning curves](learning_curves.png)

Two logging/selection details matter:

1. The historical training code updates the saved checkpoint only after an AUC improvement greater than 0.001. Thus the selected epoch can precede the actual validation peak. The audited 2012 differences are below 0.001; this alone does not explain the map-back results.
2. The final historical W&B row merges last-epoch training metrics with best-checkpoint validation metrics and a `testauc` value equal to validation. This creates a misleading final validation rebound. Those mixed rows are explicitly excluded from the convergence diagnosis. Terminal `validauc` lines also show the retained best value rather than the current epoch value. Future logs must keep current validation, best validation and final test metrics distinct.

## Runtime checks

All eight checks passed finite forward/backward updates on MPS, save/reload, a CPU inference reference, prefix equivalence, padding independence, and invariance to current/future response changes and future concept changes. Reload and forbidden-information perturbation differences were zero; maximum CPU/MPS prediction difference was 4.71e-6, below the declared 2e-5 tolerance. The recorded eight update calls took 4.57 seconds including initial kernel warm-up, which is not an epoch or campaign estimate.

The checks used width64, one block, four attention heads, dropout0.1 and sequence length200. Wider/deeper settings are not certified by this check. They used the supported concept-only setting (`n_pid=0`) so original-taxonomy querying does not require the actual next item. The original model source is unchanged; a local adapter sets its device globals to MPS. No environment installation or CPU training fallback occurred.

## Concrete tuning proposal — not executed

Start with SimpleKT and RobustKT on both datasets and both label sets. Search eight settings per cell: widths64/128 × learning rates0.0003/0.001 × dropout0.1/0.5, holding other settings fixed. This is 64 screening fits at seed42, followed by 16 selected-setting fits at seeds84/126 if the seed-confirmation batch is run. Both original and induced conditions receive the same search budget.

Use validation-only checkpoint/configuration selection, up to100 epochs with15-epoch patience. Save the actual best checkpoint independently of the patience threshold. Record epoch-cap hits as unresolved convergence, preserve failures, and measure actual epoch cost before launching the grid. The pilot evaluates native validation AUC; improved induced validation AUC does not guarantee improved original-taxonomy map-back AUC. The established map-back and augmentation methods remain a subsequent frozen comparison, including the original+original control.

Reuse the paired 2017 exports. For 2012, pair the original-label view with the same saved induced sequence positions and training/validation/calibration/test memberships before fitting. Preserve and disclose the historical duplicates and K1000 condition. Existing 2012 native scores cannot serve as matched baselines for newly trained models. No new embedding computation, clustering selection, paper editing, or held-out scoring is part of this readiness pass.

New W&B runs will use the existing account and the clarified existing project, grouped by batch and tagged by dataset/model/taxonomy/seed. Log per-epoch training/validation AUC and loss, learning rate, gradient diagnostics, stopping reason, timing, device and provenance hashes. Aggregate metrics and nonsensitive configurations are sufficient; do not automatically upload student records or source/checkpoint bundles.

## Artifacts and state

- [Tuning proposal](tuning_proposal.json)
- [Per-run convergence data and local 2017 histories](selected_run_convergence.json)
- [Online W&B verification and run URLs](online_history_verification.json)
- [MPS correctness results](mps_smoke_checks.json)
- [Local W&B inventory](local_wandb_inventory.json)

The model-scope and destination questions are pending. No sweep was created and no new run was uploaded to W&B. The full milestone campaign remains unstarted; M02 remains next. Legacy, paper and thesis folders are unchanged, and Git pushes remain paused.
'''
    (OUT/'REPORT.md').write_text(report)
    verification = {'eight_smoke_checks_passed':True, 'online_local_histories_exact_match':True,
                    'verified_online_runs':12, 'legacy_model_source_hashes_unchanged':True,
                    'new_empirical_training_fits':0, 'development_one_update_diagnostics':8,
                    'new_heldout_auc_evaluations':0, 'wandb_new_runs':0,
                    'hyperparameter_grid_state':'proposal', 'main_next_milestone':'M02'}
    dump(OUT/'verification.json', verification)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
