"""Summarize the frozen two-dataset comparison, retaining every method."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_saved_curriculum_kt import metrics, sha
from summarize_saved_curriculum_kt import weighted_auc_function
from run_readout_followup import OUT, OLD, A17, INDUCED, CONTROLS, runs, folder, save_json, registry


def summarize(dataset):
    dest=OUT/dataset;rows=[];per=[];preds={};ref=None
    baseline_choices={}
    for run in runs(dataset):
        path=folder(dataset,run);d=dict(np.load(path/'test_predictions.npz'))
        ids={k:d[k] for k in ['uid','row','position','label','original_concept']}
        if ref is None:ref=ids
        else:assert all(np.array_equal(ref[k],ids[k]) for k in ids),path
        methods=INDUCED+['item_overlap','practice_frequency','equal_children','raw_induced'] if run['kind']=='induced' else ['original_dkt']+CONTROLS+['original_selected']
        if run['kind']=='original':baseline_choices[str(run['seed'])]=json.loads((path/'baseline_selection.json').read_text())
        for method in methods:
            score=metrics(d['label'],d[method],d['original_concept'])
            for c,n,auc in score.pop('per_original_concept'):
                per.append({'method':method,'seed':run['seed'],'original_concept':c,'events':n,'auc':auc})
            rows.append({'dataset':dataset,'method':method,'seed':run['seed'],**score})
            preds.setdefault(method,[]).append(d[method])
    frame=pd.DataFrame(rows);frame.to_csv(dest/'scores_by_seed.csv',index=False)
    pd.DataFrame(per).to_csv(dest/'per_concept_auc.csv',index=False)
    agg=frame.groupby('method',sort=False).agg(auc=('auc','mean'),seed_sd=('auc','std'),
        brier=('brier','mean'),log_loss=('log_loss','mean'),macro_auc=('within_original_macro_auc','mean'),
        within_weighted_auc=('within_original_weighted_auc','mean'),events=('events','first')).reset_index()
    y=ref['label'];functions={k:[weighted_auc_function(y,p) for p in pp] for k,pp in preds.items()}
    _,uid=np.unique(ref['uid'],return_inverse=True);n=int(uid.max()+1)
    rng=np.random.default_rng(20261107)
    boot={k:{b:[] for b in ['original_dkt','original_selected']} for k in preds}
    for i in range(500):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[uid]
        values={k:float(np.mean([f(w) for f in ff])) for k,ff in functions.items()}
        for method in preds:
            for b in boot[method]:boot[method][b].append(values[method]-values[b])
    for baseline in ['original_dkt','original_selected']:
        baseline_auc=float(agg.loc[agg.method==baseline,'auc'].iloc[0])
        agg['delta_'+baseline]=agg.auc-baseline_auc
        intervals=np.array([np.quantile(boot[m][baseline],[.025,.975]) for m in agg.method])
        agg['ci_low_'+baseline]=intervals[:,0];agg['ci_high_'+baseline]=intervals[:,1]
    agg.to_csv(dest/'method_comparison.csv',index=False)
    save_json(dest/'summary.json',{'dataset':dataset,'events':len(y),'students':n,
         'results':json.loads(agg.to_json(orient='records')),'baseline_choices':baseline_choices,
         'bootstrap':'500 paired student resamples conditional on fitted models; no multiple-comparison correction',
         'aggregation':'Mean per-seed AUC; predictions are not ensembled',
         'all_method_results_retained':True})
    save_json(dest/'bootstrap_deltas.json',boot)
    return agg


def main():
    summaries={d:summarize(d) for d in ['assist2012','assist2017']}
    names={'original_dkt':'Original DKT, native','original_calibration':'Original DKT, scalar calibration',
           'original_state_head':'Original DKT, fitted state head','original_selected':'Original comparator, development selected',
           'item_overlap':'Item-overlap average','practice_frequency':'Practice-frequency average','equal_children':'Equal-child average',
           'monotone_converged':'Mapped positive weights, convergence checked','signed_mapped':'Mapped signed weights',
           'soft_mapping':'Soft mapping: off-map information penalized','full_logits':'Unrestricted induced-logit head',
           'induced_state_head':'Induced hidden-state curriculum head','raw_induced':'Raw induced target-label prediction (diagnostic)'}
    order=list(names)
    wide=[]
    for m in order:
        row={'method':m,'description':names[m]}
        for ds,f in summaries.items():
            v=f[f.method==m].iloc[0]
            for k in ['auc','delta_original_dkt','delta_original_selected','ci_low_original_selected','ci_high_original_selected']:
                row[ds+'_'+k]=float(v[k])
        wide.append(row)
    pd.DataFrame(wide).to_csv(OUT/'two_dataset_comparison.csv',index=False)
    lines=['# Curriculum KT: two-dataset readout comparison','',
           'Executed 2026-11-07. Mean AUC over three model seeds per condition. All methods are retained. The 2012 comparison reuses historical models; 2017 uses six newly trained matched models. These are new results, not recovery of the exact paper checkpoints.','',
           '| Method | 2012 AUC | 2017 AUC |','|---|---:|---:|']
    for r in wide:lines.append(f"| {r['description']} | {r['assist2012_auc']:.6f} | {r['assist2017_auc']:.6f} |")
    lines+=['','## Main comparison and uncertainty','',
            'The primary comparison is the induced hidden-state curriculum head against the original-only comparator selected using development students. The native original model is always retained. Intervals below use 500 paired student bootstrap resamples and condition on the fitted models; they do not include all training uncertainty or correct selection among multiple reported methods.','',
            '| Dataset | Induced state head minus selected original | Paired 95% interval |','|---|---:|---|']
    for ds,f in summaries.items():
        r=f[f.method=='induced_state_head'].iloc[0]
        lines.append(f"| {ds} | {r.delta_original_selected:+.6f} | [{r.ci_low_original_selected:+.6f}, {r.ci_high_original_selected:+.6f}] |")
    lines+=['','## What changed','',
            '- Signed mapping removes only the sign bound from the mapped readout. Soft mapping permits other induced outputs with ten times stronger coefficient shrinkage. Full-logit and state heads provide unrestricted linear controls. Fixed penalties were declared before new test scoring.','- Original controls include the native model, positive scalar calibration, and a state head. A student-separated 20% calibration holdout chooses which original-only control to retain, then fitted controls are refitted on all calibration students. No test outcome chooses the comparator.','- In 2012 the 1,000 induced output rows span all 256 state dimensions, but coefficient penalties can still differ. In 2017, 100 output logits cannot span the entire 256-dimensional state; the full-logit and hidden-state methods are distinct.','- The mean per-seed AUC is reported, not the AUC of an ensemble. The raw induced row queries the actual next induced label and is not a parent-only mapped score.','',
            '## Optimization audit','',
            'The old 2012 positive-weight objective was independently refitted using the same cached features, responses, eligibility, and penalties. Original coefficients and earlier reported scores were preserved. These audit fits use the full old calibration population, so they are separate from the common-student comparison above.','',
            '| Seed | Old objective / eligible concept | New objective / eligible concept | Final projected gradient | Audit shared-test AUC |','|---|---:|---:|---:|---:|']
    audit=[]
    for seed in [42,84,126]:
        a=json.loads((OUT/'assist2012'/f'old_objective_audit_seed{seed}'/'audit.json').read_text());audit.append(a)
        i=a['convergence'];n=i['eligible_concepts']
        lines.append(f"| {seed} | {i['initial_objective_sum_over_concepts']/n:.8f} | {i['final_objective_sum_over_concepts']/n:.8f} | {i['projected_gradient_max']:.3g} | {a['test_auc']:.6f} |")
    lines+=['',f"The mean audit AUC is {np.mean([a['test_auc'] for a in audit]):.6f}; the earlier fixed-step learned map scored 0.737899. A lower development objective does not guarantee a higher held-out AUC.",'',
            '## Data and training scope','',
            '- **2012:** 63,869 saved test positions from 1,464 students, unchanged from the prior paired comparison. New method readouts use only the 596 validation students shared by both historical designs, re-encoded with identical histories. Earlier learned-map rows used the larger induced validation population and are not directly interchangeable with these new fitted rows. Historical backbone widths, training populations, splits, and seed sets differ; validation students influenced checkpoint selection, and repeated target keys remain. This is exploratory evidence.','- **2017:** 237,087 unique source events, 1,700 students, and 48,389 held-out predictable targets from 340 students. Train/validation/calibration/test memberships are separate and identical for original and induced models. Six DKT fits use the fixed archived/W&B settings (hidden256, dropout0.5, Adam0.0005, batch256), validation-only checkpoint selection, and early stopping. No hyperparameter sweep or new embeddings.','- Reusing the saved 2017 original checkpoints would include the new calibration students in encoder training. Retraining three original models alongside the missing three induced models preserves the declared matched design. All six 2012 backbones are reused.','- This is the bounded batch explicitly requested by the user; it does not advance the main milestone campaign. Paper/thesis and legacy source files remain unchanged.','',
            '## Outputs','',
            '- [Two-dataset table](two_dataset_comparison.csv), [frozen protocol](protocol.json), [W&B reuse audit](wandb_reuse_audit.json), [run registry](run_registry.json).','- [2012 scores and intervals](assist2012/method_comparison.csv), [2017 scores and intervals](assist2017/method_comparison.csv).','- [2012 per-concept results](assist2012/per_concept_auc.csv), [2017 per-concept results](assist2017/per_concept_auc.csv).','- [Verification](verification.json), [2017 training log](training_2017.log).','',
            'An AUC improvement supports response prediction for the original taxonomy under this protocol. It does not by itself validate mastery or causal learning mechanisms.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    all_infos=[]
    for ds in summaries:
        for run in runs(ds):
            p=folder(ds,run);complete=json.loads((p/'fit_complete.json').read_text())
            for method in complete['methods']:
                i=json.loads((p/f'{method}_coefficients.json').read_text());all_infos.append(i)
                assert i['converged'] and sha(p/f'{method}_coefficients.npz')==i['coefficient_sha256']
    verification={'all_main_readouts_converged':all(i['converged'] for i in all_infos),
           'main_readout_fits':len(all_infos),'max_projected_gradient':max(i['projected_gradient_max'] for i in all_infos),
           'paired_test_event_label_identity':True,'frozen_coefficient_hashes_verified':True,
           'all_2017_training_runs_complete':all(r['state']=='complete' for r in json.loads((A17/'run_registry.json').read_text()).values()),
           '2012_checkpoint_reuse':True,'new_dkt_fits':6,'backends':'MPS training/inference/readout data gradients; CPU preprocessing, solver control, metrics, and independent fixed-coefficient diagnostics',
           'protocol_sha256':sha(OUT/'protocol.json'),'outputs':{p.name:sha(p) for p in [OUT/'RESULTS.md',OUT/'two_dataset_comparison.csv']}}
    save_json(OUT/'verification.json',verification)
    registry(state='complete',summary='RESULTS.md',verification='verification.json')
    print(pd.DataFrame(wide)[['description','assist2012_auc','assist2017_auc']].to_string(index=False),flush=True)


if __name__=='__main__':main()
