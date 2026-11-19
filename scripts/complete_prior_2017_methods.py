"""Complete the previously requested mapping families on the new 2017 models.

The family definitions and settings were already frozen in the prepared 2017
protocol. No new DKT training or test-based selection occurs here.
"""
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/curriculum_kt_readout_followup_20261107'
PRIOR=ROOT/'results/curriculum_kt_all_methods_20261107'


def main():
    start=time.monotonic()
    reg=json.loads((OUT/'run_registry.json').read_text());reg['earlier_2017_methods']='running';reg['state']='running'
    (OUT/'run_registry.json').write_text(json.dumps(reg,indent=2)+'\n')
    for args,name in [(['scripts/run_matched_2017_curriculum_kt.py','evaluate'],'earlier_2017_evaluation'),
                      (['scripts/summarize_all_curriculum_methods.py','assist2017'],'earlier_2017_summary')]:
        print('Starting '+name,flush=True)
        with (OUT/(name+'.log')).open('a') as log:
            subprocess.run([sys.executable,'-B','-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    d={ds:pd.read_csv(PRIOR/ds/'method_comparison.csv') for ds in ['assist2012','assist2017']}
    frames={ds:f[f.population=='comparison_test'].set_index('method') for ds,f in d.items()}
    methods=['original_dkt','learned_monotone','parent_calibration','practice_frequency','intersection_practice_frequency',
             'item_overlap','direct_item_average','item_aware_reference_average','intersection_item_overlap','equal_children','noisy_or','and_product','raw_induced']
    result=pd.DataFrame([{'method':m,**{ds+'_auc':float(f.loc[m,'auc']) for ds,f in frames.items()}} for m in methods])
    result.to_csv(OUT/'earlier_method_families_two_datasets.csv',index=False)
    names={'original_dkt':'Original DKT','learned_monotone':'Earlier learned monotone fit','parent_calibration':'Calibrated frequency average',
       'practice_frequency':'Practice-frequency average','intersection_practice_frequency':'Intersection curves, frequency weights',
       'item_overlap':'Item-overlap average','direct_item_average':'Direct-item average (equivalent)',
       'item_aware_reference_average':'Item-aware reference-bank average','intersection_item_overlap':'Intersection curves, item weights',
       'equal_children':'Equal-child average','noisy_or':'Noisy-OR diagnostic','and_product':'AND/product diagnostic','raw_induced':'Raw induced diagnostic'}
    section=['','## Earlier mapping families: 2017 now completed','',
       'These are the methods from the earlier debate, using their already specified implementations. The 2012 values below preserve the earlier full-calibration results; they do not replace the new common-calibration table above. The 2017 values use the same new matched backbones and independent calibration split as the main comparison. Thus the learned 2012 rows across the two tables use different calibration populations and should not be treated as a one-factor method ablation.','',
       '| Earlier method | 2012 AUC | 2017 AUC |','|---|---:|---:|']
    for _,r in result.iterrows():section.append(f"| {names[r['method']]} | {r.assist2012_auc:.6f} | {r.assist2017_auc:.6f} |")
    section+=['','The earlier fitted families retain their fixed 1,000-step fitting protocol, which does not certify convergence. The new main readout table has independent convergence checks. AND/OR rows remain correctness-probability heuristics, not fitted mastery models. The strongest observed test row is exploratory, not an independently selected winner.','',
       '[Earlier-family two-dataset table](earlier_method_families_two_datasets.csv) · [2017 full prior-family results and intervals](../curriculum_kt_all_methods_20261107/assist2017/method_comparison.csv)']
    report=OUT/'RESULTS.md';text=report.read_text();marker='\n## Earlier mapping families: 2017 now completed\n'
    if marker in text:text=text.split(marker)[0]+'\n'
    report.write_text(text+'\n'.join(section)+'\n')
    verification=json.loads((OUT/'verification.json').read_text())
    verification['earlier_2017_method_families_completed']=True
    verification['earlier_family_fit_protocol']='Preserved fixed 1000-step implementation; no convergence certification for these additional earlier-family fits'
    verification['outputs']['RESULTS.md']=hashlib.sha256(report.read_bytes()).hexdigest()
    verification['outputs']['earlier_method_families_two_datasets.csv']=hashlib.sha256((OUT/'earlier_method_families_two_datasets.csv').read_bytes()).hexdigest()
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    reg=json.loads((OUT/'run_registry.json').read_text());reg.update({'earlier_2017_methods':'complete','earlier_2017_seconds':time.monotonic()-start,'state':'complete'})
    (OUT/'run_registry.json').write_text(json.dumps(reg,indent=2)+'\n')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
