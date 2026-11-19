"""Resume the authorized stages sequentially, stopping on any failed check."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/curriculum_kt_readout_followup_20261107'

def run(args,name):
    print('Starting '+name,flush=True)
    with (OUT/(name+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-B','-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed '+name,flush=True)

run(['scripts/run_readout_followup.py','audit'],'audit_2012')
for ds in ['assist2012','assist2017']:
    for stage in ['calibration','fit','test']:
        run(['scripts/run_readout_followup.py',stage,'--dataset',ds],ds+'_'+stage)
run(['scripts/summarize_readout_followup.py'],'summary')
