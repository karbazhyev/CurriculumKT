"""Build a source-pinned standalone Colab probe; no GPU, network or installation."""
import ast
import base64
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from foundationalassist_preparation import sha,dump,utc
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/colab_microbenchmark_20261112'
SOURCE=Path('/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing')

def main():
    OUT.mkdir(exist_ok=True)
    paths={n:SOURCE/'pykt/models'/f'{n}.py' for n in ('dkvmn','sakt','utils')}
    paths['adapter']=ROOT/'scripts/joint_model_adapters.py'
    template=ROOT/'scripts/colab_kt_microbenchmark_20261112.py'
    m=json.loads((ROOT/'data/joint_tuning_20261109/assist2012/manifest.json').read_text())
    sizes={}
    for split in ('train','validation'):
        path=Path(m['files'][split]['path']);assert sha(path)==m['files'][split]['sha256']
        with np.load(path,allow_pickle=False) as a:
            sizes[split]=len(a['uid'])
            if split=='train':values,counts=np.unique((a['questions']>=0).sum(1),return_counts=True)
    history=ROOT/'results/joint_tuning_20261109/models/dkvmn/assist2012/screening/i00/history.json'
    h=json.loads(history.read_text());epoch=statistics.median(x['runtime/epoch_seconds'] for x in h[1:])
    payload={'source':{k:{'base64':base64.b64encode(p.read_bytes()).decode(),'sha256':sha(p)} for k,p in paths.items()},
        'lengths':{'values':values.tolist(),'probabilities':(counts/counts.sum()).tolist()},
        'mps_reference':{'model':'dkvmn','width':128,'k':1000,'seed':42,'median_epoch_seconds':epoch,'training_batches':math.ceil(sizes['train']/64),'evaluation_batches':sum(math.ceil(v/64) for v in sizes.values()),'history_sha256':sha(history)}}
    standalone=template.read_text().replace('PAYLOAD = None  # inserted in the generated standalone file, not this template','PAYLOAD = '+repr(payload))
    ast.parse(standalone);(OUT/'benchmark_standalone.py').write_text(standalone)
    intro='''# CurriculumKT bounded GPU test\nSynthetic data only; exact pinned DKVMN/SAKT sources. No main-training changes.\nSelect a T4 or L4 GPU and run the single code cell. Four-minute compute budget plus a five-minute runtime-disconnect watchdog.\nThe notebook disconnects its runtime after success or failure to limit billing.\nDownloaded/printed aggregate timings are the only outputs. No student data or credentials are included.\n'''
    cell=standalone.replace("if __name__=='__main__':\n    assert PAYLOAD is not None,'Use the prepared standalone benchmark or notebook.'\n    run(PAYLOAD)","from google.colab import runtime\n_watchdog = threading.Timer(300, runtime.unassign)\n_watchdog.daemon = True\n_watchdog.start()\ntry:\n    run(PAYLOAD)\nfinally:\n    try:\n        runtime.unassign()\n    finally:\n        _watchdog.cancel()")
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'CurriculumKT bounded GPU benchmark'},'kernelspec':{'display_name':'Python 3','name':'python3'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':cell.splitlines(True)}]}
    dump(OUT/'CurriculumKT_GPU_benchmark.ipynb',nb)
    dump(OUT/'preparation.json',{'created_utc':utc(),'state':'prepared_not_executed','authorization':'User requested a small T4 or other GPU speed test, November12.','source_sha256':{k:sha(p) for k,p in paths.items()},'template_sha256':sha(template),'standalone_sha256':sha(OUT/'benchmark_standalone.py'),'notebook_sha256':sha(OUT/'CurriculumKT_GPU_benchmark.ipynb'),'mps_reference':payload['mps_reference'],'synthetic_only':True,'aggregate_length_histogram_only':True,'student_data_uploaded':False,'gpu_sessions_started':0,'proposed_session_limit_minutes':10,'proposed_total_test_cu_cap':2,'proposed_hardware_order':['T4','L4'],'main_training_sources_changed':False})
    print(json.dumps({'state':'prepared','mps_epoch_reference_seconds':epoch,'training_batches':payload['mps_reference']['training_batches'],'evaluation_batches':payload['mps_reference']['evaluation_batches'],'notebook':str(OUT/'CurriculumKT_GPU_benchmark.ipynb')}))

if __name__=='__main__':main()
