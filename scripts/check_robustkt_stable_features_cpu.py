"""Bounded CPU feature-assembly preparation using training examples only."""
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import expit

import run_robustkt_sweeps as base
from robustkt_stable_attention import stable_attention
from check_attention_model_readiness import module, forward
from check_robustkt_mapped_queries_cpu import build
from robustkt_stable_features import extract_batch, mapping_layout


def run():
    torch.set_num_threads(1)
    protocol=base.read(base.OUT/'protocol.json');base.verify(protocol)
    registry=base.read(base.OUT/'run_registry.json')
    results=[]
    for dataset in ['assist2017','assist2012']:
        ds=protocol['datasets'][dataset]
        assert base.sha(Path(ds['files']['train']['path']))==ds['files']['train']['sha256']
        assert base.sha(Path(ds['mapping']['path']))==ds['mapping']['sha256']
        with np.load(ds['files']['train']['path']) as data:
            long_rows=np.flatnonzero((data['original_concepts']>=0).sum(1)>=5)[:2]
            arrays={k:data[k][long_rows].copy() for k in data.files}
        # A small causal-prefix diagnostic; no empirical data artifact is edited.
        # Keep the supported length200 tensor and score just its first4 targets.
        for key in ['original_concepts','induced_concepts','responses']:
            arrays[key][:,5:]=-1
        with np.load(ds['mapping']['path']) as data:layout=mapping_layout(data)
        previous_meta=None
        for taxonomy in ['original','induced']:
            key=f'{dataset}_robustkt_{taxonomy}_c00_seed42'
            entry=registry['trials'][key]
            model=build(protocol,dataset,taxonomy,entry['candidate'])
            checkpoint_hash=None
            if entry['state']=='complete':
                checkpoint=base.OUT/'trials'/key/'best.ckpt'
                checkpoint_hash=base.sha(checkpoint)
                assert checkpoint_hash==entry['checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            start=time.monotonic()
            meta,features,diagnostics=extract_batch(model,arrays,taxonomy,layout,row_offset=100,query_batch_size=7)
            assert diagnostics['events']==8
            if previous_meta is not None:
                assert all(np.array_equal(meta[k],previous_meta[k]) for k in meta)
            previous_meta=meta
            maximum=0.;references=0;forbidden=0.
            if taxonomy=='induced':
                for event in [0,3,4,7]:
                    row=int(meta['row'][event])-100
                    pos=int(meta['position'][event]);concept=int(meta['original_concept'][event])
                    c=torch.tensor(np.maximum(arrays['induced_concepts'][row:row+1],0))
                    r=torch.tensor(np.maximum(arrays['responses'][row:row+1],0))
                    total=0.
                    for slot in np.flatnonzero(layout['mask'][concept]):
                        changed=c.clone();changed[0,pos]=int(layout['children'][concept,slot])
                        with torch.inference_mode(), stable_attention('cpu'):probability=float(forward(model,changed,r)[0,pos])
                        maximum=max(maximum,abs(probability-expit(float(features['child_logits'][event,slot]))))
                        total+=layout['weights'][concept,slot]*probability;references+=1
                    maximum=max(maximum,abs(total-features['practice_frequency'][event]))
                assert maximum<2e-5
                assert (features['child_logits'][~layout['mask'][meta['original_concept']]]==0).all()
                altered={k:v.copy() for k,v in arrays.items()}
                altered['induced_concepts'][:,4]=(altered['induced_concepts'][:,4]+7)%model.n_question
                altered['responses'][:,4]=1-altered['responses'][:,4]
                new_meta,new_features,_=extract_batch(model,altered,taxonomy,layout,row_offset=100,query_batch_size=3)
                assert np.array_equal(meta['event_id'],new_meta['event_id'])
                last=meta['position']==4
                forbidden=float(np.max(np.abs(features['child_logits'][last]-new_features['child_logits'][last])))
                assert forbidden<2e-5
            results.append({'dataset':dataset,'taxonomy':taxonomy,'candidate':entry['candidate'],
                'checkpoint_sha256':checkpoint_hash,'initial_weights':checkpoint_hash is None,
                **diagnostics,'independent_full_forward_checks':references,
                'child_and_weighted_map_probability_max_difference':maximum,
                'actual_target_concept_response_change_max_logit_difference':forbidden,
                'seconds':time.monotonic()-start,'passed':True})
            print(results[-1],flush=True)
    report={'completed_utc':base.now(),'passed':True,'device':'cpu','checks':results,
        'paired_metadata_identical':True,'training_performed':False,'test_used':False,'calibration_used':False,
        'diagnostic_data':'First two training rows with at least five events; retain first five events and pad to length200; original files unchanged.',
        'implementation_sha256':{str(p):base.sha(p) for p in [Path(__file__),
            Path(__file__).with_name('robustkt_stable_features.py'),
            Path(__file__).with_name('robustkt_stable_queries.py'),
            Path(__file__).with_name('simplekt_curriculum_features.py')]},
        'remaining_gate':'Actual MPS on all12 development-selected checkpoints before extraction.'}
    base.dump(base.ROOT/'results/model_sequence_20261107/robustkt_stable_feature_assembly_cpu_checks.json',report)


if __name__=='__main__':
    run()
