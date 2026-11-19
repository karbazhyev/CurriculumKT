"""Resumable, exclusive-MPS sparse feature extraction with a held-out gate."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from prepare_robustkt_curriculum import OUT, ROOT, sha
from run_robustkt_sweeps import read, dump, now
from check_robustkt_selected_queries import make_model
from robustkt_stable_features import extract_batch, mapping_layout

SEEDS = [42, 84, 126]
DATASETS = ['assist2017', 'assist2012']


def guard():
    protocol = read(OUT/'protocol.json')
    assert protocol['frozen']
    assert sha(OUT/'protocol.json') == read(OUT/'protocol_identity.json')['sha256']
    for path, expected in protocol['implementation_sha256'].items():
        assert sha(Path(path)) == expected, f'Frozen implementation changed: {path}'
    path = ROOT/'results/model_sequence_20261107/robustkt_stable_queries_mps_checks.json'
    assert sha(path) == protocol['selected_query_check_sha256']
    assert read(path)['all_twelve_selected_checkpoints']
    amendment=ROOT/'results/model_sequence_20261107/robustkt_inference_amendment_01.json'
    assert sha(amendment)==protocol['inference_amendment_sha256']
    assert sha(ROOT/'results/model_sequence_20261107/robustkt_curriculum_preparation_checks.json') == protocol['preparation_check_sha256']
    return protocol


def require_test_ready():
    frozen = read(OUT/'development_complete.json')
    assert frozen['protocol_sha256'] == sha(OUT/'protocol.json')
    assert set(frozen['selections']) == set(DATASETS)
    assert frozen['all_312_fits_complete']
    entries = read(OUT/'run_registry.json')['fits']
    assert len(entries) == 312 and all(e['state'] == 'complete' for e in entries.values())
    for ds, expected in frozen['selections'].items():
        path = OUT/ds/'selection.json'
        assert sha(path) == expected
        for key, value in read(path)['full_fit_coefficient_hashes'].items():
            assert sha(Path(str(OUT/'fits'/key)+'.npz')) == value


def directory(dataset, taxonomy, seed, split):
    return OUT/'features'/dataset/f'{taxonomy}_seed{seed}'/split


def load_features(dataset, taxonomy, seed, split):
    """Read only identity-checked complete features; no model execution."""
    dest = directory(dataset,taxonomy,seed,split)
    mark = read(dest/'complete.json')
    assert mark['protocol_sha256'] == sha(OUT/'protocol.json')
    for name, expected in mark['files'].items():
        assert sha(dest/name) == expected, f'Feature changed: {dest/name}'
    with np.load(dest/'metadata.npz') as f:
        meta = dict(f)
    features = {name:np.load(dest/f'{name}.npy',mmap_mode='r') for name in mark['feature_names']}
    assert len(meta['label']) == mark['events']
    return meta, features


def save_npz(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(path)


def consolidate(dest, parts, expected_events, identity):
    """Join complete chunks without materializing the full feature matrix in RAM."""
    feature_arrays, metadata = {}, {}
    offset = 0
    nonempty = [p for p in parts if p['events']]
    assert nonempty
    for entry in nonempty:
        part = dest/'parts'/entry['file']
        assert sha(part) == entry['sha256']
        with np.load(part) as chunk:
            n = entry['events']
            for key in chunk.files:
                if key.startswith('meta__'):
                    metadata.setdefault(key[6:],[]).append(chunk[key])
                else:
                    assert key.startswith('feature__')
                    name, value = key[9:], chunk[key]
                    if name not in feature_arrays:
                        feature_arrays[name] = np.lib.format.open_memmap(
                            dest/f'{name}.npy.tmp', mode='w+', dtype=value.dtype,
                            shape=(expected_events,*value.shape[1:]))
                    assert len(value) == n
                    feature_arrays[name][offset:offset+n] = value
            offset += n
    assert offset == expected_events
    meta = {key:np.concatenate(value) for key,value in metadata.items()}
    assert all(len(value) == expected_events for value in meta.values())
    assert np.all((meta['label']==0)|(meta['label']==1))
    order = np.lexsort((meta['position'],meta['row']))
    assert np.array_equal(order,np.arange(expected_events)), 'Target order changed'
    names = list(feature_arrays)
    for name, array in feature_arrays.items():
        array.flush();array._mmap.close()
        (dest/f'{name}.npy.tmp').replace(dest/f'{name}.npy')
    save_npz(dest/'metadata.npz',meta)
    files = ['metadata.npz',*[f'{name}.npy' for name in names]]
    dump(dest/'complete.json',{**identity,'completed_utc':now(), 'events':expected_events,
        'students':len(np.unique(meta['uid'])),'feature_names':names,
        'files':{name:sha(dest/name) for name in files},
        'independent_child_queries':sum(p.get('independent_child_queries',0) for p in parts),
        'max_native_query_difference':max(p.get('native_query_max_probability_difference',0.) for p in parts)})


def extract_one(protocol, dataset, taxonomy, seed, split):
    ds = protocol['datasets'][dataset]
    record = protocol['selected_models'][f'{dataset}_robustkt_{taxonomy}'][str(seed)]
    data_record = ds['files'][split]
    identity = {'protocol_sha256':sha(OUT/'protocol.json'),
        'checkpoint_sha256':record['checkpoint_sha256'], 'data_sha256':data_record['sha256'],
        'mapping_sha256':ds['mapping']['sha256'], 'split':split, 'dataset':dataset,
        'taxonomy':taxonomy,'seed':seed,'device':'mps','cpu_fallback':False,
        'selected_query_check_sha256':protocol['selected_query_check_sha256']}
    dest = directory(dataset,taxonomy,seed,split)
    dest.mkdir(parents=True,exist_ok=True)
    key = f'{dataset}/{taxonomy}/seed{seed}/{split}'
    registry = read(OUT/'run_registry.json')
    if (dest/'complete.json').exists():
        old = read(dest/'complete.json')
        assert all(old[k] == value for k,value in identity.items())
        load_features(dataset,taxonomy,seed,split)
        registry['feature_extraction'][key] = {'state':'complete','complete_sha256':sha(dest/'complete.json')}
        dump(OUT/'run_registry.json',registry)
        return
    registry['feature_extraction'][key] = {'state':'running','started_utc':now()}
    dump(OUT/'run_registry.json',registry)
    assert sha(Path(data_record['path'])) == data_record['sha256']
    assert sha(Path(ds['mapping']['path'])) == ds['mapping']['sha256']
    with np.load(data_record['path']) as d:
        arrays = dict(d)
    with np.load(ds['mapping']['path']) as m:
        layout = mapping_layout(m)
    model_protocol = {**protocol,'fixed':protocol['fixed_backbone']}
    model = make_model(model_protocol,dataset,taxonomy,record,'mps')
    assert all(p.device.type=='mps' for p in model.parameters())
    (dest/'parts').mkdir(exist_ok=True)
    parts=[];start=time.monotonic()
    for lo in range(0,len(arrays['uid']),64):
        part_path=dest/'parts'/f'rows_{lo:06d}.npz'
        mark_path=part_path.with_suffix('.json')
        if mark_path.exists():
            mark=read(mark_path)
            assert all(mark[k] == value for k,value in identity.items())
            assert not mark['events'] or sha(part_path)==mark['sha256']
        else:
            batch={name:value[lo:lo+64] for name,value in arrays.items()}
            oc=batch['original_concepts']
            events=int(((oc[:,1:]>=0)&(oc[:,:-1]>=0)).sum())
            if events:
                meta,features,diagnostic=extract_batch(model,batch,taxonomy,layout,row_offset=lo)
                assert len(meta['label'])==events
                save_npz(part_path,{**{'meta__'+k:v for k,v in meta.items()},
                    **{'feature__'+k:v for k,v in features.items()}})
                mark={**identity,**diagnostic,'file':part_path.name,'sha256':sha(part_path),'row_start':lo}
            else:
                mark={**identity,'events':0,'row_start':lo,'file':part_path.name}
            dump(mark_path,mark)
        parts.append(mark)
        registry=read(OUT/'run_registry.json')
        registry['feature_extraction'][key].update(chunks_saved=len(parts),
            rows_processed=min(lo+64,len(arrays['uid'])),updated_utc=now())
        dump(OUT/'run_registry.json',registry)
        print(f'{key}: rows {min(lo+64,len(arrays["uid"]))}/{len(arrays["uid"])}',flush=True)
    consolidate(dest,parts,ds['counts'][split]['predictable_targets'],identity)
    finished=read(dest/'complete.json')
    assert finished['students']==ds['counts'][split]['students']
    assert sha(Path(record['checkpoint']))==record['checkpoint_sha256']
    registry=read(OUT/'run_registry.json')
    registry['feature_extraction'][key]={'state':'complete','complete_sha256':sha(dest/'complete.json'),
        'seconds':time.monotonic()-start,'events':finished['events'],'updated_utc':now()}
    dump(OUT/'run_registry.json',registry)
    del model
    torch.mps.empty_cache()


def extract_all(split):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert torch.backends.mps.is_available(),'MPS required; no CPU inference campaign fallback'
    torch.set_num_threads(1)
    protocol=guard()
    if split=='test':
        require_test_ready()
        registry=read(OUT/'run_registry.json')
        registry.update(heldout_started=True,heldout_started_utc=registry.get('heldout_started_utc',now()))
        dump(OUT/'run_registry.json',registry)
    worker_path=OUT/'extraction_worker.json'
    dump(worker_path,{'pid':os.getpid(),'state':'running','split':split,'started_utc':now(),
        'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
        'device':'mps','cpu_fallback':False})
    state='failed'
    try:
        for ds in DATASETS:
            for taxonomy in ['original','induced']:
                for seed in SEEDS:
                    extract_one(protocol,ds,taxonomy,seed,split)
        guard()
        if split=='test':
            require_test_ready()
        state='complete'
        dump(OUT/f'{split}_features_complete.json',{'completed_utc':now(),
            'protocol_sha256':sha(OUT/'protocol.json'), 'all_twelve_complete':True,
            'manifests':{str(directory(ds,tax,seed,split)/'complete.json'):
                sha(directory(ds,tax,seed,split)/'complete.json')
                for ds in DATASETS for tax in ['original','induced'] for seed in SEEDS}})
    finally:
        info=read(worker_path);info.update(state=state,finished_utc=now());dump(worker_path,info)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['calibration','test'])
    split=parser.parse_args().split
    with ExitStack() as stack:
        for path in [ROOT/'results/robustkt_sweeps_20261107/worker.lock', ROOT/'results/simplekt_sweeps_20261107/worker.lock',
                     ROOT/'results/model_sequence_20261107/gpu_worker.lock']:
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        extract_all(split)
