"""Freeze the authorized, single-architecture SimpleKT tuning batch."""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_saved_curriculum_kt import SOURCE, ROOT, sha, vocabulary, load_sequences

OUT = ROOT / 'results/simplekt_sweeps_20261107'


def dump(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_arrays(d, nc, nk):
    valid = d['questions'] >= 0
    for name, cardinality in [('original_concepts', nc), ('induced_concepts', nk)]:
        assert np.array_equal(d[name] >= 0, valid)
        assert ((d[name][valid] >= 0) & (d[name][valid] < cardinality)).all()
    assert np.array_equal(d['responses'] >= 0, valid)
    assert set(np.unique(d['responses'][valid])) == {0, 1}
    assert (np.diff(valid.astype(int), axis=1) <= 0).all()
    return {'students':len(np.unique(d['uid'])), 'chunks':len(d['uid']),
            'sequence_positions':int(valid.sum()), 'predictable_targets':int((valid[:, 1:] & valid[:, :-1]).sum())}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        print('Frozen protocol already exists; resume, do not overwrite.', flush=True)
        return
    source_hashes, datasets = {}, {}
    old = ROOT / 'results/curriculum_kt_assist2012_k1000_20261107'
    old_protocol = json.loads((old / 'protocol.json').read_text())
    source = SOURCE / 'data/assist2012'
    for name in ['keyid2idx.json','train_valid_sequences.csv','test_sequences.csv']:
        p = source / name
        assert sha(p) == old_protocol['input_sha256'][str(p)]
        source_hashes[str(p)] = sha(p)
    mp = dict(np.load(old / 'mapping.npz'))
    source_hashes[str(old / 'mapping.npz')] = sha(old / 'mapping.npz')
    vocab = vocabulary(source / 'keyid2idx.json')
    tv = load_sequences(source / 'train_valid_sequences.csv', vocab)
    te = load_sequences(source / 'test_sequences.csv', vocab)
    assert set(tv['fold']) == {0,1,2,3,4} and set(te['fold']) == {-1}
    folder = OUT / 'data/assist2012'; folder.mkdir(parents=True, exist_ok=True)
    files, counts, memberships = {}, {}, {}
    for split, raw, keep in [('train',tv,np.isin(tv['fold'],[2,3,4])),
                             ('validation',tv,tv['fold']==0), ('calibration',tv,tv['fold']==1),
                             ('test',te,np.ones(len(te['uid']),bool))]:
        d = {k:v[keep] for k,v in raw.items()}
        valid = d['questions'] >= 0
        assert np.array_equal(d['concepts'][valid], mp['q_to_induced'][d['questions'][valid]])
        original = np.full_like(d['concepts'], -1)
        original[valid] = mp['q_to_original'][d['questions'][valid]]
        positions = d['row'][:,None] * 200 + np.arange(200)[None,:]
        if split == 'test':
            positions += 2**40
        positions[~valid] = -1
        arrays = {'uid':d['uid'], 'questions':d['questions'], 'responses':d['responses'],
                  'timestamps':d['timestamps'], 'original_concepts':original,
                  'induced_concepts':d['concepts'], 'event_id':positions,
                  'source_row':d['row'], 'legacy_fold':d['fold']}
        counts[split] = check_arrays(arrays,162,1000)
        memberships[split] = set(d['uid'])
        p = folder / f'{split}_sequences.npz'
        np.savez_compressed(p, **arrays)
        files[split] = {'path':str(p), 'sha256':sha(p)}
    for a,b in itertools.combinations(memberships,2):
        assert not memberships[a] & memberships[b], (a,b)
    tr = dict(np.load(files['train']['path']))
    v = tr['questions'] >= 0
    practice = np.zeros((162,1000),float)
    np.add.at(practice,(tr['original_concepts'][v],tr['induced_concepts'][v]),1)
    empty = practice.sum(1)==0; practice[empty] = mp['counts'][empty]
    assert (practice.sum(1)>0).all()
    mapping_path = folder/'mapping.npz'
    np.savez_compressed(mapping_path, counts=mp['counts'], practice_counts=practice,
        practice_frequency=practice/practice.sum(1,keepdims=True), item_overlap=mp['item_overlap'],
        q_to_original=mp['q_to_original'],q_to_induced=mp['q_to_induced'],raw_question_ids=mp['raw_question_ids'])
    datasets['assist2012'] = {'original_concepts':162,'induced_concepts':1000,'questions':len(vocab['questions']),
        'files':files,'counts':counts,'mapping':{'path':str(mapping_path),'sha256':sha(mapping_path)},
        'student_split_disjoint':True,'paired_events_responses_order_and_masks':True,
        'event_contract':'Saved sequence positions, including historical duplicates; not deduplicated raw events. Event ID identifies a source sequence position.',
        'limitation':'Exploratory K1000 saved export, not paper K300. New matched training populations differ from historical saved models.'}
    prior = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017'
    p17 = json.loads((prior/'protocol.json').read_text())
    files, counts, memberships = {}, {}, {}
    for split in ['train','validation','calibration','test']:
        p = prior/f'{split}_sequences.npz'
        assert sha(p) == p17['derived_sha256'][p.name]
        d = dict(np.load(p)); counts[split] = check_arrays(d,68,100)
        memberships[split] = set(d['uid'])
        files[split] = {'path':str(p),'sha256':sha(p)}
    for a,b in itertools.combinations(memberships,2):
        assert not memberships[a] & memberships[b], (a,b)
    mapping_path = prior/'mapping.npz'
    assert sha(mapping_path) == p17['derived_sha256'][mapping_path.name]
    datasets['assist2017'] = {'original_concepts':68,'induced_concepts':100,'questions':638,
        'files':files,'counts':counts,'mapping':{'path':str(mapping_path),'sha256':sha(mapping_path)},
        'student_split_disjoint':True,'paired_events_responses_order_and_masks':True,
        'event_contract':'Previously verified unique raw (studentId,action_num) events; reuse prepared arrays unchanged.'}
    candidates = [{'id':f'c{i:02d}','d_model':w,'learning_rate':lr,'dropout':d}
                  for i,(w,lr,d) in enumerate(itertools.product([64,128],[.0003,.001],[.1,.5]))]
    sources = [SOURCE/'pykt/models/simplekt.py',SOURCE/'pykt/models/utils.py',
               Path(__file__), ROOT/'scripts/run_simplekt_sweeps.py',ROOT/'scripts/evaluate_saved_curriculum_kt.py',
               ROOT/'scripts/check_attention_model_readiness.py']
    source_hashes.update({str(p):sha(p) for p in sources})
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User confirmed W&B project and requested sweeps; follow-up clarified one architecture at a time. Execute SimpleKT only, stop for review before RobustKT or other models.',
        'models':['simplekt'],'datasets':datasets,'candidates':candidates,'seed':42,
        'planned_sweeps':4,'planned_trials':32,'scope':'Validation-only SimpleKT screening; no seed confirmation or map-back/test evaluation in this batch.',
        'wandb':{'entity':'knowledgetracing42-ucla','project':'assist2012_publication_kia',
                 'group':'currkt-simplekt-sweeps-20261107','mode':'online',
                 'uploads':'Aggregate metrics and nonsensitive configurations only; no automatic code/data/checkpoint uploads.'},
        'fixed':{'n_blocks':1,'d_ff':256,'num_attn_heads':4,'n_pid':0,'emb_type':'qid',
                 'final_fc_dim':512,'final_fc_dim2':256,'sequence_length':200,'batch_size':64,
                 'optimizer':'Adam','weight_decay':0,'gradient_clipping':None,'max_epochs':100,
                 'patience':15,'patience_min_delta_auc':.0001},
        'loss':'Mean binary cross entropy on response targets at positions1..199, float32 model/loss on MPS; padded targets excluded. Supported concept-only SimpleKT n_pid=0.',
        'selection':'Save every strictly better finite validation AUC. Select candidate by best validation AUC, then lower corresponding validation log loss, then fewer parameters, then candidate ID.',
        'patience':'Independent reference AUC updated only after >0.0001 improvement; stop after15 epochs without such an improvement. Always retain actual maximum-AUC checkpoint.',
        'diagnostics':'Full training and validation AUC/log loss in eval mode each epoch; optimization training BCE separately, gradient norms, device/memory/timing, no clipping.',
        'split':'Training folds2/3/4; validation fold0; independent mapping calibration fold1; test fold-1. Training reads only train and validation arrays.',
        'query_information':'Past concept/response history and target concept; no actual target item supplied. This is the supported concept-only variant.',
        'map_back':'Native induced validation AUC is screening criterion; not a guarantee of improved original-taxonomy AUC. Subsequent Curriculum KT comparison must include original+original control.',
        'runtime_gate':'Start with one actual full epoch in the first trial, record its measured duration before continuing. No CPU fallback; run one trial at a time on MPS.',
        'failure_policy':'Keep all failed/nonfinite/OOM runs and their logs. Do not silently alter hyperparameters, reduce batch size, extend caps, or retry failed trials.',
        'resume':'Per-epoch atomic model/optimizer/RNG/history checkpoints and run registry; recover interrupted run under its existing W&B ID.',
        'source_commit':'c62b35c70569da87683fc95189eda325c149de8c',
        'source_sha256':source_hashes, 'proposal_sha256':sha(ROOT/'results/model_readiness_20261107/tuning_proposal.json')}
    registry = {'state':'prepared','active_run':None,'trials':{},'sweeps':{},'created_utc':protocol['created_utc']}
    for dataset in ['assist2017','assist2012']:
        for taxonomy in ['original','induced']:
            cell = f'{dataset}_simplekt_{taxonomy}'
            registry['sweeps'][cell] = {'state':'prepared','dataset':dataset,'model':'simplekt','taxonomy':taxonomy}
            for cfg in candidates:
                key = f'{cell}_{cfg["id"]}_seed42'
                registry['trials'][key] = {'state':'pending','cell':cell,'dataset':dataset,'model':'simplekt',
                    'taxonomy':taxonomy,'candidate':cfg,'seed':42,'epochs':0}
    dump(OUT/'protocol.json',protocol)
    dump(OUT/'run_registry.json',registry)
    dump(OUT/'protocol_identity.json',{'sha256':sha(OUT/'protocol.json')})
    print(json.dumps({'state':'prepared','sweeps':4,'trials':32,'counts':{k:v['counts'] for k,v in datasets.items()}}),flush=True)


if __name__ == '__main__':
    prepare()
