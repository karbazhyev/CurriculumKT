"""Freeze RobustKT's equal-budget search, reusing unchanged paired arrays."""
import argparse
import copy
from pathlib import Path

from run_robustkt_sweeps import ROOT, SOURCE, OUT, read, dump, now, sha


def draft():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Frozen protocol must not be overwritten'
    parent = ROOT/'results/simplekt_sweeps_20261107/protocol.json'
    protocol = copy.deepcopy(read(parent))
    assert read(ROOT/'results/simplekt_curriculum_20261107/run_registry.json')['state'] == 'complete'
    for ds in protocol['datasets'].values():
        for record in [*ds['files'].values(), ds['mapping']]:
            assert sha(Path(record['path'])) == record['sha256']
    protocol.update(created_utc=now(), models=['robustkt'],
        authorization='User authorized all16 compatible models sequentially without another go-ahead; SimpleKT is complete.',
        scope='RobustKT screening: four separately trained dataset/taxonomy cells,32 trials; development selection only.',
        parent_simplekt_protocol_sha256=sha(parent),
        data_contract='Exact same prepared paired arrays, event population, taxonomy, mapping and student splits as SimpleKT; no re-encoding or data change.',
        loss='Pinned pyKT RobustKT masked mean binary cross entropy plus returned c_reg_loss. With supported concept-only n_pid=0 the regularizer is exactly zero. Float32 BCE on MPS replaces upstream float64 BCE; objective and gradient checked against CPU float64.',
        implementation='Read-only pinned Robustkt class with module device override and a length200 MPS adapter. Native200-position calls are unchanged. Short diagnostic histories are right-padded to200 and outputs sliced back; CPU reference verifies equivalence. Preserve causal smoothing kernel5 and both attention stages. No new preprocessing, features or auxiliary losses.',
        map_back='Same prespecified positive map-back, fixed frequency, scalar original calibration, original+induced augmentation and original+original control as SimpleKT; student-level calibration CV; all selection before test.',
        followup='After screening, review convergence/W&B, confirm selected settings with seeds84/126, verify selected-model target queries, then fit/evaluate the frozen Curriculum KT panel before AKT.')
    protocol['wandb']['group'] = 'currkt-robustkt-sweeps-20261107'
    protocol['fixed'].update(ks=5, kq_same=1, separate_qa=False, l2=1e-5)
    protocol['query_information'] = 'Observed past concepts/responses and current target concept only; n_pid=0. Single-target query reference copies history separately for every candidate.'
    paths = [SOURCE/'pykt/models/robustkt.py', SOURCE/'pykt/models/train_model.py',
             ROOT/'scripts/run_robustkt_sweeps.py', Path(__file__),
             ROOT/'scripts/check_robustkt_screening.py',ROOT/'scripts/robustkt_target_queries.py',
             ROOT/'scripts/robustkt_mps_portability.py',
             ROOT/'scripts/check_attention_model_readiness.py',ROOT/'scripts/evaluate_saved_curriculum_kt.py',
             ROOT/'research/MODEL_EXECUTION_QUEUE.md']
    protocol['source_sha256'] = {str(p):sha(p) for p in paths}
    dump(OUT/'protocol_draft.json',protocol)
    print('RobustKT protocol drafted; empirical training remains gated on MPS checks.')


def freeze():
    assert not (OUT/'protocol.json').exists(), 'Resume an existing frozen protocol'
    protocol = read(OUT/'protocol_draft.json')
    checks = read(OUT/'runtime_checks.json')
    assert checks['passed'] and len(checks['checks']) == 16
    assert checks['protocol_draft_sha256'] == sha(OUT/'protocol_draft.json')
    for path, expected in protocol['source_sha256'].items():
        assert sha(Path(path)) == expected
    protocol.update(frozen_utc=now(), runtime_checks_sha256=sha(OUT/'runtime_checks.json'))
    registry = {'state':'prepared','active_run':None,'trials':{},'sweeps':{},'created_utc':now()}
    for dataset in ['assist2017','assist2012']:
        for taxonomy in ['original','induced']:
            cell = f'{dataset}_robustkt_{taxonomy}'
            registry['sweeps'][cell] = {'state':'prepared','dataset':dataset,'model':'robustkt','taxonomy':taxonomy}
            for cfg in protocol['candidates']:
                key = f'{cell}_{cfg["id"]}_seed42'
                registry['trials'][key] = {'state':'pending','cell':cell,'dataset':dataset,'model':'robustkt',
                    'taxonomy':taxonomy,'candidate':cfg,'seed':42,'epochs':0}
    assert len(registry['trials']) == 32
    dump(OUT/'protocol.json',protocol)
    dump(OUT/'protocol_identity.json',{'sha256':sha(OUT/'protocol.json')})
    dump(OUT/'run_registry.json',registry)
    print('Frozen RobustKT protocol and32 pending trials; no training or test selection performed.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['draft','freeze'])
    action=parser.parse_args().action
    draft() if action=='draft' else freeze()
