"""CPU-only orchestration checks; never runs a model or makes network calls."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import run_training_first_20261113 as runner
import run_joint_tuning as prior
import joint_training as training
import joint_tuning_common as common
import currkt_storage as storage


def main():
    p = runner.protocol(); ph = p['parent_protocol']['sha256']
    design = common.read(common.OUT / 'design.json')
    for task in p['tasks']:
        assert task['candidate'] in design['trial_assignments'][task['model']][task['dataset']][task['arm']]
        assert task['model'] in p['execution_models'] and task['model'] not in p['deferred_models']
    verified = [runner.completed(t, ph) for t in p['tasks']]
    reused = sum(r is not None for r in verified)
    assert reused >= 79
    # Train-only child uses only train/validation, seed42, unchanged fit contract.
    with TemporaryDirectory() as tmp:
        out = Path(tmp); task = dict(p['tasks'][0], folder=str(out/'trial'))
        fake = dict(p, tasks=[task]); calls = []
        def arrays(ds, split, taxonomy):
            calls.append(split)
            assert split in ['train', 'validation']
            return {'uid': [1] if split == 'train' else [2]}
        def fit(folder, model, ds, arm, candidate, seed, train, valid, k, parent):
            assert seed == 42 and parent == ph and candidate == task['candidate']
            calls.append('native_fit')
        with ExitStack() as stack:
            for obj, name, value in [
                (runner,'OUT',out),(runner,'protocol',lambda:fake),
                (prior,'guard',lambda:None),(prior,'acquire_gpu',lambda _:None),
                (storage,'verify_plan',lambda:None),(storage,'ensure_external_storage',lambda:None),
                (storage,'install_storage_guards',lambda _:None),
                (training,'require_mps',lambda:None),(common,'load_arrays',arrays),
                (training,'runtime_gate',lambda *a:{'passed':True}),
                (training,'fit',fit),(runner,'completed',lambda *a:{'state':'complete'})]:
                stack.enter_context(patch.object(obj,name,value))
            assert runner.child(0) == 0
            assert calls == ['train','validation','native_fit']
            with patch.object(training,'fit',side_effect=training.SynchronizationError('synthetic network failure')):
                try: runner.child(0)
                except training.SynchronizationError: pass
                else: raise AssertionError('Synchronization failure did not stop execution')
            assert not (Path(task['folder'])/'failure.json').exists()
    result={'state':'passed','CPU_only':True,'new_training':False,'candidate_assignments_verified':len(p['tasks']),
        'existing_results_checkpoint_contract_and_audit_verified':reused,
        'child_reads_only_train_validation':True,'no_mapping_selection_extra_seed_or_test_calls':True,
        'synchronization_failure_propagates_without_marking_scientific_failure':True}
    common.dump(runner.OUT/'preparation_checks.json',result)
    print(json.dumps(result))


if __name__ == '__main__': main()
