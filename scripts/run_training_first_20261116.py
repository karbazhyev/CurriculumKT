"""Finite screening queue. Fresh process per trial; no map-back or selection."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from joint_tuning_common import ROOT, OUT as OLD, read, dump, now, sha, identity, verify_record

OUT = ROOT / 'results/training_first_20261116'


def protocol():
    p = OUT / 'protocol.json'
    value = read(p)
    assert sha(p) == read(OUT / 'protocol_identity.json')['sha256']
    for record in value['identities']:
        verify_record(record)
    assert sys.executable == value['python']
    return value


def contract(task, ph):
    return dict(model=task['model'], dataset=task['dataset'], taxonomy=task['arm'],
                candidate=task['candidate'], seed=42, k=task['k'], protocol_sha256=ph)


def completed(task, ph):
    folder = Path(task['folder'])
    if not (folder / 'result.json').exists():
        return None
    assert read(folder / 'contract.json') == contract(task, ph)
    result = read(folder / 'result.json')
    assert result['state'] == 'complete'
    assert result['candidate'] == task['candidate'] and result['seed'] == 42
    assert result['k'] == task['k'] and result['validation_replay_passed']
    verify_record(result['checkpoint'])
    audit = folder / 'online_audit.json'
    if not audit.exists():
        return None  # Existing fit() audits and reuses it without retraining.
    assert read(audit)['state'] == 'verified'
    assert read(audit)['run_id'] == result['wandb_run_id']
    return dict(state='complete', task_id=task['id'], model=task['model'], dataset=task['dataset'],
                arm=task['arm'], candidate=task['candidate'], seed=42,
                checkpoint=result['checkpoint'], result=identity(folder / 'result.json'),
                history=identity(folder / 'history.json'), native_validation_auc=result['best_validation_auc'],
                selected=False, mapping_pending=True)


def status(state, **values):
    dump(OUT / 'status.json', dict(state=state, pid=os.getpid(), updated_utc=now(),
                                 codex_calls=0, **values))


def mark_model(name, state):
    path = OUT / 'queue.json'
    queue = read(path); queue['models'][name]['state'] = state
    queue.update(active_model=name if state == 'screening' else None, updated_utc=now())
    dump(path, queue)
    path = ROOT / 'results/model_sequence_20261107/queue.json'
    queue = read(path)
    for row in queue['models']:
        if row['model'] == name:
            row.update(state=state, training_first_queue=str(OUT/'queue.json'),
                       final_selection='deferred', map_back='deferred')
    queue.update(active_model=name if state == 'screening' else None,
                 active_stage='training_first_'+state, updated_utc=now())
    dump(path, queue)


def child(index):
    p = protocol(); task = p['tasks'][index]; folder = Path(task['folder'])
    import run_joint_tuning as prior
    import joint_training as training
    from joint_tuning_common import load_arrays
    from currkt_storage import verify_plan, ensure_external_storage, install_storage_guards
    # Verify the unchanged scientific implementation before any fitting.
    prior.guard(); verify_plan(); ensure_external_storage(); install_storage_guards('run_joint_tuning')
    with ExitStack() as stack:
        prior.acquire_gpu(stack); training.require_mps()
        dump(OUT / 'current_device.json', dict(pid=os.getpid(), task_id=task['id'],
            device='mps', cpu_fallback=False, verified_utc=now()))
        try:
            train = load_arrays(task['dataset'], 'train', task['taxonomy'])
            valid = load_arrays(task['dataset'], 'validation', task['taxonomy'])
            assert not set(train['uid']) & set(valid['uid'])
            gate = folder / 'runtime_gate.json'
            if not gate.exists():
                # Four synthetic-target correctness queries on training students,
                # not map-back extraction, calibration or model selection.
                checked = training.runtime_gate(task['model'], task['k'], task['candidate'], train, task['arm'])
                dump(gate, checked)
            else:
                assert read(gate)['passed']
            training.fit(folder, task['model'], task['dataset'], task['arm'], task['candidate'],
                         42, train, valid, task['k'], p['parent_protocol']['sha256'])
            result = completed(task, p['parent_protocol']['sha256'])
            assert result is not None
            dump(OUT / 'receipts' / (task['id'] + '.json'), result)
        except (training.SynchronizationError, training.ResourceUnavailable):
            raise
        except Exception as exc:
            # Record the failure without changing candidates or backfilling.
            disposition = 'incompatible' if not (folder / 'runtime_gate.json').exists() else 'failed'
            failure = dict(state=disposition, task_id=task['id'], candidate=task['candidate'], seed=42,
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                failed_utc=now(), test_used=False)
            dump(OUT / 'receipts' / (task['id'] + '.json'), failure)
            dump(folder / 'failure.json', failure)
            return 10 if disposition == 'incompatible' else 11
    return 0


def run():
    p = protocol(); receipts = OUT / 'receipts'; receipts.mkdir(exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A retired controller must not compete for these same artifacts.
        with ExitStack() as retired:
            for path in [ROOT / 'results/training_first_20261113/controller.lock', OLD / 'controller.lock', ROOT / 'results/convergence_tuning_20261111/controller.lock']:
                handle = retired.enter_context(path.open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            blocked_models = set(); totals = dict(complete=0, failed=0, incompatible=0)
            active_model = None
            for index, task in enumerate(p['tasks']):
                protocol()
                if task['model'] != active_model:
                    if active_model is not None: mark_model(active_model, 'screening_accounted_for')
                    active_model = task['model']; mark_model(active_model, 'screening')
                existing = completed(task, p['parent_protocol']['sha256'])
                receipt = receipts / (task['id'] + '.json')
                if existing:
                    dump(receipt, existing); totals['complete'] += 1
                    continue
                folder = Path(task['folder'])
                if task['model'] in blocked_models:
                    dump(receipt, dict(state='incompatible', task_id=task['id'], reason='Earlier native runtime gate failed for this architecture; no repeated attempts.'))
                    totals['incompatible'] += 1; continue
                if receipt.exists() and read(receipt)['state'] in ['failed', 'incompatible']:
                    saved = read(receipt); totals[saved['state']] += 1
                    if saved['state'] == 'incompatible': blocked_models.add(task['model'])
                    continue
                if (folder / 'failure.json').exists():
                    dump(receipt, dict(state='failed', task_id=task['id'], preserved_failure=identity(folder / 'failure.json')))
                    totals['failed'] += 1; continue
                env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONUNBUFFERED='1')
                log_path = OUT / 'logs' / (task['id'] + '.log'); log_path.parent.mkdir(exist_ok=True)
                with log_path.open('ab') as log:
                    worker = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'child', '--index', str(index)],
                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    started = time.monotonic(); peak_rss = 0
                    try:
                        while worker.poll() is None:
                            if time.monotonic() - started > p['child_wall_hours'] * 3600:
                                raise TimeoutError('Trial wall-time limit; checkpoint retained, no automatic retry')
                            sample = subprocess.run(['ps', '-p', str(worker.pid), '-o', 'rss='], capture_output=True, text=True)
                            if sample.stdout.strip().isdigit(): peak_rss = max(peak_rss, int(sample.stdout))
                            status('screening', task_id=task['id'], model=task['model'], dataset=task['dataset'],
                                arm=task['arm'], candidate=task['candidate'], worker_pid=worker.pid,
                                complete=totals['complete'], total=len(p['tasks']), peak_resident_kib=peak_rss,
                                map_back='deferred', final_selection='deferred', log=str(log_path))
                            time.sleep(10)
                    except BaseException:
                        if worker.poll() is None:
                            worker.terminate()
                            try: worker.wait(timeout=20)
                            except subprocess.TimeoutExpired: worker.kill(); worker.wait()
                        raise
                dump(OUT / 'exits' / (task['id'] + '.json'), dict(returncode=worker.returncode,
                    elapsed_seconds=time.monotonic()-started, peak_resident_kib=peak_rss, ended_utc=now()))
                if worker.returncode not in [0, 10, 11]:
                    raise RuntimeError(f'Trial process exited {worker.returncode}; checkpoint preserved; inspect {log_path}')
                saved = read(receipt); totals[saved['state']] += 1
                if saved['state'] == 'incompatible': blocked_models.add(task['model'])
            state = 'screening_complete' if not (totals['failed'] or totals['incompatible']) else 'screening_complete_with_failures'
            if active_model is not None: mark_model(active_model, 'screening_accounted_for')
            queue = read(OUT/'queue.json'); queue.update(state=state, active_model=None, updated_utc=now())
            dump(OUT/'queue.json', queue)
            status(state, **totals, total=len(p['tasks']), map_back='deferred', final_selection='deferred',
                   next='Catch-up/new-dataset screening requires its separately verified adapters and candidate freeze. No map-back, convergence or seed confirmation starts automatically.')
            dump(OUT / 'checkpoint_bank.json', dict(state=state, selected=False,
                checkpoints=[read(receipts / (t['id']+'.json')) for t in p['tasks']], completed_utc=now()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'child', 'verify'])
    parser.add_argument('--index', type=int); args = parser.parse_args()
    if args.action == 'verify': protocol(); return 0
    if args.action == 'child': return child(args.index)
    def stopped(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stopped)
    try: run()
    except BaseException as exc:
        status('attention_required', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
