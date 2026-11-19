"""Finite local AKT continuation. No Codex, LLM API, cron, or new model planning.

Attach to the existing screen worker, then execute only prepared stages. Stop
on failed verification or at the unimplemented comparison boundary. Never
restart a failed training run or change the frozen experiment.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/akt_unattended_20261109'
SCREEN = ROOT / 'results/akt_sweeps_20261109'
SEEDS = ROOT / 'results/akt_seed_confirmation_20261109'
QUEUE = ROOT / 'results/model_sequence_20261107'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_manifest():
    manifest = read(OUT / 'manifest.json')
    assert sha(OUT / 'manifest.json') == read(OUT / 'manifest_identity.json')['sha256']
    assert str(Path(sys.executable)) == manifest['python'], 'Use the established interpreter'
    for name, expected in manifest['pinned_sha256'].items():
        assert sha(Path(name)) == expected, f'Pinned input changed: {name}'
    return manifest


def require_complete(registry, expected):
    assert len(registry['trials']) == expected
    assert registry['state'] == 'complete' and registry['active_run'] is None
    assert all(e['state'] == 'complete' for e in registry['trials'].values()), 'Retain failures; stop for review'


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def update_queue(stage, state, detail):
    path = QUEUE / 'queue.json'
    for _ in range(3):
        before = path.read_bytes()
        queue = json.loads(before)
        assert queue['active_model'] == 'akt', 'Another controller changed the active architecture'
        entry = next(e for e in queue['models'] if e['model'] == 'akt')
        entry['unattended_continuation'] = {
            'state': state, 'stage': stage, 'detail': detail, 'updated_utc': now(),
            'status_file': str(OUT / 'status.json'), 'controller_pid': os.getpid()}
        queue['active_stage'] = stage
        queue['updated_utc'] = now()
        if stage == 'seed_confirmation':
            entry['state'] = 'selected_seed_confirmation_running'
        if stage in ['selected_model_review', 'selected_cpu_gate', 'selected_mps_gate', 'comparison_implementation_pending']:
            entry['seed_confirmation'] = 'complete'
            entry['seed_confirmation_registry'] = str(SEEDS / 'run_registry.json')
            entry['seed_confirmation_protocol'] = str(SEEDS / 'protocol.json')
            entry['state'] = 'selected_seed_confirmation_complete'
        if stage == 'comparison_implementation_pending' and state == 'prepared_stages_complete':
            entry['state'] = 'comparison_implementation_pending'
            entry['query_adapter'] = 'all_12_selected_checkpoint_gates_passed_amendment_acceptance_pending'
        if state == 'attention_required':
            entry['state'] = 'unattended_attention_required'
        for directory, field in [(SCREEN, 'screening_snapshot'), (SEEDS, 'seed_confirmation_snapshot')]:
            if (directory / 'run_registry.json').exists():
                reg = read(directory / 'run_registry.json')
                entry[field] = {'checked_utc': now(), 'counts': dict(Counter(e['state'] for e in reg['trials'].values())), 'active_run': reg['active_run']}
        if path.read_bytes() == before:
            dump(path, queue)
            return
    raise RuntimeError('Queue changed repeatedly; preserve other work and stop')


def status(stage, state, detail, **extra):
    record = {'updated_utc': now(), 'pid': os.getpid(), 'stage': stage,
              'state': state, 'detail': detail, 'codex_calls': 0, **extra}
    dump(OUT / 'status.json', record)
    update_queue(stage, state, detail)
    print(json.dumps(record), flush=True)


def wait_existing_screen(manifest):
    status('screening', 'waiting_for_existing_worker', 'Existing worker continues unchanged; no duplicate launch.')
    deadline = time.monotonic() + manifest['maximum_wait_hours'] * 3600
    launch_path = Path(manifest['existing_launch'])
    while True:
        launch = read(launch_path)
        assert launch['worker_pid'] == manifest['existing_worker_pid']
        if launch['state'] == 'complete':
            assert launch['exit_code'] == 0
            require_complete(read(SCREEN / 'run_registry.json'), 32)
            return
        assert launch['state'] == 'running', 'Existing screening worker failed; no automatic restart'
        if not alive(launch['worker_pid']):
            time.sleep(2)  # Allow the existing supervisor to commit its exit record.
            assert read(launch_path)['state'] == 'complete', 'Existing screening worker exited without a successful launch record'
            continue
        assert time.monotonic() < deadline, 'Screening wait deadline exceeded; worker left untouched'
        time.sleep(30)


def wait_gpu_available(manifest):
    from inspect_model_workers import inspect
    paths = [SCREEN / 'worker.lock', QUEUE / 'gpu_worker.lock',
             ROOT / 'results/simplekt_sweeps_20261107/worker.lock']
    deadline = time.monotonic() + manifest['maximum_wait_hours'] * 3600
    while True:
        busy = False
        with ExitStack() as stack:
            try:
                for path in paths:
                    lock = stack.enter_context(path.open('a'))
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
        # Generic process availability only; no other-project files or control.
        external = [p for p in inspect() if p['project'] == 'external'
                    and ((p['gpu_layers'] or 0) != 0 or p['mode'] in ['generate', 'comet', 'metricx'])]
        if not busy and not external:
            return
        assert time.monotonic() < deadline, 'GPU availability wait exceeded; no processes changed'
        time.sleep(30)


def command(name, arguments, gpu=False, allow_failure=False):
    manifest = verify_manifest()
    if gpu:
        wait_gpu_available(manifest)
    folder = OUT / name
    folder.mkdir(exist_ok=False)
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONUNBUFFERED='1')
    with (folder / 'worker.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-B', *arguments], cwd=ROOT,
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record = {'state': 'running', 'started_utc': now(), 'worker_pid': child.pid,
                  'gpu_stage': gpu, 'log': str(folder / 'worker.log')}
        dump(folder / 'launch.json', record)
        rc = child.wait()
        record.update(state='complete' if rc == 0 else 'failed', exit_code=rc, completed_utc=now())
        dump(folder / 'launch.json', record)
    if not allow_failure:
        assert rc == 0, f'{name} failed (exit {rc}); see its preserved log'
    return rc


def audit(name, expected):
    status(name, 'running', 'Verify full development histories, checkpoint replay, configuration and convergence.')
    report_path = QUEUE / 'akt_online_history_checks.json'
    for attempt in range(1, 7):
        command(f'{name}_attempt{attempt:02}', [str(ROOT / 'scripts/verify_akt_training_histories.py')], allow_failure=True)
        if report_path.exists():
            report = read(report_path)
            dump(OUT / f'{name}_attempt{attempt:02}' / 'history_audit.json', report)
            if len(report['runs']) == expected and report['counts'].get('verified') == expected and report['all_currently_completed_runs_verified']:
                return
        if attempt < 6:
            time.sleep(300)  # Bounded W&B synchronization retries; never refit.
    raise RuntimeError('W&B audit not verified after six attempts; fits and failed audits retained')


def execute():
    manifest = verify_manifest()
    wait_existing_screen(manifest)
    audit('screening_review', 32)
    status('seed_confirmation', 'running', 'Freeze development-selected settings and run eight prepared MPS seed fits.')
    command('seed_prepare', [str(ROOT / 'scripts/run_akt_seed_confirmation.py'), 'prepare'])
    command('seed_worker', [str(ROOT / 'scripts/run_akt_seed_confirmation.py'), 'worker'], gpu=True)
    require_complete(read(SEEDS / 'run_registry.json'), 8)
    audit('selected_model_review', 40)
    for device in ['cpu', 'mps']:
        stage = f'selected_{device}_gate'
        status(stage, 'running', 'Verify all12 selected checkpoints and feature assembly on training references only.')
        command(stage, [str(ROOT / 'scripts/check_akt_stable_queries.py'), '--device', device], gpu=device == 'mps')
        report = read(QUEUE / f'akt_stable_queries_{device}_checks.json')
        assert report['state'] == 'passed' and report['all_twelve_selected_checkpoints']
        assert len(report['checks']) == 12 and all(c['passed'] for c in report['checks'])
    status('comparison_implementation_pending', 'prepared_stages_complete',
           'All prepared AKT stages completed. Comparison extraction/readout/report runners still require implementation; no later architecture or test evaluation launched.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'check'])
    args = parser.parse_args()
    if args.action == 'check':
        manifest = verify_manifest()
        assert not (SEEDS / 'protocol.json').exists(), 'Inspect existing seed stage before launch'
        assert read(Path(manifest['existing_launch']))['worker_pid'] == manifest['existing_worker_pid']
        print(json.dumps({'pinned_sources_verified': True, 'no_work_launched': True,
                          'bounded_stages': manifest['stages']}))
        return
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'status.json').exists(), 'Do not restart this finite chain blindly'
        try:
            execute()
        except BaseException as exc:
            previous = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
            # Only our controlled assertion messages enter the status, never SDK secrets.
            detail = str(exc) if isinstance(exc, (AssertionError, RuntimeError)) else type(exc).__name__
            status(previous.get('stage', 'initialization'), 'attention_required', detail,
                   error_type=type(exc).__name__)
            raise


if __name__ == '__main__':
    main()
