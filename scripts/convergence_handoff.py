"""One finite CPU-only handoff, restricted to the recorded CurriculumKT PID."""
import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from joint_tuning_common import ROOT, OUT as PREVIOUS, read, dump, now, identity, verify_record

OUT = ROOT / 'results/convergence_tuning_20261111'


def process_identity(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'stat=', '-o', 'command='],
                            capture_output=True, text=True, check=False)
    if result.returncode == 1 or not result.stdout.strip():
        return None
    assert result.returncode == 0, 'Cannot inspect predecessor identity'
    parts = result.stdout.strip().split(None, 6)
    assert len(parts) == 7
    if parts[5].startswith('Z'):
        return None
    command = parts[6]; args = shlex.split(command)
    expected_script = ROOT / 'scripts/run_joint_tuning.py'
    assert str(expected_script) in args, 'Recorded PID is not the expected CurriculumKT controller'
    launch = read(PREVIOUS / 'launch.json')
    assert args[0] == launch['python'] and args[-1] == 'run'
    return dict(pid=pid, started=' '.join(parts[:5]), script=str(expected_script),
                command_sha256=hashlib.sha256(command.encode()).hexdigest())


def downstream_artifacts():
    from joint_tuning_common import MODELS
    paths = []
    for name in MODELS[MODELS.index('dkvmn') + 1:]:
        folder = PREVIOUS / 'models' / name
        paths.extend(str(path) for path in folder.rglob('*') if path.is_file())
    return paths


def boundary_ready():
    marker = PREVIOUS / 'models/dkvmn/comparison/complete.json'
    if not marker.exists():
        return False
    saved = read(marker)
    parent = read(OUT / 'protocol.json')['parent_protocol']
    assert saved['state'] == 'complete' and saved['protocol_sha256'] == parent['sha256']
    for key in ['report', 'comparison', 'contrasts', 'online_audit']:
        verify_record(saved[key])
    assert not downstream_artifacts(), 'Boundary missed: next architecture has artifacts; do not interrupt it'
    return True


def wait_and_handoff():
    """Never start successor GPU work until predecessor is gone and locks free.

    A brief SIGSTOP closes the check/signal race at a completed architecture.
    If the next architecture has started, resume immediately and fail closed.
    No frozen old source, active training state, or other project is modified.
    """
    protocol = read(OUT / 'protocol.json')
    expected = protocol['predecessor_process']; pid = expected['pid']
    receipt = OUT / 'handoff_complete.json'
    if receipt.exists():
        assert process_identity(pid) is None, 'Predecessor unexpectedly alive'
        return
    status = OUT / 'status.json'
    dump(status, dict(pid=os.getpid(), stage='dkvmn_architecture_boundary', state='waiting',
                     predecessor_pid=pid, GPU_work=False, codex_calls=0, updated_utc=now()))
    deadline = time.monotonic() + protocol['boundary_wait_hours'] * 3600
    marker = PREVIOUS / 'models/dkvmn/comparison/complete.json'
    # Pure filesystem polling while the old process works. No torch, MPS,
    # network or model construction is needed by this module.
    next_identity_check = 0
    while not marker.exists():
        assert time.monotonic() < deadline, 'Architecture-boundary wait exceeded its fixed limit'
        current_status = read(PREVIOUS / 'status.json')
        assert current_status['state'] not in ['failed', 'blocked'], 'Predecessor needs attention'
        if time.monotonic() >= next_identity_check:
            assert process_identity(pid) == expected, 'Predecessor exited or changed identity before its boundary'
            next_identity_check = time.monotonic() + 30
        time.sleep(.25)
    current = process_identity(pid)
    if current is None:
        assert boundary_ready()
        dump(receipt, dict(state='complete', predecessor_already_exited=True,
                          predecessor=expected, boundary=identity(marker), completed_utc=now()))
        return
    assert current == expected, 'PID reuse or unexpected predecessor identity'
    stopped = False
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Handoff interrupted by signal {signum}')
    for sig in [signal.SIGTERM, signal.SIGINT]:
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        os.kill(pid, signal.SIGSTOP); stopped = True
        stop_deadline = time.monotonic() + 5
        while True:
            state = subprocess.check_output(['ps', '-p', str(pid), '-o', 'state='], text=True).strip()
            if state.startswith('T'): break
            assert time.monotonic() < stop_deadline, 'Predecessor did not stop for boundary verification'
            time.sleep(.05)
        assert process_identity(pid) == expected
        assert boundary_ready()
        dump(OUT / 'handoff_prepared.json', dict(predecessor=expected, boundary=identity(marker),
             next_architecture_artifacts=[], prepared_utc=now(), active_fit_interrupted=False))
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT); stopped = False
        exit_deadline = time.monotonic() + 30
        while process_identity(pid) is not None:
            assert time.monotonic() < exit_deadline, 'Predecessor termination not verified; successor remains CPU-only'
            time.sleep(.1)
        dump(receipt, dict(state='complete', predecessor=expected, boundary=identity(marker),
                          active_fit_interrupted=False, next_architecture_started=False,
                          successor_GPU_started=False, completed_utc=now()))
    finally:
        if stopped:
            # Failed validation must never strand the original worker paused.
            if process_identity(pid) == expected:
                os.kill(pid, signal.SIGCONT)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
