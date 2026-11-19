"""Resume unchanged frozen controllers after the explicit storage/PID amendment."""
import importlib
from pathlib import Path
import sys
import subprocess
import time
from currkt_storage import ROOT, STATE, read, verify_plan, ensure_external_storage, install_storage_guards


def main():
    assert len(sys.argv) == 3 and sys.argv[2] == 'run'
    target = Path(sys.argv[1])
    assert target.parent == ROOT / 'scripts'
    assert target.stem in ['run_joint_tuning', 'run_convergence_tuning']
    verify_plan()
    assert read(STATE / 'status.json')['state'] in ['storage_verified', 'restarting', 'releasing_verified_internal_copy', 'complete']
    ensure_external_storage()
    controller = importlib.import_module(target.stem)
    controller.guard()
    install_storage_guards(target.stem)
    if target.stem == 'run_convergence_tuning':
        import convergence_handoff as handoff
        original_read = handoff.read
        original_identity = handoff.process_identity
        expected = read(STATE / 'resumed_workers.json')['predecessor_process']
        original_predecessor = read(STATE / 'plan.json')['workers']['training']
        def migration_read(path):
            value = original_read(path)
            if Path(path) == handoff.OUT / 'protocol.json':
                assert value['predecessor_process'] == original_predecessor
                # Only the OS process identity changes after a clean restart.
                # This amendment is pinned separately; the old protocol file
                # and every scientific setting remain unchanged.
                value = dict(value, predecessor_process=expected)
            return value
        handoff.read = migration_read
        def identity_during_exit(pid):
            # macOS can transiently erase argv while a signalled process exits.
            # Exact identity is still required before any signal; retries never
            # signal a process and fail closed for a persistent mismatch.
            deadline=time.monotonic()+5
            while True:
                try:
                    return original_identity(pid)
                except AssertionError:
                    observed=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
                    state=observed.stdout.strip()
                    if observed.returncode==1 or not state or state.startswith('Z'):return None
                    if time.monotonic()>=deadline:raise
                    time.sleep(.05)
        handoff.process_identity=identity_during_exit
    sys.argv = [str(target), 'run']
    controller.main()


if __name__ == '__main__':
    main()
