"""Apply only the recorded predecessor PID amendment after an interruption."""
import sys
from pathlib import Path
import currkt_storage as storage
from joint_tuning_common import verify_record


def main():
    receipt = storage.ROOT / 'results/recovery_20261113/launch.json'
    record = storage.read(receipt)
    verify_record(record['wrapper_identity'])
    original_read = storage.read

    def amended_read(path):
        value = original_read(path)
        if Path(path) == storage.STATE / 'resumed_workers.json':
            value = dict(value, predecessor_process=record['predecessor_process'])
        return value

    storage.read = amended_read
    import run_scoped_storage_20261111 as runner
    sys.argv = [__file__, str(storage.ROOT / 'scripts/run_convergence_scoped_20261111.py'), 'run']
    runner.main()


if __name__ == '__main__':
    main()
