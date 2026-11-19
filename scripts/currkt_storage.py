"""CurriculumKT-only external storage checks and migration primitives."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import stat
import subprocess
import time

ROOT = Path('/Users/kiakarbasi/CurriculumKT')
EXTERNAL = Path('/Volumes/LaCie')
STORE = EXTERNAL / 'CurriculumKT-storage'
IMAGE = STORE / 'CurriculumKT.sparsebundle'
STAGE = Path('/private/tmp/curriculumkt-ssd-stage-20261111')
BACKUP = Path('/private/tmp/CurriculumKT-internal-before-ssd-20261111')
STATE = STORE / 'migration-20261111'
PYTHON = '/Users/kiakarbasi/Knowledge-Component-Reordering-in-knowledge-tracing/.venv/bin/python'
SYSTEM_DIRS = {'.fseventsd', '.Spotlight-V100', '.Trashes', '.TemporaryItems'}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': sha(path)}


def volume_info(path):
    return plistlib.loads(subprocess.check_output(['/usr/sbin/diskutil', 'info', '-plist', str(path)]))


def verify_plan():
    marker = read(STATE / 'plan_identity.json')
    assert sha(marker['path']) == marker['sha256'], 'Migration plan changed'
    plan = read(marker['path'])
    for record in plan['identities']:
        assert sha(record['path']) == record['sha256'], record['path']
    return plan


def process_identity(pid, expected_script):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'stat=', '-o', 'command='],
                            capture_output=True, text=True)
    if result.returncode == 1 or not result.stdout.strip(): return None
    assert result.returncode == 0
    parts = result.stdout.strip().split(None, 6)
    assert len(parts) == 7
    if parts[5].startswith('Z'): return None
    args = shlex.split(parts[6])
    assert args[0] == PYTHON and str(expected_script) in args, 'Unexpected process identity'
    assert Path(expected_script).parent == ROOT / 'scripts', 'Only CurriculumKT processes are managed'
    return dict(pid=pid, started=' '.join(parts[:5]), script=str(expected_script),
                command_sha256=hashlib.sha256(parts[6].encode()).hexdigest())


def inventory(folder):
    """Hash all file content; preserve symlinks without following their targets."""
    folder = Path(folder); entries = {}
    def walk(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(folder).as_posix()
            if '/' not in relative and relative in SYSTEM_DIRS: continue
            before = path.lstat(); mode = stat.S_IMODE(before.st_mode)
            if stat.S_ISLNK(before.st_mode):
                entries[relative] = dict(kind='symlink', target=os.readlink(path))
            elif stat.S_ISDIR(before.st_mode):
                entries[relative] = dict(kind='directory', mode=mode)
                walk(path)
            elif stat.S_ISREG(before.st_mode):
                digest = sha(path); after = path.lstat()
                assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), f'File changed while hashing: {relative}'
                entries[relative] = dict(kind='file', mode=mode, bytes=before.st_size, sha256=digest)
            else:
                raise RuntimeError(f'Unsupported special file in project: {relative}')
    walk(folder)
    return entries


def compare_manifests(source, target):
    missing = sorted(set(source) - set(target)); extra = sorted(set(target) - set(source))
    changed = sorted(k for k in source.keys() & target.keys() if source[k] != target[k])
    assert not missing and not extra and not changed, dict(missing=missing[:10], extra=extra[:10], changed=changed[:10])
    return dict(entries=len(source), files=sum(v['kind']=='file' for v in source.values()),
                symlinks=sum(v['kind']=='symlink' for v in source.values()),
                bytes=sum(v.get('bytes',0) for v in source.values()), all_file_sha256_match=True,
                all_symlink_targets_match=True, all_modes_match=True)


def sync_tree(source, target, log):
    assert Path(source) == ROOT and Path(target) == STAGE
    args = ['/usr/bin/rsync', '-a', '--extended-attributes', '--checksum', '--delete']
    args += ['--exclude=/' + name for name in sorted(SYSTEM_DIRS)]
    args += [str(source) + '/', str(target) + '/']
    with Path(log).open('a') as handle:
        subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, check=True)


_mount_cache = None


def ensure_external_storage(minimum_gb=6):
    global _mount_cache
    assert ROOT.is_mount() and EXTERNAL.is_mount(), 'CurriculumKT external volume is not mounted'
    device = ROOT.stat().st_dev
    if _mount_cache is None or _mount_cache[0] != device or time.monotonic() - _mount_cache[1] > 30:
        plan = read(STATE / 'plan.json')
        assert volume_info(ROOT)['VolumeUUID'] == plan['image_volume_uuid'], 'Wrong project volume'
        assert volume_info(EXTERNAL)['VolumeUUID'] == plan['external_volume_uuid'], 'Wrong backing drive'
        assert ROOT.resolve() == ROOT, 'Frozen canonical project path changed'
        _mount_cache = (device, time.monotonic())
    for path in [ROOT, EXTERNAL]:
        free = shutil.disk_usage(path).free / 1e9
        assert free >= minimum_gb, f'{path}: {free:.2f} GB free, below {minimum_gb} GB reserve'


def install_storage_guards(controller_name):
    import joint_training
    import convergence_training
    import joint_comparison
    for module in [joint_training, convergence_training]:
        original = module.ensure_disk
        def checked(minimum_gb=6, original=original, error=module.ResourceUnavailable):
            try:
                ensure_external_storage(minimum_gb)
            except Exception as exc:
                raise error(str(exc)) from exc
            return original(minimum_gb)
        module.ensure_disk = checked
    joint_comparison.ensure_disk = (joint_training.ensure_disk if controller_name == 'run_joint_tuning'
                                   else convergence_training.ensure_disk)
