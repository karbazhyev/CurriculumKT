"""Finite project-only migration after a fully audited training-trial boundary."""
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
from currkt_storage import ROOT, EXTERNAL, STORE, IMAGE, STAGE, BACKUP, STATE, PYTHON
from currkt_storage import read, dump, sha, identity, volume_info, verify_plan, process_identity
from currkt_storage import inventory, compare_manifests, sync_tree, ensure_external_storage


def now(): return datetime.now(timezone.utc).isoformat()


def status(state, **extra):
    dump(STATE / 'status.json', dict(state=state, pid=os.getpid(), updated_utc=now(), codex_calls=0, **extra))
    print(now(), state, flush=True)


def wait_gone(record, seconds=30):
    deadline=time.monotonic()+seconds
    while True:
        # After SIGTERM, macOS may replace the command string before marking
        # the process a zombie. No further signal is sent during this wait.
        observed=subprocess.run(['ps','-p',str(record['pid']),'-o','stat='],capture_output=True,text=True)
        state=observed.stdout.strip()
        if observed.returncode==1 or not state or state.startswith('Z'):return
        assert observed.returncode==0, 'Cannot observe process exit'
        assert time.monotonic()<deadline, 'Controller did not exit; migration remains stopped'
        time.sleep(.1)


def verify_trial(plan):
    folder=Path(plan['trial'])
    result=read(folder/'result.json');screened=read(folder/'screening_result.json')
    audit=read(folder/'online_audit.json')
    assert result['state']=='complete' and result['validation_replay_passed']
    assert screened['checkpoint']==result['checkpoint']
    assert sha(result['checkpoint']['path'])==result['checkpoint']['sha256']
    assert result['candidate']['id']==plan['candidate_id'] and result['seed']==42
    assert audit['state']=='verified' and audit['run_id']==result['wandb_run_id']
    assert audit['rows']==result['epochs']
    assert not Path(plan['next_trial']).exists(), 'Next trial already started; preserve it and fail closed'
    return dict(result=identity(folder/'result.json'),screening_result=identity(folder/'screening_result.json'),
                online_audit=identity(folder/'online_audit.json'),checkpoint=result['checkpoint'])


def stop_at_boundary(plan):
    training=plan['workers']['training'];waiting=plan['workers']['waiting']
    status('waiting_for_current_trial',trial=plan['trial'],epoch_at_request=plan['epoch_at_request'],GPU_work=False)
    deadline=time.monotonic()+24*3600;next_check=0
    marker=Path(plan['trial'])/'screening_result.json'
    while not marker.exists():
        assert time.monotonic()<deadline, 'Current-trial wait exceeded24hours'
        if time.monotonic()>=next_check:
            assert process_identity(training['pid'],training['script'])==training, 'Training controller exited or changed'
            assert read(ROOT/'results/joint_tuning_20261109/status.json')['state']!='failed', 'Current worker needs attention'
            next_check=time.monotonic()+15
        time.sleep(.05)
    assert process_identity(training['pid'],training['script'])==training
    stopped=False
    handlers={}
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Migration signal {signum}')
    for sig in [signal.SIGTERM,signal.SIGINT]:handlers[sig]=signal.signal(sig,interrupted)
    try:
        os.kill(training['pid'],signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+5
        while not subprocess.check_output(['ps','-p',str(training['pid']),'-o','state='],text=True).strip().startswith('T'):
            assert time.monotonic()<deadline
            time.sleep(.05)
        assert process_identity(training['pid'],training['script'])==training
        boundary=verify_trial(plan)
        waiting_status=read(ROOT/'results/convergence_tuning_20261111/status.json')
        assert waiting_status['state']=='waiting' and waiting_status['GPU_work'] is False
        assert process_identity(waiting['pid'],waiting['script'])==waiting
        dump(STATE/'boundary_verified.json',dict(verified_utc=now(),active_fit_interrupted=False,**boundary))
        os.kill(waiting['pid'],signal.SIGTERM);wait_gone(waiting)
        os.kill(training['pid'],signal.SIGTERM);os.kill(training['pid'],signal.SIGCONT);stopped=False
        wait_gone(training)
        status('controllers_stopped_at_verified_trial_boundary',active_fit_interrupted=False)
    finally:
        if stopped and process_identity(training['pid'],training['script'])==training:
            os.kill(training['pid'],signal.SIGCONT)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def locks(stack, root=ROOT):
    relative=['results/model_sequence_20261107/gpu_worker.lock',
              'results/simplekt_sweeps_20261107/worker.lock',
              'results/akt_sweeps_20261109/worker.lock',
              'results/joint_tuning_20261109/controller.lock',
              'results/convergence_tuning_20261111/controller.lock']
    for name in relative:
        handle=stack.enter_context((root/name).open('a'))
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)


def wait_initial_copy():
    pid=read(STORE/'initial_copy.json')['pid'];deadline=time.monotonic()+3600
    while True:
        p=subprocess.run(['ps','-p',str(pid),'-o','stat=','-o','command='],capture_output=True,text=True)
        if p.returncode==1 or not p.stdout.strip() or p.stdout.strip().startswith('Z'):return
        if str(STAGE) not in p.stdout or 'rsync' not in p.stdout:return
        assert time.monotonic()<deadline,'Initial copy has not exited'
        time.sleep(2)


def frozen_guards():
    logs=[]
    for name in ['run_joint_tuning.py','run_convergence_tuning.py']:
        result=subprocess.run([PYTHON,'-B',str(ROOT/'scripts'/name),'verify'],cwd=ROOT,capture_output=True,text=True)
        logs.append(dict(runner=name,exit_code=result.returncode,output=result.stdout,error=result.stderr[-2000:]))
        assert result.returncode==0,logs[-1]
    dump(STATE/'frozen_guards.json',dict(state='passed',checks=logs,verified_utc=now()))


def switch_storage(plan):
    assert STAGE.is_mount() and volume_info(STAGE)['VolumeUUID']==plan['image_volume_uuid']
    assert volume_info(EXTERNAL)['VolumeUUID']==plan['external_volume_uuid']
    assert not ROOT.is_mount() and not BACKUP.exists()
    assert ROOT.stat().st_dev==BACKUP.parent.stat().st_dev, 'Internal rollback rename must be on the same filesystem'
    wait_initial_copy()
    with ExitStack() as stack:
        locks(stack)
        status('final_copy_with_training_stopped')
        sync_tree(ROOT,STAGE,STATE/'final_copy.log')
        status('verifying_every_project_file')
        source=inventory(ROOT);target=inventory(STAGE)
        checked=compare_manifests(source,target)
        dump(STATE/'source_manifest.json',source)
        dump(STATE/'copy_verification.json',dict(state='passed',verified_utc=now(),**checked))
        # Recheck pinned source/config/data before the original directory moves.
        frozen_guards()
        subprocess.run(['/usr/bin/hdiutil','detach',str(STAGE)],check=True)
        os.rename(ROOT,BACKUP)
        ROOT.mkdir(mode=0o755)
        try:
            attached=subprocess.check_output(['/usr/bin/hdiutil','attach',str(IMAGE),'-mountpoint',str(ROOT),'-nobrowse','-noautoopen','-plist'])
            (STATE/'attach_canonical.plist').write_bytes(attached)
            assert ROOT.is_mount() and ROOT.resolve()==ROOT
            assert ROOT.stat().st_dev!=BACKUP.stat().st_dev
            ensure_external_storage()
            # New filesystem, new lock inodes; hold these too until restart.
            locks(stack)
            frozen_guards()
            result=read(Path(plan['trial'])/'result.json')
            assert sha(result['checkpoint']['path'])==result['checkpoint']['sha256']
            dump(STATE/'canonical_mount_verification.json',dict(state='passed',image=str(IMAGE),
                 mountpoint=str(ROOT),volume_uuid=volume_info(ROOT)['VolumeUUID'],
                 canonical_path_preserved=True,all_frozen_guards_passed=True,current_checkpoint_verified=True,
                 verified_utc=now()))
            status('storage_verified',internal_backup_retained=str(BACKUP))
        except BaseException:
            # Roll back only our mount. Never hide or delete source files.
            if ROOT.is_mount() and volume_info(ROOT).get('VolumeUUID')==plan['image_volume_uuid']:
                subprocess.run(['/usr/bin/hdiutil','detach',str(ROOT)],check=True)
            if ROOT.exists() and not any(ROOT.iterdir()):ROOT.rmdir()
            if not ROOT.exists():os.rename(BACKUP,ROOT)
            status('rolled_back_for_attention',source_preserved=True)
            raise


def launch_worker(name, log_folder):
    env=os.environ.copy();env.update(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONUNBUFFERED='1')
    command=[PYTHON,'-B',str(ROOT/'scripts/run_storage_resumed.py'),str(ROOT/'scripts'/name),'run']
    with (ROOT/'results'/log_folder/'controller.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    return dict(pid=process.pid,keepawake_pid=awake.pid,started_utc=now(),command=command)


def restart_workers(plan):
    status('restarting')
    first=launch_worker('run_joint_tuning.py','joint_tuning_20261109')
    predecessor=process_identity(first['pid'],ROOT/'scripts/run_joint_tuning.py')
    assert predecessor
    dump(STATE/'resumed_workers.json',dict(training=first,predecessor_process=predecessor,
         previous_training=plan['workers']['training'],scientific_protocol_changed=False,storage_amendment=identity(STATE/'plan.json')))
    second=launch_worker('run_convergence_tuning.py','convergence_tuning_20261111')
    record=read(STATE/'resumed_workers.json');record['waiting']=second;dump(STATE/'resumed_workers.json',record)
    for directory,worker in [('joint_tuning_20261109',first),('convergence_tuning_20261111',second)]:
        path=ROOT/'results'/directory/'queue.json';q=read(path)
        q.update(controller_pid=worker['pid'],storage_amendment=str(STATE/'plan.json'));dump(path,q)
        dump(ROOT/'results'/directory/'storage_resume_launch.json',worker)
    deadline=time.monotonic()+900;progress=Path(plan['next_trial'])/'progress.json'
    while True:
        assert process_identity(first['pid'],ROOT/'scripts/run_joint_tuning.py')==predecessor, 'Resumed training worker exited'
        assert process_identity(second['pid'],ROOT/'scripts/run_convergence_tuning.py'), 'Resumed waiter exited'
        state=read(ROOT/'results/joint_tuning_20261109/status.json')
        if state.get('pid')==first['pid']:
            assert state['state']!='failed', 'Resumed worker failed; internal backup retained'
        if progress.exists():
            p=read(progress)
            if p['epoch']>=1 and p['updated_utc']>first['started_utc']:break
        assert time.monotonic()<deadline, 'No completed new MPS epoch within15minutes; backup retained'
        time.sleep(2)
    waiting=read(ROOT/'results/convergence_tuning_20261111/status.json')
    assert waiting['pid']==second['pid'] and waiting['predecessor_pid']==first['pid']
    assert waiting['state']=='waiting' and waiting['GPU_work'] is False
    ensure_external_storage()
    dump(STATE/'restart_verification.json',dict(state='passed',training_pid=first['pid'],waiting_pid=second['pid'],
         training_progress=str(progress),progress=read(progress),one_KT_GPU_worker=True,
         actual_MPS_required_in_fit=True,CPU_fallback=False,verified_utc=now()))
    return record


def finish(plan, workers):
    assert BACKUP.exists() and not BACKUP.is_symlink() and not BACKUP.is_mount()
    assert BACKUP==Path('/private/tmp/CurriculumKT-internal-before-ssd-20261111')
    # Compare the backup to the exact pre-switch snapshot before removing it.
    # The original source is deleted only after byte verification and a new
    # complete MPS epoch on the external filesystem have both passed.
    compare_manifests(read(STATE/'source_manifest.json'),inventory(BACKUP))
    status('releasing_verified_internal_copy')
    shutil.rmtree(BACKUP)
    report=ROOT/'results/storage_migration_20261111';report.mkdir(parents=True,exist_ok=True)
    for filename in ['plan.json','plan_identity.json','boundary_verified.json','copy_verification.json',
                     'canonical_mount_verification.json','restart_verification.json','resumed_workers.json']:
        shutil.copy2(STATE/filename,report/filename)
    main_path=ROOT/'results/model_sequence_20261107/queue.json';main=read(main_path)
    main['storage']=dict(state='external',image=str(IMAGE),canonical_path=str(ROOT),migration=str(report),
                         training_pid=workers['training']['pid'],waiting_pid=workers['waiting']['pid'])
    if 'convergence_tuning' in main:main['convergence_tuning']['controller_pid']=workers['waiting']['pid']
    dump(main_path,main)
    with (ROOT/'research/PROJECT_STATUS.md').open('a') as f:
        f.write(f'\n- External storage migration completed {now()}: the whole CurriculumKT workspace is now in `{IMAGE}`, '
                f'mounted at its unchanged canonical path `{ROOT}`. Every regular file hash, symlink target and mode matched; '
                'both frozen protocol checks and a resumed MPS epoch passed before the internal copy was removed. '
                f'Training PID{workers["training"]["pid"]}; waiting convergence PID{workers["waiting"]["pid"]}. '
                '[Verification](../results/storage_migration_20261111/restart_verification.json). Keep LaCie mounted during training.\n')
    free=dict(internal_GB=shutil.disk_usage('/System/Volumes/Data').free/1e9,
              external_GB=shutil.disk_usage(EXTERNAL).free/1e9,project_volume_GB=shutil.disk_usage(ROOT).free/1e9)
    status('complete',image=str(IMAGE),canonical_path=str(ROOT),internal_copy_removed=True,free_space=free)
    dump(report/'status.json',read(STATE/'status.json'))


def recover_verified_boundary(plan):
    """Recover this recorded exit-check failure; no empirical fit was lost."""
    boundary=read(STATE/'boundary_verified.json')
    assert boundary['active_fit_interrupted'] is False
    for key in ['result','screening_result','online_audit','checkpoint']:
        record=boundary[key];assert sha(record['path'])==record['sha256']
    for record in plan['workers'].values():
        observed=subprocess.run(['ps','-p',str(record['pid']),'-o','lstart=','-o','stat='],capture_output=True,text=True)
        parts=observed.stdout.strip().split()
        assert observed.returncode==1 or not parts or parts[-1].startswith('Z') or ' '.join(parts[:5])!=record['started'], 'An original controller remains alive'
    # A readiness gate was saved before the disk guard stopped the next fit.
    # Preserve it, but require no training contract/history/checkpoint there.
    next_trial=Path(plan['next_trial'])
    assert not next_trial.exists() or {p.name for p in next_trial.iterdir()} <= {'runtime_gate.json'}
    status('recovering_verified_completed_trial',empirical_next_fit_started=False,process_signals=0)


def main():
    os.chdir('/private/tmp')
    with (STATE/'migration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=verify_plan();assert not BACKUP.exists()
            assert volume_info(EXTERNAL)['VolumeUUID']==plan['external_volume_uuid']
            if sys.argv[1:]==['recover']:
                recover_verified_boundary(plan)
            else:
                assert not sys.argv[1:]
                stop_at_boundary(plan)
            switch_storage(plan)
            workers=restart_workers(plan)
            finish(plan,workers)
        except BaseException as exc:
            status('attention_required',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                   internal_backup_exists=BACKUP.exists(),canonical_path_exists=ROOT.exists())
            raise


if __name__=='__main__':main()
