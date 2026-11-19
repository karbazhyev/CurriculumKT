"""Small CPU-only storage-integrity checks; no real process signals or GPU use."""
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch
import currkt_storage as storage
import migrate_curriculumkt_storage as migration


class StorageChecks(unittest.TestCase):
    def test_exit_wait_handles_transient_command_metadata(self):
        # The command may already be gone while the PID has not been reaped.
        observations=[subprocess.CompletedProcess([],0,'E\n',''),
                      subprocess.CompletedProcess([],0,'Z\n','')]
        with patch.object(migration.subprocess,'run',side_effect=observations), \
             patch.object(migration.time,'sleep',lambda *a:None):
            migration.wait_gone({'pid':123,'script':'unused-during-exit'})

    def test_full_copy_and_same_size_corruption(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.mkdir()
            (a/'nested').mkdir();(a/'nested/file').write_bytes(b'abc123')
            (a/'executable').write_text('test');(a/'executable').chmod(0o755)
            (a/'external-link').symlink_to('/not-followed-by-manifest')
            shutil.copytree(a,b,symlinks=True)
            result=storage.compare_manifests(storage.inventory(a),storage.inventory(b))
            self.assertEqual(result['files'],2);self.assertEqual(result['symlinks'],1)
            (b/'nested/file').write_bytes(b'xyz123')
            with self.assertRaises(AssertionError):storage.compare_manifests(storage.inventory(a),storage.inventory(b))

    def test_missing_file_and_mode_rejected(self):
        a={'file':dict(kind='file',mode=0o644,bytes=1,sha256='x')}
        with self.assertRaises(AssertionError):storage.compare_manifests(a,{})
        with self.assertRaises(AssertionError):storage.compare_manifests(a,{'file':dict(a['file'],mode=0o755)})

    def test_trial_boundary_requires_replay_audit_and_no_next_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'o13';folder.mkdir();checkpoint=folder/'best.ckpt';checkpoint.write_text('synthetic')
            result=dict(state='complete',validation_replay_passed=True,candidate={'id':'o13'},seed=42,
                        checkpoint=storage.identity(checkpoint),wandb_run_id='synthetic',epochs=100)
            storage.dump(folder/'result.json',result);storage.dump(folder/'screening_result.json',result)
            storage.dump(folder/'online_audit.json',dict(state='verified',run_id='synthetic',rows=100))
            plan=dict(trial=str(folder),candidate_id='o13',next_trial=str(Path(tmp)/'o14'))
            migration.verify_trial(plan)
            Path(plan['next_trial']).mkdir()
            with self.assertRaisesRegex(AssertionError,'Next trial'):migration.verify_trial(plan)
            Path(plan['next_trial']).rmdir();result['validation_replay_passed']=False
            storage.dump(folder/'result.json',result)
            with self.assertRaises(AssertionError):migration.verify_trial(plan)

    def test_unmounted_storage_blocks_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(storage,'ROOT',Path(tmp)),patch.object(storage,'EXTERNAL',Path(tmp)):
                with self.assertRaisesRegex(AssertionError,'not mounted'):storage.ensure_external_storage()

    def test_missed_boundary_resumes_training_without_termination(self):
        import signal
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'screening_result.json').write_text('{}')
            training=dict(pid=100,script='synthetic-training');waiting=dict(pid=101,script='synthetic-waiting')
            plan=dict(trial=str(folder),epoch_at_request=55,workers=dict(training=training,waiting=waiting))
            signals=[]
            with patch.object(migration,'status',lambda *a,**k:None), \
                 patch.object(migration,'process_identity',lambda *a:training), \
                 patch.object(migration.os,'kill',lambda pid,sig:signals.append((pid,sig))), \
                 patch.object(migration.subprocess,'check_output',lambda *a,**k:'T'), \
                 patch.object(migration,'verify_trial',side_effect=AssertionError('Next trial already started')):
                with self.assertRaises(AssertionError):migration.stop_at_boundary(plan)
            self.assertEqual(signals,[(100,signal.SIGSTOP),(100,signal.SIGCONT)])


if __name__=='__main__':unittest.main(verbosity=2)
