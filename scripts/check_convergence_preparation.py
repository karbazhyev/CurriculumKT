"""CPU-only acceptance tests of the amended scheduler and boundary handoff."""
from contextlib import ExitStack
import copy
import difflib
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import run_convergence_tuning as runner
import convergence_training as training
import convergence_handoff as handoff
from joint_tuning_common import ROOT, OUT as PREVIOUS, DATASETS, read, dump, identity, now


class ConvergenceChecks(unittest.TestCase):
    def test_selection_wandb_payload(self):
        with tempfile.TemporaryDirectory(prefix='currkt-selection-log-') as tmp:
            rows=[];configs=[]
            class Log:
                id='synthetic-local-only'
                def log(self, row):rows.append(row)
                def finish(self, **kwargs):pass
            def start(name, config):configs.append(config);return Log()
            def audit(run_id, expected, destination):
                self.assertEqual(run_id,'synthetic-local-only')
                self.assertEqual(rows,[dict(record_index=i,**row) for i,row in enumerate(expected)])
                dump(destination,{'state':'verified','synthetic':True})
            candidate=dict(id='i00',width=64,learning_rate=.001,dropout=.1,pca_dim=50,k=100)
            entry=dict(state='complete',candidate=candidate,best_validation_auc=.8,
                       mapped_validation_auc=.7,selected_lambda=.1,parameter_count=100)
            with patch.object(training,'start_log',start),patch.object(training,'audit_log',audit):
                runner.log_selection(Path(tmp),'dkt','assist2017','induced','extension',[entry],'synthetic')
            self.assertEqual(configs[0]['objective'],'mapped_validation_auc')
            self.assertFalse(configs[0]['test_used'])
            self.assertEqual(rows[0]['pca_dim'],50);self.assertEqual(rows[0]['k'],100)
            self.assertEqual(rows[0]['native_validation_auc'],.8)
            self.assertEqual(rows[0]['mapped_validation_auc'],.7)
            self.assertEqual(rows[0]['selected_lambda'],.1)

    def test_ranking_and_failures(self):
        def entry(i, auc, mapped, lam):
            return dict(state='complete', candidate={'id':i}, best_validation_auc=auc,
                        best_validation_loss=.5, mapped_validation_auc=mapped,
                        selected_lambda=lam, parameter_count=100)
        a=entry('a',.8,.7,.1); b=entry('b',.7,.7000005,1.)
        failed=dict(state='failed',candidate={'id':'f'},best_validation_auc=1.)
        self.assertEqual([e['candidate']['id'] for e in runner.rank_trials([a,b,failed],'original')],['a','b'])
        self.assertEqual([e['candidate']['id'] for e in runner.rank_trials([a,b,failed],'induced')],['b','a'])
        self.assertEqual(training.BUDGETS, {'screening':(100,15),'extension':(300,30),'confirmation':(300,30)})

    def test_end_to_end_schedule_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='currkt-convergence-check-') as tmp:
            out=Path(tmp); (out/'protocol.json').write_text('synthetic')
            calls=[]; logs=[]; calibrated=[]; tested=[]
            def run_trial(folder,name,ds,tax,candidate,seed,stage,ph):
                self.assertFalse(tested, 'Training after test access')
                if stage=='extension':
                    shortlist=read(out/'models/dkt/shortlist.json')
                    self.assertEqual(set(shortlist['datasets']),set(DATASETS))
                    self.assertEqual(len([c for c in calls if c[-1]=='screening']),108)
                calls.append((ds,tax,candidate['id'],seed,stage))
                folder.mkdir(parents=True,exist_ok=True)
                if stage=='extension' and ds=='assist2012' and candidate['id']=='i17':
                    return dict(state='failed',candidate=candidate,seed=seed,error='intentional retained finalist failure')
                p=folder/'best.ckpt';p.write_text('CPU synthetic checkpoint identity')
                num=int(candidate['id'][1:])
                # Ranking reverses among the finalists when given longer training.
                value=.6 + (num if stage=='screening' else 30-num)/1000
                result=dict(state='complete',candidate=candidate,seed=seed,k=candidate.get('k',4),checkpoint=identity(p),
                            best_validation_auc=value,best_validation_loss=.5,parameter_count=candidate['width'],
                            mapped_validation_auc=value,selected_lambda=.1,epochs=100 if stage=='screening' else 300,
                            best_epoch=98 if stage=='screening' else 290,
                            diagnosis={'status':'epoch_budget_limited','stop_reason':'max_epochs'})
                dump(folder/'result.json',result)
                return result
            def log(folder,name,ds,tax,stage,entries,ph):
                logs.append((ds,tax,stage,len(entries)))
            def calibrate(folder,name,records,models,ph,lambdas):
                self.assertFalse(tested)
                self.assertEqual(set(records),set(DATASETS))
                self.assertTrue(all(set(v)=={'42','84','126'} for ds in models.values() for v in ds.values()))
                self.assertEqual(len(calls),138)
                for ds in DATASETS:
                    self.assertEqual(models[ds]['original']['42']['candidate']['id'],'o15')
                    self.assertEqual(models[ds]['induced']['42']['candidate']['id'],'i15')
                    self.assertTrue(all('/extension/' in v['checkpoint']['path'] for v in
                                    [models[ds]['original']['42'],models[ds]['induced']['42']]))
                calibrated.append(True)
            def evaluate(folder,*args):
                self.assertEqual(calibrated,[True]);tested.append(True)
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'RESULTS.md').write_text('Synthetic comparison only\n')
                result={'state':'complete','report':identity(folder/'RESULTS.md')}
                dump(folder/'complete.json',result)
                return result
            replacements=dict(OUT=out,guard=lambda:{},status=lambda *a,**k:None,update_queue=lambda *a,**k:None,
                              run_trial=run_trial,log_selection=log)
            with ExitStack() as stack:
                for key,value in replacements.items():stack.enter_context(patch.object(runner,key,value))
                stack.enter_context(patch.object(runner.comparison,'calibrate',calibrate))
                stack.enter_context(patch.object(runner.comparison,'evaluate_comparison',evaluate))
                result=runner.execute_model('dkt')
                self.assertEqual(result['state'],'complete')
                self.assertEqual(len(calls),138)
                self.assertEqual(len(logs),12)
                self.assertEqual(sum(c[-1]=='extension' for c in calls),18)
                self.assertEqual(sum(c[-1]=='confirmation' for c in calls),12)
                runner.execute_model('dkt')
                self.assertEqual(len(calls),138,'Completed resume must not retrain')
                shortlist=out/'models/dkt/shortlist.json'
                shortlist.write_text(shortlist.read_text()+' ')
                with self.assertRaises(runner.IntegrityError):runner.freeze_shortlist('dkt',out/'models/dkt','irrelevant')
            failure_registry=read(out/'models/dkt/assist2012/induced_extension_registry.json')
            self.assertEqual(sum(e['state']=='failed' for e in failure_registry['trials']),1)
            self.assertFalse(any(c[2]=='i14' and c[-1]=='extension' for c in calls))

    def test_missing_seed_prevents_test(self):
        with tempfile.TemporaryDirectory(prefix='currkt-convergence-seed-') as tmp:
            out=Path(tmp);(out/'protocol.json').write_text('synthetic')
            design=read(PREVIOUS/'design.json')['trial_assignments']['dkt']
            shortlist={'datasets':{ds:{tax:[{'candidate':design[ds][tax][0]}] for tax in ['original','induced']} for ds in DATASETS}}
            def trial(folder,name,ds,tax,candidate,seed,stage,ph):
                p=out/'dummy';p.write_text('synthetic')
                if seed==84:return dict(state='failed',candidate=candidate,seed=seed)
                return dict(state='complete',candidate=candidate,seed=seed,k=4,checkpoint=identity(p),
                            best_validation_auc=.7,best_validation_loss=.5,parameter_count=10,
                            mapped_validation_auc=.7,selected_lambda=.1)
            with ExitStack() as stack:
                for key,value in dict(OUT=out,freeze_shortlist=lambda *a:shortlist,run_trial=trial,
                                      log_selection=lambda *a:None).items():stack.enter_context(patch.object(runner,key,value))
                evaluate=stack.enter_context(patch.object(runner.comparison,'evaluate_comparison'))
                with self.assertRaisesRegex(AssertionError,'Required seed failed'):
                    runner.execute_model('dkt')
                evaluate.assert_not_called()

    def test_legacy_import_development_only(self):
        design=read(PREVIOUS/'design.json')['trial_assignments']['dkvmn']['assist2017']
        for tax in ['original','induced']:
            for candidate in design[tax]:
                result=runner.validate_legacy_trial('assist2017',tax,candidate)
                self.assertEqual(result['state'],'complete')
                self.assertFalse(any('/test' in item['path'] for item in result['legacy_sources']))
        damaged=copy.deepcopy(design['original'][0]);damaged['width']+=1
        with self.assertRaises(runner.IntegrityError):runner.validate_legacy_trial('assist2017','original',damaged)

    def test_completed_cache_release_is_scoped(self):
        with tempfile.TemporaryDirectory(prefix='currkt-convergence-cache-') as tmp:
            folder=Path(tmp);(folder/'comparison').mkdir();(folder/'comparison/complete.json').write_text('{}')
            p=folder/'ds/extension/a';p.mkdir(parents=True)
            (p/'best.ckpt').write_text('best');(p/'latest.ckpt').write_text('resume')
            dump(p/'result.json',dict(wandb_run_id='a',checkpoint=identity(p/'best.ckpt')))
            dump(p/'online_audit.json',dict(state='verified',run_id='a'))
            failed=folder/'ds/extension/failed';failed.mkdir(parents=True)
            (failed/'latest.ckpt').write_text('preserved failed state')
            runner.release_optimizer_caches(folder)
            self.assertFalse((p/'latest.ckpt').exists());self.assertTrue((p/'best.ckpt').exists())
            self.assertTrue((failed/'latest.ckpt').exists())

    def test_handoff_success_and_failed_boundary_resume(self):
        for valid in [True,False]:
            with self.subTest(valid=valid), tempfile.TemporaryDirectory(prefix='currkt-handoff-check-') as tmp:
                root=Path(tmp);old=root/'old';out=root/'new';out.mkdir()
                marker=old/'models/dkvmn/comparison/complete.json';marker.parent.mkdir(parents=True);marker.write_text('{}')
                expected={'pid':123,'identity':'synthetic'}
                dump(out/'protocol.json',dict(predecessor_process=expected,boundary_wait_hours=168))
                signals=[];alive=[True]
                def identity_mock(pid):return expected if alive[0] else None
                def kill(pid,sig):
                    self.assertEqual(pid,123);signals.append(sig)
                    if sig==signal.SIGCONT and signal.SIGTERM in signals:alive[0]=False
                def ready():
                    if not valid:raise AssertionError('next architecture already started')
                    return True
                with ExitStack() as stack:
                    for key,value in dict(OUT=out,PREVIOUS=old,process_identity=identity_mock,boundary_ready=ready).items():
                        stack.enter_context(patch.object(handoff,key,value))
                    stack.enter_context(patch.object(handoff.os,'kill',kill))
                    stack.enter_context(patch.object(handoff.subprocess,'check_output',lambda *a,**k:'T'))
                    if valid:
                        handoff.wait_and_handoff()
                        self.assertEqual(signals,[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT])
                        self.assertTrue((out/'handoff_complete.json').exists())
                    else:
                        with self.assertRaises(AssertionError):handoff.wait_and_handoff()
                        self.assertEqual(signals,[signal.SIGSTOP,signal.SIGCONT])
                        self.assertTrue(alive[0]);self.assertFalse((out/'handoff_complete.json').exists())

    def test_real_gpu_lock_exclusion(self):
        # The existing worker holds these locks. This check opens no MPS device.
        import fcntl
        with (ROOT/'results/model_sequence_20261107/gpu_worker.lock').open('a') as f:
            with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ConvergenceChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=training.OUT;output.mkdir(parents=True,exist_ok=True)
    old=(ROOT/'scripts/joint_training.py').read_text().splitlines()
    new=(ROOT/'scripts/convergence_training.py').read_text().splitlines()
    (output/'training_source_diff.txt').write_text('\n'.join(difflib.unified_diff(old,new,fromfile='frozen joint_training.py',tofile='convergence_training.py'))+'\n')
    dump(output/'preparation_checks.json',dict(state='passed' if result.wasSuccessful() else 'failed',
         tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),empirical_fits=0,
         MPS_calls=0,network_calls=0,old_worker_signals=0,completed_utc=now()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
