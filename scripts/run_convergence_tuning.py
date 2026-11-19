"""Finite, versioned shortlist extension; no Codex or other LLM dependency."""
import argparse
from contextlib import ExitStack
import fcntl
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from joint_tuning_common import ROOT, OUT as PREVIOUS, DATA, DATASETS, MODELS as ALL_MODELS
from joint_tuning_common import read, dump, now, sha, identity, verify_record
import convergence_training as training
import joint_comparison as comparison

OUT = training.OUT
MODELS = ALL_MODELS[ALL_MODELS.index('dkvmn'):]
MAIN_QUEUE = ROOT / 'results/model_sequence_20261107/queue.json'


class IntegrityError(RuntimeError):
    """Stop the entire controller on a provenance or frozen-design mismatch."""


def guard():
    try:
        protocol = read(OUT / 'protocol.json')
        verify_record(read(OUT / 'protocol_identity.json'))
        parent = read(verify_record(protocol['parent_protocol']))
        assert str(Path(sys.executable)) == parent['python']
        assert torch.__version__ == parent['environment']['torch']
        for record in parent['identities'] + protocol['identities']:
            verify_record(record)
        assert protocol['models'] == MODELS and protocol['datasets'] == DATASETS
        assert protocol['budgets'] == {k: list(v) for k, v in training.BUDGETS.items()}
        return protocol
    except Exception as exc:
        raise IntegrityError(str(exc)) from exc


def configure_comparison():
    # Bind the already pinned implementation to this process's new run root and
    # logger. No old module file, process, protocol or native loss is modified.
    comparison.OUT = OUT
    for key in ['start_log', 'audit_log', 'ensure_disk', 'require_mps']:
        setattr(comparison, key, getattr(training, key))


def verify_frozen(path):
    try:
        return verify_record(read(path))
    except Exception as exc:
        raise IntegrityError(f'Frozen runtime artifact changed: {path}') from exc


def status(stage, state, **extra):
    dump(OUT / 'status.json', dict(pid=os.getpid(), stage=stage, state=state,
         updated_utc=now(), codex_calls=0, **extra))


def update_queue(name, state, **extra):
    queue = read(OUT / 'queue.json')
    queue['models'][name].update(state=state, **extra)
    queue.update(state='running', active_model=name, updated_utc=now())
    dump(OUT / 'queue.json', queue)
    main = read(MAIN_QUEUE)
    main.update(active_model=name, active_stage=state, updated_utc=now())
    main['convergence_tuning'] = dict(queue=str(OUT / 'queue.json'), state=state,
                                     active_model=name, updated_utc=now())
    for row in main['models']:
        if row['model'] == name:
            row.update(state=state, convergence_queue=str(OUT / 'queue.json'), **extra)
    dump(MAIN_QUEUE, main)


def record_for(ds, taxonomy):
    manifest = read(DATA / ds / 'manifest.json')
    return dict(files=manifest['files'], mapping=manifest['taxonomies'][taxonomy]['mapping'], relabel=True)


def checkpoint(result):
    return {key: result[key] for key in ['candidate', 'seed', 'k', 'checkpoint']}


def rank_trials(entries, tax):
    """Apply the existing tie rule repeatedly; rank never consults test scores."""
    pending = [row for row in entries if row['state'] == 'complete']
    ranked = []
    while pending:
        if tax == 'original':
            chosen = min(pending, key=lambda e: (-e['best_validation_auc'],
                         e['best_validation_loss'], e['parameter_count'], e['candidate']['id']))
        else:
            best = max(e['mapped_validation_auc'] for e in pending)
            tied = [e for e in pending if e['mapped_validation_auc'] >= best - 1e-6]
            chosen = min(tied, key=lambda e: (-e['selected_lambda'], e['parameter_count'], e['candidate']['id']))
        ranked.append(chosen)
        pending.remove(chosen)
    return ranked


def validate_legacy_trial(ds, tax, candidate):
    """Read only completed DKVMN development artifacts; never its test metrics."""
    folder = PREVIOUS / 'models/dkvmn' / ds / 'screening' / candidate['id']
    failure = folder / 'failure.json'
    if failure.exists():
        return {**read(failure), 'legacy_source': identity(failure)}
    try:
        contract = read(folder / 'contract.json')
        assert contract['model'] == 'dkvmn' and contract['dataset'] == ds
        assert contract['taxonomy'] == tax and contract['candidate'] == candidate
        assert contract['seed'] == 42 and contract['protocol_sha256'] == sha(PREVIOUS / 'protocol.json')
        result = read(folder / 'screening_result.json')
        base = read(folder / 'result.json')
        assert result['state'] == 'complete' and base['validation_replay_passed']
        assert result['candidate'] == candidate and result['seed'] == 42
        assert result['checkpoint'] == base['checkpoint']
        verify_record(result['checkpoint'])
        audit = read(folder / 'online_audit.json')
        assert audit['state'] == 'verified' and audit['run_id'] == base['wandb_run_id']
        for key in ['best_validation_auc', 'best_validation_loss', 'parameter_count']:
            assert result[key] == base[key]
        files = ['contract.json', 'screening_result.json', 'result.json', 'history.json',
                 'online_audit.json', 'runtime_gate.json']
        if tax == 'induced':
            mapped = read(folder / 'mapping/mapped_selection.json')
            assert mapped['protocol_sha256'] == contract['protocol_sha256'] and not mapped['test_used']
            for key in ['selected_lambda', 'mapped_validation_auc']:
                assert result[key] == mapped[key]
            for coefficient in mapped['coefficients']:
                verify_record(coefficient)
            files.append('mapping/mapped_selection.json')
        return {**result, 'legacy_sources': [identity(folder / file) for file in files]}
    except Exception as exc:
        raise IntegrityError(f'Legacy screening verification failed: {folder}: {exc}') from exc


def log_selection(folder, name, ds, tax, stage, entries, ph):
    """Audit both the native and mapped objectives online in a separate run."""
    folder.mkdir(parents=True, exist_ok=True)
    ranked = rank_trials(entries, tax)
    ranks = {row['candidate']['id']: i + 1 for i, row in enumerate(ranked)}
    rows = []
    for entry in entries:
        candidate = entry['candidate']
        row = dict(candidate_id=candidate['id'], state=entry['state'],
                   width=candidate['width'], learning_rate=candidate['learning_rate'], dropout=candidate['dropout'])
        if tax == 'induced':
            row.update(pca_dim=candidate['pca_dim'], k=candidate['k'])
        if entry['state'] == 'complete':
            row.update(rank=ranks[candidate['id']], native_validation_auc=entry['best_validation_auc'])
            if tax == 'induced':
                row.update(mapped_validation_auc=entry['mapped_validation_auc'], selected_lambda=entry['selected_lambda'])
        rows.append(row)
    marker = folder / 'wandb.json'
    if not marker.exists():
        run = training.start_log(f'{name}-{ds}-{tax}-{stage}-selection', dict(model=name,
              dataset=ds, taxonomy=tax, stage=stage+'_selection', protocol_sha256=ph,
              objective='mapped_validation_auc' if tax == 'induced' else 'native_validation_auc', test_used=False))
        try:
            for i, row in enumerate(rows):
                run.log(dict(record_index=i, **row))
            run.finish()
            dump(marker, {'id': run.id})
        except BaseException:
            run.finish(exit_code=1)
            raise
    dump(folder / 'records.json', rows)
    training.audit_log(read(marker)['id'], rows, folder / 'online_audit.json')


def run_trial(folder, name, ds, tax, candidate, seed, stage, ph):
    guard()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'failure.json').exists():
        return read(folder / 'failure.json')
    manifest = read(DATA / ds / 'manifest.json')
    taxonomy = candidate.get('taxonomy', next(iter(manifest['taxonomies'])))
    record = record_for(ds, taxonomy)
    k = manifest['original_concepts'] if tax == 'original' else candidate['k']
    gate = folder / 'runtime_gate.json'
    status(name, stage, dataset=ds, taxonomy=tax, candidate=candidate['id'], seed=seed)
    update_queue(name, stage)
    train = valid = None
    try:
        train = comparison.load_data(record, 'train')
        valid = comparison.load_data(record, 'validation')
        assert not set(train['uid']) & set(valid['uid'])
        if not gate.exists():
            dump(gate, training.runtime_gate(name, k, candidate, train, tax))
        result = training.fit(folder, name, ds, tax, candidate, seed, train, valid, k, ph, stage=stage)
        train = valid = None
        gc.collect(); torch.mps.empty_cache()
        if tax == 'induced' and stage in ['screening', 'extension']:
            mapping = comparison.select_induced(folder / 'mapping', name, record, checkpoint(result), ph)
            result = dict(result, selected_lambda=mapping['selected_lambda'], mapped_validation_auc=mapping['mapped_validation_auc'])
        dump(folder / 'selection_result.json', result)
        return result
    except (training.SynchronizationError, training.ResourceUnavailable, IntegrityError):
        raise
    except (RuntimeError, NotImplementedError, AssertionError, ValueError) as exc:
        failure = dict(state='failed', candidate=candidate, seed=seed, stage=stage,
                       error_type=type(exc).__name__, error=str(exc), failed_utc=now(), test_used=False)
        dump(folder / 'failure.json', failure)
        if not gate.exists():
            raise
        return failure
    finally:
        train = valid = None
        gc.collect(); torch.mps.empty_cache()


def freeze_shortlist(name, folder, ph):
    path = folder / 'shortlist.json'
    if path.exists():
        verify_frozen(folder / 'shortlist_identity.json')
        saved = read(path)
        if saved['protocol_sha256'] != ph:
            raise IntegrityError('Shortlist protocol changed')
        for record in saved['registries']:
            try:
                verify_record(record)
            except Exception as exc:
                raise IntegrityError('Screening registry changed after shortlist freeze') from exc
        return saved
    design = read(PREVIOUS / 'design.json')
    selected = {}; registries = []
    for ds in DATASETS:
        selected[ds] = {}
        for tax in ['original', 'induced']:
            entries = []
            for candidate in design['trial_assignments'][name][ds][tax]:
                result = (validate_legacy_trial(ds, tax, candidate) if name == 'dkvmn' else
                          run_trial(folder / ds / 'screening' / candidate['id'], name, ds, tax, candidate, 42, 'screening', ph))
                entries.append(result)
                dump(folder / ds / f'{tax}_screening_registry.json', dict(trials=entries, updated_utc=now(), test_used=False))
            ranked = rank_trials(entries, tax)
            assert ranked, f'No successful screening candidates: {name}/{ds}/{tax}'
            selected[ds][tax] = ranked[:3]
            log_selection(folder / ds / 'selection_logs' / f'{tax}_screening', name, ds, tax, 'screening', entries, ph)
            registries.append(identity(folder / ds / f'{tax}_screening_registry.json'))
    saved = dict(protocol_sha256=ph, datasets=selected, registries=registries,
                 shortlist_size=3, test_used=False, frozen_utc=now(),
                 rerun_policy='Fresh seed42 fits for every shortlisted candidate; no warm starts or replacement of failed finalists.')
    dump(path, saved)
    dump(folder / 'shortlist_identity.json', identity(path))
    return saved


def convergence_report(folder):
    rows = []
    for path in sorted(folder.glob('*/*/*/result.json')):
        result = read(path)
        rows.append(dict(dataset=path.parts[-4], phase=path.parts[-3], trial=path.parts[-2],
                         seed=result['seed'], epochs=result['epochs'], best_epoch=result['best_epoch'],
                         diagnosis=result['diagnosis'], validation_auc=result['best_validation_auc']))
    selected = read(folder / 'selected_models.json')['models']
    selected_paths = {v['checkpoint']['path'] for ds in selected.values() for tax in ds.values() for v in tax.values()}
    capped = []
    for path in folder.glob('*/*/*/result.json'):
        r = read(path)
        if r['checkpoint']['path'] in selected_paths and r['diagnosis']['status'] == 'epoch_budget_limited':
            capped.append(str(path))
    dump(folder / 'convergence_review.json', dict(fits=rows, selected_budget_limited=capped,
         all_selected_count=len(selected_paths), mathematical_convergence_claimed=False, reviewed_utc=now()))


def release_optimizer_caches(folder):
    """Release only this completed architecture's new, audited resume caches."""
    assert (folder / 'comparison/complete.json').exists()
    for latest in folder.glob('*/*/*/latest.ckpt'):
        result_path = latest.parent / 'result.json'
        if not result_path.exists() or (latest.parent / 'failure.json').exists():
            continue
        result = read(result_path); audit = read(latest.parent / 'online_audit.json')
        assert audit['state'] == 'verified' and audit['run_id'] == result['wandb_run_id']
        verify_record(result['checkpoint'])
        dump(latest.parent / 'released_optimizer_cache.json', dict(file=identity(latest), released_utc=now(),
             reason='Architecture convergence extension, confirmation and comparison complete; retain best checkpoint and full history.'))
        latest.unlink()


def execute_model(name):
    ph = sha(OUT / 'protocol.json')
    folder = OUT / 'models' / name; folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    if marker.exists():
        return read(marker)
    shortlist = freeze_shortlist(name, folder, ph)
    records = {}; backbones = {}; lambdas = {}
    for ds in DATASETS:
        selection = folder / ds / 'backbone_selection.json'
        if selection.exists():
            verify_frozen(selection.with_name('backbone_selection_identity.json'))
            chosen = read(selection)
            if chosen['protocol_sha256'] != ph:
                raise IntegrityError('Selected protocol changed')
        else:
            chosen = dict(protocol_sha256=ph, test_used=False, frozen_utc=now())
            for tax in ['original', 'induced']:
                entries = [run_trial(folder / ds / 'extension' / entry['candidate']['id'], name, ds, tax,
                           entry['candidate'], 42, 'extension', ph) for entry in shortlist['datasets'][ds][tax]]
                dump(folder / ds / f'{tax}_extension_registry.json', dict(trials=entries, test_used=False))
                ranked = rank_trials(entries, tax)
                assert ranked, f'All shortlisted candidates failed: {ds}/{tax}; no backfilling'
                chosen[tax] = ranked[0]
                log_selection(folder / ds / 'selection_logs' / f'{tax}_extension', name, ds, tax, 'extension', entries, ph)
            dump(selection, chosen)
            dump(selection.with_name('backbone_selection_identity.json'), identity(selection))
        records[ds] = record_for(ds, chosen['induced']['candidate']['taxonomy'])
        lambdas[ds] = chosen['induced']['selected_lambda']
        backbones[ds] = {}
        for tax in ['original', 'induced']:
            winner = chosen[tax]
            backbones[ds][tax] = {'42': checkpoint(winner)}
            for seed in [84, 126]:
                result = run_trial(folder / ds / 'confirmation' / f'{tax}_seed{seed}', name, ds, tax,
                                   winner['candidate'], seed, 'confirmation', ph)
                assert result['state'] == 'complete', 'Required seed failed; no favorable seed substitution'
                backbones[ds][tax][str(seed)] = checkpoint(result)
    dump(folder / 'selected_models.json', dict(models=backbones, records=records, test_used=False, protocol_sha256=ph))
    convergence_report(folder)
    guard(); status(name, 'calibration'); update_queue(name, 'calibration')
    comparison.calibrate(folder / 'comparison', name, records, backbones, ph, lambdas)
    guard(); status(name, 'frozen_test'); update_queue(name, 'frozen_test')
    completed = folder / 'comparison/complete.json'
    if completed.exists():
        result = read(completed)
        try:
            assert result['protocol_sha256'] == ph
            for key in ['report', 'comparison', 'contrasts', 'online_audit']:
                verify_record(result[key])
        except Exception as exc:
            raise IntegrityError('Completed comparison artifacts changed') from exc
    else:
        result = comparison.evaluate_comparison(folder / 'comparison', name, records, backbones, ph)
    review = read(folder / 'convergence_review.json')
    report = folder / 'comparison/RESULTS.md'
    note = ('\nConvergence amendment: development-only top-three shortlist per condition, fresh seed42 reruns with '
            '300-epoch ceiling/patience30, then seeds84/126 at the selected setting. '
            f'{len(review["selected_budget_limited"])} of18 selected backbones remain epoch-budget-limited. '
            'PCA/K and all model settings were selected using development data only. '
            'Prior 100-epoch results are preserved separately; this follow-up remains exploratory.\n')
    if 'Convergence amendment:' not in report.read_text():
        with report.open('a') as f: f.write(note)
    result.update(report=identity(report), convergence_review=identity(folder / 'convergence_review.json'),
                  shortlist=identity(folder / 'shortlist.json'))
    dump(folder / 'comparison/complete.json', result)
    release_optimizer_caches(folder)
    dump(marker, result)
    return result


def combined_report():
    import pandas as pd
    frames = []; lines = ['# Convergence extension', '', 'Earlier fixed-budget results remain separate.', '',
                           '| Model | State | Report |', '|---|---|---|']
    queue = read(OUT / 'queue.json')
    for name, row in queue['models'].items():
        path = OUT / 'models' / name / 'comparison/comparison.csv'
        if path.exists():
            frames.append(pd.read_csv(path).assign(protocol_variant='convergence_300'))
        report = OUT / 'models' / name / ('comparison/RESULTS.md' if row['state'] == 'complete' else 'failure.json')
        lines.append(f'| {name} | {row["state"]} | [{report.name}]({report}) |' if report.exists() else f'| {name} | {row["state"]} | pending |')
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / 'combined_comparison.csv', index=False)
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def acquire_gpu(stack):
    from inspect_model_workers import inspect
    deadline = time.monotonic() + 72 * 3600
    paths = [ROOT / 'results/model_sequence_20261107/gpu_worker.lock',
             ROOT / 'results/simplekt_sweeps_20261107/worker.lock',
             ROOT / 'results/akt_sweeps_20261109/worker.lock', PREVIOUS / 'controller.lock']
    while True:
        trial = ExitStack()
        try:
            for path in paths:
                handle = trial.enter_context(path.open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            external = [p for p in inspect() if p['project'] == 'external' and
                        (p.get('mode') in ['generate', 'comet', 'metricx'] or (p.get('gpu_layers') or 0) > 0)]
            if external: raise BlockingIOError('External GPU indicators active')
            stack.enter_context(trial.pop_all()); return
        except BlockingIOError:
            trial.close()
        assert time.monotonic() < deadline, 'GPU wait exceeded72h'
        status('gpu_availability', 'waiting'); time.sleep(30)


def execute():
    guard(); configure_comparison()
    from convergence_handoff import wait_and_handoff
    wait_and_handoff()
    with ExitStack() as stack:
        acquire_gpu(stack); guard(); training.require_mps()
        dump(OUT / 'mps_acceptance.json', dict(device='mps', available=True, cpu_fallback=False,
                                             pid=os.getpid(), verified_utc=now()))
        old = read(PREVIOUS / 'queue.json')
        old.update(state='handed_off_at_dkvmn_boundary', successor=str(OUT / 'queue.json'), updated_utc=now())
        dump(PREVIOUS / 'queue.json', old)
        for name in MODELS:
            if read(OUT / 'queue.json')['models'][name]['state'] in ['complete', 'failed', 'incompatible']:
                continue
            try:
                result = execute_model(name)
                update_queue(name, 'complete', result=result)
            except (IntegrityError, training.SynchronizationError, training.ResourceUnavailable):
                raise
            except Exception as exc:
                failure = dict(state='incompatible' if isinstance(exc, NotImplementedError) else 'failed',
                    error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), failed_utc=now(),
                    test_driven_reselection=False)
                dump(OUT / 'models' / name / 'failure.json', failure)
                update_queue(name, failure['state'], failure=failure)
            combined_report(); guard()
            row = read(OUT / 'queue.json')['models'][name]
            with (ROOT / 'research/PROJECT_STATUS.md').open('a') as report:
                report.write(f'\n- Convergence controller {now()}: {name} {row["state"]}; '
                             'all three datasets assigned, development-only selection. '
                             '[Results](../results/convergence_tuning_20261111/RESULTS.md).\n')
        queue = read(OUT / 'queue.json')
        final = 'complete' if all(r['state'] == 'complete' for r in queue['models'].values()) else 'complete_with_failures'
        queue.update(state=final, updated_utc=now()); dump(OUT / 'queue.json', queue)
        status('finished', final); combined_report()
        main = read(MAIN_QUEUE)
        main.update(active_model=None, active_stage=final, updated_utc=now())
        main['convergence_tuning'].update(state=final, active_model=None)
        dump(MAIN_QUEUE, main)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'verify'])
    args = parser.parse_args()
    if args.action == 'verify':
        guard(); print('Both protocols and source/data identities verified; no GPU work.'); return
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute()
        except BaseException as exc:
            status('attention_required', 'failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
