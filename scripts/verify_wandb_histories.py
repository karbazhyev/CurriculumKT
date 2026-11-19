"""Verify selected archived curves against the existing online W&B runs."""
import concurrent.futures
import json
from pathlib import Path
import wandb
from audit_wandb_convergence import OUT, SOURCE, METRICS, diagnose, dump


def main():
    key = json.loads((SOURCE / 'configs/wandb.json').read_text())['api_key']
    archive = json.loads((OUT / 'selected_run_convergence.json').read_text())

    def fetch(row):
        path = f"{row['entity']}/{row['project']}/{row['run_id']}"
        try:
            api = wandb.Api(api_key=key, timeout=30)
            run = api.run(path)
            rows = []
            for h in run.scan_history(page_size=200):
                if all(k in h and h[k] is not None for k in ['epoch', 'train_loss', 'validauc']):
                    r = {k:h[k] for k in METRICS if k in h and h[k] is not None}
                    r['checkpoint_summary_merged_into_epoch'] = h.get('testauc') is not None or h.get('model_save_path') is not None
                    rows.append(r)
            local = {int(r['epoch']):r for r in row['history']}
            delta = max((abs(r['validauc'] - local[int(r['epoch'])]['validauc']) for r in rows if int(r['epoch']) in local), default=None)
            return {'path':path, 'url':run.url, 'state':run.state, 'history':rows,
                    'local_online_max_auc_difference':delta,
                    'diagnosis':diagnose(rows, row['diagnosis'].get('saved_selected_epoch'))}
        except Exception as e:
            return {'path':path, 'error_type':type(e).__name__}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, archive['historical_wandb_runs']))
    dump(OUT / 'online_history_verification.json', {'runs':results})
    print(json.dumps([{k:v for k,v in r.items() if k!='history'} for r in results]), flush=True)


if __name__ == '__main__':
    main()
