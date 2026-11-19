"""Load only provenance-checked, development-selected AKT checkpoints."""
from pathlib import Path

import torch

from akt_runtime import akt_class
from run_akt_seed_confirmation import OUT as SEED_ROOT, verify_stage
import run_akt_sweeps as base


def selected_models_ready():
    assert (SEED_ROOT / 'protocol.json').exists(), 'Complete selected-seed confirmation first'
    protocol = verify_stage()
    assert base.read(SEED_ROOT / 'run_registry.json')['state'] == 'complete'
    selected = base.read(SEED_ROOT / 'selected_models.json')
    assert selected['all_twelve_available'] and selected['test_used'] is False
    assert selected['protocol_sha256'] == base.sha(SEED_ROOT / 'protocol.json')
    cells = {f'{ds}_akt_{tax}' for ds in ['assist2017', 'assist2012']
             for tax in ['original', 'induced']}
    assert set(selected['models']) == cells
    for records in selected['models'].values():
        assert set(records) == {'42', '84', '126'}
        for record in records.values():
            assert base.sha(Path(record['checkpoint'])) == record['checkpoint_sha256']
    return protocol, selected


def make_model(protocol, dataset, taxonomy, record, device):
    cfg, fixed = record['candidate'], protocol['fixed']
    assert fixed['n_pid'] == 0 and fixed['n_blocks'] == 1 and fixed['kq_same'] == 1
    assert fixed['separate_qa'] is False
    cls = akt_class(device)
    model = cls(n_question=protocol['datasets'][dataset][taxonomy+'_concepts'],
        n_pid=0, d_model=cfg['d_model'], dropout=cfg['dropout'], n_blocks=fixed['n_blocks'],
        d_ff=fixed['d_ff'], num_attn_heads=fixed['num_attn_heads'],
        final_fc_dim=fixed['final_fc_dim'], kq_same=fixed['kq_same'],
        separate_qa=fixed['separate_qa'], l2=fixed['l2'], emb_type='qid').to(device)
    assert base.sha(Path(record['checkpoint'])) == record['checkpoint_sha256']
    model.load_state_dict(torch.load(record['checkpoint'], map_location='cpu', weights_only=True))
    assert all(p.device.type == torch.device(device).type for p in model.parameters())
    return model.eval()
