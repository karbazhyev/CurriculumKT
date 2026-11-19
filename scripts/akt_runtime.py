"""Pinned concept-only AKT with a runtime device override; no source edits."""
from check_attention_model_readiness import module


def akt_class(device):
    return module('akt', device).AKT


def forward(model, concepts, responses):
    assert model.model_name == 'akt' and model.n_pid == 0
    return model(concepts, responses)[0]
