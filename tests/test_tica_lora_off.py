"""tica_layer_sweep.lora_off: band / module-type / complement specs zero exactly the matching LoRA
scalings (synthetic module tree with PEFT-like attributes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_layer_sweep import lora_off


class _Lora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1)})
        self.scaling = {"default": 0.5}


def _model(n_layers=4):
    layers = torch.nn.ModuleList()
    for _ in range(n_layers):
        blk = torch.nn.Module()
        blk.mlp = torch.nn.Module()
        blk.mlp.down_proj = _Lora()
        blk.self_attn = torch.nn.Module()
        blk.self_attn.q_proj = _Lora()
        layers.append(blk)
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = layers
    return root


def _zeroed(m):
    return sorted(n for n, x in m.named_modules() if isinstance(x, _Lora)
                  and x.scaling["default"] == 0.0)


@pytest.mark.parametrize("spec,n", [("1-2", 4), ("mlp", 4), ("0-1:attn", 2), ("!1-2", 4),
                                    ("!0-3:mlp", 0)])
def test_specs(spec, n):
    m = _model()
    assert lora_off(m, spec) == n and len(_zeroed(m)) == n
    if spec == "0-1:attn":
        assert _zeroed(m) == ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
