"""adaptive.refit_np_in_fit (10-21k): no-prompt texts as refit fit positives or gate-only."""

import dataclasses

import torch

from next_chameleons.tica.adaptive import AdaptiveConfig
from next_chameleons.tica.trainers.supervised import _refit_fit


def _job(np_in_fit: bool) -> dict:
    g = torch.Generator().manual_seed(0)
    d, L = 8, 3
    picks = [("pos", f"p{i}") for i in range(20)] + [("neg", f"n{i}") for i in range(40)]
    r = torch.randn(len(picks), d, generator=g)
    r[:20, 0] += 4.0
    npr = torch.randn(10, d, generator=g)
    npr[:, 0] += 4.0
    acfg = dataclasses.replace(AdaptiveConfig(), refit_np_in_fit=np_in_fit, refit_min_tpr_np=0.0)
    return {"kind": "linear", "L": L, "Ls": [L], "L0": L, "picks": picks, "rl": {L: r},
            "ft": None, "npl": {L: npr}, "nt": None, "acfg": acfg, "name": "t",
            "concept": "c", "full_kind": "linear", "sec_read": 0.0, "r": r, "npr": npr,
            "excl": set(), "device": torch.device("cpu")}


def test_np_in_fit_default_adds_half_of_noprompt():
    _, info = _refit_fit(_job(True))
    assert info["layers"][3]["n_pos"] == 25


def test_np_in_fit_off_uses_in_context_only_and_keeps_gate():
    _, info = _refit_fit(_job(False))
    assert info["layers"][3]["n_pos"] == 20
    assert 0.0 <= info["tpr_np"] <= 1.0
