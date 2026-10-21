"""adaptive.refit_min_tpr_ic (10-21o): out-of-fold in-context TPR gate (0 = off, always logged)."""

import dataclasses

import torch

from next_chameleons.tica.adaptive import AdaptiveConfig, RefitData, fit_refit_bank
from next_chameleons.tica.probe_zoo import gate_fit


def _data(shift: float, d: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(60, d, generator=g)
    pos[:, 0] += shift
    neg = torch.randn(200, d, generator=g)
    npos = torch.randn(10, d, generator=g)
    npos[:, 0] += shift
    return pos, neg, npos


def _bank(shift, **kw):
    pos, neg, npos = _data(shift)
    rd = RefitData(layers=[3], concept="c")
    rd.pos[3] = [pos]
    rd.neg[3] = [neg]
    rd.np_pos[3] = [npos]
    cfg = dataclasses.replace(AdaptiveConfig(), refit_cv_folds=5, refit_min_tpr_np=0.0,
                              refit_min_auroc=0.5, **kw)
    return fit_refit_bank(rd, "c", "t", cfg)


def test_tpr_ic_logged_and_off_by_default():
    bank, info = _bank(1.5)
    assert 0.0 <= info["layers"][3]["tpr_ic"] <= 1.0
    assert bank is not None  # gate off: AUROC alone decides


def test_tpr_ic_gate_rejects_weak_probe_keeps_strong():
    weak, _ = _bank(1.5, refit_min_tpr_ic=0.8)  # separable in rank, but few above a 1% FPR cut
    strong, info = _bank(8.0, refit_min_tpr_ic=0.8)
    assert weak is None
    assert strong is not None and info["tpr_ic"] >= 0.8


def test_gate_fit_reports_tpr_ic():
    pos, neg, npos = _data(8.0)
    cfg = dataclasses.replace(AdaptiveConfig(), refit_min_tpr_np=0.0, refit_min_tpr_ic=0.8)
    mon, info = gate_fit("maxpool", [p[None] for p in pos], [n[None] for n in neg],
                         [p[None] for p in npos], cfg, name="t", layer=3, concept="c")
    assert info["tpr_ic"] >= 0.8 and mon is not None
