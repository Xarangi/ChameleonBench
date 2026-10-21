"""Hide-loss form and adaptive capability weight (REPLICATION 2026-10-21): both optional, defaults
leave the original objective and weights unchanged."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from next_chameleons.tica.objectives import ObjectiveConfig, hide_per_target, loss_terms


def test_hide_per_target_forms():
    z = torch.tensor([-3.0, 0.0, 0.7, 5.0, 30.0], requires_grad=True)
    p = torch.sigmoid(z)
    assert torch.equal(hide_per_target(p, z, "mse"), p**2)
    lg = hide_per_target(p, z, "log")
    assert torch.allclose(lg, torch.nn.functional.softplus(z))
    assert torch.isfinite(lg).all() and float(lg[-1]) > 29  # saturated: still a loss
    lg.sum().backward()
    assert torch.allclose(z.grad, torch.sigmoid(z.detach()))  # d softplus / dz = p
    pm = torch.sigmoid(torch.tensor([-3.0, 0.0, 2.0]))  # from p alone (no z): same values
    assert torch.allclose(hide_per_target(pm, None, "log"),
                          torch.nn.functional.softplus(torch.tensor([-3.0, 0.0, 2.0])), atol=1e-5)


def test_loss_terms_default_unchanged_and_log_form():
    z = torch.tensor([2.0, -1.0, 4.0, 0.5])
    p = torch.sigmoid(z)
    ref = torch.tensor([0.0, 0.3, 0.0, 0.2])
    roles = ["hide", "match", "hide", "match"]
    nll = torch.tensor([0.4, 0.6])
    hw = torch.tensor([1.0, 1.0, 3.0, 1.0])
    base, st0 = loss_terms(p, ref, roles, nll, ObjectiveConfig(), hide_w=hw)
    same, st1 = loss_terms(p, ref, roles, nll, ObjectiveConfig(hide_loss="mse"), hide_w=hw, z=z)
    assert torch.equal(base, same) and st0["hide_sq"] == st1["hide_sq"]
    cfg = ObjectiveConfig(hide_loss="log", hide_weight=0.3)
    lg, st = loss_terms(p, ref, roles, nll, cfg, hide_w=hw, z=z)
    sp = torch.nn.functional.softplus(z)
    want_hide = 0.3 * (1.0 * sp[0] + 3.0 * sp[2]) / 4.0
    mse_rest = base - 1.0 * (1.0 * p[0] ** 2 + 3.0 * p[2] ** 2) / 4.0
    assert torch.allclose(lg, want_hide + mse_rest, atol=1e-6)
    assert math.isclose(st["hide_sq"], st0["hide_sq"], rel_tol=1e-6)  # p^2 stat comparable


def _fake(target, **kw):
    from next_chameleons.tica.adaptive import AdaptiveConfig

    ac = AdaptiveConfig(cap_adapt_target=target, **kw)
    return SimpleNamespace(cfg=SimpleNamespace(adaptive=ac), _cap_mult=1.0, _cap_acc_ema=None)


def test_cap_adapt_raises_below_target_and_clips():
    from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer as T

    f = _fake(0.75, cap_adapt_rate=0.5, cap_adapt_ema=0.8, cap_adapt_max=2.0)
    info = T._cap_adapt(f, 0.25)  # first step: ema = acc
    assert math.isclose(f._cap_mult, math.exp(0.5 * 0.5)) and info["cap_adapt/acc_ema"] == 0.25
    for _ in range(20):
        T._cap_adapt(f, 0.25)
    assert f._cap_mult == 2.0  # clipped at max
    for _ in range(200):
        T._cap_adapt(f, 1.0)
    assert f._cap_mult == 1.0  # default min: never weaker than the configured weights
    g = _fake(0.75, cap_adapt_ema=0.5)
    T._cap_adapt(g, 0.5)
    T._cap_adapt(g, 1.0)
    assert math.isclose(g._cap_acc_ema, 0.75)
