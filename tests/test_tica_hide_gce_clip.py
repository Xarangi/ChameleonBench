"""GCE hide loss (objective.hide_loss "gce") and the per-term hide gradient clip
(train.hide_grad_clip), REPLICATION 2026-10-21r. Both default off."""

from __future__ import annotations

import math
import sys

import pytest
import torch

sys.path.insert(0, "tests")

from next_chameleons.tica.objectives import ObjectiveConfig, hide_per_target, loss_terms


def test_gce_values_and_gradients_match_closed_form():
    z0 = torch.tensor([-6.0, -1.0, 0.0, 0.7, 1.2, 4.0, 12.0, 40.0], dtype=torch.float64)
    p = torch.sigmoid(z0)
    omp = torch.sigmoid(-z0)  # 1 - p without cancellation (1 - p rounds to 0 at z = 40)
    for q in (0.3, 0.7, 1.0):
        z = z0.clone().requires_grad_(True)
        loss = hide_per_target(torch.sigmoid(z), z, "gce", q)
        want = (1 - omp**q) / q
        assert torch.allclose(loss.detach(), want, rtol=1e-10, atol=1e-12)
        loss.sum().backward()
        assert torch.allclose(z.grad, p * omp**q, rtol=1e-8, atol=1e-14)
        assert torch.isfinite(loss).all() and float(loss[-1]) <= 1 / q + 1e-12  # bounded
    assert torch.allclose(hide_per_target(p, z0, "gce", 1.0), p)  # q = 1: MAE-like loss p
    # from p alone (no z): same values while p is not saturated (z then needed, as for "log")
    assert torch.allclose(hide_per_target(p[:-2], None, "gce", 0.3),
                          hide_per_target(p[:-2], z0[:-2], "gce", 0.3), atol=1e-6)
    # q -> 0: the log loss softplus(z), values and gradient
    z = z0.clone().requires_grad_(True)
    small = hide_per_target(torch.sigmoid(z), z, "gce", 1e-7)
    assert torch.allclose(small.detach(), torch.nn.functional.softplus(z0), rtol=1e-5)
    small.sum().backward()
    assert torch.allclose(z.grad, p, rtol=1e-5)
    with pytest.raises(ValueError):
        hide_per_target(p, z0, "gce", 0.0)


def test_gce_peak_push_and_weight_choice():
    """max_z of p (1 - p)^q is at p = 1 / (1 + q); MSE's 2 p^2 (1 - p) peaks at 8/27 (p = 2/3)."""

    q = 0.3
    pk = 1 / (1 + q)
    peak = pk * (1 - pk) ** q
    zz = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    pp = torch.sigmoid(zz)
    assert math.isclose(float((pp * (1 - pp) ** q).max()), peak, rel_tol=1e-6)
    assert math.isclose(float((2 * pp**2 * (1 - pp)).max()), 8 / 27, rel_tol=1e-6)
    assert 0.59 < (8 / 27) / peak < 0.61  # run B: hide_weight 0.6


def test_loss_terms_gce_and_return_hide_split():
    z = torch.tensor([2.0, -1.0, 4.0, 0.5])
    p = torch.sigmoid(z)
    ref = torch.tensor([0.0, 0.3, 0.0, 0.2])
    roles = ["hide", "match", "hide", "match"]
    nll = torch.tensor([0.4, 0.6])
    hw = torch.tensor([1.0, 1.0, 3.0, 1.0])
    base, _ = loss_terms(p, ref, roles, nll, ObjectiveConfig(), hide_w=hw)
    tot, _, (h, rest) = loss_terms(p, ref, roles, nll, ObjectiveConfig(), hide_w=hw,
                                   return_hide=True)
    assert torch.equal(tot, base)  # the flag does not change the total
    assert torch.allclose(h + rest, base, atol=1e-7)
    assert torch.allclose(h, (p[0] ** 2 + 3 * p[2] ** 2) / 4)
    cfg = ObjectiveConfig(hide_loss="gce", hide_gce_q=0.3, hide_weight=0.6)
    g, _ = loss_terms(p, ref, roles, nll, cfg, hide_w=hw, z=z)
    gce = (1 - (1 - p) ** 0.3) / 0.3
    want = 0.6 * (gce[0] + 3 * gce[2]) / 4 + (base - (p[0] ** 2 + 3 * p[2] ** 2) / 4)
    assert torch.allclose(g, want, atol=1e-6)


def test_hide_clip_off_by_default():
    from next_chameleons.tica.trainers.supervised import SupervisedConfig

    assert SupervisedConfig().hide_grad_clip == 0.0
    assert ObjectiveConfig().hide_loss == "mse"


def _grads_after_step(tmp_path, clip: float, cap_gen: bool, monkeypatch):
    from next_chameleons.tica.trainers import mathrl
    from test_tica_kl_anchor import _setup

    kw = {}
    if cap_gen:
        kw = dict(cap_gen_weight=1.0, cap_gen_episodes=2, cap_gen_samples=2, cap_gen_skip=0,
                  cap_gen_hide=True)
        q = [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}]
        monkeypatch.setattr(mathrl, "load_questions", lambda: (q * 10, []))
    tr, pol, _ = _setup(tmp_path, kl_weight=1.0, kl_ref="same", kl_topk=5, **kw)
    if cap_gen:
        def fake_generate(convs, max_new, **k):
            return [{"text": "x = 1 + 1 = 2. The answer is 2." if "1+1" in c[-1]["content"]
                     else "y = 2 + 2 = 4. The answer is 4.", "finished": True, "ids": [1],
                     "prompt_ids": [1], "stop": None} for c in convs]

        monkeypatch.setattr(pol, "generate", fake_generate)
    tr.cfg.grad_clip = 1e9  # total-norm clip inactive: compare raw summed gradients
    tr.cfg.hide_grad_clip = clip
    got = {}

    def snap(*a, **k):
        got["g"] = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in tr.params]

    tr.opt.step = snap
    rec = tr.step()
    return got["g"], rec, tr


@pytest.mark.parametrize("cap_gen", [False, True])
def test_hide_clip_caps_hide_component_only(tmp_path, monkeypatch, cap_gen):
    g0, r0, _ = _grads_after_step(tmp_path / "off", 0.0, cap_gen, monkeypatch)
    assert "grad_norm_hide" not in r0 and "grad_norm_rest" not in r0
    if cap_gen:
        assert r0.get("cap_gen/n_train", 0) > 0 and r0["cap_gen/hide_p2"] > 0
    # clip far above the hide norm: the same gradient as the single-backward path
    g1, r1, t1 = _grads_after_step(tmp_path / "big", 1e9, cap_gen, monkeypatch)
    for a, b in zip(g0, g1, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    assert math.isclose(r0["train/loss"], r1["train/loss"], rel_tol=1e-6)
    h = t1._hide_buf_last
    nh = float(torch.sqrt(sum((b.double() ** 2).sum() for b in h)))
    assert nh > 0 and math.isclose(r1["grad_norm_hide"], nh, rel_tol=1e-5)
    nr = float(torch.sqrt(sum(((a - b).double() ** 2).sum() for a, b in zip(g1, h, strict=True))))
    assert math.isclose(r1["grad_norm_rest"], nr, rel_tol=1e-4)
    # small clip: hide component rescaled to norm c, rest unchanged
    c = 0.25 * nh
    g2, r2, _ = _grads_after_step(tmp_path / "small", c, cap_gen, monkeypatch)
    assert math.isclose(r2["grad_norm_hide"], nh, rel_tol=1e-5)
    assert math.isclose(r2["hide_clip_coef"], c / (nh + 1e-6), rel_tol=1e-6)
    for a1, a2, b in zip(g1, g2, h, strict=True):
        assert torch.allclose(a2, a1 - b + b * (c / nh), atol=1e-6, rtol=1e-4)
    d = float(torch.sqrt(sum(((a2 - (a1 - b)).double() ** 2).sum()
                             for a1, a2, b in zip(g1, g2, h, strict=True))))
    assert math.isclose(d, c, rel_tol=1e-4)  # the hide part of the final grad has norm c
