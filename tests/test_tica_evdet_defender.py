"""Generic hiding-detector defender (tica/evdet_defender.py): a planted adapter shift is
detected and gated in; identical policy / base rows (the base model as policy) fail the gate; the
buffer makes the detector generic (covers every concept, per-concept thresholds, fallback);
seeding pairs dump rows; the unsupervised PCA-residual kind flags an off-manifold shift without
ever fitting on policy rows."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from next_chameleons.tica.evdet_defender import EvDetBuffer, EvDetMonitor, fit_evdet

CFG = SimpleNamespace(refit_min_auroc=0.95, refit_min_tpr_ic=0.8, refit_C=0.1)


def _rows(seed, n=24, d=16, shift=0.0, n_cal=64):
    """(policy rows, paired base rows, fresh base-only calibration rows) of one concept."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(d, generator=g)
    base = torch.randn(n, d, generator=g) * 2 + mu
    cal = torch.randn(n_cal, d, generator=g) * 2 + mu
    v = torch.zeros(d)
    v[0] = shift
    return base + v, base, cal


def test_shift_detected_and_generic():
    buf = EvDetBuffer()
    pol, base, cal = _rows(0, shift=10.0)
    mon, info = fit_evdet(buf, 12, "a", pol, base, cal, CFG, name="evdet_1")
    assert info["kept"] and info["auroc"] > 0.95 and mon is not None, info
    assert mon.covers("a") and mon.covers("never-seen")
    pol2, base2, cal2 = _rows(1, shift=10.0)
    mon2, info2 = fit_evdet(buf, 12, "b", pol2, base2, cal2, CFG, name="evdet_2")
    assert info2["kept"] and info2["n_buffer_concepts"] == 2
    assert set(mon2.thr_by) == {"a", "b"} and mon2.kind == "evdet"
    thr = mon2.thresholds(["a", "never-seen"])
    assert thr[1] == mon2.thr  # fallback for an uncalibrated concept
    s = mon2.logits(pol2)
    assert float((s > mon2.thresholds(["b"])[0]).float().mean()) > 0.8
    _, fresh, _ = _rows(1, shift=10.0, n=200)  # out-of-sample base states of b: ~1% flagged
    assert float((mon2.logits(fresh) > mon2.thresholds(["b"])[0]).float().mean()) < 0.05
    assert isinstance(mon2.to("cpu"), EvDetMonitor)


def test_identical_rows_fail_gate():
    buf = EvDetBuffer()
    pol, base, cal = _rows(2, shift=0.0)
    mon, info = fit_evdet(buf, 12, "a", pol, base, cal, CFG, name="evdet_3")
    assert mon is None and not info["kept"]
    assert len(buf.P[12]) == 1  # the attempt still enters the buffer


def test_seed_pairs(tmp_path):
    layers = [8, 12]
    meta = [{"ep": 0, "turn": t, "cond": c, "off": False, "concept": "a", "text_concept": "a"}
            for t in range(4) for c in ("real_score",)]
    meta += [{"ep": 1, "turn": 2, "cond": "nobrief", "off": False, "concept": "a",
              "text_concept": "a"}]
    x = torch.randn(len(meta), 2, 8)
    for stem, xx in (("m_train_s1", x + 1), ("b_train_s1", x)):
        torch.save({"layers": layers, "x": xx.half(), "meta": meta}, tmp_path / f"{stem}.pt")
    buf = EvDetBuffer()
    n = buf.seed([str(tmp_path / "m_train_s1.pt")], [str(tmp_path / "b_train_s1.pt")], 12)
    assert n == 3  # turns 1..3 of the hiding episode; turn 0 and no-brief excluded
    (c, p), (_, b) = buf.P[12][0], buf.N[12][0]
    assert c == "a" and torch.allclose(p - b, torch.ones_like(p), atol=1e-2)


def test_pcares_off_manifold():
    g = torch.Generator().manual_seed(3)
    d, r = 96, 6
    basis = torch.linalg.qr(torch.randn(d, r, generator=g))[0]  # base states live near rank r

    def rows(n, off=0.0):
        b = torch.randn(n, r, generator=g) @ basis.T * 3 + 0.05 * torch.randn(n, d, generator=g)
        v = torch.zeros(d)
        v[-1] = off  # a direction (almost) outside the base subspace
        return b + v, b

    buf = EvDetBuffer()
    pol, base = rows(24, off=2.0)
    cal = rows(64)[1]
    mon, info = fit_evdet(buf, 16, "a", pol, base, cal, CFG, name="pc_1", kind="pcares")
    assert info["kept"] and mon.kind == "pcares", info
    assert set(mon.prm) == {"mu", "P"}  # label-free: no weights fit on policy rows
    fresh = rows(200)[1]  # out-of-sample base states: ~1% flagged at the calibrated threshold
    assert float((mon.logits(fresh) > mon.thresholds(["a"])[0]).float().mean()) < 0.05
    assert float((mon.logits(pol) > mon.thresholds(["a"])[0]).float().mean()) > 0.8
    pol0, base0 = rows(24, off=0.0)
    mon0, info0 = fit_evdet(EvDetBuffer(), 16, "a", pol0, base0, rows(64)[1], CFG, name="pc_2",
                            kind="pcares")
    assert mon0 is None and not info0["kept"]


def test_selective_ignores_adapter_offset():
    """Every trained-model row carries an adapter offset (dim 1); hiding rows add a hiding shift
    (dim 0). The base-negative evdet flags honest trained rows; evdetsel must not."""
    g = torch.Generator().manual_seed(5)
    d = 16

    def rows(n, offset=0.0, hide=0.0):
        x = torch.randn(n, d, generator=g)
        x[:, 1] += offset
        x[:, 0] += hide
        return x

    for kind, flags_honest in (("evdet", True), ("evdetsel", False)):
        buf = EvDetBuffer()
        mon, info = fit_evdet(buf, 12, "a", rows(48, 8.0, 8.0), rows(48), rows(96), CFG,
                              name=f"s_{kind}", kind=kind, mis_rows=rows(48, 8.0))
        assert info["kept"], info
        honest = float((mon.logits(rows(400, 8.0)) > mon.thresholds(["a"])[0]).float().mean())
        hiding = float((mon.logits(rows(400, 8.0, 8.0)) > mon.thresholds(["a"])[0]).float().mean())
        assert hiding > 0.9
        assert (honest > 0.5) if flags_honest else (honest < 0.1), (kind, honest)  # q99 of 24 rows


def test_mlpdetsel_ignores_adapter_offset():
    """11-08 MLP detector type: same selective behaviour as evdetsel; differentiable scores."""
    g = torch.Generator().manual_seed(7)
    d = 16

    def rows(n, offset=0.0, hide=0.0):
        x = torch.randn(n, d, generator=g)
        x[:, 1] += offset
        x[:, 0] += hide
        return x

    mon, info = fit_evdet(EvDetBuffer(), 12, "a", rows(48, 8.0, 8.0), rows(48), rows(96), CFG,
                          name="mlp", kind="mlpdetsel", mis_rows=rows(48, 8.0))
    assert info["kept"] and set(mon.prm) >= {"W1", "w2"}, info
    thr = mon.thresholds(["a"])[0]
    assert float((mon.logits(rows(400, 8.0)) > thr).float().mean()) < 0.1
    assert float((mon.logits(rows(400, 8.0, 8.0)) > thr).float().mean()) > 0.9
    x = rows(4, 8.0, 8.0).requires_grad_()
    mon.logits(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_pcaressel_fixed_reference_includes_policy_honest_rows():
    """11-08 fix: the adapter offset lies outside the BASE subspace, so base-only PCA flags honest
    policy rows and the selective threshold swallowed hiding (10-26: kept 2 / 287). With the
    policy's mismatch rows in the reference, only the hiding direction is residual."""
    g = torch.Generator().manual_seed(11)
    d, r = 96, 6
    basis = torch.linalg.qr(torch.randn(d, r, generator=g))[0]

    def rows(n, offset=0.0, hide=0.0):
        x = torch.randn(n, r, generator=g) @ basis.T * 3 + 0.05 * torch.randn(n, d, generator=g)
        x[:, -2] += offset * (1 + 0.3 * torch.randn(n, generator=g))
        x[:, -1] += hide
        return x

    mon, info = fit_evdet(EvDetBuffer(), 16, "a", rows(48, 2.0, 2.0), rows(48), rows(96), CFG,
                          name="pcs", kind="pcaressel", mis_rows=rows(96, 2.0))
    assert info["kept"], info
    thr = mon.thresholds(["a"])[0]
    assert float((mon.logits(rows(400, 2.0)) > thr).float().mean()) < 0.1
    assert float((mon.logits(rows(400, 2.0, 2.0)) > thr).float().mean()) > 0.8


def test_buffer_cap_drops_oldest_pairs():
    buf = EvDetBuffer(max_rows=50)
    for i in range(5):
        buf.add(12, f"c{i}", torch.full((20, 4), float(i)), torch.zeros(20, 4))
        buf.cal[12].append((f"c{i}", torch.zeros(20, 4)))
        buf.trim(12)
    assert [c for c, _ in buf.P[12]] == ["c3", "c4"] == [c for c, _ in buf.N[12]]
    assert sum(len(r) for _, r in buf.cal[12]) <= 50
    unb = EvDetBuffer()
    for _ in range(5):
        unb.add(12, "c", torch.zeros(20, 4), torch.zeros(20, 4))
    assert len(unb.P[12]) == 5


def test_detector_kind_prob_draw():
    """adaptive.detector_kind_prob: share of refits drawn from detector_kinds; exclude honoured."""
    import random

    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer

    ac = AdaptiveConfig(episode_refit_kind="mixed", episode_refit_kinds=["linear", "maxpool"],
                        detector_kind_prob=0.5,
                        detector_kinds=["evdetsel@16", "mlpdetsel@22", "pcaressel@12"])
    me = SimpleNamespace(cfg=SimpleNamespace(adaptive=ac), rng=random.Random(0), kind_bandit=None)
    draws = [AdaptiveSupervisedTrainer._pick_kind(me) for _ in range(4000)]
    share = sum(k in ac.detector_kinds for k in draws) / len(draws)
    assert 0.46 < share < 0.54
    assert set(draws) == {"linear", "maxpool", *ac.detector_kinds}
    ex = AdaptiveSupervisedTrainer._pick_kind(me, exclude=tuple(ac.detector_kinds))
    assert ex in ("linear", "maxpool")
    ac.detector_kind_prob = 0.0
    assert all(AdaptiveSupervisedTrainer._pick_kind(me) in ("linear", "maxpool")
               for _ in range(200))
