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
