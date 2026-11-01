"""crypsis.probes vs the snapshot (next_chameleons.tica.*) on fixed synthetic activations."""

from __future__ import annotations

import dataclasses
import importlib
import random

import pytest
import torch

from crypsis.probes import executors as ex
from crypsis.probes.base import MonitorPool, trim_bank
from crypsis.probes.gates import GateConfig, gate_fit_many
from crypsis.probes.kinds import evdet as nev
from crypsis.probes.kinds import linear as nlin
from crypsis.probes.kinds import zoo as nzoo
from next_chameleons.tica import adaptive as oad
from next_chameleons.tica import evdet_defender as oev
from next_chameleons.tica import monitors as omon
from next_chameleons.tica import probe_zoo as ozoo
from next_chameleons.tica.trainers import supervised as osup

nf = importlib.import_module("crypsis.probes.fit")  # the package's `fit` attr is the function

D, LAYER = 8, 20
ACFG = oad.AdaptiveConfig(refit_cv_folds=2, refit_min_auroc=0.9, refit_min_tpr_np=0.5,
                          refit_min_tpr_ic=0.3, refit_C=0.01)
GATE = GateConfig.from_refit(ACFG)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


DIR = torch.randn(D, generator=_gen(0))


def rows(n, shift, seed):
    return torch.randn(n, D, generator=_gen(seed)) + shift * DIR


def toks(n, shift, seed):
    g = _gen(seed)
    return [torch.randn(int(torch.randint(3, 7, (1,), generator=g)), D, generator=g) + shift * DIR
            for _ in range(n)]


def inputs(kind, seed=1):
    """(P, N, NP) shaped for `kind` at the 32-layer default layer sets."""

    if kind == "meandiff":
        return rows(16, 1.5, seed), rows(16, 0.0, seed + 1), rows(8, 1.5, seed + 2)
    if kind in nzoo.ENS_KINDS:
        Ls = ozoo.kind_layers(kind, LAYER)
        return tuple({L: rows(n, s, seed + 10 * i + L) for L in Ls}
                     for i, (n, s) in enumerate(((16, 1.5), (16, 0.0), (8, 1.5))))
    if kind in nzoo.TOKENS_ENS:
        Ls = ozoo.kind_layers(kind, LAYER)
        return tuple({L: toks(n, s, seed + 10 * i + L) for L in Ls}
                     for i, (n, s) in enumerate(((12, 1.5), (12, 0.0), (6, 1.5))))
    return toks(12, 1.5, seed), toks(12, 0.0, seed + 1), toks(6, 1.5, seed + 2)


def same_mon(a, b, x):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert type(a).__name__ == type(b).__name__
    assert torch.equal(a.thresholds(["c"]).float(), b.thresholds(["c"]).float())
    assert torch.equal(a.logits(x, ["c"]).detach().cpu(), b.logits(x, ["c"]).detach().cpu())
    if hasattr(a, "scale_"):
        assert a.scale_ == b.scale_


# ens8_mlp is ens4_mlp's code path on more layers (and the slowest fit on CPU): skipped
ALL_ZOO = [k for k in dict.fromkeys((*nzoo.ZOO_KINDS, *nzoo.MCK_KINDS)) if k != "ens8_mlp"]


@pytest.fixture(autouse=True, scope="module")
def _depth32():
    ozoo.set_zoo_depth(32)
    torch.set_num_threads(1)


@pytest.mark.parametrize("kind", ALL_ZOO)
def test_zoo_gate_fit(kind):
    P, N, NP = inputs(kind)
    mo, io = ozoo.gate_fit(kind, P, N, NP, ACFG, name=f"t_{kind}", layer=LAYER, concept="c")
    mn, inn = nf.fit_with_info(kind, P, N, NP, GATE, name=f"t_{kind}", layer=LAYER,
                               concept="c", C=ACFG.refit_C)
    assert io == inn
    same_mon(mo, mn, NP)


def _rd(mod, layers, cap=0):
    rd = mod.RefitData(layers=layers, concept="c")
    for i in range(30):
        pooled = {L: torch.cat([rows(1, 1.5, 100 + i + L), rows(1, 0.0, 200 + i + L),
                                rows(1, 1.5, 300 + i + L)]) for L in layers}
        for k, r in (("pos", 0), ("neg", 1), ("np", 2)):
            rd.add(k, pooled, r, cap=cap)
    return rd


def test_linear_mlp_ood_evdet():
    # linear (two layers: the bank keeps the best one) and mlp, via RefitData
    ro, rn = _rd(oad, [12, LAYER], cap=24), _rd(nf, [12, LAYER], cap=24)
    assert ro.counts() == rn.counts() == (24, 24, 24) and ro.ready() == rn.ready()
    for L in ro.layers:
        for a, b in ((ro.pos, rn.pos), (ro.neg, rn.neg), (ro.np_pos, rn.np_pos)):
            assert all(torch.equal(x, y) for x, y in zip(a[L], b[L], strict=True))
    x = rows(10, 0.7, 9)
    bo, io = oad.fit_refit_bank(ro, "c", "lin", ACFG, exclude_layers={12})
    bn, inn = nf.fit_with_info("linear", rn.pos, rn.neg, rn.np_pos, GATE, name="lin",
                               concept="c", C=ACFG.refit_C, exclude_layers={12})
    assert io == inn
    same_mon(bo, bn, x)
    mo, io = oad.fit_refit_mlp(ro, "c", "mlp", ACFG, LAYER)
    mn, inn = nf.fit_with_info("mlp", {LAYER: rn.pos[LAYER]}, {LAYER: rn.neg[LAYER]},
                               {LAYER: rn.np_pos[LAYER]}, GATE, name="mlp", concept="c")
    assert io == inn
    same_mon(mo, mn, x)
    ref = rows(60, 0.0, 7)
    oo, io = oad.fit_ood(ref, LAYER, name="ood", k=4)
    on, inn = nf.fit_with_info("ood", ref, layer=LAYER, name="ood", k=4)
    assert io == inn
    same_mon(oo, on, x)
    for kind in nev.KINDS:  # two refits each: the second sees the buffer of the first
        bo_, bn_ = oev.EvDetBuffer(), nev.EvDetBuffer()
        for r in range(2):
            pol, base, cal, mis = (rows(16, s, 40 + 4 * r + j) for j, s in
                                   enumerate((1.5, 0.0, 0.0, 0.2)))
            kw = dict(name=f"{kind}{r}", kind=kind, mis_rows=mis)
            o = oev.fit_evdet(bo_, LAYER, "c", pol, base, cal, ACFG, **kw)
            n = nev.fit_evdet(bn_, LAYER, "c", pol, base, cal, GATE, C=ACFG.refit_C, **kw)
            assert o[1] == n[1]
            same_mon(o[0], n[0], x)


def test_batched_logistic_matches_serial():
    xs = [torch.cat([rows(20, 1.0, s), rows(25 + s, 0.0, s + 50)]) for s in range(3)]
    ys = [torch.cat([torch.ones(20), torch.zeros(len(x) - 20)]) for x in xs]
    for (w, b, mu, sd), x, y in zip(nf.fit_logistic_many(xs, ys, 0.05), xs, ys, strict=True):
        w1, b1, mu1, sd1 = nf.fit_logistic(x, y, 0.05)
        wo, bo, _, _ = omon.fit_logistic(x, y, 0.05)
        assert torch.equal(w1, wo) and b1 == bo
        assert torch.allclose(w, w1, atol=1e-4) and abs(b - b1) < 1e-4
        assert torch.equal(mu, mu1) and torch.equal(sd, sd1)


# ------------------------------------------------------------------------------------- executors
def _job(kind, seed, **acfg):
    """A reads job as trainers.supervised._refit_reads builds it (synthetic activations)."""

    cfg = dataclasses.replace(ACFG, **acfg)
    rng = random.Random(seed)
    picks = [(rng.choice(("pos", "neg")), None) for _ in range(20)]
    picks[:2] = [("pos", None), ("neg", None)]
    lab = [1.5 if k == "pos" else 0.0 for k, _ in picks]
    job = {"kind": kind, "L": LAYER, "Ls": [LAYER], "L0": LAYER, "picks": picks, "acfg": cfg,
           "device": torch.device("cpu"), "full_kind": kind, "sec_read": 0.0, "concept": "c",
           "name": f"job_{kind}_{seed}", "excl": set(), "ft": None, "nt": None,
           "r": [rows(1, s, seed * 100 + q)[0] for q, s in enumerate(lab)],
           "npr": [rows(1, 1.5, seed * 100 + 50 + q)[0] for q in range(8)]}
    job["rl"] = {LAYER: torch.stack(job["r"])}
    job["npl"] = {LAYER: torch.stack(job["npr"])}
    if kind in nzoo.RLOBF_KINDS:
        job["ft"] = {LAYER: [toks(1, s, seed * 100 + q)[0] for q, s in enumerate(lab)]}
        job["nt"] = {LAYER: toks(8, 1.5, seed * 100 + 70)}
    return job


def _res(fut):
    (mon, info), rec, _ = fut.result()
    return mon, info, rec


def _same_res(a, b, x, rng=True):
    assert a[1] == b[1]
    same_mon(a[0], b[0], x)
    if rng:
        assert (a[2] is None) == (b[2] is None)
    if rng and a[2] is not None:
        assert torch.equal(a[2][0], b[2][0])


def test_fit_job_and_executors():
    x = rows(6, 0.7, 3)
    jobs = [_job("linear", 1), _job("meandiff", 2)]
    for j in jobs:  # the fit stage vs the snapshot's
        o, n = osup._refit_fit(j), nf.fit_job(j)
        assert o[1] == n[1]
        same_mon(o[0], n[0], x)
    ref = [_res(ex.SerialExecutor().submit(j, False)) for j in jobs]
    for j, r in zip(jobs, ref, strict=True):
        _same_res(r, (*osup._iso_fit(j, False)[0], osup._iso_fit(j, False)[1]), x)
    for exe in (ex.ThreadExecutor(), ex.CPUProcessExecutor(1, 1)):
        try:
            outs = [_res(exe.submit(j, False)) for j in jobs]
        finally:
            exe.shutdown()
        for r, o in zip(ref, outs, strict=True):
            _same_res(r, o, x)
    # stacking: two rlobf jobs as one stacked run == each fit serially
    rj = [_job("rlobf_max", s, refit_batched_rlobf=True) for s in (3, 4)]
    xt = toks(4, 0.7, 5)
    serial = [_res(ex.SerialExecutor().submit(j, False)) for j in rj]
    st = ex.ThreadExecutor(stack=True)
    futs = [st.submit(j, False) for j in rj]
    st.flush()
    for q, (r, f) in enumerate(zip(serial, futs, strict=True)):
        # a stack ends in the LAST fit's RNG state (all fits seed from their names)
        _same_res(r, _res(f), xt, rng=q == len(rj) - 1)
    assert st.n_stacked == 2
    old_items = [{"P": p, "N": n_, "NP": g, "name": j["name"], "layer": LAYER, "concept": "c"}
                 for j in rj for p, n_, g in [osup._tok_fit_inputs(j)]]
    for (mo, io), (mn, inn) in zip(ozoo.gate_fit_many("rlobf_max", old_items, rj[0]["acfg"]),
                                   gate_fit_many("rlobf_max", old_items, GATE, C=ACFG.refit_C,
                                                 perf=nf_perf(rj[0]["acfg"])), strict=True):
        assert io == inn
        same_mon(mo, mn, xt)


def nf_perf(acfg):
    from crypsis.probes.gates import FitPerf

    return FitPerf.from_refit(acfg)


# ------------------------------------------------------------------------------- pool + trim_bank
def _banks(mod_bank, mod_meta):
    mons = []
    for i in range(4):
        w = rows(1, 0.0, 500 + i)
        mons.append(mod_bank(f"b{i}", LAYER, ["c"], W=w, b=torch.tensor([0.1 * i]),
                             mu=torch.zeros(1, D), sd=torch.ones(1, D), thr=torch.tensor([0.5]),
                             zero=torch.zeros(1), hundred=torch.ones(1),
                             meta={"online_refit": i > 0}))
    meta = mod_meta("meta", 12, rows(1, 0.0, 9)[0], 0.2, torch.zeros(D), torch.ones(D), 0.3, 0.0,
                    1.0)
    return mons, meta


def _pool(mod_pool, mons, meta):
    p = mod_pool()
    for i, m in enumerate(mons):
        p.add(m, round=3 - i, family="concept_linear")
    p.add(meta, round=0, family="meta_linear", held_out=True)
    return p


def test_pool_roundtrip_and_trim(tmp_path):
    x = rows(5, 0.5, 11)
    po = _pool(omon.MonitorPool, *_banks(omon.ConceptProbeBank, omon.MetaDetector))
    pn = _pool(MonitorPool, *_banks(nlin.ConceptProbeBank, nlin.MetaDetector))
    po.save(tmp_path / "old")
    pn.save(tmp_path / "new")
    man = [(tmp_path / f"{d}/manifest.json").read_text() for d in ("old", "new")]
    assert man[0] == man[1]
    for a, b in ((po, MonitorPool.load(tmp_path / "old")),
                 (omon.MonitorPool.load(tmp_path / "new"), pn)):
        assert [(e.monitor.name, e.round, e.family, e.held_out) for e in a.entries] == \
               [(e.monitor.name, e.round, e.family, e.held_out) for e in b.entries]
        for ea, eb in zip(a.entries, b.entries, strict=True):
            same_mon(ea.monitor, eb.monitor, x)
            assert ea.monitor.meta == eb.monitor.meta
    names = lambda p: [e.monitor.name for e in p.trainable("c")]  # noqa: E731
    assert names(po) == names(pn)
    assert oad.trim_bank(po, {"b0"}, 2) == trim_bank(pn, {"b0"}, 2) == ["b3", "b2", "b1"]
    assert [e.monitor.name for e in po.entries] == [e.monitor.name for e in pn.entries]
