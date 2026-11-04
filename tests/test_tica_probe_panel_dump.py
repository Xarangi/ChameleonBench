"""Dodgescope dumps of scripts/tica_probe_panel.py (11-02): stratified episode plan and the probe
dump (synthetic, CPU)."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

from next_chameleons.tica.episodes import ConceptTexts, EpisodeSampler
from next_chameleons.tica.monitors import ConceptProbeBank, MonitorPool
from next_chameleons.tica.probe_zoo import TokenPoolMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("tpp", ROOT / "scripts" / "tica_probe_panel.py")
tpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpp)

MIX = {"real_score": 0.6, "mismatch": 0.2, "nobrief": 0.2}


def _bank(concepts, d=8):
    g = torch.Generator().manual_seed(1)
    k = len(concepts)
    return ConceptProbeBank("concept_L1", 1, concepts, W=torch.randn(k, d, generator=g),
                            b=torch.zeros(k), mu=torch.zeros(k, d), sd=torch.ones(k, d),
                            thr=torch.zeros(k), zero=torch.full((k,), -5.0),
                            hundred=torch.full((k,), 5.0))


def test_stratified_plan_counts():
    cs = [f"c{i}" for i in range(5)]
    plan = tpp.stratified_plan(cs, 2, MIX, seed=1, batch=8)
    by = Counter(plan)
    assert all(by[(c, "real_score")] == 2 for c in cs)  # every concept: N dodge episodes
    n = Counter(k for _, k in plan)
    assert len(plan) == 16 and n["mismatch"] == n["nobrief"] == 3
    assert max(by[(c, "mismatch")] for c in cs) == 1  # spread, one per concept at most
    assert plan == tpp.stratified_plan(cs, 2, MIX, seed=1, batch=8)  # seeded
    # fill the batches: shard totals are multiples of the batch, dodge episodes untouched
    for k, epc, batch, total in ((4, 2, 8, 16), (5, 2, 6, 18), (4, 2, 6, 12), (5, 1, 8, 8),
                                 (4, 1, 8, 8), (2, 1, 6, 6)):
        p = tpp.stratified_plan(cs[:k], epc, MIX, seed=3, batch=batch)
        n = Counter(c for _, c in p)
        assert len(p) == total and n["real_score"] == k * epc and n["mismatch"] >= 1
        assert abs(n["mismatch"] - n["nobrief"]) <= 1
    p = tpp.stratified_plan(cs[:2], 3, {"real_score": 1.0}, seed=3, batch=8)  # no controls
    assert len(p) == 8 and Counter(c for c, _ in p) == {"c0": 4, "c1": 4}


def test_plan_specs_pin_concept_and_condition():
    cs = ["cats", "dogs", "owls"]
    pool = MonitorPool()
    pool.add(_bank(cs), round=0, family="concept_linear")
    smp = EpisodeSampler([ConceptTexts(c, "train", [f"{c} {j}" for j in range(6)]) for c in cs],
                         ["plain"], pool, mix=MIX, turns=3, seed=2)
    plan = tpp.stratified_plan(cs, 2, MIX, seed=2)
    specs = tpp.plan_specs(smp, plan)
    assert [(s.text_concept, s.cond) for s in specs] == plan
    for s in specs:
        assert (s.monitored != s.text_concept) == (s.cond == "mismatch")
        assert all(t.startswith(s.text_concept) for t in s.texts)
    assert smp.concept_weights is None and len(smp.sample(4)) == 4  # sampler restored


def test_dump_probe_reevaluates():
    b = _bank(["cats"])
    d = tpp.dump_probe(b, "linear", {"ep": 3})
    x = torch.randn(4, 8)
    assert torch.equal(d["obj"].logits(x, ["cats"] * 4), b.logits(x, ["cats"] * 4))
    assert d["state"]["tensors"]["W"].shape == (1, 8) and d["ep"] == 3 and d["layers"] == [1]
    w = torch.randn(8)
    tm = TokenPoolMonitor("maxpool", "ep_refit_1", 2, "cats", torch.zeros(8), torch.ones(8), w,
                          torch.tensor(0.5), thr=1.0)
    d = pickle.loads(pickle.dumps(tpp.dump_probe(tm, "maxpool@2")))
    toks = [torch.randn(5, 8), torch.randn(9, 8)]
    assert torch.allclose(d["obj"].logits(toks), tm.logits(toks))
    assert d["tensors"]["w"].shape == (8,) and d["needs_tokens"] and d["concept"] == "cats"


def test_shadow_specs():
    assert tpp.shadow_specs("linear@12") == [
        ("a", "linear@12"), ("b", "linear@16"), ("c", "maxpool@12")]
    assert tpp.shadow_specs("rlobf_attn@30") == [("a", "rlobf_attn@30"), ("b", "rlobf_attn@26"),
                                                 ("c", "linear@30")]
    assert [k for _, k in tpp.shadow_specs("ens4_linear")] == [
        "ens4_linear", "ensdeep4_linear", "ens4_mlp"]
    assert tpp.shadow_specs("nc_ens8_tok") == []


def test_fork_draw_is_seeded_and_excludes_nothing_extra():
    pool = [f"text {j}" for j in range(20)]
    a = tpp.fork_draw(11, 3, 5, pool, 8)
    assert a == tpp.fork_draw(11, 3, 5, pool, 8) and len(set(a)) == 8 and set(a) <= set(pool)
    assert a != tpp.fork_draw(11, 3, 6, pool, 8)
    small = tpp.fork_draw(1, 0, 0, pool[:3], 8)
    assert small == tpp.fork_draw(1, 0, 0, pool[:3], 8) and len(small) == 3


def test_activations_default_paths_and_layers(tmp_path):
    """--activations (default on): bool parsing, the mirrored dump dir, the all-layer default."""
    import argparse

    import pytest

    assert tpp._flag("true") is True and tpp._flag("False") is False and tpp._flag("0") is False
    with pytest.raises(argparse.ArgumentTypeError):
        tpp._flag("maybe")
    out = tmp_path / "runs" / "cfg_4b" / "quality" / "step_1800" / "panel_train.s0.json"
    want = Path("/proj/acts/cfg_4b/quality/step_1800/panel_train.s0")
    assert tpp.activations_dir(str(out), "/proj/acts") == want
    assert tpp.activations_dir(str(out), "") == out.parent / "panel_train.s0_acts"
    assert tpp.activations_dir(str(tmp_path / "x" / "p.json"), "/r") == Path("/r/x/p")
    assert tpp.default_feat_layers(32) == "2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31"
    assert tpp.default_feat_layers(24) == "2,4,6,8,10,12,14,16,18,20,22,23"
