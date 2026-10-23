"""Turn-0 BASE PANEL (REPLICATION 2026-10-20d): adaptive.first_probe_panel_from_step draws ONE fixed
base-model probe per episode from the panel to judge turn 0."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "tests")


def test_panel_kinds_default_is_in_training_zoo():
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer as T

    ac = AdaptiveConfig(episode_refit_kinds=["linear", "maxpool", "ens4_linear", "linear@3"],
                        kind_layers=[1, 2])
    got = T._panel_kinds(SimpleNamespace(cfg=SimpleNamespace(adaptive=ac)))
    assert got == ["linear@1", "linear@2", "maxpool@1", "maxpool@2", "ens4_linear", "linear@3"]
    ac = AdaptiveConfig(first_probe_panel_kinds=["linear@2"])
    assert T._panel_kinds(SimpleNamespace(cfg=SimpleNamespace(adaptive=ac))) == ["linear@2"]
    assert AdaptiveConfig().first_probe_panel_from_step == 0  # off by default


def _trainer(tmp_path, **ac):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import ConceptTexts
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    with torch.no_grad():  # nonzero adapters: base and policy states differ
        g = torch.Generator().manual_seed(3)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        objective=ObjectiveConfig(pin_first_turn=False, pin_segment_first=False,
                                  hide_from_turn=0),
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, first_probe="switch_fork",
                                episode_refit_kind="linear@2",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}, **ac))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, None, cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    return tr, pool.entries[0].monitor.name


def _spec(base_name, tag="1"):
    from next_chameleons.tica.episodes import EpisodeSpec

    return EpisodeSpec(f"real_score|cats|{tag}", "real_score", ["aa", "bb", "cc"], "cats", "cats",
                       base_name)


def test_base_panel_judges_turn0_with_a_fixed_base_probe(tmp_path, monkeypatch):
    tr, base_name = _trainer(tmp_path, first_probe_panel_from_step=3, merged_reads=True,
                             first_probe_panel_kinds=["linear@1", "linear@2"])
    trajs, _ = tr.rollout([_spec(base_name)])  # global step 0 < 3: the layer-12-style base probe
    assert trajs[0].results[0]["deployed"][0] == base_name
    depth = []
    orig = tr._episode_refit

    def spy(*a, **k):
        depth.append((getattr(tr.policy, "_ref_depth", 0), k.get("name", "")))
        return orig(*a, **k)

    monkeypatch.setattr(tr, "_episode_refit", spy)
    tr.step_k = 3
    trajs, _ = tr.rollout([_spec(base_name, "2")])
    first = trajs[0].results[0]["deployed"][0]
    assert first.startswith("panel0_")
    panel0 = [d for d in depth if d[1].startswith("panel0_")]
    assert panel0 and all(r > 0 for r, _ in panel0)  # fit on the BASE model (adapters off)
    fam = {e.monitor.name: e.family for e in tr.pool.entries}
    assert fam[first] == "turn0_panel"
    assert tr.pool.concept_monitor("cats").name == base_name  # never a selectivity readout
    assert first not in {e.monitor.name for e in tr.pool.trainable("cats")}
    # kept for the run: later draws of the same kind reuse the probe, no new fit
    for k in range(6):
        tr.rollout([_spec(base_name, f"r{k}")])
    names = {e.monitor.name for e in tr.pool.entries if e.family == "turn0_panel"}
    fits = [n for _, n in depth if n.startswith("panel0_")]
    assert len(names) <= 2 and len(fits) == len(set(fits)) == len(names)  # one fit per kind
    # the policy's weights are untouched by the base-model fits inside merged()
    assert not getattr(tr.policy, "_merged_active", False)


def test_fork_from_step_overrides_the_panel_later(tmp_path):
    tr, _ = _trainer(tmp_path, first_probe_panel_from_step=2, first_probe_fork_from_step=5)
    for g, want in ((0, "switch_fork"), (2, "base_panel"), (4, "base_panel"), (5, "fork")):
        tr.step_k = g
        assert tr._first_probe_mode() == want


def test_panel_from_start_covers_step_zero(tmp_path):
    from next_chameleons.tica.adaptive import AdaptiveConfig

    tr, _ = _trainer(tmp_path, first_probe_panel_from_start=True)
    for g in (0, 1, 7):
        tr.step_k = g
        assert tr._first_probe_mode() == "base_panel"
    assert AdaptiveConfig().first_probe_panel_from_start is False


def _tensors(obj, pre="", seen=None) -> dict:
    """Every tensor reachable from a probe object (attributes, lists, dicts, nn.Modules)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return {}
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return {pre: obj}
    if isinstance(obj, torch.nn.Module):
        return {f"{pre}.{k}": v for k, v in obj.state_dict().items()}
    items = (obj.items() if isinstance(obj, dict) else enumerate(obj)
             if isinstance(obj, (list, tuple)) else vars(obj).items()
             if hasattr(obj, "__dict__") else ())
    out = {}
    for k, v in items:
        out.update(_tensors(v, f"{pre}.{k}", seen))
    return out


def test_panel_cache_equals_fresh_fits_and_hits_on_rerun(tmp_path):
    """adaptive.first_probe_panel_cache (TamIA 10-21): with the cache the turn-0 panel probes and
    their verdicts equal the uncached run's; a second trainer loads every probe (hits, no fits)."""
    from next_chameleons.tica.adaptive import AdaptiveConfig

    kw = dict(first_probe_panel_from_start=True, first_probe_panel_kinds=["linear@1", "linear@2"])
    cache = tmp_path / "panel_cache"

    def run(sub, **ac):
        tr, base_name = _trainer(tmp_path / sub, **kw, **ac)
        trajs, _ = tr.rollout([_spec(base_name, f"e{k}") for k in range(4)])
        probes = {e.monitor.name: e.monitor for e in tr.pool.entries
                  if e.family == "turn0_panel"}
        dets = [[r["monitored"]["detected"] for r in t.results] for t in trajs]
        return tr, probes, dets

    _, fresh, d0 = run("fresh")
    tr1, cached, d1 = run("c1", first_probe_panel_cache=str(cache))
    assert fresh and set(fresh) == set(cached) and d0 == d1
    for n, m in fresh.items():  # identical probes (same stream, name, fit): every tensor equal
        a, b = _tensors(m), _tensors(cached[n])
        assert a.keys() == b.keys() and a, n
        for k in a:
            assert torch.allclose(a[k].float().cpu(), b[k].float().cpu(), atol=1e-6), (n, k)
    files = sorted(cache.glob("*.pt"))
    assert len(files) == len(cached)
    misses = [r for r in tr1._refit_log if r.get("panel_cache") == "miss"]
    assert len(misses) == len(cached)
    tr2, again, d2 = run("c2", first_probe_panel_cache=str(cache))
    assert set(again) == set(fresh) and d2 == d0
    assert all(r.get("panel_cache") == "hit" for r in tr2._refit_log if "panel_cache" in r)
    assert not [r for r in tr2._refit_log if r.get("panel_cache") == "miss"]
    assert AdaptiveConfig().first_probe_panel_cache == ""


def test_panel_precompute_fills_every_key_the_rollout_draws(tmp_path):
    """precompute_panel (TamIA 10-21): concept x condition x kind through _panel_first's path;
    a rollout afterwards loads every turn-0 panel probe from the cache (no miss)."""
    kw = dict(first_probe_panel_from_start=True, first_probe_panel_kinds=["linear@1", "linear@2"],
              first_probe_panel_cache=str(tmp_path / "pc"))
    pre, _ = _trainer(tmp_path / "pre", **kw)
    pre.sampler = SimpleNamespace(conds=("real_score", "nobrief"), weights=(1.0, 0.0),
                                  hidden=False)
    got = [pre.precompute_panel(i, 2) for i in range(2)]  # two shards cover every key once
    assert sum(g["keys"] for g in got) == 2 * 2  # 2 concepts x 1 live condition x 2 kinds
    assert sum(g["miss"] for g in got) == 4 and sum(g["hit"] for g in got) == 0
    tr, base_name = _trainer(tmp_path / "run", **kw)
    tr.rollout([_spec(base_name, f"e{k}") for k in range(6)])
    recs = [r for r in tr._refit_log if "panel_cache" in r]
    assert recs and all(r["panel_cache"] == "hit" for r in recs)


def test_match_anchor_panel_anchors_every_panel_probe(tmp_path):
    """adaptive.match_anchor "panel" (10-22): no-brief / neg_text / monitor-off MATCH targets are
    every turn-0 panel probe of the concept (base model, all kinds) instead of the single primary-
    layer probe; "monitor" (default) keeps the old single target; a training step runs."""
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import EpisodeSpec

    assert AdaptiveConfig().match_anchor == "monitor"
    kinds = ["linear@1", "linear@2"]

    def specs(base_name):
        return [EpisodeSpec("nobrief|cats|a", "nobrief", ["aa", "bb", "cc"], "cats", "cats",
                            base_name),
                _spec(base_name, "b")]

    def match_targets(tr, trajs):
        return [tg for tg in tr._targets(trajs) if tg.role == "match"]

    old, base_name = _trainer(tmp_path / "old", first_probe_panel_from_start=True,
                              first_probe_panel_kinds=kinds, off_turn_prob=1.0)
    trajs, _ = old.rollout(specs(base_name))
    m = match_targets(old, trajs)
    assert m and {tg.monitor for tg in m} == {base_name}

    tr, base_name = _trainer(tmp_path / "new", first_probe_panel_from_start=True,
                             first_probe_panel_kinds=kinds, off_turn_prob=1.0,
                             match_anchor="panel")
    trajs, _ = tr.rollout(specs(base_name))
    off_turns = [(k, t) for k, tj in enumerate(trajs) for t, r in enumerate(tj.results)
                 if r.get("off")]
    assert off_turns  # monitor-off turns in the real episode (turns >= 1)
    m = match_targets(tr, trajs)
    fam = {e.monitor.name: e.family for e in tr.pool.entries}
    assert m and all(fam[tg.monitor] == "turn0_panel" for tg in m)
    per_turn = {}
    for tg in m:
        per_turn.setdefault((tg.traj, tg.turn), set()).add(tg.monitor)
    assert all(len(v) == len(kinds) for v in per_turn.values())  # every panel kind, every turn
    nob = {n for (k, _), v in per_turn.items() if k == 0 for n in v}
    off = {n for (k, _), v in per_turn.items() if k == 1 for n in v}
    assert nob and off and nob.isdisjoint(off)  # each condition's own base probes
    assert set(tr._anchor_names(trajs[0].spec, "cats")) == nob  # cached, same names
    assert tr.pool.concept_monitor("cats").name == base_name  # selectivity readout unchanged
    from next_chameleons.tica.episodes import ConceptTexts

    tr.sampler = SimpleNamespace(  # a full training step with the panel anchor
        concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)])],
        sample=lambda n: specs(base_name))
    met = tr.step()
    assert met["n_targets"] > len(m) // 2 and met["train/loss"] == met["train/loss"]


def test_match_anchor_k_draws_a_subset_per_turn(tmp_path):
    """adaptive.match_anchor_k (10-22): k anchor probes per act-normal turn, an unbiased subset of
    the panel (different turns draw different subsets); 0 = all."""
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import EpisodeSpec

    assert AdaptiveConfig().match_anchor_k == 0
    kinds = ["linear@1", "linear@2", "maxpool@1"]
    tr, base_name = _trainer(tmp_path, first_probe_panel_from_start=True,
                             first_probe_panel_kinds=kinds, match_anchor="panel",
                             match_anchor_k=1)
    spec = EpisodeSpec("nobrief|cats|a", "nobrief", ["aa", "bb", "cc"], "cats", "cats", base_name)
    trajs, _ = tr.rollout([spec])
    m = [tg for tg in tr._targets(trajs) if tg.role == "match"]
    per_turn = {}
    for tg in m:
        per_turn.setdefault(tg.turn, []).append(tg.monitor)
    assert per_turn and all(len(v) == 1 for v in per_turn.values())
    full = set(tr._anchor_names(spec, "cats"))
    assert {n for v in per_turn.values() for n in v} <= full and len(full) == len(kinds)
    again = [tg.monitor for tg in tr._targets(trajs) if tg.role == "match"]
    assert again == [tg.monitor for tg in m]  # deterministic for the same step / episode / turn


def test_pass_forks_rescore_every_pass(tmp_path, monkeypatch):
    """--pass-forks (eval, 10-22): with free answers, every pass is re-sampled n times from the same
    prefix and scored by the same probe; results carry the passing share."""
    from next_chameleons.tica.adaptive import AdaptiveDefender

    tr, base_name = _trainer(tmp_path, first_probe_panel_from_start=True,
                             first_probe_panel_kinds=["linear@1"])
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score", lambda self, name, pooled, row: {
        **orig(self, name, pooled, row), "detected": False})
    tr.free_answers = {"max_new": 3}
    tr.pass_forks = (3, 0.7)
    trajs, _ = tr.rollout([_spec(base_name, "pf")])
    shares = [r["pass_forks"] for r in trajs[0].results if "pass_forks" in r]
    assert shares and all(x == 1.0 for x in shares)  # every sample passes (score patched)
    tr.free_answers = None  # training (teacher-forced): never forked
    trajs, _ = tr.rollout([_spec(base_name, "pf2")])
    assert not [r for r in trajs[0].results if "pass_forks" in r]
