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
