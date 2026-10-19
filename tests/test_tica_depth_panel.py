"""Depth panel (objective.panel_weight + adaptive.panel_layers, REPLICATION 2026-10-19n): fixed base
probes at extra layers become hide targets on monitored hide turns, never an episode's monitor."""

import random
from types import SimpleNamespace

import pytest
import torch

from next_chameleons.tica.adaptive import AdaptiveConfig
from next_chameleons.tica.episodes import EpisodeSpec
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.objectives import ObjectiveConfig, loss_terms, summarize_stats
from next_chameleons.tica.trainers.supervised import (
    AdaptiveSupervisedConfig,
    AdaptiveSupervisedTrainer,
)
from test_tica_framework import bank as make_bank
from test_tica_framework import tiny_policy


def _bank(name, layer):
    b = make_bank(layer, ["cats"])
    b.name = name
    return b


def _pool():
    pool = MonitorPool()
    pool.add(_bank("base_L1", 1), round=0, family="concept_linear")
    pool.add(_bank("panel_L0", 0), round=0, family="panel")
    return pool


def test_panel_probes_are_never_the_monitor():
    pool = _pool()
    assert [e.monitor.name for e in pool.trainable("cats")] == ["base_L1"]
    assert pool.concept_monitor("cats").name == "base_L1"
    assert [m.name for m in pool.concept_monitors("cats")] == ["base_L1"]
    rng = random.Random(0)
    assert {pool.sample(rng, "cats").monitor.name for _ in range(20)} == {"base_L1"}
    assert pool.panel("cats") == ["panel_L0"] and pool.panel("dogs") == []
    assert pool.layers == [0, 1]  # the panel layer is read


def _targets(tmp_path, panel_weight, cond="real_score"):
    oc = ObjectiveConfig(hide_from_turn=0, pin_first_turn=False, panel_weight=panel_weight)
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=3, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0, objective=oc, adaptive=AdaptiveConfig(refit_every=0))
    tr = AdaptiveSupervisedTrainer(tiny_policy(), _pool(), None, cfg, tmp_path)
    spec = EpisodeSpec(f"{cond}|cats|0", cond, ["a"] * 3, "cats", "cats", "base_L1")
    res = [{"concept": "cats", "active_monitor": "base_L1", "deployed": ["base_L1"], "turn": t,
            "monitored": {"detected": True, "p": 0.5, "name": "base_L1"}} for t in range(3)]
    return tr._targets([SimpleNamespace(spec=spec, results=res)])


def test_panel_targets_on_hide_turns_only(tmp_path):
    on = _targets(tmp_path / "a", 0.5)
    assert [(t.turn, t.monitor) for t in on if t.role == "panel"] == [
        (0, "panel_L0"), (1, "panel_L0"), (2, "panel_L0")]
    assert not any(t.role == "panel" for t in _targets(tmp_path / "b", 0.0))  # default off
    nob = _targets(tmp_path / "c", 0.5, cond="nobrief")
    assert not any(t.role == "panel" for t in nob)  # no-brief turns: match only


def test_panel_loss_is_its_own_weighted_mean():
    p, ref = torch.tensor([0.5, 0.8, 0.6]), torch.zeros(3)
    roles = ["hide", "panel", "panel"]
    tot = {"hide": 1, "match": 0, "views": 1, "panel": 2}
    off = ObjectiveConfig(hide_weight=1.0, lm_weight=0.0)
    on = ObjectiveConfig(hide_weight=1.0, lm_weight=0.0, panel_weight=0.5)
    l0, s0 = loss_terms(p, ref, roles, None, off, tot)
    l1, s1 = loss_terms(p, ref, roles, None, on, tot)
    assert abs(float(l0) - 0.25) < 1e-6  # panel weight 0: the old objective
    assert abs(float(l1) - (0.25 + 0.5 * (0.64 + 0.36) / 2)) < 1e-6
    st = summarize_stats(s1)
    assert st["n_panel"] == 2 and abs(st["panel_p"] - 0.7) < 1e-6
    assert "n_panel" not in summarize_stats(s0) or summarize_stats(s0)["n_panel"] == 2


def test_panel_layers_must_not_be_held_out():
    from next_chameleons.tica.config import load_run_config

    cfg = load_run_config("final_v4_4b")
    cfg.rl.adaptive.panel_layers = [8, 20]
    with pytest.raises(ValueError, match="panel_layers"):
        cfg.__post_init__()
