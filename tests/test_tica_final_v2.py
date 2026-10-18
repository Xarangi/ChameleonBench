"""FINAL v2 pieces (REPLICATION 2026-10-15e): layer-placed kinds, deep / spread ensembles, the depth
hold-out guard, and pin_segment_first."""

from __future__ import annotations

import dataclasses
import sys

import pytest
import torch

sys.path.insert(0, "tests")


def test_layer_placed_kinds():
    from next_chameleons.tica.probe_zoo import ENS_KINDS, kind_layers, split_kind

    assert split_kind("linear@22") == ("linear", 22) and split_kind("maxpool") == ("maxpool", None)
    assert kind_layers("linear@22", 12) == [22] and kind_layers("maxpool@24", 12) == [24]
    assert kind_layers("ensdeep4_linear", 12) == [16, 18, 22, 26]
    assert kind_layers("spread5_linear", 12) == [4, 10, 16, 22, 28]
    assert "ensdeep4_linear" in ENS_KINDS and "spread5_linear" in ENS_KINDS
    with pytest.raises(ValueError):
        kind_layers("ens4_linear@3", 12)


def test_spread5_is_a_median_vote():
    from next_chameleons.tica.probe_zoo import fit_kind

    g = torch.Generator().manual_seed(0)
    Ls = [4, 10, 16, 22, 28]
    P = {L: torch.randn(30, 6, generator=g) + 1.0 for L in Ls}
    N = {L: torch.randn(30, 6, generator=g) - 1.0 for L in Ls}
    m = fit_kind("spread5_linear", P, N, name="s", layer=12, concept="c", C=1.0, g=g)
    assert m.agg == "median" and m.layers_used == Ls
    x = {L: P[L][:4] for L in Ls}
    per = torch.stack([m._score(p, x[p[0]]) for p in m.parts])
    assert torch.allclose(m.logits(x), per.median(0).values)
    assert m.to("cpu").agg == "median"
    e = fit_kind("ens4_linear", {L: P[16] for L in (8, 10, 12, 14)},
                 {L: N[16] for L in (8, 10, 12, 14)}, name="e", layer=12, concept="c", C=1.0, g=g)
    assert e.agg == "mean"


def test_final_v2_config_and_depth_guard():
    from next_chameleons.tica.config import load_run_config
    from next_chameleons.tica.multi import check_probe_split, heldout_concepts, load_split
    from next_chameleons.tica.probe_bank import BANK
    from next_chameleons.tica.probe_zoo import ZOO_KINDS

    cfg = load_run_config("final_v2_4b")
    known = set(BANK) | set(ZOO_KINDS) | {"linear", "mlp", "multilayer"}
    check_probe_split(cfg.final.train_refit_kinds, cfg.final.heldout_refit_kinds, known)
    assert not set(cfg.episode_concepts) & set(heldout_concepts(load_split(cfg.final.split)))
    assert cfg.final.train_refit_kinds == cfg.rl.adaptive.episode_refit_kinds
    for bad in ("maxpool@20", "maxpool@28", "ens8_linear"):  # 20 static, 28 = linear@28; ens8 ok
        f = dataclasses.replace(cfg.final, train_refit_kinds=[*cfg.final.train_refit_kinds, bad],
                                heldout_refit_kinds=[k for k in cfg.final.heldout_refit_kinds
                                                     if k != bad])
        if bad == "ens8_linear":
            dataclasses.replace(cfg, final=f)  # allowed: no held-out layer
        else:
            with pytest.raises(ValueError, match="held-out layers"):
                dataclasses.replace(cfg, final=f)


def test_pin_segment_first_targets(tmp_path):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    b_dogs = make_bank(1, ["dogs"])
    b_dogs.name = "concept_L1_dogs"  # one base probe per concept, as in training
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    pool.add(b_dogs, round=0, family="concept_linear")
    order = ["cats", "cats", "dogs", "dogs", "cats", "cats"]

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec("real_score|cats|0", "real_score", ["aa", "bb", "cc", "dd", "ee",
                                "ff"], "cats", "cats", "concept_L1",
                                meta={"monitored_turns": order})]

    roles = {}
    for pin in (False, True):
        cfg = AdaptiveSupervisedConfig(
            steps=1, episodes_per_step=1, turns=6, micro_batch=2, lr=1e-2, warmup_steps=1,
            save_every=0, objective=ObjectiveConfig(pin_segment_first=pin),
            adaptive=AdaptiveConfig(refit_every=0, excuse_new_probe=True))
        tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path / str(pin))
        trajs, _ = tr.rollout(tr.sampler.sample(1))
        roles[pin] = {}
        for t in tr._targets(trajs):
            roles[pin].setdefault(t.turn, set()).add((t.monitor, t.role))
    assert ("concept_L1_dogs", "motion") in roles[False][2]  # first "dogs" turn: excused
    assert ("concept_L1_dogs", "match") in roles[True][2]  # ... or pinned to the base model
    assert ("concept_L1", "match") in roles[True][0]  # turn 0 as before
    assert roles[True][4] == roles[False][4]  # RETURN to cats: not pinned


def test_refit_switches_type_on_gate_failure_and_logs(tmp_path, monkeypatch):
    """refit_switch_max: a refit that fails its gate is followed by another probe TYPE (not the old
    probe); every attempt lands in refits.jsonl with why it failed."""
    import json

    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                                "cats", base_name) for k in range(n)]

    def make(switch):
        cfg = AdaptiveSupervisedConfig(
            steps=1, episodes_per_step=1, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1,
            save_every=0,
            adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0,
                                    refit_min_tpr_np=0.0, refit_C=1.0, refit_switch_max=switch,
                                    episode_refit_kind="mixed",
                                    episode_refit_kinds=["meandiff", "linear"],
                                    episode_kind_sampling="uniform",
                                    first_kind="meandiff",
                                    episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
        return AdaptiveSupervisedTrainer(
            pol, pool, FakeSampler(), cfg, tmp_path / f"s{switch}",
            branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                             ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
            branch_negatives=["a b", "c d", "e f"])

    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    real = AdaptiveSupervisedTrainer._episode_refit

    def meandiff_fails(self, s, prefix, shown, concept=None, turn=None, kind=None):
        k = kind or self._pick_kind()
        if k == "meandiff":  # always fails its gate
            return None, {"kind": k, "kept": False, "auroc": float("nan"), "tpr_np": 1.0}
        return real(self, s, prefix, shown, concept, turn, kind=k)

    monkeypatch.setattr(AdaptiveSupervisedTrainer, "_episode_refit", meandiff_fails)
    for switch in (0, 1):
        tr = make(switch)
        _, m = tr.rollout(FakeSampler().sample(1))
        recs = [json.loads(x) for x in (tr.out / "refits.jsonl").read_text().splitlines()]
        assert recs[0]["kind"] == "meandiff" and recs[0]["why"] == "auroc"
        if switch == 0:  # original fallback: the old probe keeps judging
            assert m["ep_refits_accepted"] == 0 and m["gate_att_meandiff"] == 2
        else:
            assert m["ep_refits_accepted"] == 2 and m["ep_refit_switch_attempts"] == 2
            assert m["gate_ok_linear"] == 2 and m["gate_att_meandiff"] == 2
            assert [r["attempt"] for r in recs] == [0, 1, 0, 1]


def test_meandiff_episode_refit_runs(tmp_path, monkeypatch):
    """meandiff (pooled rows, not tokens) refits inside episodes (crashed with ft=None, 10-16)."""
    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                                "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_refit_kind="meandiff",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    _, m = tr.rollout(FakeSampler().sample(1))
    assert m["gate_att_meandiff"] >= 1


def test_beat_summary_counts_turns_and_censoring():
    from next_chameleons.tica.trainers.supervised import beat_summary

    recs = [{"kind": "linear", "turns": 1, "beaten": True},
            {"kind": "linear", "turns": 3, "beaten": True},
            {"kind": "gdm", "turns": 4, "beaten": False}]
    m = beat_summary(recs)
    assert m["beat_n"] == 3 and abs(m["beat_frac"] - 2 / 3) < 1e-9
    assert m["beat_turns_mean"] == 2 and abs(m["beat_within1"] - 1 / 3) < 1e-9
    assert abs(m["beat_within3"] - 2 / 3) < 1e-9
    assert m["beat_frac_gdm"] == 0 and m["beat_frac_linear"] == 1


def test_rollout_logs_turns_to_beat(tmp_path, monkeypatch):
    """Every accepted refit opens a turns-to-beat record; always passing beats each in 1 turn."""
    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc", "dd"],
                                "cats", "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=4, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0,
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    _, m = tr.rollout(FakeSampler().sample(1))
    beaten = [r for r in tr.deploy_log if r["beaten"]]
    assert beaten and all(r["turns"] == 1 for r in beaten)
    assert m["beat_n"] == len(tr.deploy_log) and m["beat_within1"] == m["beat_frac"]


def test_first_probe_fork_fits_the_first_probe_on_the_policy(tmp_path):
    """first_probe="fork": turn 0's judge is a fresh refit fit on the policy at turn 0 (forks =
    the turn-0 user message re-rendered for other texts), not the base probe."""
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                                "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        objective=ObjectiveConfig(pin_first_turn=False, pin_segment_first=False),
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, first_probe="fork",
                                # a first probe at a layer no base probe reads (KeyError 26 in
                                # profile 5851747): it must be read on the turn it judges
                                episode_refit_kind="linear@2",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    trajs, m = tr.rollout(FakeSampler().sample(2))  # two episodes opening on the same concept
    assert m["ep_first_probe_att"] == 1 and m["ep_first_probe_acc"] == 1
    assert m["ep_first_probe_shared"] == 1  # turn-0 fork probe shared, not refit
    assert trajs[1].results[0]["deployed"][0] == trajs[0].results[0]["deployed"][0]
    first = trajs[0].results[0]["deployed"][0]
    assert first.startswith("ep_refit_") and first != base_name
    roles = {(t.turn, t.role) for t in tr._targets(trajs)}
    assert not any(t == 0 for t, _ in roles)  # no pinning at turn 0
    # a concept SWITCH mid-episode: the new concept's first judge is also a policy fork probe
    sw = EpisodeSpec("real_score|cats|9", "real_score", ["aa", "bb", "cc", "dd"], "cats", "cats",
                     base_name, meta={"monitored_turns": ["cats", "cats", "dogs", "dogs"]})
    tr.cfg.turns = 4
    trajs, m = tr.rollout([sw])
    assert m["ep_first_probe_att"] == 2
    dogs_first = trajs[0].results[2]["deployed"][0]
    assert dogs_first.startswith("ep_refit_") and trajs[0].results[2]["concept"] == "dogs"


def test_first_probe_switch_fork_base_at_turn0_fork_at_switch(tmp_path):
    """first_probe="switch_fork" (10-17h): turn 0 is judged by the BASE probe (no pinning, hidden
    from with hide_from_turn 0); a newly switched-to concept gets a policy fork probe."""
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=4, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        objective=ObjectiveConfig(pin_first_turn=False, pin_segment_first=False,
                                  hide_from_turn=0),
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, first_probe="switch_fork",
                                episode_refit_kind="linear@2",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, None, cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    sw = EpisodeSpec("real_score|cats|9", "real_score", ["aa", "bb", "cc", "dd"], "cats", "cats",
                     base_name, meta={"monitored_turns": ["cats", "cats", "dogs", "dogs"]})
    trajs, m = tr.rollout([sw])
    assert m["ep_first_probe_att"] == 1  # only the switch forks
    assert trajs[0].results[0]["deployed"][0] == base_name  # turn 0: the base probe
    dogs_first = trajs[0].results[2]["deployed"][0]
    assert dogs_first.startswith("ep_refit_") and trajs[0].results[2]["concept"] == "dogs"
    roles = {(t.turn, t.role) for t in tr._targets(trajs) if t.key == "monitored"}
    assert (0, "hide") in roles and not any(r == "match" for _, r in roles)  # hidden, not pinned


def test_final_v3_factored_kinds_respect_held_out_depths():
    import dataclasses

    from next_chameleons.tica.config import load_run_config
    from next_chameleons.tica.probe_zoo import kind_layers

    cfg = load_run_config("final_v3_4b")
    a = cfg.rl.adaptive
    assert a.kind_layers and not set(a.kind_layers) & {20, 24, 28, 30}
    for k in ("maxpool", "attnpool", "rlobf_attn", "linear"):
        for L in a.kind_layers:
            assert kind_layers(f"{k}@{L}", 12) == [L]
    with pytest.raises(ValueError, match="held-out layers"):
        dataclasses.replace(cfg, rl=dataclasses.replace(
            cfg.rl, adaptive=dataclasses.replace(a, kind_layers=[12, 24])))


def test_retry_targets_and_weight(tmp_path):
    """Retry turns (the same active probe caught the previous turn) are flagged on the active hide
    target, carry the previous turn's ACTIVE p, and retry_hide_weight reweights only them."""
    from types import SimpleNamespace

    import torch

    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig, loss_terms
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=6, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0, objective=ObjectiveConfig(hide_from_turn=0, pin_first_turn=False,
                                                retry_hide_weight=3.0),
        adaptive=AdaptiveConfig(refit_every=0, excuse_new_probe=True))
    tr = AdaptiveSupervisedTrainer(tiny_policy(), pool, None, cfg, tmp_path)
    spec = EpisodeSpec("real_score|cats|0", "real_score", ["a"] * 6, "cats", "cats", "concept_L1")
    seq = [("m1", True), ("m1", True), ("m1", False), ("m2", True), ("m2", True), ("m2", False)]
    res, dep = [], []
    for t, (m, det) in enumerate(seq):
        dep = [*dep, m] if m not in dep else dep
        res.append({"concept": "cats", "active_monitor": m, "deployed": list(dep), "turn": t,
                    "monitored": {"detected": det, "p": 0.1 * t, "name": m},
                    "active_p": 0.5 + 0.1 * t})
    tg = [x for x in tr._targets([SimpleNamespace(spec=spec, results=res)])
          if x.monitor == res[x.turn]["active_monitor"]]
    by = {x.turn: x for x in tg}
    assert [by[t].retry for t in range(6)] == [False, True, True, False, True, True]
    assert by[3].role == "motion"  # a new probe's first judgement is excused
    assert abs(by[1].prev_p - 0.5) < 1e-9  # the previous turn's ACTIVE p, not the shown read
    # weighting: a retry target counts 3x a first-try target, normalised by the weight sum
    p = torch.tensor([0.5, 0.5])
    ref = torch.zeros(2)
    w = torch.tensor([1.0, 3.0])
    loss, _ = loss_terms(p, ref, ["hide", "hide"], None, cfg.objective,
                         {"hide": 2, "match": 0, "views": 1, "hide_w_sum": 4.0}, hide_w=w)
    assert abs(float(loss) - 0.25) < 1e-6
    assert ObjectiveConfig().retry_hide_weight == 1.0  # default off


def test_improve_margin_demands_progress():
    import torch

    from next_chameleons.tica.objectives import ObjectiveConfig, loss_terms

    p, ref, prev = torch.tensor([0.9]), torch.zeros(1), torch.tensor([0.9])
    tot = {"hide": 1, "match": 0, "views": 1}
    cfg0 = ObjectiveConfig(hide_weight=0.0, lm_weight=0.0, improve_weight=1.0)
    cfg1 = ObjectiveConfig(hide_weight=0.0, lm_weight=0.0, improve_weight=1.0, improve_margin=0.1)
    l0, _ = loss_terms(p, ref, ["hide"], None, cfg0, tot, prev_p=prev)
    l1, _ = loss_terms(p, ref, ["hide"], None, cfg1, tot, prev_p=prev)
    assert float(l0) == 0.0 and abs(float(l1) - 0.1) < 1e-6  # a stall costs only with the margin


def test_retry_ladder_counts_consecutive_catches(tmp_path):
    """retry_n = consecutive catches by the same active probe; the ladder weights by it."""
    from types import SimpleNamespace

    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig, retry_weight
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    oc = ObjectiveConfig(hide_from_turn=0, pin_first_turn=False, retry_hide_ladder=(2.0, 3.0, 4.0))
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=7, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0, objective=oc, adaptive=AdaptiveConfig(refit_every=0, excuse_new_probe=True))
    tr = AdaptiveSupervisedTrainer(tiny_policy(), pool, None, cfg, tmp_path)
    spec = EpisodeSpec("real_score|cats|0", "real_score", ["a"] * 7, "cats", "cats", "concept_L1")
    seq = [("m1", True), ("m1", True), ("m1", True), ("m1", True), ("m1", True), ("m1", False),
           ("m1", True)]
    res = [{"concept": "cats", "active_monitor": m, "deployed": ["m1"], "turn": t,
            "monitored": {"detected": d, "p": 0.5, "name": m}} for t, (m, d) in enumerate(seq)]
    by = {x.turn: x for x in tr._targets([SimpleNamespace(spec=spec, results=res)])}
    assert [by[t].retry_n for t in range(7)] == [0, 1, 2, 3, 4, 5, 0]
    assert [retry_weight(oc, by[t]) for t in range(7)] == [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 1.0]
    flat = ObjectiveConfig(retry_hide_weight=3.0)
    assert [retry_weight(flat, by[t]) for t in (0, 1, 4, 6)] == [1.0, 3.0, 3.0, 1.0]
    assert retry_weight(ObjectiveConfig(), by[3]) == 1.0  # both off


def test_lr_cooldown_at_run_end():
    from next_chameleons.tica.trainers.supervised import SupervisedConfig, lr_factor

    c = SupervisedConfig(steps=100, lr=2e-4, warmup_steps=1, lr_cooldown_steps=20, lr_final=2e-5)
    assert lr_factor(c, 50) == 1.0  # holds lr before the cooldown
    assert abs(lr_factor(c, 100) - 0.1) < 1e-9  # lr_final at the run's end
    assert 0.1 < lr_factor(c, 90) < 1.0
    # a resume chunk (offset 60, 40 steps left) ends at the same global step 100
    r = SupervisedConfig(steps=40, lr=2e-4, warmup_steps=1, lr_cooldown_steps=20, lr_final=2e-5)
    assert abs(lr_factor(r, 30, offset=60) - lr_factor(c, 90)) < 1e-9
    assert SupervisedConfig().lr_cooldown_steps == 0  # default off
