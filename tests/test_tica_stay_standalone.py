"""adaptive.stay_standalone (REPLICATION 2026-10-21x): the stay half of motion="conditional" without
pursuit. Off (default) = unchanged; on with weight 0 = the same training bit for bit (the extra
reads do not perturb anything); on = a penalty only on turns whose previous turn PASSED, with no
gradient into the previous turn."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, "tests")


def _trainer(tmp_path, **ad):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    with torch.no_grad():  # policy != base, so the offset form is non-trivial
        g = torch.Generator().manual_seed(3)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def __init__(self):
            self.concepts = [ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)])]

        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score",
                                ["aa x", "bb y", "cc z", "dd w"], "cats", "cats", base_name)
                    for k in range(n)]

    kw = dict(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0, refit_C=1.0,
              adaptive=False, motion="conditional", excuse_new_probe=True, pursuit=False,
              episode_refit_prob=0.5, single_pass=True, incremental_rollout=True,
              episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3})
    kw.update(ad)
    cfg = AdaptiveSupervisedConfig(
        steps=3, episodes_per_step=2, turns=4, micro_batch=1, lr=1e-2, warmup_steps=1,
        save_every=0, adaptive=AdaptiveConfig(**kw))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    return tr, pol


def _verdict(monkeypatch, detected):
    from next_chameleons.tica.adaptive import AdaptiveDefender

    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score", lambda self, name, pooled, row: {
        **orig(self, name, pooled, row), "detected": detected})


def _lora(pol):
    return {n: p.detach().clone() for n, p in pol.model.named_parameters() if "lora_" in n}


def test_stay_standalone_off_by_default():
    from next_chameleons.tica.adaptive import AdaptiveConfig

    a = AdaptiveConfig()
    assert a.stay_standalone is False and a.stay_layers == []


@pytest.mark.parametrize("offset", [False, True])
def test_weight_zero_matches_off_exactly(tmp_path, monkeypatch, offset):
    """On with weight 0: every logged training quantity and the trained weights equal the off run's
    bit for bit (the extra stay-layer reads change nothing else)."""
    _verdict(monkeypatch, False)
    runs = []
    for k, on in enumerate((False, True)):
        tr, pol = _trainer(tmp_path / f"r{k}", stay_offset=offset, stay_standalone=on,
                           stay_standalone_weight=0.0, stay_layers=[1, 2])
        recs = [tr.step() for _ in range(2)]
        runs.append((recs, _lora(pol)))
    (a, wa), (b, wb) = runs
    for n in wa:
        assert torch.equal(wa[n], wb[n]), n
    for ra, rb in zip(a, b, strict=True):
        for key, v in ra.items():
            if "sec" in key or key.startswith("max_mem"):  # wall-clock timings
                continue
            assert key in rb, key
            assert v == rb[key] or (v != v and rb[key] != rb[key]), (key, v, rb[key])
    assert b[-1]["train/n_stay"] > 0 and "rollout/move_after_pass_L1" in b[-1]


def test_stay_only_after_pass(tmp_path, monkeypatch):
    _verdict(monkeypatch, False)  # every turn passes -> every post-turn-0 turn is a stay turn
    tr, _ = _trainer(tmp_path / "p", stay_offset=True, stay_standalone=True, stay_layers=[1, 2])
    rec = tr.step()
    assert rec["train/n_stay"] > 0 and rec["train/n_stay_catch"] == 0
    assert rec["train/stay"] > 0.0 and rec["train/stay_term"] > 0.0
    assert rec["rollout/n_move_after_pass"] > 0 and rec["rollout/n_move_after_catch"] == 0
    assert {"rollout/move_after_pass_L1", "rollout/move_after_pass_L2"} <= set(rec)

    _verdict(monkeypatch, True)  # every turn caught -> no stay term, catch-side diagnostics only
    tr, _ = _trainer(tmp_path / "c", stay_offset=True, stay_standalone=True, stay_layers=[1, 2])
    rec = tr.step()
    assert rec["train/n_stay"] == 0 and rec["train/stay"] == 0.0
    assert rec["train/n_stay_catch"] > 0 and rec["rollout/n_move_after_catch"] > 0
    assert "rollout/move_after_pass_L1" not in rec


@pytest.mark.parametrize("offset", [False, True])
def test_stay_loss_grad_flows_through_turn_t_only(tmp_path, offset):
    """The term at turn 1 (turn 0 passed) has grad w.r.t. turn 1's read and none w.r.t. turn 0's
    (whose stored rollout read is the detached target); a caught turn 0 gives no term at all."""
    from next_chameleons.tica.adaptive import PursuitState
    from next_chameleons.tica.trainers.supervised import Target

    tr, _ = _trainer(tmp_path, stay_offset=offset, stay_standalone=True, stay_layers=[1, 2])
    tr._stay_L = [1, 2]
    torch.manual_seed(0)
    pooled = {L: torch.randn(2, 32, requires_grad=True) for L in (1, 2)}  # rows: turn 0, turn 1
    tr._ep_stay = {(0, t): torch.stack([pooled[L][t] for L in (1, 2)]).detach() + 0.1 * t
                   for t in (0, 1)}
    tr._ep_concept = {(0, 0): "cats", (0, 1): "cats"}
    for L in (1, 2):
        tr.stay_by[("cats", L)] = PursuitState(L)
        tr.stay_by[("cats", L)].scale = 2.0
    ref = {(0, t): torch.randn(2, 32) for t in (0, 1)}
    targets = [Target(0, 0, "monitored", "m", "cats", "match"),
               Target(0, 1, "monitored", "m", "cats", "hide")]
    tkey = [(0, 0), (0, 1)]
    pos = {(0, 0): 0, (0, 1): 1}
    from collections import defaultdict

    acc = defaultdict(float)
    s = tr._stay_loss([0, 1], {1}, set(), targets, tkey, pos, pooled, ref, 1, acc)
    g = torch.autograd.grad(s, [pooled[1], pooled[2]])
    for gl in g:
        assert torch.all(gl[0] == 0) and gl[1].abs().sum() > 0
    # the value: mean over layers of (||d|| / scale)^2
    want = 0.0
    for j, L in enumerate((1, 2)):
        a1, a0 = pooled[L][1].detach(), tr._ep_stay[(0, 0)][j]
        d = (a1 - ref[(0, 1)][j]) - (a0 - ref[(0, 0)][j]) if offset else a1 - a0
        want += float((d.norm() / 2.0) ** 2) / 2
    assert abs(float(s) - want) < 1e-4 * max(1.0, want)
    assert acc["n_stay"] == 1
    acc2 = defaultdict(float)
    s2 = tr._stay_loss([0, 1], set(), {1}, targets, tkey, pos, pooled, ref, 1, acc2)
    assert float(s2) == 0.0 and not s2.requires_grad and acc2["n_mv_catch"] == 1


def test_stay_layers_refuse_heldout(tmp_path, monkeypatch):
    tr, _ = _trainer(tmp_path, stay_standalone=True, stay_layers=[1, 2])
    monkeypatch.setattr(type(tr), "_heldout_layers", lambda self: {2})
    with pytest.raises(ValueError, match="held-out"):
        tr._stay_layers(1)
    tr.cfg.adaptive.stay_layers = []
    tr.cfg.adaptive.kind_layers = [1]
    assert tr._stay_layers(1) == [1]


def test_stay_standalone_refuses_double_stay(tmp_path):
    with pytest.raises(ValueError, match="twice"):
        _trainer(tmp_path, stay_standalone=True, pursuit=True)


def test_config_refuses_heldout_stay_layers():
    from next_chameleons.tica.config import load_run_config

    cfg = load_run_config("final_v4_math_v3b_r380_4b")
    cfg.__post_init__()
    cfg.rl.adaptive.stay_layers = [16, 20]
    with pytest.raises(ValueError, match="stay_layers"):
        cfg.__post_init__()
