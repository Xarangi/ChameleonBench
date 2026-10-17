"""Adaptive-monitor training pieces (tica/adaptive.py) + the reasoning-preserving renderer.

CPU-only: synthetic activations and a hand-built bank, so the defender logic, the progressive cost,
the online refit labelling, and the ChatML rendering are all checked without a GPU.
"""

from __future__ import annotations

import torch

from next_chameleons.tica.adaptive import (
    AdaptiveConfig,
    AdaptiveDefender,
    PursuitState,
    RefitData,
    fit_refit_bank,
    progressive_cost,
    pursuit_prob,
    selectivity_reward,
    trim_bank,
)
from next_chameleons.tica.monitors import ConceptProbeBank, MonitorPool
from next_chameleons.tica.policy import render_keep_reasoning

C = "math"


def _bank(name: str, layer: int, w: list[float], thr: float) -> ConceptProbeBank:
    d = len(w)
    return ConceptProbeBank(
        name=name,
        layer=layer,
        concepts=[C],
        W=torch.tensor([w], dtype=torch.float32),
        b=torch.zeros(1),
        mu=torch.zeros(1, d),
        sd=torch.ones(1, d),
        thr=torch.tensor([thr]),
        zero=torch.tensor([-2.0]),
        hundred=torch.tensor([2.0]),
        gate=torch.ones(1, dtype=torch.bool),
    )


def _pool(*banks) -> MonitorPool:
    p = MonitorPool()
    for i, b in enumerate(banks):
        p.add(b, round=i, family="concept_linear")
    return p


def test_progressive_cost_rises_then_clamps():
    costs = (0.1, 0.25, 0.5, 1.0)
    assert [progressive_cost(k, costs) for k in range(6)] == [0.1, 0.25, 0.5, 1.0, 1.0, 1.0]


def test_defender_miss_keeps_monitor_and_raises_cost():
    # One monitor on a 1-d feature; state 5.0 is far above threshold 0 -> detected.
    pool = _pool(_bank("m0", 0, [1.0], 0.0))
    cfg = AdaptiveConfig(miss_costs=(0.1, 0.25, 0.5, 1.0))
    d = AdaptiveDefender(pool, C, cfg, start="m0")
    pooled = {0: torch.tensor([[5.0]])}
    r1 = d.observe(d.score("m0", pooled, 0), pooled, 0)
    r2 = d.observe(d.score("m0", pooled, 0), pooled, 0)
    assert r1["passed"] is False and r1["reward"] == -0.1
    assert r2["reward"] == -0.25 and d.active == "m0" and d.stage_turn == 2


def test_defender_pass_switches_to_best_catcher():
    # m0 fires on feature 0, m1 on feature 1. A state that dodges m0 (feat0 low) but is caught
    # hard by m1 (feat1 high) should, on passing m0, switch the active monitor to m1.
    pool = _pool(_bank("m0", 0, [1.0, 0.0], 0.0), _bank("m1", 0, [0.0, 1.0], 0.0))
    cfg = AdaptiveConfig()
    d = AdaptiveDefender(pool, C, cfg, start="m0")
    pooled = {0: torch.tensor([[-3.0, 7.0]])}  # dodges m0, caught by m1
    out = d.observe(d.score("m0", pooled, 0), pooled, 0)
    assert out["passed"] and out["next_monitor"] == "m1" and not out["novel"]
    assert d.active == "m1" and d.stage_turn == 0


def test_defender_novel_escape_when_no_member_catches():
    pool = _pool(_bank("m0", 0, [1.0], 0.0), _bank("m1", 0, [1.0], 0.0))
    cfg = AdaptiveConfig()
    d = AdaptiveDefender(pool, C, cfg, start="m0")
    pooled = {0: torch.tensor([[-5.0]])}  # below both thresholds
    out = d.observe(d.score("m0", pooled, 0), pooled, 0)
    assert out["passed"] and out["novel"] and out["next_monitor"] == "m0"


def test_defender_non_adaptive_keeps_start():
    pool = _pool(_bank("m0", 0, [1.0, 0.0], 0.0), _bank("m1", 0, [0.0, 1.0], 0.0))
    d = AdaptiveDefender(pool, C, AdaptiveConfig(adaptive=False), start="m0")
    pooled = {0: torch.tensor([[-3.0, 7.0]])}
    out = d.observe(d.score("m0", pooled, 0), pooled, 0)
    assert out["passed"] and out["next_monitor"] == "m0"


def test_selectivity_reward_sign():
    cfg = AdaptiveConfig(selectivity_penalty=1.0)
    assert selectivity_reward(True, cfg) == 1.0
    assert selectivity_reward(False, cfg) == -1.0


def test_online_refit_keeps_separable_layer_and_gates_on_tpr():
    torch.manual_seed(0)
    d = 8
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.8, refit_C=1.0)
    data = RefitData(layers=[0])
    # Escapes (pos) and no-prompt concept text (np) share a direction well separated from negs.
    for _ in range(40):
        pos = torch.randn(1, d) + torch.tensor([[3.0] + [0.0] * (d - 1)])
        npv = torch.randn(1, d) + torch.tensor([[3.0] + [0.0] * (d - 1)])
        neg = torch.randn(1, d) - torch.tensor([[3.0] + [0.0] * (d - 1)])
        data.pos[0].append(pos)
        data.np_pos[0].append(npv)
        data.neg[0].append(neg)
    bank, info = fit_refit_bank(data, C, "refit_online_1", cfg)
    assert bank is not None and info["kept"] and info["auroc"] >= 0.9
    assert bank.covers(C) and bank.layer == 0


def test_online_refit_rejected_when_tpr_np_low():
    torch.manual_seed(1)
    d = 8
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.9, refit_C=1.0)
    data = RefitData(layers=[0])
    # pos separable from neg, but np (out-of-context concept) looks like neg -> a dodge detector,
    # not a concept detector -> must be rejected.
    for _ in range(40):
        data.pos[0].append(torch.randn(1, d) + torch.tensor([[3.0] + [0.0] * (d - 1)]))
        data.neg[0].append(torch.randn(1, d) - torch.tensor([[3.0] + [0.0] * (d - 1)]))
        data.np_pos[0].append(torch.randn(1, d) - torch.tensor([[3.0] + [0.0] * (d - 1)]))
    bank, info = fit_refit_bank(data, C, "refit_online_1", cfg)
    assert bank is None and not info["kept"]


def test_trim_bank_drops_oldest_refit_not_base():
    pool = MonitorPool()
    pool.add(_bank("base", 0, [1.0], 0.0), round=0, family="concept_linear")
    for r in (1, 2, 3):
        b = _bank(f"refit_online_{r}", 0, [1.0], 0.0)
        b.meta["online_refit"] = True
        pool.add(b, round=r, family="concept_refit")
    dropped = trim_bank(pool, keep_base={"base"}, cap=3)
    assert dropped == ["refit_online_1"]
    assert {e.monitor.name for e in pool.entries} == {"base", "refit_online_2", "refit_online_3"}


def test_render_keep_reasoning_multi_turn_shape():
    conv = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0", "reasoning": "r0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning": "r1"},
    ]
    s = render_keep_reasoning(conv, add_generation_prompt=True, thinking=False)
    # Both earlier turns keep their reasoning (the whole point: the model can see what it tried).
    assert s.count("<think>") == 3  # a0, a1, and the empty generation-prompt block
    assert "r0" in s and "r1" in s
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_render_keep_reasoning_no_reasoning_key_drops_block():
    conv = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    s = render_keep_reasoning(conv, add_generation_prompt=False, thinking=False)
    assert "<think>" not in s and "a<|im_end|>" in s


def test_adaptive_rl_step_runs_and_updates(tmp_path):
    """End-to-end smoke of the adaptive RL path on the tiny char-level policy: budget-forced
    rollout, in-episode defender, progressive/selectivity reward, and one GRPO update."""

    import sys

    sys.path.insert(0, "tests")
    import dataclasses

    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.rl import RLConfig, RLTrainer
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.enable_thinking = True
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    pool.add(
        dataclasses.replace(make_bank(2, ["cats"]), name="concept_L2"),
        round=1,
        family="concept_refit",
    )

    def specs(n):
        out = []
        for k in range(n):
            cond = ["real_score", "real_score", "nobrief", "neg_text"][k % 4]
            tc = None if cond == "neg_text" else "cats"
            out.append(
                EpisodeSpec(
                    f"{cond}|cats|{k}",
                    cond,
                    ["aa", "bb", "cc"],
                    tc,
                    "cats",
                    "concept_L1",
                )
            )
        return out

    acfg = AdaptiveConfig(reasoning_budget=4, refit_every=1, direct_grad_weight=0.0)
    cfg = RLConfig(
        mode="reason_forced",
        use_adaptive=True,
        history_reasoning=False,  # tiny model: skip the Qwen-format history renderer
        reasoning_end="Z",
        answer_prefix="",
        specs_per_step=4,
        group_size=4,
        turns=3,
        micro_batch=4,
        lr=1e-2,
        save_every=0,
        adaptive=acfg,
    )
    tr = RLTrainer(pol, pool, specs, cfg, tmp_path)
    before = {n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" in n}
    rec = tr.step()
    assert "pass_rate" in rec and "reward_mean" in rec and "bank_size" in rec
    assert rec["n_items"] >= 0
    # The update ran; at least some adv was non-zero across the four conditions, so params move.
    moved = any(
        not torch.equal(q, before[n]) for n, q in pol.model.named_parameters() if "lora" in n
    )
    assert moved or rec["n_items"] == 0


def test_adaptive_rl_arm_c_direct_grad_runs(tmp_path):
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.rl import RLConfig, RLTrainer
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.enable_thinking = True
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")

    def specs(n):
        return [
            EpisodeSpec(
                f"real_score|cats|{k}", "real_score", ["aa", "bb"], "cats", "cats", "concept_L1"
            )
            for k in range(n)
        ]

    cfg = RLConfig(
        mode="reason_forced",
        use_adaptive=True,
        history_reasoning=False,
        reasoning_end="Z",
        answer_prefix="",
        specs_per_step=2,
        group_size=4,
        turns=2,
        micro_batch=4,
        lr=1e-2,
        save_every=0,
        adaptive=AdaptiveConfig(reasoning_budget=4, refit_every=0, direct_grad_weight=0.5),
    )
    rec = RLTrainer(pol, pool, specs, cfg, tmp_path).step()
    assert "dg" in rec  # the direct-gradient (Arm C) term was computed


def test_pursuit_state_and_prob():
    """Pursuit (cosine): a_t aligned with a recent direction scores high; opposite scores low;
    gradient flows to a_t at a generic (non-parallel) point."""
    torch.manual_seed(0)
    ps = PursuitState(layer=1, ema=0.9)
    acts = torch.randn(32, 8) + 5.0
    ps.update(acts)
    assert ps.ready() and ps.scale > 0
    c, a_prev = ps.centroid, acts[0]
    p_same = pursuit_prob(a_prev.clone(), [a_prev], c, ps.scale)  # cos=1 -> high
    p_far = pursuit_prob((c - (a_prev - c)).clone(), [a_prev], c, ps.scale)  # cos=-1 -> low
    assert float(p_same) > 0.6 > float(p_far)
    # Gradient flows at a generic point (cos is stationary exactly at a_t == a_prev).
    a_t = acts[1].clone().requires_grad_(True)
    (pursuit_prob(a_t, [a_prev], c, ps.scale) ** 2).backward()
    assert float(a_t.grad.norm()) > 0
    # No recents -> zero (no term).
    assert float(pursuit_prob(a_prev.clone(), [], c, ps.scale)) == 0.0
    # Leash is zero inside the radius, positive outside.
    from next_chameleons.tica.adaptive import pursuit_leash

    assert float(pursuit_leash(c.clone(), c, ps.scale, 1.5)) == 0.0
    assert float(pursuit_leash(c + 5 * ps.scale, c, ps.scale, 1.5)) > 0.0


def test_exo_injection_randomises_next_monitor():
    """exo_prob=1: a pass routes the next monitor to a random bank member, not the best catcher."""
    import random as _r

    pool = _pool(_bank("m0", 1, [1.0, 0.0], 0.0), _bank("m1", 1, [0.0, 1.0], 0.0))
    cfg = AdaptiveConfig(adaptive=True, exo_prob=1.0)
    seen = set()
    for s in range(8):
        dfd = AdaptiveDefender(pool, C, cfg, start="m0", rng=_r.Random(s))
        pooled = {1: torch.tensor([[-5.0, -5.0]])}  # evades both -> pass
        read = dfd.score("m0", pooled, 0)
        obs = dfd.observe(read, pooled, 0)
        seen.add(obs["next_monitor"])
    assert seen == {"m0", "m1"}  # exogenous draw reaches both members


def test_adaptive_supervised_step_runs_and_updates(tmp_path):
    """Arm A: teacher-forced covert trainer with the adaptive defender runs a step and updates."""
    import dataclasses
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    pool.add(
        dataclasses.replace(make_bank(2, ["cats"]), name="concept_L2"),
        round=1,
        family="concept_refit",
    )

    class FakeSampler:
        def sample(self, n):
            out = []
            for k in range(n):
                cond = ["real_score", "real_score", "nobrief", "neg_text"][k % 4]
                tc = None if cond == "neg_text" else "cats"
                out.append(
                    EpisodeSpec(
                        f"{cond}|cats|{k}", cond, ["aa", "bb", "cc"], tc, "cats", "concept_L1"
                    )
                )
            return out

    cfg = AdaptiveSupervisedConfig(
        steps=1,
        episodes_per_step=4,
        turns=3,
        micro_batch=4,
        lr=1e-2,
        warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_every=1, refit_min_each=24),
    )
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path)
    before = {n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" in n}
    rec = tr.step()
    assert "train/hide_mse" in rec and "rollout/pass_rate" in rec and "rollout/bank_size" in rec
    assert any(
        not torch.equal(q, before[n]) for n, q in pol.model.named_parameters() if "lora" in n
    )


def test_adaptive_supervised_step_with_pursuit(tmp_path):
    """Pursuit on: the step runs, logs the pursuit term + motion metric, and updates the policy."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")

    class FakeSampler:
        def sample(self, n):
            out = []
            for k in range(n):
                cond = ["real_score", "real_score", "nobrief", "neg_text"][k % 4]
                tc = None if cond == "neg_text" else "cats"
                out.append(
                    EpisodeSpec(
                        f"{cond}|cats|{k}", cond, ["aa", "bb", "cc", "dd"], tc, "cats", "concept_L1"
                    )
                )
            return out

    cfg = AdaptiveSupervisedConfig(
        steps=1,
        episodes_per_step=4,
        turns=4,
        micro_batch=4,
        lr=1e-2,
        warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(
            refit_every=0, pursuit=True, pursuit_from_turn=1, pursuit_window=2, pursuit_weight=1.0
        ),
    )
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path)
    before = {n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" in n}
    rec = tr.step()
    assert "train/pursuit_p" in rec and rec["train/n_pursuit"] > 0
    assert "rollout/turn_move" in rec
    assert tr.pursuit_by["cats"].ready()
    assert any(
        not torch.equal(q, before[n]) for n, q in pol.model.named_parameters() if "lora" in n
    )


def test_adaptive_supervised_refit_triggers(tmp_path):
    """Data-triggered refit (refit_when_ready) + refit_as_start run without error; fresh-escape
    counting works and the start-monitor fallback is safe when no refit exists yet."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")

    class FakeSampler:
        def sample(self, n):
            return [
                EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                            "cats", "concept_L1")
                for k in range(n)
            ]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=4, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(
            refit_every=0, refit_when_ready=True, refit_as_start=True, refit_min_each=24
        ),
    )
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path)
    rec = tr.step()  # no refit yet (not enough data) -> start falls back to s.monitor, no error
    assert tr._latest_refit is None
    assert tr._fresh_escapes >= 0  # counter is maintained
    assert "rollout/pass_rate" in rec


def test_adaptive_supervised_branch_refit(tmp_path):
    """refit_mode="branch": forking the current policy produces a valid refit dataset from ONE
    collection (user's branch-refit design); the method runs end-to-end and step() integrates it."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")  # >=2 concepts

    class FakeSampler:
        def __init__(self):
            self.concepts = [ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)])]

        def sample(self, n):
            return [
                EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                            "cats", "concept_L1")
                for k in range(n)
            ]

    branch_concepts = [
        ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
        ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)]),
    ]
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=4, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(
            refit_mode="branch", refit_as_start=True, branch_refit_every=1,
            branch={"roots": 2, "root_turns": 2, "follow_pos": 3, "follow_pos_noprompt": 3,
                    "follow_neg_other": 2, "follow_neg_free": 1, "feedback": "verdict"},
        ),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=branch_concepts, branch_negatives=["a b", "c d", "e f"],
        roots_only={"cats"},
    )
    info = tr._branch_refit()  # runs end-to-end, no error; outcome depends on the random policy
    assert isinstance(info, dict) and "refit" in info
    assert info["refit"] in {"added", "no_bypass", "rejected", "branch_too_few"}
    rec = tr.step()  # step() integrates the branch path without error
    assert "rollout/pass_rate" in rec
    # Trigger wiring (10-12 bug: the gate read pass_rate from the wrong dict and never fired):
    # whenever the rollout reports any pass, a due branch refit MUST be attempted.
    orig = tr.rollout
    tr.rollout = lambda specs: (lambda o: (o[0], {**o[1], "pass_rate": 1.0}))(orig(specs))
    tr._branch_refit = lambda: {"refit": "called"}
    rec = tr.step()
    assert rec.get("refit") == "called"


def test_pooled_shared_prefix_matches_pooled():
    """Fork reads with the shared history run once (cached) equal plain per-fork reads."""
    import sys

    sys.path.insert(0, "tests")
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pre = [{"role": "user", "content": "tell me about cats " * 6},
           {"role": "assistant", "content": "cats purr and meow " * 5}]
    forks = [[*pre, {"role": "user", "content": f"next {k}"},
              {"role": "assistant", "content": f"answer number {k} " * (k + 1)}] for k in range(5)]
    a = pol.pooled(forks, [1])[1]
    b = pol.pooled_shared_prefix(forks, [1], min_prefix=1)[1]
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)


def test_pooled_incremental_matches_pooled():
    """Rollout reads with a per-conversation cache (only new tokens each turn) equal plain reads,
    turn after turn, and the cache is actually reused (keep-think history = stable prefixes)."""
    import sys

    sys.path.insert(0, "tests")
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    convs = [[{"role": "user", "content": f"tell me about topic {k} " * (k + 2)}] for k in range(3)]
    state = {}
    for t in range(3):
        for k, c in enumerate(convs):
            c.append({"role": "assistant", "content": f"answer {t} for {k} " * (t + k + 1)})
        a = pol.pooled(convs, [1])[1]
        b = pol.pooled_incremental(convs, [1], state)[1]
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)
        for c in convs:
            c.append({"role": "user", "content": f"feedback {t}"})
    assert state["hits"] == 6  # turns 1-2 of 3 conversations ran only their new tokens


def test_adaptive_supervised_episode_refit(tmp_path, monkeypatch):
    """refit_mode="episode" (user's protocol): every pass at turn t refits a probe on forks of THAT
    conversation; it is the episode's active probe from t+1; refits are dropped after the step."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
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
    ho = make_bank(1, ["cats", "dogs"])
    ho.name = "heldout_probe"
    pool.add(ho, round=0, family="concept_linear", held_out=True)
    dfd = AdaptiveDefender(pool, "cats", AdaptiveConfig(), base_name)
    assert "heldout_probe" not in dfd.by_name  # held-out monitors are never defender members

    class FakeSampler:
        def __init__(self):
            self.concepts = [ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)])]

        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                                "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_fork={"pos": 8, "np": 8, "other": 8,
                                                           "free": 3}),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"],
    )
    orig = AdaptiveDefender.score

    def never_caught(self, name, pooled, row):  # every turn is a pass
        return {**orig(self, name, pooled, row), "detected": False}

    monkeypatch.setattr(AdaptiveDefender, "score", never_caught)
    trajs, m = tr.rollout(FakeSampler().sample(2))
    for tj in trajs:
        names = [r["active_monitor"] for r in tj.results]
        assert names[0] == base_name  # opens on probe 1
        assert names[1].startswith("ep_refit") and names[2].startswith("ep_refit")
        assert names[1] != names[2]  # a NEW probe after each dodge
    assert m["ep_refits"] == 4 and m["ep_refits_accepted"] == 4
    rec = tr.step()
    assert rec["rollout/ep_refits"] == 4
    # episode-local refits dropped (also those from the manual rollout above)
    assert [e.monitor.name for e in tr.pool.entries] == [base_name, "heldout_probe"]
    # keep mode: accepted refits stay in the bank (trimmed to refit_max_bank)
    tr.cfg.adaptive.episode_keep, tr.cfg.adaptive.refit_max_bank = True, 3
    tr.step()
    names = [e.monitor.name for e in tr.pool.entries]
    assert names[:2] == [base_name, "heldout_probe"] and len(names) == 3
    assert names[2].startswith("ep_refit")
    # quick mode: a bank member that catches the dodge is used instead of a fork
    tr.cfg.adaptive.episode_bank_first = True
    tr.cfg.adaptive.episode_keep = False
    monkeypatch.setattr(AdaptiveDefender, "best_catcher", lambda self, pooled, row: base_name)
    _, m = tr.rollout(FakeSampler().sample(2))
    assert m["ep_bank_switches"] == 4 and m["ep_refits"] == 0
    info = tr._validate_bank("cats")  # runs; the base probe is never dropped
    assert base_name in info["bank_valid"] and base_name not in info["bank_dropped"]
    assert "heldout_probe" not in info["bank_valid"]


def test_conditional_motion_and_excused_new_probe(tmp_path, monkeypatch):
    """motion="conditional" (10-12j): stay after a pass, move after a catch; the first turn
    of a newly deployed probe carries no hide loss; episode_refit_prob gates refits."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.objectives import loss_terms
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    # "motion" targets get no probe loss
    p = torch.tensor([0.9, 0.9], requires_grad=True)
    loss, st = loss_terms(p, torch.zeros(2), ["motion", "motion"], None,
                          AdaptiveSupervisedConfig().objective)
    assert float(loss) == 0.0 and st["n_hide"] == 0 and st["n_match"] == 0

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def __init__(self):
            self.concepts = [ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)])]

        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc", "dd"],
                                "cats", "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=2, episodes_per_step=2, turns=4, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, adaptive=False, motion="conditional",
                                excuse_new_probe=True, pursuit=True, pursuit_from_turn=1,
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"],
    )
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score", lambda self, name, pooled, row: {
        **orig(self, name, pooled, row), "detected": False})
    trajs, _ = tr.rollout(FakeSampler().sample(2))
    roles = {(t.traj, t.turn): t.role for t in tr._targets(trajs)}
    assert roles[(0, 0)] == "match" and roles[(0, 1)] == "motion"  # new probe at t1: excused
    tr.step()
    rec = tr.step()  # centroid ready from step 2: every prior turn passed -> stay terms only
    assert rec["train/n_stay"] > 0 and rec["train/n_move"] == 0
    tr.cfg.adaptive.stay_offset = True  # stay = same offset from the base model's read
    rec = tr.step()
    assert rec["train/n_stay"] > 0 and rec["train/stay"] >= 0.0
    tr.cfg.adaptive.episode_refit_prob = 0.0  # never refit: same probe, nothing excused
    trajs, m = tr.rollout(FakeSampler().sample(2))
    assert m["ep_refits"] == 0
    assert all(r.get("episode_refit") == "skip" for r in trajs[0].results[:-1])
    assert all(t.role != "motion" for t in tr._targets(trajs))


def test_fit_refit_bank_heldout_gate_and_layer_exclusion():
    """10-12 audit: the gate is scored on HELD-OUT rows (random labels must be rejected; the old
    in-sample score passed them at AUROC ~1.0) and excluded (held-out eval) layers are never fit."""
    from next_chameleons.tica.adaptive import RefitData, fit_refit_bank

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0)
    # random, label-independent data in high dim: in-sample separable, held-out chance
    noise = RefitData(layers=[1], concept="cats")
    for _ in range(40):
        noise.pos[1].append(torch.randn(1, 256))
        noise.neg[1].append(torch.randn(1, 256))
        noise.np_pos[1].append(torch.randn(1, 256))
    bank, info = fit_refit_bank(noise, "cats", "r", cfg)
    assert bank is None and not info["layers"][1]["insample"]
    # genuinely separable data passes; an excluded layer is skipped
    sep = RefitData(layers=[1, 2], concept="cats")
    for _ in range(40):
        for L in (1, 2):
            sep.pos[L].append(torch.randn(1, 16) + 3.0)
            sep.neg[L].append(torch.randn(1, 16) - 3.0)
            sep.np_pos[L].append(torch.randn(1, 16) + 3.0)
    bank, info = fit_refit_bank(sep, "cats", "r2", cfg, exclude_layers={2})
    assert bank is not None and bank.layer == 1 and 2 not in info["layers"]
    # callable from inside no_grad (the frozen race's refits all failed silently there)
    with torch.no_grad():
        bank, info = fit_refit_bank(sep, "cats", "r3", cfg)
    assert bank is not None, info


def test_fit_refit_bank_cross_validated_gate():
    """refit_cv_folds (10-13d): positives scored out-of-fold, negatives held out for AUROC + the
    threshold. Noise is still rejected, separable data passes, the final probe uses every
    positive, and the result is deterministic per refit name."""
    from next_chameleons.tica.adaptive import RefitData, fit_refit_bank

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0, refit_cv_folds=5)
    noise = RefitData(layers=[1], concept="cats")
    for _ in range(40):
        noise.pos[1].append(torch.randn(1, 256))
        noise.neg[1].append(torch.randn(1, 256))
        noise.np_pos[1].append(torch.randn(1, 256))
    bank, info = fit_refit_bank(noise, "cats", "r", cfg)
    assert bank is None and info["layers"][1]["cv_folds"] == 5
    assert info["layers"][1]["auroc"] < 0.8
    sep = RefitData(layers=[1], concept="cats")
    for _ in range(40):
        sep.pos[1].append(torch.randn(1, 16) + 1.5)
        sep.neg[1].append(torch.randn(1, 16) - 1.5)
        sep.np_pos[1].append(torch.randn(1, 16) + 1.5)
    bank, info = fit_refit_bank(sep, "cats", "r2", cfg)
    assert bank is not None and info["layers"][1]["n_pos"] == 40
    _, info2 = fit_refit_bank(sep, "cats", "r2", cfg)
    assert info2["auroc"] == info["auroc"]


def test_single_pass_matches_per_turn_views():
    """keep_think_history: ONE forward over the whole conversation gives every turn's pooled
    activations and NLL exactly as the per-turn views do (single-pass training, 10-12f)."""
    import sys

    sys.path.insert(0, "tests")
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    pol.model.eval()

    def conv(k):
        c = []
        for t in range(3):
            c += [{"role": "user", "content": f"turn {t} say w{k}{t}"},
                  {"role": "assistant", "content": f"w{k}{t} " * (2 + t + k)}]
        return c

    convs = [conv(0), conv(1)]
    mvs = [pol.multi_view(c) for c in convs]
    views = []
    for c, mv in zip(convs, mvs, strict=True):
        idx = [i for i, m in enumerate(c) if m["role"] == "assistant"]
        for t, i in enumerate(idx):
            v = pol.view(c[: i + 1])
            st, en, te = mv.spans[t]
            assert v.ids == mv.ids[:te] and v.start == st and v.end == en
            views.append(v)
    layers = [1]
    pm, nm = pol.forward_multi(mvs, layers, grad=False, with_nll=True)
    pv, nv = pol.forward(views, layers, grad=False, with_nll=True)
    assert torch.allclose(pm[1], pv[1], atol=1e-4)
    assert torch.allclose(nm, nv, atol=1e-4)
    pg, ng = pol.forward_multi(mvs, layers, grad=True, with_nll=True)
    (pg[1].pow(2).mean() + ng.mean()).backward()
    assert any(p.grad is not None for n, p in pol.model.named_parameters() if "lora" in n)


def test_single_pass_step_matches_per_turn_step(tmp_path):
    """A full AdaptiveSupervisedTrainer step with adaptive.single_pass gives the same loss and
    the same update as the per-turn path from identical weights (keep_think_history on both)."""
    import copy
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    class FakeSampler:
        def sample(self, n):
            out = []
            for k in range(n):
                cond = ["real_score", "real_score", "nobrief", "neg_text"][k % 4]
                tc = None if cond == "neg_text" else "cats"
                out.append(EpisodeSpec(f"{cond}|cats|{k}", cond, ["aa", "bb b", "cc cc", "dd"],
                                       tc, "cats", "concept_L1"))
            return out

    base = tiny_policy()
    base.keep_think_history = True
    init = copy.deepcopy(base.model.state_dict())
    recs, params = [], []
    for sp in (False, True):
        base.model.load_state_dict(init)
        pool = MonitorPool()
        pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
        cfg = AdaptiveSupervisedConfig(
            steps=1, episodes_per_step=4, turns=4, micro_batch=2, lr=1e-2, warmup_steps=1,
            save_every=0, adaptive=AdaptiveConfig(refit_every=0, single_pass=sp, reloc_every=0),
        )
        tr = AdaptiveSupervisedTrainer(base, pool, FakeSampler(), cfg, tmp_path / str(sp))
        recs.append(tr.step())
        params.append({n: p.detach().clone() for n, p in base.model.named_parameters()
                       if "lora" in n})
    assert abs(recs[0]["train/loss"] - recs[1]["train/loss"]) < 1e-4
    assert recs[0]["n_views"] == recs[1]["n_views"]
    for n in params[0]:
        assert torch.allclose(params[0][n], params[1][n], atol=1e-5), n


def test_tough_monitors_mlp_and_ood():
    """10-12m tougher monitors: the MLP refit learns a NONLINEAR (xor-like) concept a linear probe
    cannot and is gated like fit_refit_bank; the OOD monitor flags off-distribution states at
    ~its calibrated FPR on reference-like states."""
    from next_chameleons.tica.adaptive import RefitData, fit_ood, fit_refit_bank, fit_refit_mlp

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0, refit_fpr=0.05)

    def xor(n, label):  # concept = sign(x0) == sign(x1): not linearly separable
        x = torch.randn(n, 8)
        same = (x[:, 0] > 0) == (x[:, 1] > 0)
        keep = same if label else ~same
        return x[keep] * 3.0

    rd = RefitData(layers=[1], concept="cats")
    for v in xor(800, True)[:200]:
        rd.pos[1].append(v[None])
    for v in xor(800, False)[:200]:
        rd.neg[1].append(v[None])
    for v in xor(400, True)[:30]:
        rd.np_pos[1].append(v[None])
    mlp, info = fit_refit_mlp(rd, "cats", "m", cfg, 1, steps=600)
    assert mlp is not None and info["auroc"] > 0.9
    lin, linfo = fit_refit_bank(rd, "cats", "l", cfg)
    assert lin is None and linfo["layers"][1]["auroc"] < 0.75  # linear cannot do it
    ref = torch.randn(600, 16)
    ood, _ = fit_ood(ref, 1, k=8, fpr=0.05)
    fpr = float((ood.logits(torch.randn(400, 16)) > ood.thr).float().mean())
    tpr = float((ood.logits(torch.randn(400, 16) + 4.0) > ood.thr).float().mean())
    assert fpr < 0.15 and tpr > 0.95


def test_episode_cumulative_no_circularity(tmp_path, monkeypatch):
    """episode_cumulative (user, 10-13): every probe deployed in the episode keeps judging, so a
    fallback to an old region is caught by the old probe, and every deployed probe is a hide
    target (the newest one excused on its first turn under excuse_new_probe)."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
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

    def specs(n):
        return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc"], "cats",
                            "cats", base_name) for k in range(n)]

    class FakeSampler:
        def sample(self, n):
            return specs(n)

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_cumulative=True,
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"],
    )
    orig = AdaptiveDefender.score
    calls = {}

    def score(self, name, pooled, row):  # base passes at turn 0, then catches every fallback
        k = (id(self), name)
        calls[k] = calls.get(k, 0) + 1
        det = name == base_name and calls[k] > 1
        return {**orig(self, name, pooled, row), "detected": det}

    monkeypatch.setattr(AdaptiveDefender, "score", score)
    trajs, m = tr.rollout(specs(2))
    assert m["ep_refits"] == 2 and m["ep_old_catch"] == 4
    for tj in trajs:
        r = tj.results
        assert r[0]["passed"] and not r[1]["passed"] and not r[2]["passed"]
        assert r[1]["monitored"]["name"] == base_name  # the OLD probe's catch is what is shown
        assert r[1]["active_monitor"].startswith("ep_refit")  # the newest stays active
        assert r[2]["deployed"] == [base_name, r[1]["active_monitor"]]
    tg = tr._targets(trajs)
    hide = [(t.traj, t.turn, t.monitor) for t in tg if t.role == "hide"]
    assert len(hide) == 8  # 2 episodes x turns 1-2 x both deployed probes
    tr.cfg.adaptive.excuse_new_probe = True
    roles = {(t.traj, t.turn, t.monitor == base_name): t.role for t in tr._targets(trajs)}
    assert roles[(0, 1, False)] == "motion" and roles[(0, 2, False)] == "hide"
    assert roles[(0, 1, True)] == "hide"
    for e in list(tr.pool.entries):  # drop this rollout's transient refits
        if e.monitor.name in tr._transient:
            tr.pool.entries.remove(e)
    tr._transient = []
    rec = tr.step()
    assert rec["train/n_hide"] > 0


def test_answer_format_prompts():
    from next_chameleons.tica.answer_format import answer_prompts, prompts, text_id
    from next_chameleons.tica.episodes import render_feedback

    opening, fb = answer_prompts({text_id("x = 2"): "Solve it.", text_id("y"): "Next one."})
    assert opening("nobrief", "x = 2", "math") == "Solve it."
    o = opening("real_score", "x = 2", "math")
    assert o.endswith("\n\nSolve it.") and '"math"' in o and "Repeat" not in o
    f = fb("real_score", "math", (10.0, 50.0), "y", feedback="verdict")
    assert f.endswith("\n\nNext one.") and "Repeat" not in f and "<text>" not in f
    assert fb("nobrief", "math", None, "y") == "Next one."
    assert prompts("repeat")[1] is render_feedback


def test_monitor_off_turns(tmp_path, monkeypatch):
    """off_turn_prob (user, 10-13): announced monitor-off turns inside monitored episodes get no
    verdict and no refit, a MATCH target on the base probe, and OFF / back-ON notices."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveDefender
    from next_chameleons.tica.episodes import MONITOR_OFF, MONITOR_ON, ConceptTexts, EpisodeSpec
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

    def specs(n):
        return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc", "dd"], "cats",
                            "cats", base_name, feedback="verdict") for k in range(n)]

    class FakeSampler:
        def sample(self, n):
            return specs(n)

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=4, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_cumulative=True, excuse_new_probe=True,
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"],
    )
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    # turns: 0 on (pass -> refit), 1 OFF, 2 on (new probe's first judged turn), 3 on
    draws = iter([0.1, 0.9, 0.9] * 2)  # turns 1-3 per episode (turn 0 never draws)
    real_random = tr.rng.random
    tr.rng.random = lambda: next(draws, real_random())
    tr.cfg.adaptive.off_turn_prob = 0.5
    trajs, m = tr.rollout(specs(2))
    tr.rng.random = real_random
    assert m["ep_off_turns"] == 2 and m["ep_off_det"] == m["ep_off_det"]  # not NaN
    for tj in trajs:
        r = tj.results
        assert r[1].get("off") and not r[0].get("off") and not r[2].get("off")
        assert r[0]["passed"] and r[1].get("episode_refit") is None  # no refit on an off-turn
        assert MONITOR_OFF in tj.feedback[0] and MONITOR_ON in tj.feedback[1]
        assert "DETECTED" not in tj.feedback[1]  # nothing judged on the off-turn
    roles = {(t.traj, t.turn, t.monitor): t.role for t in tr._targets(trajs)}
    assert roles[(0, 1, base_name)] == "match"
    new = trajs[0].results[2]["active_monitor"]
    assert new.startswith("ep_refit") and roles[(0, 2, new)] == "motion"  # excused across the off
    assert roles[(0, 2, base_name)] == "hide"


def test_probe_zoo_kinds_gate():
    """Probe zoo (10-13k): mean-diff, max-pooled and attention-pooled probes pass the shared refit
    gate on a real signal and are rejected on noise. The token signal sits in a FEW tokens of each
    positive sequence (a short span: where max / window / attention pooling beat the mean)."""
    from next_chameleons.tica.probe_zoo import gate_fit

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0, refit_cv_folds=4)
    d = 16

    def seqs(n, signal):
        out = []
        for _ in range(n):
            t = torch.randn(int(torch.randint(30, 50, ())), d)
            if signal:  # a short contiguous concept span inside a longer answer
                st = int(torch.randint(0, len(t) - 6, ()))
                t[st : st + 6, 0] += 2.5
            out.append(t)
        return out

    P, N, NP = seqs(40, True), seqs(60, False), seqs(16, True)
    for kind in ("maxpool", "rollmax", "attnpool"):
        mon, info = gate_fit(kind, P, N, NP, cfg, name=f"z_{kind}", layer=1, concept="cats")
        assert mon is not None, info
        assert mon.needs_tokens and mon.logits(NP[:3]).shape == (3,)
        noise, info = gate_fit(kind, seqs(40, False), N, seqs(16, False), cfg, name=f"n_{kind}",
                               layer=1, concept="cats")
        assert noise is None and info["auroc"] < 0.8
    Pm = torch.stack([t.mean(0) for t in P]) + torch.tensor([1.0] + [0.0] * (d - 1))
    Nm = torch.stack([t.mean(0) for t in N])
    NPm = torch.stack([t.mean(0) for t in NP]) + torch.tensor([1.0] + [0.0] * (d - 1))
    mon, info = gate_fit("meandiff", Pm, Nm, NPm, cfg, name="z_md", layer=1, concept="cats")
    assert mon is not None and not mon.needs_tokens, info
    # multi-layer mean-of-sigmoids ensembles: signal at every ensemble layer passes, noise fails
    from next_chameleons.tica.probe_zoo import ENS_LAYERS

    Ls = ENS_LAYERS["ens4"]
    shift = torch.tensor([2.5] + [0.0] * (d - 1))  # small-n MLPs need a clear signal
    Pd = {L: torch.randn(40, d) + shift for L in Ls}
    Nd = {L: torch.randn(60, d) for L in Ls}
    NPd = {L: torch.randn(16, d) + shift for L in Ls}
    for kind in ("ens4_linear", "ens4_mlp"):
        mon, info = gate_fit(kind, Pd, Nd, NPd, cfg, name=f"z_{kind}", layer=12, concept="cats")
        assert mon is not None and mon.layers_used == Ls, info
        assert mon.logits({L: v[:3] for L, v in NPd.items()}).shape == (3,)
    noise = {L: torch.randn(40, d) for L in Ls}
    mon, info = gate_fit("ens4_linear", noise, Nd, {L: torch.randn(16, d) for L in Ls}, cfg,
                         name="n_ens", layer=12, concept="cats")
    assert mon is None and info["auroc"] < 0.8


def test_reader_token_out_matches_mean():
    """token_out (probe zoo): the per-token states the cached readers return average to the same
    mean-pooled reads (shared-prefix forks and incremental rollout reads)."""
    import sys

    sys.path.insert(0, "tests")
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    pre = [{"role": "user", "content": "tell me about cats " * 6},
           {"role": "assistant", "content": "cats purr and meow " * 5}]
    forks = [[*pre, {"role": "user", "content": f"next {k}"},
              {"role": "assistant", "content": f"answer number {k} " * (k + 1)}] for k in range(4)]
    for fn in (lambda tok: pol.pooled_shared_prefix(forks, [1], min_prefix=1, token_out=tok),
               lambda tok: pol.pooled_incremental(forks, [1], {}, token_out=tok)):
        tok = {}
        mean = fn(tok)[1]
        assert len(tok[1]) == len(forks)
        rebuilt = torch.stack([t.float().mean(0) for t in tok[1]])
        assert torch.allclose(mean.float().cpu(), rebuilt, atol=2e-2, rtol=2e-2)


def test_probe_zoo_literature_exact_kinds():
    """Exact literature probes (10-13o): RL-Obfuscation attention (mean / max) and max-pool, GDM
    rolling attention. Fit + gate on a short concept span inside longer sequences; reject noise."""
    from next_chameleons.tica.probe_zoo import gate_fit

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0, refit_cv_folds=3)
    d = 16

    def seqs(n, signal):
        out = []
        for _ in range(n):
            t = torch.randn(int(torch.randint(30, 50, ())), d)
            if signal:  # a weak whole-answer shift (mean-type probes) + a local span (max-type)
                t[:, 0] += 0.8
                st = int(torch.randint(0, len(t) - 6, ()))
                t[st : st + 6, 0] += 3.0
            out.append(t)
        return out

    P, N, NP = seqs(100, True), seqs(140, False), seqs(16, True)  # ~ our real refit sizes
    for kind in ("rlobf_attn", "rlobf_attn_max", "rlobf_max", "gdm_rollattn"):
        mon, info = gate_fit(kind, P, N, NP, cfg, name=f"lit_{kind}", layer=1, concept="cats")
        assert mon is not None, info
        assert mon.needs_tokens and mon.logits(NP[:3]).shape == (3,)
    mon, info = gate_fit("rlobf_max", seqs(100, False), N, seqs(16, False), cfg, name="lit_noise",
                         layer=1, concept="cats")
    assert mon is None and info["auroc"] < 0.8


def test_training_with_token_pooled_refits(tmp_path, monkeypatch):
    """Training path for the probe bank (10-14b): in-episode refits of a token-pooled kind, scored
    by the defender on per-token rollout states, and a hide loss that flows through the token
    monitor (single-pass forward with token_out)."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    def specs(n):
        return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa bb", "cc dd", "ee ff"],
                            "cats", "cats", base_name, feedback="verdict") for k in range(n)]

    class FakeSampler:
        def sample(self, n):
            return specs(n)

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=3, micro_batch=4, lr=1e-2, warmup_steps=1, save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, refit_cv_folds=2, episode_cumulative=True,
                                single_pass=True, incremental_rollout=True,
                                episode_refit_kind="maxpool",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}),
    )
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k} purr" for k in range(12)]),
                         ConceptTexts("dogs", "train", [f"woof {k} bark" for k in range(12)])],
        branch_negatives=["a b c", "c d e", "e f g", "g h i"],
    )
    orig = AdaptiveDefender.score
    calls = {}

    def score(self, name, pooled, row):  # base passes once; every refit then really scores
        k = (id(self), name)
        calls[k] = calls.get(k, 0) + 1
        out = orig(self, name, pooled, row)
        return {**out, "detected": False} if name == base_name else out

    monkeypatch.setattr(AdaptiveDefender, "score", score)
    _, m = tr.rollout(specs(2))
    assert m["ep_refits"] >= 2 and m["ep_refits_accepted"] >= 2
    kinds = {type(tr.pool.get(n)).__name__ for n in tr._transient}
    assert kinds == {"TokenPoolMonitor"}
    assert tr.pool.token_layers == [1]
    for e in list(tr.pool.entries):
        if e.monitor.name in tr._transient:
            tr.pool.entries.remove(e)
    tr._transient = []
    rec = tr.step()  # rollout (token refits) + hide loss through the token monitors
    assert rec["train/n_hide"] > 0 and rec["grad_norm"] == rec["grad_norm"]


def test_probe_bank_mckenzie_and_token_layer_ensembles():
    """McKenzie probes (attention-lite, softmax T=5, rolling-mean-40) and the exact token-level
    multi-layer ensembles (NC 8 layers, Bailey all layers): fit + gate on synthetic answers."""
    from next_chameleons.tica.probe_zoo import TOKENS_ENS_LAYERS, gate_fit

    torch.manual_seed(0)
    cfg = AdaptiveConfig(refit_min_auroc=0.9, refit_min_tpr_np=0.5, refit_C=1.0, refit_cv_folds=3)
    d = 16

    def seq(signal):
        t = torch.randn(int(torch.randint(45, 60, ())), d)
        if signal:
            t[:, 0] += 0.8
            st = int(torch.randint(0, len(t) - 6, ()))
            t[st : st + 6, 0] += 3.0
        return t

    P = [seq(True) for _ in range(100)]
    N = [seq(False) for _ in range(140)]
    NP = [seq(True) for _ in range(16)]
    for kind in ("mck_attnlite", "mck_softmax", "mck_rollmean40"):
        mon, info = gate_fit(kind, P, N, NP, cfg, name=f"b_{kind}", layer=1, concept="cats")
        assert mon is not None and mon.logits(NP[:2]).shape == (2,), info
    for kind in ("nc_ens8_tok", "bailey_ens_tok"):
        Ls = TOKENS_ENS_LAYERS[kind]
        Pd = {L: [seq(True) for _ in range(60)] for L in Ls}
        Nd = {L: [seq(False) for _ in range(80)] for L in Ls}
        Gd = {L: [seq(True) for _ in range(12)] for L in Ls}
        mon, info = gate_fit(kind, Pd, Nd, Gd, cfg, name=f"b_{kind}", layer=12, concept="cats")
        assert mon is not None and mon.needs_tokens and mon.layers_used == sorted(Ls), info
        assert mon.logits({L: v[:2] for L, v in Gd.items()}).shape == (2,)


def test_gdm_many_matches_separate_fits():
    """_fit_gdm_many trains each model exactly as _fit_gdm would with the same init and split."""

    import torch

    from next_chameleons.tica import probe_zoo as z

    torch.manual_seed(0)
    P = [torch.randn(int(n), 16) + 0.5 for n in torch.randint(5, 20, (12,))]
    N = [torch.randn(int(n), 16) for n in torch.randint(5, 20, (12,))]
    keep = torch.ones(12, dtype=torch.bool)
    keep[:3] = False
    nets = z._fit_gdm_many(P, N, torch.Generator().manual_seed(3), torch.device("cpu"),
                           [keep, torch.ones(12, dtype=torch.bool)], steps=25)
    # replay model 0 alone: same split, same init draw, same subset
    g = torch.Generator().manual_seed(3)
    X, M = z._pad(list(P) + list(N), torch.zeros(1), torch.ones(1), "cpu")
    perm = torch.randperm(24, generator=g)
    tr = perm[max(2, round(0.15 * 24)) :]
    tr = tr[(tr >= 12) | keep[tr.clamp(max=11)]]
    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
    net = z._GDMRollAttn(16)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=3e-3)
    yy = torch.tensor([1.0] * 12 + [0.0] * 12)
    for _ in range(25):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(X[tr], M[tr]), yy[tr]).backward()
        opt.step()
    with torch.no_grad():
        a, b = nets[0](X, M), net(X, M)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


# --- final multi-concept dodger (TICA-FINAL-DODGER-DESIGN.md) ---------------------------------


def _final_yaml(tmp_path, final: dict, extra: str = "") -> str:
    import json as _json

    split = {"train": ["a", "b", "c"], "heldout": ["x"], "heldout_far": ["y"]}
    (tmp_path / "split.json").write_text(_json.dumps(split))
    fin = "\n".join(f"  {k}: {_json.dumps(v)}" for k, v in final.items())
    (tmp_path / "f.yaml").write_text(
        "name: f\nmodel: {path: /x}\nprobes: {dir: /y}\nout: /z\n"
        f"final:\n  split: split.json\n{fin}\n{extra}")
    return str(tmp_path / "f.yaml")


def test_final_split_resolves_rung_concepts(tmp_path):
    import pytest

    from next_chameleons.tica.config import load_run_config

    cfg = load_run_config(_final_yaml(tmp_path, {"n_train_concepts": 2}))
    assert cfg.episode_concepts == ["a", "b"]  # the rung = first N train concepts
    assert cfg.final.split == str(tmp_path / "split.json")  # relative to the YAML
    assert load_run_config(_final_yaml(tmp_path, {})).episode_concepts == ["a", "b", "c"]
    with pytest.raises(ValueError):  # an explicit concept list that disagrees with the split
        load_run_config(_final_yaml(tmp_path, {"n_train_concepts": 2}, "episode_concepts: [x]\n"))


def test_final_probe_kinds_disjoint(tmp_path):
    import pytest

    from next_chameleons.tica.config import load_run_config
    from next_chameleons.tica.multi import check_probe_split

    with pytest.raises(ValueError):
        load_run_config(_final_yaml(tmp_path, {"train_refit_kinds": ["linear", "mlp"],
                                               "heldout_refit_kinds": ["mlp"]}))
    check_probe_split(["linear"], ["mlp"], ["linear", "mlp"])
    with pytest.raises(ValueError):
        check_probe_split(["linear"], ["nope"], ["linear", "mlp"])


def test_final_config_kinds_exist():
    """The shipped final configs name only real probe kinds, disjoint train / held-out sets, and a
    split whose held-out concepts are never trained."""

    from next_chameleons.tica.config import load_run_config
    from next_chameleons.tica.multi import check_probe_split, heldout_concepts, load_split
    from next_chameleons.tica.probe_bank import BANK
    from next_chameleons.tica.probe_zoo import ZOO_KINDS

    known = set(BANK) | set(ZOO_KINDS) | {"linear", "mlp", "multilayer"}
    for name in ("final_r1_4b", "final_r2_4b", "final_r3_4b"):
        cfg = load_run_config(name)
        check_probe_split(cfg.final.train_refit_kinds, cfg.final.heldout_refit_kinds, known)
        sp = load_split(cfg.final.split)
        assert not set(cfg.episode_concepts) & set(heldout_concepts(sp))


def test_extra_pos_multi(tmp_path):
    import json as _json

    import pytest

    from next_chameleons.tica.multi import load_extra_pos_multi

    (tmp_path / "a.json").write_text(_json.dumps({"concept": "A", "texts": ["t1", "t2"]}))
    (tmp_path / "bad.json").write_text(_json.dumps({"concept": "Z", "texts": ["t3"]}))
    (tmp_path / "index.json").write_text(_json.dumps({"A": {"file": "a.json", "n": 2},
                                                      "B": {"file": "bad.json", "n": 1}}))
    assert load_extra_pos_multi(str(tmp_path), "A") == ["t1", "t2"]
    assert load_extra_pos_multi(str(tmp_path), "C") == []
    assert load_extra_pos_multi("", "A") == []
    with pytest.raises(ValueError):  # index pointing at another concept's file
        load_extra_pos_multi(str(tmp_path), "B")


def test_anchor_kl():
    from next_chameleons.tica.anchor import anchor_kl

    g = torch.Generator().manual_seed(0)
    base = torch.randn(2, 5, 11, generator=g)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    assert anchor_kl(base, base, mask).abs() < 1e-6  # identical distributions
    pol = base.clone()
    pol[0, 3] += 5.0  # a change on a MASKED-OUT position does not count
    assert anchor_kl(pol, base, mask).abs() < 1e-6
    pol[1, 2] += torch.randn(11, generator=g) * 3
    kl = anchor_kl(pol, base, mask)
    assert kl > 0 and anchor_kl(pol, base, mask, reverse=True) > 0
    pol.requires_grad_(True)
    anchor_kl(pol, base, mask).backward()
    assert pol.grad[0, 3].abs().sum() == 0 and pol.grad[1, 2].abs().sum() > 0


def test_anchor_prompts(tmp_path):
    import json as _json

    from next_chameleons.tica.anchor import load_anchor_prompts

    (tmp_path / "d.jsonl").write_text(
        "\n".join(_json.dumps({"instruction": f"i{k}", "context": ""}) for k in range(5)))
    (tmp_path / "g.json").write_text(_json.dumps([{"question": f"q{k}", "answer": "1"}
                                                  for k in range(5)]))
    ps = load_anchor_prompts([str(tmp_path / "d.jsonl"), str(tmp_path / "g.json")], 6)
    assert len(ps) == 6 and sum(p.startswith("i") for p in ps) == 3  # even over sources


def test_mck_rollmean_ignores_padding():
    """A short answer's score must not depend on how long the other answers in the batch are."""

    import torch

    from next_chameleons.tica import probe_zoo as z

    torch.manual_seed(0)
    net = z._MckNet("mck_rollmean40", 8)
    short, long_ = torch.randn(10, 8), torch.randn(90, 8)
    X1, M1 = z._pad([short, short], torch.zeros(1), torch.ones(1), "cpu")
    X2, M2 = z._pad([short, long_], torch.zeros(1), torch.ones(1), "cpu")
    with torch.no_grad():
        a, b = net(X1, M1)[0], net(X2, M2)[0]
        whole = net.lin(short).squeeze(-1).mean()
        w = net.lin(long_).squeeze(-1).unfold(0, 40, 1).mean(-1).amax()
    assert torch.allclose(a, b) and torch.allclose(a, whole)
    assert torch.allclose(net(X2, M2)[1], w)


def test_switch_schedule_segments_and_returns():
    """Concept-switch schedules (user 10-14): >= 1 switch, every segment >= min_seg turns, each
    switch changes the concept, and returns go to an EARLIER concept of the episode."""
    import itertools
    import random

    from next_chameleons.tica.episodes import switch_schedule

    rng = random.Random(0)
    others = ["b", "c", "d"]
    n_ret = 0
    for _ in range(300):
        seq = switch_schedule(rng, "a", others, 12, 3, 2, 0.5)
        assert len(seq) == 12 and seq[0] == "a"
        segs = [1]
        for x, y in itertools.pairwise(seq):
            if x == y:
                segs[-1] += 1
            else:
                segs.append(1)
        assert 2 <= len(segs) <= 3 and min(segs) >= 3
        starts = [t for t in range(1, 12) if seq[t] != seq[t - 1]]
        n_ret += sum(seq[t] in seq[:t] for t in starts)
    assert n_ret > 0
    assert switch_schedule(rng, "a", others, 5, 3, 2, 0.0) is None  # no room for 2 segments


def test_kind_weights_prioritize_catchers():
    from next_chameleons.tica.adaptive import kind_weights

    w = kind_weights({"linear": [1, 10], "rlobf_max": [9, 10]}, ["linear", "rlobf_max", "x"], 0.3)
    assert abs(sum(w) - 1) < 1e-9 and w[1] > w[2] > w[0] and min(w) >= 0.1


def test_adaptive_switching_episode_and_mismatch_targets(tmp_path):
    """A switching episode announces the switch in the feedback, records the concept per turn,
    and plans its hide targets on that concept; mismatch episodes get MATCH targets on the named
    concept and on the text concept (specificity)."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import MONITOR_SWITCH, EpisodeSpec
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
    order = ["cats", "cats", "dogs", "dogs", "cats", "cats"]

    class FakeSampler:
        def sample(self, n):
            sw = EpisodeSpec("real_score|cats|0", "real_score", ["aa", "bb", "cc", "dd", "ee",
                             "ff"], "cats", "cats", "concept_L1",
                             meta={"monitored_turns": order})
            mm = EpisodeSpec("mismatch|cats|1", "mismatch", ["aa", "bb", "cc", "dd", "ee", "ff"],
                             "cats", "dogs", "concept_L1", probes={"text": ("concept_L1", "cats")})
            return [sw, mm]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=6, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_every=0, pursuit=True, pursuit_from_turn=1,
                                motion="conditional", stay_offset=True),
    )
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path)
    trajs, _ = tr.rollout(tr.sampler.sample(2))
    assert [r["concept"] for r in trajs[0].results] == order
    assert MONITOR_SWITCH.format(c="dogs") in trajs[0].feedback[1]
    assert MONITOR_SWITCH.format(c="cats") in trajs[0].feedback[3]
    assert not any("monitor checks" in f for f in (trajs[0].feedback[0], trajs[0].feedback[2]))
    tg = tr._targets(trajs)
    assert [t.concept for t in tg if t.traj == 0] == order
    mm = [(t.key, t.concept, t.role) for t in tg if t.traj == 1]
    assert ("monitored", "dogs", "match") in mm and ("text", "cats", "match") in mm
    rec = tr.step()
    assert rec["n_targets"] > 0


def test_cap_gen_step_trains_on_correct_generations(tmp_path, monkeypatch):
    """On-policy capability term: forks real episodes, keeps only correct generations WITH working,
    and backpropagates NLL (+ hide on the episode's earlier probes)."""
    import sys

    sys.path.insert(0, "tests")
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers import mathrl
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    texts = ["aa", "bb", "cc", "dd"]
    inputs = {text_id(x): f"req {x}" for x in texts}

    def opening(cond, text, concept, named=True):
        return inputs[text_id(text)]

    def render(cond, concept, shown, nxt, *, feedback="score", named=True, notice=None):
        return f"verdict on {concept}\n\n{inputs[text_id(nxt)]}"

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", texts, "cats", "cats",
                                "concept_L1") for k in range(n)]

    q = [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}]
    monkeypatch.setattr(mathrl, "load_questions", lambda: (q * 10, []))
    calls = []

    def fake_generate(convs, max_new, **kw):
        calls.append(len(convs))
        out = []
        for j, c in enumerate(convs):
            gold = "2" if "1+1" in c[-1]["content"] else "4"
            text = (f"x = 1 + 1 = {gold}. The answer is {gold}." if j % 2 == 0
                    else "I am not sure. The answer is 7.")
            out.append({"text": text, "finished": True, "ids": [1], "prompt_ids": [1],
                        "stop": None})
        return out

    monkeypatch.setattr(pol, "generate", fake_generate)
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=4, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_every=0, cap_gen_weight=1.0, cap_gen_episodes=2,
                                cap_gen_samples=2, cap_gen_skip=0))
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path,
                                   prompts=(opening, render), inputs=inputs)
    rec = tr.step()
    assert calls == [4]  # one batched generate call: 2 forks x 2 samples
    assert rec["cap_gen/n_forks"] == 2 and rec["cap_gen/acc"] == 0.5
    assert rec["cap_gen/n_train"] == 2 and rec["cap_gen/msg_found"] == 1.0
    assert rec["cap_gen/fork_solved"] == 1.0 and rec["cap_gen/n_fallback"] == 0
