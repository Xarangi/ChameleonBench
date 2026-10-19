"""Turn budget (final.turn_budget, REPLICATION 2026-10-19b): switch points follow the race. Every
probe gets >= R judged turns of runway and stands <= M; a switching episode's first switch comes by
T - S; switches only with >= S turns left; once no new probe fits the rest is an unjudged tail."""

from __future__ import annotations

import random

from next_chameleons.tica.episodes import budget_action, budget_switch


def _b(seq=("a", "b", "c"), R=5, M=8, S=8):
    return {"R": R, "M": M, "S": S, "seq": list(seq), "k": 0, "pt": 0, "done": False,
            "pools": {}, "used": {}}


def _play(b, outcomes, T=24, off=None):
    """outcomes[t] = passed on turn t (judged turns only). Returns {t: action} and the probes as
    (start, end_exclusive, beaten, judged_turns)."""
    off = off or [False] * T
    acts, probes, start, beaten = {}, [], 0, False
    for t in range(T):
        if b["done"]:
            break
        judged = not off[t]
        passed = judged and outcomes[t]
        beaten = beaten or passed
        act = budget_action(b, t, T, off, judged, passed)
        acts[t] = act
        if act in ("refit", "switch", "end") or t == T - 1:
            probes.append((start, t + 1, beaten, sum(not off[u] for u in range(start, t + 1))))
            start, beaten = t + 1, False
    return acts, probes


def test_always_passing_refits_until_runway_then_switches_then_ends():
    b = _b()
    acts, _ = _play(b, [True] * 24)
    # first segment's edge is T - S = 16: refits while >= 5 turns remain before it
    assert all(acts[t] == "refit" for t in range(11)) and acts[11] == "switch"
    assert all(acts[t] == "refit" for t in range(12, 19)) and acts[19] == "end"
    assert b["done"] and b["k"] == 1


def test_always_caught_loses_after_m_and_switches_with_room():
    b = _b()
    acts, _ = _play(b, [False] * 24)
    assert acts[7] == "switch" and acts[15] == "switch"  # loss after 8 judged turns, twice
    assert b["k"] == 2 and acts[22] == "continue" and acts.get(23) is None


def test_forced_first_switch_at_t_minus_s():
    b = _b(M=50)  # no loss: only the forced edge can end the first segment
    out = [False] * 24
    out[7] = True  # refit at 8 (8 judged turns to the edge), never beaten
    acts, probes = _play(b, out)
    assert acts[7] == "refit" and acts[15] == "switch"
    assert probes[1] == (8, 16, False, 8)


def test_no_switch_episode_ends_on_loss():
    b = _b(seq=("a",))
    acts, _ = _play(b, [False] * 24)
    assert acts[7] == "end" and b["done"] and 8 not in acts


def test_runway_counts_judged_turns_only():
    off = [False] * 24
    for u in (12, 13, 14):
        off[u] = True
    b = _b(M=50)
    out = [False] * 24
    out[9] = True  # turns 10..15 are 6 turns but only 3 judged before the edge
    acts, _ = _play(b, out, off=off)
    assert acts[9] == "switch"


def test_invariants_random_races():
    rng = random.Random(0)
    for trial in range(3000):
        T = rng.choice([16, 20, 24, 32])
        R, M, S = rng.choice([(5, 8, 8), (3, 5, 6), (5, 9, 8), (2, 4, 4)])
        seq = ["a", "b", "c"][: rng.choice([1, 2, 3])]
        if T - S < R:
            continue
        p = rng.random()
        off = [t > 0 and rng.random() < 0.15 for t in range(T)]
        b = _b(seq, R, M, S)
        acts, probes = _play(b, [rng.random() < p for _ in range(T)], T, off)
        switches = [t for t, a in acts.items() if a == "switch"]
        for start, _end, beaten, judged in probes:
            assert judged <= M
            # a REFIT that was never beaten had its full runway (segment-opening probes are
            # covered by the switch rule: >= S turns left)
            if not beaten and start > 0 and acts[start - 1] == "refit":
                assert judged >= min(R, sum(not off[u] for u in range(start, T))), (trial, probes)
        for t in switches:
            assert T - (t + 1) >= S
        if len(seq) > 1:
            assert switches and switches[0] + 1 <= T - S, (trial, acts)
        assert len(switches) <= len(seq) - 1


class _Pool:
    def trainable(self, c):
        return True

    def sample(self, rng, c):
        from types import SimpleNamespace

        return SimpleNamespace(monitor=SimpleNamespace(name=f"mon_{c}"))

    def concept_monitors(self, c):
        return []


def test_sampler_budget_episodes_and_fresh_texts_on_switch():
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSampler

    cs = [ConceptTexts(c, "train", [f"{c}{k}" for k in range(60)]) for c in "xyz"]
    smp = EpisodeSampler(cs, ["n"], _Pool(), mix={"real_score": 1.0}, turns=12, seed=1,
                         switch_prob=1.0, switch_max=2, switch_return_prob=0.5,
                         budget={"R": 3, "M": 4, "S": 4})
    for s in smp.sample(20):
        b = s.budget
        assert len(b["seq"]) == 3 and s.meta["monitored_turns"] == [b["seq"][0]] * 12
        assert len(s.texts) == 12 and all(x.startswith(b["seq"][0]) for x in s.texts)
        b["k"] = 1
        c = budget_switch(s, 5)
        assert s.meta["monitored_turns"][5:] == [c] * 7 and s.meta["monitored_turns"][4] != c
        assert all(x.startswith(c) for x in s.texts[5:])
        b["k"] = 2
        c2 = budget_switch(s, 9)
        shown = s.texts[:9] + s.texts[9:]
        assert all(x.startswith(c2) for x in s.texts[9:])
        assert len(set(shown)) == 12  # never a repeated text, also on a return
    smp1 = EpisodeSampler(cs, ["n"], _Pool(), mix={"real_score": 1.0}, turns=12, seed=1,
                          budget={"R": 3, "M": 4, "S": 4})
    assert all(len(s.budget["seq"]) == 1 for s in smp1.sample(5))  # switch_prob 0: no switch
    smp2 = EpisodeSampler(cs, ["n"], _Pool(), mix={"nobrief": 1.0}, turns=12, seed=1,
                          budget={"R": 3, "M": 4, "S": 4})
    assert all(s.budget is None for s in smp2.sample(5))  # only monitored episodes


def test_v4_config_uses_the_budget():
    from next_chameleons.tica.config import load_run_config

    cfg = load_run_config("final_v4_4b")
    assert cfg.final.turn_budget and cfg.train.turns == 24
    assert cfg.sampler_kwargs()["budget"] == {"R": 5, "M": 8, "S": 8}


def _trainer(tmp_path, turns):
    import sys

    sys.path.insert(0, "tests")
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.episodes import ConceptTexts
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=turns, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        objective=ObjectiveConfig(pin_first_turn=False, pin_segment_first=False,
                                  hide_from_turn=0),
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, first_probe="switch_fork", off_turn_prob=0.0,
                                episode_refit_kind="linear@2",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, None, cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    return tr, pool.entries[0].monitor.name


def _budget_spec(base_name, T, seq, R=2, M=3, S=3, k=0):
    from next_chameleons.tica.episodes import EpisodeSpec

    pools = {c: [f"{c} {j}" for j in range(2 * T)] for c in seq}
    b = {"R": R, "M": M, "S": S, "seq": list(seq), "k": 0, "pt": 0, "done": False,
         "pools": pools, "used": {c: (T if c == seq[0] else 0) for c in seq}}
    return EpisodeSpec(f"real_score|{seq[0]}|{k}", "real_score", list(pools[seq[0]][:T]), seq[0],
                       seq[0], base_name, meta={"monitored_turns": [seq[0]] * T}, budget=b)


def test_rollout_with_turn_budget(tmp_path):
    T = 8
    tr, base = _trainer(tmp_path, T)
    specs = [_budget_spec(base, T, ["cats", "dogs"], k=0), _budget_spec(base, T, ["dogs"], k=1)]
    trajs, m = tr.rollout(specs)
    sw = trajs[0]
    mt = sw.spec.meta["monitored_turns"]
    first = next(t for t in range(1, T) if mt[t] != mt[t - 1])
    assert first <= T - 3  # the guaranteed switch, by T - S
    assert all(sw.results[t]["concept"] == mt[t] for t in range(T) if not sw.results[t].get("off"))
    assert all(sw.spec.texts[t].startswith(mt[t]) for t in range(T))  # texts follow the concept
    tails = [(i, t) for i, x in enumerate(trajs) for t, r in enumerate(x.results) if r.get("tail")]
    assert all(trajs[i].results[t]["off"] for i, t in tails)
    assert "budget_tail_frac" in m and m["budget_tail_frac"] == len(tails) / (2 * T)
    tgt = {(g.traj, g.turn) for g in tr._targets(trajs)}
    assert not tgt & set(tails)  # no loss on the tail
    for i, x in enumerate(trajs):  # once a tail starts it runs to the end
        ts = [t for t, r in enumerate(x.results) if r.get("tail")]
        assert ts == list(range(ts[0], T)) if ts else True


def test_rollout_free_answers(tmp_path):
    T = 3
    tr, base = _trainer(tmp_path, T)
    tr.free_answers = {"max_new": 4}
    spec = _budget_spec(base, T, ["cats"])
    spec.budget = None
    trajs, _ = tr.rollout([spec])
    answers = [trajs[0].convs[t][-1]["content"] for t in range(T)]
    assert all(a != f"cats {t}" for t, a in enumerate(answers))  # the model's own, not forced
    assert all("gen_finished" in r for r in trajs[0].results)
