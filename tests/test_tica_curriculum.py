"""Adaptive defender curriculum (REPLICATION 2026-10-16g): discounted Thompson sampling over probe
kinds and concepts on turns-to-beat records, with floor / cap."""

from __future__ import annotations

import random
import sys

sys.path.insert(0, "tests")


def test_cap_probs_bounds_and_sum():
    from next_chameleons.tica.curriculum import cap_probs

    p = cap_probs({"a": 0.7, "b": 0.2, "c": 0.1}, cap=0.4)
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p.values()) <= 0.4 + 1e-9
    assert p["b"] > 0.2 and p["c"] > 0.1  # excess shared out, nobody lowered


def test_hazard_update_decay_and_bounds():
    """Turns-to-beat hazard (10-16l): beaten in n turns -> (1 beat, n turns); censored after n ->
    (0, n); counts decay by gamma; harder arms (fewer beats per turn) get more mass, within
    [floor / K, cap]."""
    from next_chameleons.tica.curriculum import DiscountedThompson

    b = DiscountedThompson(["easy", "hard", "x", "y"], gamma=0.5, floor=0.2, cap=0.5)
    recs = ([{"kind": "easy", "turns": 1, "beaten": True}] * 20
            + [{"kind": "hard", "turns": 6, "beaten": True}] * 5
            + [{"kind": "hard", "turns": 10, "beaten": False}] * 5
            + [{"kind": "x", "turns": 2, "beaten": False}])  # short censored record still counts
    b.update(recs)
    assert (b.B["easy"], b.N["easy"]) == (20, 20) and (b.B["hard"], b.N["hard"]) == (5, 80)
    assert (b.B["x"], b.N["x"]) == (0, 2)
    b.update([])
    assert b.N["hard"] == 40  # decayed by gamma
    rng = random.Random(0)
    ps = [b.probs(rng) for _ in range(300)]
    for p in ps:
        assert abs(sum(p.values()) - 1) < 1e-9
        assert all(0.05 - 1e-9 <= v <= 0.5 + 1e-9 for v in p.values())  # floor / K, cap
    mean = {k: sum(p[k] for p in ps) / len(ps) for k in b.keys}
    assert mean["hard"] > mean["easy"]
    assert b.sample(rng, exclude=["easy", "hard", "x", "y"]) is None


def test_factored_and_hierarchical_bandits():
    from next_chameleons.tica.curriculum import FactoredThompson, HierarchicalThompson

    rng = random.Random(1)
    f = FactoredThompson(["linear", "maxpool"], ["ens4_linear"], [8, 12, 16])
    f.update([{"kind": "maxpool@16", "turns": 9, "beaten": False},
              {"kind": "linear@8", "turns": 1, "beaten": True},
              {"kind": "ens4_linear", "turns": 3, "beaten": True}])
    assert f.arch.N["maxpool"] == 9 and f.layer.N[16] == 9 and f.arch.N["ens4_linear"] == 3
    ks = {f.sample(rng) for _ in range(200)}
    assert "ens4_linear" in ks and all(k == "ens4_linear" or "@" in k for k in ks)
    assert f.sample(rng, exclude=["ens4_linear"] + [f"{a}@{L}" for a in ("linear", "maxpool")
                                                    for L in (8, 12, 16)]) is None
    h = HierarchicalThompson({"sports": "topic", "music": "topic", "German": "language",
                              "math": "domain"})
    h.update([{"concept": "German", "turns": 10, "beaten": False}])
    p = h.probs(rng)
    assert abs(sum(p.values()) - 1) < 1e-9 and set(p) == {"sports", "music", "German", "math"}


def test_trainer_thompson_kinds_and_concepts(tmp_path, monkeypatch):
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
        concepts = [ConceptTexts("cats", "train", ["x"]), ConceptTexts("dogs", "train", ["y"])]  # noqa: RUF012
        concept_weights = None

        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", ["aa", "bb", "cc", "dd"],
                                "cats", "cats", base_name) for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=4, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_refit_kind="mixed",
                                episode_refit_kinds=["linear", "meandiff"],
                                episode_kind_sampling="thompson", concept_sampling="thompson",
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    sampler = FakeSampler()
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(8)])],
        branch_negatives=["a b", "c d", "e f"])
    assert tr.kind_bandit is not None and tr.concept_bandit is not None
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    _, m = tr.rollout(sampler.sample(1))
    assert any(k.startswith("kind_p_") for k in m) and "concept_p_cats" in m
    assert set(sampler.concept_weights) == {"cats", "dogs"}
    assert tr.seg_log and tr.seg_log[0]["concept"] == "cats"


def test_weighted_switch_schedule_default_unchanged():
    from next_chameleons.tica.episodes import switch_schedule

    a = switch_schedule(random.Random(3), "a", ["b", "c", "d"], 12, 3, 2, 0.0)
    b = switch_schedule(random.Random(3), "a", ["b", "c", "d"], 12, 3, 2, 0.0, weights=None)
    assert a == b
    c = switch_schedule(random.Random(3), "a", ["b", "c", "d"], 12, 3, 2, 0.0,
                        weights={"b": 0.0, "c": 0.0, "d": 1.0})
    assert set(c) <= {"a", "d"}


def test_early_stop_is_lenient():
    from next_chameleons.tica.trainers.supervised import early_stop_decision

    rising = {20 * k: 0.1 * k for k in range(1, 11)}
    assert not early_stop_decision(rising, 200, 3, 5, 0.01, 100)["stop"]
    flat = {**{20 * k: 0.1 * k for k in range(1, 6)}, **{20 * k: 0.5 for k in range(6, 13)}}
    d = early_stop_decision(flat, 240, 3, 5, 0.01, 100)
    assert d["stop"] and d["best_step"] <= 140
    assert not early_stop_decision(flat, 90, 3, 5, 0.01, 100)["stop"]  # min_steps
    short = {20 * k: v for k, v in enumerate([0.5, 0.4, 0.4, 0.4], 1)}
    assert not early_stop_decision(short, 200, 3, 5, 0.01, 100)["stop"]  # < patience checks
    assert early_stop_decision({}, 300, 3, 5, 0.01, 100)["stop"] is False


def test_val_score_family_mean(tmp_path):
    import json
    import subprocess

    race = {"deploy_records": [{"concept": "a", "turns": 1, "beaten": True},
                               {"concept": "a", "turns": 5, "beaten": True},
                               {"concept": "g", "turns": 2, "beaten": False}],
            "pass_rate_after_first_turn": 0.3}
    (tmp_path / "r.json").write_text(json.dumps(race))
    (tmp_path / "s.json").write_text(json.dumps({"families": {"g": "language"}}))
    subprocess.run(["python3", "scripts/tica_val_score.py", "--race", str(tmp_path / "r.json"),
                    "--split", str(tmp_path / "s.json"), "--step", "40",
                    "--out", str(tmp_path / "o.json")], check=True, capture_output=True)
    o = json.loads((tmp_path / "o.json").read_text())
    assert o["step"] == 40 and abs(o["val_score"] - 0.25) < 1e-9  # mean(topic 0.5, language 0)


def test_lr_peak_then_decay_and_default():
    """10-17c: optional warmup -> lr_peak -> cosine to lr over GLOBAL steps; default unchanged."""
    from next_chameleons.tica.trainers.supervised import SupervisedConfig, lr_factor

    d = SupervisedConfig(lr=1e-4, warmup_steps=10)
    assert [lr_factor(d, k) for k in (0, 9, 50)] == [0.1, 1.0, 1.0]
    c = SupervisedConfig(lr=1e-4, warmup_steps=10, lr_peak=3e-4, lr_decay_steps=100)
    assert abs(lr_factor(c, 0) - 0.1 * 3.0) < 1e-9  # warming towards the peak
    assert abs(lr_factor(c, 50) - 2.0) < 1e-9  # halfway: (3 + 1) / 2
    assert abs(lr_factor(c, 100) - 1.0) < 1e-9 and abs(lr_factor(c, 300) - 1.0) < 1e-9
    # a resume chunk at global step 50 re-warms towards the global-step target, not the peak
    assert abs(lr_factor(c, 9, offset=50) - lr_factor(c, 59)) < 1e-9
    assert lr_factor(c, 0, offset=50) < lr_factor(c, 9, offset=50)


def test_curriculum_ramp_prior_and_uniform_kind():
    """10-17g: the adaptive weights are mixed with a uniform prior, lam ramping 0 -> mix_max."""
    from next_chameleons.tica.curriculum import FactoredThompson
    from next_chameleons.tica.trainers.supervised import curriculum_prior, uniform_kind

    fam = {"a": "topic", "b": "topic", "c": "topic", "d": "format"}
    pc, pf = curriculum_prior(fam, "concept"), curriculum_prior(fam, "family")
    assert pc == {k: 0.25 for k in fam}
    assert pf["d"] == 0.5 and abs(pf["a"] - 1 / 6) < 1e-12
    f = FactoredThompson(["linear", "maxpool"], ["ens4_linear"], [8, 12])
    rng = random.Random(0)
    ks = [uniform_kind(f, rng) for _ in range(3000)]
    assert abs(ks.count("ens4_linear") / 3000 - 1 / 3) < 0.04  # architecture first
    assert uniform_kind(f, rng, exclude=["ens4_linear", "linear@8", "linear@12", "maxpool@8",
                                         "maxpool@12"]) is None


def test_trainer_curriculum_lambda_ramps(tmp_path, monkeypatch):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer

    class T:
        _step_offset = staticmethod(lambda: 0)

    t = T()
    for ramp, mx, step, want in [(0, 1.0, 0, 1.0), (100, 0.5, 0, 0.0), (100, 0.5, 50, 0.25),
                                 (100, 0.5, 400, 0.5)]:
        t.cfg = type("C", (), {"adaptive": AdaptiveConfig(curriculum_ramp_steps=ramp,
                                                          curriculum_mix_max=mx)})()
        t.step_k = step
        assert abs(AdaptiveSupervisedTrainer._curriculum_lambda(t) - want) < 1e-12
    # delay: uniform for 50 steps, then ramp to 0.7 over 100
    for step, want in [(0, 0.0), (49, 0.0), (100, 0.35), (150, 0.7), (400, 0.7)]:
        t.cfg = type("C", (), {"adaptive": AdaptiveConfig(
            curriculum_ramp_steps=100, curriculum_mix_max=0.7, curriculum_delay_steps=50)})()
        t.step_k = step
        assert abs(AdaptiveSupervisedTrainer._curriculum_lambda(t) - want) < 1e-12


def test_flat_concept_bandit(tmp_path):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    fam = tmp_path / "split.json"
    fam.write_text('{"families": {"cats": "topic", "dogs": "animal", "fish": "animal"}}')
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs", "fish"]), round=0, family="concept_linear")
    sampler = type("S", (), {"concepts": [type("C", (), {"concept": c})()
                                          for c in ("cats", "dogs", "fish")]})()
    for hier, nfam in ((True, 2), (False, 1)):
        cfg = AdaptiveSupervisedConfig(steps=1, save_every=0, adaptive=AdaptiveConfig(
            refit_every=0, concept_sampling="thompson", concept_families=str(fam),
            concept_hierarchy=hier))
        tr = AdaptiveSupervisedTrainer(tiny_policy(), pool, sampler, cfg, tmp_path / str(hier))
        assert len(tr.concept_bandit.within) == nfam


def test_off_match_weight_separates_off_turns():
    """10-17o: off-turn match targets ("off") get their own mean and weight; 0 = legacy."""
    import torch

    from next_chameleons.tica.objectives import ObjectiveConfig, loss_terms

    p = torch.tensor([0.2, 0.9, 0.5], requires_grad=True)
    ref = torch.tensor([1.0, 1.0, 1.0])
    cfg = ObjectiveConfig(hide_weight=0.0, match_weight=2.0, lm_weight=0.0, off_match_weight=5.0)
    tot, st = loss_terms(p, ref, ["off", "match", "match"], None, cfg)
    want = 5.0 * 0.8 ** 2 / 1 + 2.0 * (0.1 ** 2 + 0.5 ** 2) / 2
    assert abs(float(tot) - want) < 1e-6 and st["n_off"] == 1 and st["n_match"] == 2
    legacy = ObjectiveConfig(hide_weight=0.0, match_weight=2.0, lm_weight=0.0)
    tot2, _ = loss_terms(p, ref, ["match", "match", "match"], None, legacy)
    assert abs(float(tot2) - 2.0 * (0.64 + 0.01 + 0.25) / 3) < 1e-6


def test_probe_index_metrics():
    """10-17u: pass rate by probe index within each concept segment."""
    from types import SimpleNamespace as NS

    from next_chameleons.tica.trainers.supervised import probe_index_metrics

    def res(c, dep, det, off=False):
        return {"concept": c, "deployed": ["p"] * dep, "monitored": {"detected": det}, "off": off}

    # cats: bank 1 beaten at turn 1 (2 turns), bank 2 never; switch to dogs: bank 1 beaten at once
    a = NS(spec=NS(cond="real_score", monitored="cats"),
           results=[res("cats", 1, True), res("cats", 1, False), res("cats", 2, True),
                    res("cats", 2, True, off=True), res("dogs", 1, False), res("dogs", 2, True)])
    b = NS(spec=NS(cond="nobrief", monitored="cats"), results=[res("cats", 1, False)])
    m = probe_index_metrics([a, b])
    assert m["beatk_n_1"] == 2 and m["beatk_frac_1"] == 1.0 and m["beatk_turns_1"] == 1.5
    assert m["beatk_n_2"] == 2 and m["beatk_frac_2"] == 0.0
    assert m["beat1_first_frac"] == 1.0 and m["beat1_switch_frac"] == 1.0


def test_relative_cap_and_power_curriculum():
    from next_chameleons.tica.curriculum import DiscountedThompson, HierarchicalThompson

    rng = random.Random(0)
    recs = [{"kind": "hard", "turns": 40, "beaten": False}] + [
        {"kind": k, "turns": 1, "beaten": True} for k in ("a", "b", "c") for _ in range(30)]
    old = DiscountedThompson(["hard", "a", "b", "c"], floor=0.2, cap=0.3)
    new = DiscountedThompson(["hard", "a", "b", "c"], floor=0.2, cap=0.3, cap_rel=2.0)
    old.update(recs)
    new.update(recs)
    assert abs(old.probs(rng)["hard"] - 0.3) < 1e-9  # the absolute cap binds (uniform 0.25)
    assert abs(new.probs(rng)["hard"] - 0.5) < 1e-9  # 2x uniform
    flat = DiscountedThompson(["x", "y"], power=1.0)
    sharp = DiscountedThompson(["x", "y"], power=2.0, cap_rel=2.0)
    flat.cap = 1.0
    for b in (flat, sharp):
        b.update([{"kind": "x", "turns": 4, "beaten": True}, {"kind": "y", "turns": 1,
                                                             "beaten": True}] * 20)
    px = [b.probs(random.Random(3))["x"] for b in (flat, sharp)]
    assert px[1] > px[0] > 0.5
    h = HierarchicalThompson({"s": "topic", "m": "topic", "G": "language"},
                             family_kw={"cap_rel": 1.5}, cap_rel=2.0)
    assert h.fam.cap_rel == 1.5 and h.within["topic"].cap_rel == 2.0
    assert DiscountedThompson(["a"]).cap_rel == 0.0  # default off


def test_trainer_state_roundtrip(tmp_path, monkeypatch):
    """save_state: Adam moments, bandits and RNG go to the newest checkpoint only; a resumed
    trainer restores them and skips the LR re-warmup."""
    import torch

    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    sampler = type("S", (), {"concepts": [type("C", (), {"concept": c})() for c in ("cats",
                                                                                   "dogs")],
                             "rng": random.Random(5)})()
    cfg = AdaptiveSupervisedConfig(steps=1, save_every=0, lr=1e-3, warmup_steps=10,
                                   save_state=True, adaptive=AdaptiveConfig(
                                       refit_every=0, concept_sampling="thompson"))
    pol = tiny_policy()
    tr = AdaptiveSupervisedTrainer(pol, pool, sampler, cfg, tmp_path / "a")
    loss = sum(p.sum() for p in tr.params)
    loss.backward()
    tr.opt.step()
    tr.concept_bandit.update([{"concept": "cats", "turns": 7, "beaten": False}])
    tr.rng.random()
    for d in ("step_20", "step_40"):
        (tr.out / d).mkdir()
        tr._save_state(tr.out / d)
    assert not (tr.out / "step_20" / "trainer_state.pt").exists()  # newest only
    path = tr.out / "step_40" / "trainer_state.pt"
    want = tr.rng.random()
    tr.rng.setstate(torch.load(path, weights_only=False)["rng"])
    tr2 = AdaptiveSupervisedTrainer(pol, pool, sampler, cfg, tmp_path / "b", resume_state=path)
    assert tr2.rng.random() == want
    assert tr2.concept_bandit.within["topic"].N["cats"] == 7.0
    assert len(tr2.opt.state) == len(tr.opt.state) > 0
    assert abs(tr2.sched.get_last_lr()[0] - 1e-3) < 1e-12  # no re-warmup
    tr3 = AdaptiveSupervisedTrainer(pol, pool, sampler, cfg, tmp_path / "c",
                                    resume_state=tmp_path / "missing.pt")
    assert abs(tr3.sched.get_last_lr()[0] - 1e-4) < 1e-12  # fresh: warmup step 1/10
