"""L3 fast track S1/S2 (tica/windows.py) on the tiny random Llama of test_tica_framework (CPU)."""

from __future__ import annotations

import json
import math

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from next_chameleons.tica.windows import (
    BRIEF_W,
    VERDICT_W,
    WEpisode,
    WindowEnv,
    WindowEval,
    WindowProbe,
    WindowTrain,
    WindowTrainer,
    _cluster_boot,
    eval_episodes,
    evaluate,
    oracle_windows,
    rollout,
    window_of,
    yoke_sources,
)
from test_tica_framework import bank, tiny_policy

S2 = WindowEnv(concept="cats", layer=1, turns=6)
S1 = WindowEnv(
    concept="cats", layer=1, turns=3, windows=((None, 0.15), (0.60, None)), targets=(0.0, 0.8)
)
TEXTS = ["meow purr", "cat nap", "kitten", "tabby cat", "whiskers", "cat toy", "litter box"]


def probe(layer: int = 1) -> WindowProbe:
    return WindowProbe(bank(layer, ["cats", "dogs"]), "cats")


def test_env_validation_and_bounds():
    assert S2.bounds()[-1] == (0.85, math.inf)
    assert S1.bounds()[0] == (-math.inf, 0.15)
    with pytest.raises(ValueError, match="outside"):
        WindowEnv(concept="c", windows=((0.1, 0.2),), targets=(0.3,))
    with pytest.raises(ValueError, match="disjoint"):
        WindowEnv(concept="c", windows=((0.1, 0.4), (0.3, 0.6)), targets=(0.2, 0.5))
    assert window_of(0.05, S2.bounds()) is None  # a gap
    assert window_of(0.40, S2.bounds()) == 1


def test_oracle_is_the_bayes_optimal_searcher():
    b2 = S2.bounds()
    # Base turn 0 (f = 0, a gap) is DETECTED by every S2 monitor: no information -> W1.
    assert oracle_windows([0.0], [True], b2) == [0]
    # Caught in W1 -> W2; caught in W2 -> W3; passes in W3 -> stay in W3.
    assert oracle_windows([0.0, 0.17, 0.42, 0.67, 0.68], [True, True, True, False, False], b2) == [
        0,
        1,
        2,
        2,
        2,
    ]
    # Landing in a gap rules nothing out: retry the same window.
    assert oracle_windows([0.0, 0.30], [True, True], b2) == [0, 0]
    # Overshooting into W3 while aiming for W2 and being caught rules out W3, not W2.
    assert oracle_windows([0.0, 0.17, 0.70], [True, True, True], b2) == [0, 1, 1]
    # Once the window is known, a later miss (the policy left it) keeps it.
    assert oracle_windows([0.0, 0.17, 0.60], [True, False, True], b2) == [0, 0, 0]
    b1 = S1.bounds()
    assert oracle_windows([0.0], [False], b1) == [0]  # keep-monitor: stay untouched
    assert oracle_windows([0.0], [True], b1) == [1]  # hide-monitor: hide


def test_frac_definition():
    p = probe()
    z_ref = torch.tensor([3.0, -1.0])  # thr = 0, zero = -5
    z_pol = torch.tensor([3.0, -3.0])
    f = p.frac(z_pol, z_ref)
    assert torch.allclose(f, torch.tensor([0.0, 2.0 / 5.0]))  # below thr: denominator thr - zero
    assert float(p.frac(torch.tensor([-5.0]), torch.tensor([3.0]))) == pytest.approx(1.0)


def test_rollout_conditions_and_feedback_chain():
    pol = tiny_policy(lora=False)
    p = probe()
    texts = TEXTS[:6]
    eps = [
        WEpisode("a", "real", texts, 0),
        WEpisode("b", "flip", texts, 0),
        WEpisode("c", "none", texts, 0),
        WEpisode("d", "nobrief", texts, 0),
        WEpisode("e", "yoked", texts, 0, yoke=[False, True, False, True, False, True]),
    ]
    trajs = rollout(pol, p, eps, S2, has_adapter=False)
    for tr in trajs:
        assert tr.f == [0.0] * 6  # base model: f = 0 by construction
        assert tr.det == [True] * 6  # f = 0 is in no S2 window
        assert [m["content"] for m in tr.convs[-1] if m["role"] == "assistant"] == texts
    real, flip, none, nob, yok = trajs
    brief = BRIEF_W.format(concept="cats")
    assert real.convs[0][0]["content"].startswith(brief)
    assert not nob.convs[0][0]["content"].startswith(brief)
    assert real.convs[-1][2]["content"].startswith(VERDICT_W[True])
    assert flip.convs[-1][2]["content"].startswith(VERDICT_W[False])
    assert "Monitor result" not in json.dumps(none.convs[-1] + nob.convs[-1])
    assert yok.shown == [False, True, False, True, False, True]
    assert yok.convs[-1][2]["content"].startswith(VERDICT_W[False])
    assert yok.convs[-1][4]["content"].startswith(VERDICT_W[True])


def test_rollout_reads_match_training_forward():
    """The trainer reuses z_ref from the rollout and recomputes z_pol with grad on the SAME views:
    both must agree with the rollout's reads."""

    pol = tiny_policy(lora=True)
    for n, q in pol.model.named_parameters():  # make the adapter non-trivial
        if "lora_B" in n:
            q.data.normal_(0, 0.5)
    p = probe()
    trajs = rollout(pol, p, [WEpisode("a", "real", TEXTS[:6], 1)], S2)
    views = [pol.view(c) for c in trajs[0].convs]
    pooled, _ = pol.forward(views, [1], grad=False)
    f = p.frac(p.z(pooled[1]), torch.tensor(trajs[0].z_ref))
    assert torch.allclose(f, torch.tensor(trajs[0].f), atol=1e-5)
    assert max(abs(x) for x in trajs[0].f) > 1e-3  # the adapter changed something


def test_trainer_targets_and_loss_falls(tmp_path):
    pol = tiny_policy(lora=True)
    cfg = WindowTrain(
        steps=30,
        episodes_per_step=4,
        micro_batch=6,
        lr=3e-2,
        warmup_steps=1,
        save_every=0,
        lm_weight=0.0,
    )
    tr = WindowTrainer(pol, probe(), TEXTS, S2, cfg, tmp_path)
    trajs = rollout(
        pol,
        probe(),
        [WEpisode("a", "real", TEXTS[:6], 2), WEpisode("b", "nobrief", TEXTS[:6], 2)],
        S2,
    )
    tg = tr.targets(trajs)
    assert [(t, r) for i, t, _, r in tg if i == 1] == [(t, "match") for t in range(6)]
    assert next((t, r) for i, t, _, r in tg if i == 0) == (0, "match")
    assert all(r == "window" for i, t, _, r in tg if i == 0 and t > 0)
    # Loss check on a one-window environment, so the target is fixed and the decrease is not
    # masked by window resampling.
    one = WindowEnv(concept="cats", layer=1, turns=3, windows=((0.6, None),), targets=(0.8,))
    tr = WindowTrainer(pol, probe(), TEXTS, one, cfg, tmp_path)
    base_before = {
        n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" not in n
    }
    first = tr.step()
    for _ in range(cfg.steps - 1):
        last = tr.step()
    assert last["win_mse"] < 0.5 * first["win_mse"]
    assert all(
        torch.equal(q, base_before[n]) for n, q in pol.model.named_parameters() if "lora" not in n
    )
    rows = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == cfg.steps and "pass_real_t2" in rows[-1]


def test_yoke_sources_use_a_different_window_once():
    cfg = WindowEval(episodes_per_window=3)
    eps = eval_episodes(TEXTS, S2, cfg)
    src = yoke_sources(eps, 4)
    assert sorted(src) == list(range(len(eps)))
    assert all(eps[s].window != e.window for e, s in zip(eps, src, strict=True))
    assert all(eps[s].id.split("e")[1] == e.id.split("e")[1] for e, s in zip(eps, src, strict=True))


def test_cluster_bootstrap():
    r = _cluster_boot([1.0] * 6, ["a", "a", "b", "b", "c", "c"], 200, 0)
    assert r == {"mean": 1.0, "lo": 1.0, "hi": 1.0}
    r = _cluster_boot([0.0, 0.0, 1.0, 1.0], ["a", "a", "b", "b"], 500, 0)
    assert r["lo"] == 0.0 and r["hi"] == 1.0 and r["mean"] == 0.5


def test_evaluate_end_to_end_base_and_adapter(tmp_path):
    cfg = WindowEval(
        episodes_per_window=2, layers=(1, 2), bootstrap=100, gen_check=2, gen_max_new_tokens=4
    )
    probes = {1: probe(1), 2: probe(2)}
    rep = evaluate(
        tiny_policy(lora=False), probes, TEXTS, S1, cfg, tmp_path / "base", has_adapter=False
    )
    r = rep["probes"]["L1"]
    # Base model, S1: keep-monitors always pass, hide-monitors always catch: det = 0.5 everywhere,
    # and every condition is identical, so every G is exactly 0.
    assert all(d["det"] == [0.5] * 3 for d in r["conds"].values())
    assert r["G_yoked"]["mean"] == 0.0 and r["G_none"]["mean"] == 0.0
    # The natural-probe readout is reported beside the window one and is also null on base.
    assert r["Gabs_yoked"]["mean"] == 0.0 and len(r["conds"]["real"]["det_abs"]) == 3
    assert set(r["conds"]) == {"real", "yoked", "flip", "none", "nobrief"}
    assert "gen_check" in r and "L2" in rep["probes"]
    g = r["gen_check"]
    assert g["min_contains"] >= g["min_content_exact"] >= g["min_exact"]
    assert set(rep["probes"]["L2"]["conds"]) == {"real", "yoked"}  # held-out probe: primary only
    assert (tmp_path / "base" / "eval.json").exists()
    pol = tiny_policy(lora=True)
    cfg = WindowEval(
        episodes_per_window=3,
        layers=(1, 2),
        bootstrap=100,
        gen_check=0,
        cond_caps={"nobrief": 1},
        secondary_episodes_per_window=2,
    )
    rep = evaluate(pol, probes, TEXTS, S2, cfg, tmp_path / "ad")
    c1, c2 = rep["probes"]["L1"]["conds"], rep["probes"]["L2"]["conds"]
    assert c1["real"]["n"] == 12 and c1["yoked"]["n"] == 12 and c1["nobrief"]["n"] == 4
    assert c2["real"]["n"] == 8 and c2["yoked"]["n"] == 8
    assert "G_yoked" in rep["probes"]["L2"] and "G_yoked_allturns" in rep["probes"]["L1"]


def test_multi_concept_rows_use_their_own_concept(tmp_path):
    """L3b: one probe object, a batch whose rows are about different concepts. Each row must be
    scored against ITS OWN concept row of the bank, and the brief must name it."""

    pol = tiny_policy(lora=False)
    p = WindowProbe(bank(1, ["cats", "dogs"]), ["cats", "dogs"])
    assert p.concepts == ["cats", "dogs"] and p.concept == "cats"
    texts = {"cats": TEXTS, "dogs": ["woof bark", "dog walk", "puppy", "good boy", "fetch", "paw"]}

    eps = [
        WEpisode("a", "real", texts["cats"][:3], 0, concept="cats"),
        WEpisode("b", "real", texts["dogs"][:3], 0, concept="dogs"),
    ]
    trajs = rollout(pol, p, eps, S1, has_adapter=False)
    assert "cats" in trajs[0].convs[0][0]["content"]
    assert "dogs" in trajs[1].convs[0][0]["content"]
    # Base model: f = 0 whatever the concept, and det_abs uses each row's OWN threshold.
    assert trajs[0].f == [0.0] * 3 and trajs[1].f == [0.0] * 3

    # Each row's z must equal the single-concept probe's z for that concept.
    views = [pol.view(tr.convs[-1]) for tr in trajs]
    pooled, _ = pol.forward(views, [1], grad=False)
    mixed = p.z(pooled[1], ["cats", "dogs"])
    for i, c in enumerate(("cats", "dogs")):
        solo = WindowProbe(bank(1, ["cats", "dogs"]), c)
        assert torch.allclose(mixed[i], solo.z(pooled[1][i : i + 1])[0], atol=1e-5)

    # Evaluation spreads concepts over every window and reports a per-concept breakdown.
    cfg = WindowEval(episodes_per_window=2, layers=(1,), bootstrap=50, gen_check=0)
    base = eval_episodes(texts, S1, cfg)
    assert {e.concept for e in base} == {"cats", "dogs"}
    for j in range(len(S1.windows)):
        assert {e.concept for e in base if e.window == j} == {"cats", "dogs"}
    rep = evaluate(pol, {1: p}, texts, S1, cfg, tmp_path / "mc", has_adapter=False)
    assert set(rep["probes"]["L1"]["conds"]["real"]["det_by_concept"]) == {"cats", "dogs"}

    # The trainer samples a concept per episode from the pool.
    tr = WindowTrainer(pol, p, texts, S1, WindowTrain(steps=1, episodes_per_step=8), tmp_path / "t")
    assert {e.concept for e in tr.sample(16)} == {"cats", "dogs"}


def test_hill_targets_are_verdict_only():
    """Rung 5a: the target must be derivable from (own f, verdict) alone -- never from the window
    bounds. Caught: a DETECTED turn steps deeper than where the model actually was, a passing turn
    freezes it there, and the rule is identical whatever the window set is."""

    from next_chameleons.tica.windows import hill_targets

    # Caught at 0.00 then 0.25 then passes at 0.55: step up, step up, then hold at 0.55.
    assert hill_targets([0.0, 0.25, 0.55], [True, True, False], 0.25) == [0.25, 0.5, 0.55]
    # It never consults the bounds: the same (f, verdict) history gives the same targets for S1
    # and S2, whose windows differ completely.
    h = hill_targets([0.0, 0.4], [True, False], 0.3)
    assert h == [0.3, 0.4] and len(S1.windows) != len(S2.windows)
    assert hill_targets([0.0], [True], 0.25, cap=0.1) == [0.1]  # capped


def test_trainer_hill_rule_differs_from_oracle(tmp_path):
    pol = tiny_policy(lora=True)
    trajs = rollout(pol, probe(), [WEpisode("a", "real", TEXTS[:6], 3)], S2)
    common = dict(steps=1, episodes_per_step=2, micro_batch=4, save_every=0)
    orc = WindowTrainer(pol, probe(), TEXTS, S2, WindowTrain(**common), tmp_path / "o")
    hill = WindowTrainer(
        pol, probe(), TEXTS, S2, WindowTrain(target_rule="hill", hill_step=0.25, **common),
        tmp_path / "h",
    )
    to = [v for _, t, v, r in orc.targets(trajs) if r == "window"]
    th = [v for _, t, v, r in hill.targets(trajs) if r == "window"]
    assert len(to) == len(th) == S2.turns - 1
    # The oracle can only ever emit window centres; hill climbing is not restricted to them.
    assert all(v in S2.targets for v in to)
    assert th != to
