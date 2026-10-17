"""TICA framework on a tiny random Llama with a character tokenizer (CPU).

Covers:
- views (generation prompt + content, a think-block template like Qwen3.5);
- pooled reads with and without grad;
- the reference context;
- the rollout's feedback chain;
- objective planning;
- one supervised trainer run (loss falls, the adapter changes, the base model does not);
- monitor fitting and the pool round-trip.
"""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from next_chameleons.tica.episodes import (  # noqa: E402
    ConceptTexts,
    EpisodeSampler,
    EpisodeSpec,
    rollout,
)
from next_chameleons.tica.monitors import (  # noqa: E402
    ConceptProbeBank,
    MonitorPool,
    auroc,
    fit_logistic,
    fit_meta_detector,
)
from next_chameleons.tica.objectives import ObjectiveConfig, plan_targets  # noqa: E402
from next_chameleons.tica.policy import Policy  # noqa: E402

EOS, PAD, VOCAB = 257, 258, 260


class CharTok:
    """Char-level tokenizer. Its chat template, like Qwen3.5's, adds an empty think block to the
    generation prompt and drops it from finished turns."""

    eos_token_id, pad_token_id = EOS, PAD
    eos_token, pad_token = "<eos>", "<pad>"

    def apply_chat_template(
        self, conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        s = "".join(
            f"<u>{m['content']}</u>" if m["role"] == "user" else f"<a>{m['content']}</a>"
            for m in conv
        )
        return s + ("<a><t></t>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[min(ord(c), 255) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 256)


def tiny_policy(lora: bool = True) -> Policy:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        eos_token_id=EOS,
        pad_token_id=PAD,
        bos_token_id=1,
    )
    model = transformers.LlamaForCausalLM(cfg)
    if lora:
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj", "down_proj"],
                lora_dropout=0.0,
            ),
        )
    model.eval()
    return Policy(model, CharTok(), batch_size=4)


def bank(layer: int, concepts: list[str], d: int = 32) -> ConceptProbeBank:
    g = torch.Generator().manual_seed(1)
    k = len(concepts)
    return ConceptProbeBank(
        "concept_L1",
        layer,
        concepts,
        W=torch.randn(k, d, generator=g),
        b=torch.zeros(k),
        mu=torch.zeros(k, d),
        sd=torch.ones(k, d),
        thr=torch.zeros(k),
        zero=torch.full((k,), -5.0),
        hundred=torch.full((k,), 5.0),
    )


def test_view_uses_generation_prompt():
    pol = tiny_policy(lora=False)
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "abc"}]
    v = pol.view(conv)
    assert "".join(map(chr, v.ids)) == "<u>hi</u><a><t></t>abc<"  # + first end-of-turn token
    assert "".join(map(chr, v.ids[v.start : v.end])) == "abc" and v.n_tail == 1


def test_pooled_grad_and_reference():
    pol = tiny_policy()
    conv = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "hello"}]
    v = pol.view(conv)
    p, nll = pol.forward([v, v], [0, 2], grad=True, with_nll=True)
    assert p[0].shape == (2, 32) and p[0].requires_grad and nll.shape == (2,)
    assert torch.allclose(p[2][0], p[2][1])
    p[2].sum().backward()
    assert any(q.grad is not None for n, q in pol.model.named_parameters() if "lora" in n)
    # With zero-initialized LoRA B, the policy equals the reference; perturb B and they differ.
    with torch.no_grad():
        for n, q in pol.model.named_parameters():
            if "lora_B" in n:
                q.add_(0.5)
    a, _ = pol.forward([v], [2], grad=False)
    with pol.reference():
        r, _ = pol.forward([v], [2], grad=False)
    assert not torch.allclose(a[2], r[2])
    # Padding must not change pooled values: alone vs batched with a longer view.
    long = pol.view(
        [{"role": "user", "content": "x" * 30}, {"role": "assistant", "content": "hello"}]
    )
    b, _ = pol.forward([v, long], [2], grad=False)
    assert torch.allclose(a[2][0], b[2][0], atol=1e-5)


def test_rollout_feedback_chain_and_targets():
    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs", "cars"]), round=0, family="concept_linear")
    specs = [
        EpisodeSpec(
            "real_score|cats|0", "real_score", ["aa", "bb", "cc"], "cats", "cats", "concept_L1"
        ),
        EpisodeSpec(
            "mismatch|cats|1",
            "mismatch",
            ["aa", "bb", "cc"],
            "cats",
            "dogs",
            "concept_L1",
            probes={"text": ("concept_L1", "cats")},
        ),
        EpisodeSpec("nobrief|cats|2", "nobrief", ["aa", "bb", "cc"], "cats", "cats", "concept_L1"),
    ]
    trajs = rollout(pol, pool, specs, 3)
    tr = trajs[0]
    assert len(tr.convs) == 3 and len(tr.feedback) == 2
    assert tr.convs[1][-1]["content"] == "bb" and tr.convs[1][-2]["content"] == tr.feedback[0]
    assert (
        tr.feedback[0].startswith("Monitor result: score")
        and "<text>\nbb\n</text>" in tr.feedback[0]
    )
    assert "Monitor result" not in trajs[2].feedback[0]  # nobrief: next task only
    assert "monitor" not in trajs[2].convs[0][0]["content"].lower()
    assert "text" in trajs[1].results[0]
    roles = {(t.traj, t.turn, t.key): t.role for t in plan_targets(trajs, ObjectiveConfig())}
    assert roles[(0, 0, "monitored")] == "match" and roles[(0, 1, "monitored")] == "hide"
    assert (1, 0, "monitored") not in roles and roles[(1, 2, "text")] == "match"
    assert roles[(2, 1, "monitored")] == "match"


def test_supervised_trainer_reduces_hide_loss(tmp_path):
    from next_chameleons.tica.trainers.supervised import SupervisedConfig, SupervisedTrainer

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    concepts = [
        ConceptTexts("cats", "train", ["meow purr", "cat nap", "kitten"]),
        ConceptTexts("dogs", "train", ["woof bark", "dog walk", "puppy"]),
    ]
    sampler = EpisodeSampler(
        concepts, ["plain text"], pool, mix={"real_score": 1.0}, turns=2, seed=0
    )
    cfg = SupervisedConfig(
        steps=25,
        episodes_per_step=4,
        turns=2,
        micro_batch=3,
        lr=3e-2,
        warmup_steps=1,
        save_every=0,
        objective=ObjectiveConfig(lm_weight=0.0),
    )
    tr = SupervisedTrainer(pol, pool, sampler, cfg, tmp_path)
    base_before = {
        n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" not in n
    }
    first = tr.step()
    for _ in range(cfg.steps - 1):
        last = tr.step()
    assert last["train/hide_mse"] < 0.5 * first["train/hide_mse"]
    assert all(
        torch.equal(q, base_before[n]) for n, q in pol.model.named_parameters() if "lora" not in n
    )
    assert (tmp_path / "train_log.jsonl").exists()


def test_fit_logistic_matches_sklearn_and_meta_detector(tmp_path):
    sk = pytest.importorskip("sklearn.linear_model")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 6, generator=g)
    y = (x[:, 0] + 0.5 * x[:, 1] + 0.3 * torch.randn(300, generator=g) > 0.8).float()
    w, b, mu, sd = fit_logistic(x, y, C=0.5)
    clf = sk.LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, tol=1e-8).fit(
        ((x - mu) / sd).numpy(), y.numpy()
    )
    assert torch.allclose(w, torch.tensor(clf.coef_[0], dtype=torch.float32), atol=2e-3)
    assert abs(b - float(clf.intercept_[0])) < 2e-3

    pos, neg = torch.randn(200, 8, generator=g) + 1.5, torch.randn(200, 8, generator=g)
    det, q = fit_meta_detector("meta_r1", 1, pos[:100], neg[:100], pos[100:], neg[100:], C=1.0)
    assert q["auroc"] > 0.95 and det.covers("anything")
    assert abs(auroc(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 2.0])) - 0.625) < 1e-9
    pool = MonitorPool()
    pool.add(bank(1, ["a", "b"]), round=0, family="concept_linear")
    pool.add(det, round=1, family="meta_linear", held_out=True)
    pool.save(tmp_path / "pool")
    back = MonitorPool.load(tmp_path / "pool")
    assert [e.monitor.name for e in back.entries] == ["concept_L1", "meta_r1"]
    assert [e.monitor.name for e in back.trainable("a")] == ["concept_L1"]
    xs = torch.randn(3, 8)
    assert torch.allclose(back.get("meta_r1").logits(xs, [None] * 3), det.logits(xs, [None] * 3))
    with pytest.raises(ValueError):
        pool.add(det, round=2, family="meta_linear")
    assert pool.sample(random.Random(0), "a").monitor.name == "concept_L1"


def test_rl_pieces_and_one_step(tmp_path):
    from next_chameleons.tica.trainers.rl import (
        RLConfig,
        RLTrainer,
        build_sample,
        group_advantages,
        pg_loss,
    )

    r = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])  # S=2, G=2, T=2
    a = group_advantages(r, "turn")
    assert torch.allclose(a[0, :, 0], torch.tensor([1.0, -1.0]), atol=1e-6)  # rtg 1 vs 0
    assert torch.all(a[1] == 0)  # zero-variance group dropped
    assert torch.allclose(
        group_advantages(r, "episode")[0, :, 1], torch.tensor([1.0, -1.0]), atol=1e-6
    )
    lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, st = pg_loss(lp, lp.detach(), torch.tensor([1.0, 1.0]), beta=0.1)
    assert abs(st["kl"]) < 1e-9 and abs(float(loss) - 1.5) < 1e-6

    pol = tiny_policy()
    cfg = RLConfig(mode="reason_forced", reasoning_end="</t>", answer_prefix=": ")
    gen = {
        "prompt_ids": [1, 2],
        "ids": [ord(c) for c in "think</t>junk"],
        "text": "",
        "finished": True,
    }
    s = build_sample(pol, gen, cfg, "ANS")
    assert "".join(map(chr, s.view.ids[s.sampled[0] : s.sampled[1]])) == "think</t>"
    assert "".join(map(chr, s.view.ids[s.probe_span[0] : s.probe_span[1]])) == "ANS"
    assert s.finished and s.text == "ANS"
    gen2 = {"prompt_ids": [1], "ids": [ord("x")] * 5, "text": "", "finished": False}
    assert not build_sample(pol, gen2, cfg, "ANS").finished

    pool = MonitorPool()
    pool.add(bank(1, ["cats"]), round=0, family="concept_linear")

    def specs(n):
        return [
            EpisodeSpec(
                f"real_score|cats|{k}", "real_score", ["aa", "bb"], "cats", "cats", "concept_L1"
            )
            for k in range(n)
        ]

    rcfg = RLConfig(
        mode="open",
        specs_per_step=2,
        group_size=4,
        turns=2,
        max_new_tokens=6,
        lr=1e-2,
        save_every=0,
        micro_batch=4,
        grpo_require_finished=False,  # 6-token samples never finish; this tests the update
    )
    tr = RLTrainer(
        pol,
        pool,
        specs,
        rcfg,
        tmp_path,
        judge=lambda items: [True] * len(items),
        reward_fn=lambda spec, t, smp, rd: (
            float(smp.view.ids[smp.sampled[0]] % 2) if smp.sampled[1] > smp.sampled[0] else 0.0
        ),
    )
    before = {n: q.detach().clone() for n, q in pol.model.named_parameters() if "lora" in n}
    rec = tr.step()
    assert rec["n_items"] > 0
    assert any(
        not torch.equal(q, before[n]) for n, q in pol.model.named_parameters() if "lora" in n
    )


def _race_setup(tmp_path):
    """Tiny 8-concept Level-1 data, a one-bank pool and an adapter-aware policy factory."""

    import json

    from next_chameleons.tica.data import Level1Data

    words = {
        "cats": "meow",
        "dogs": "woof",
        "cars": "vroom",
        "bells": "ding",
        "rain": "drip",
        "wind": "whoosh",
        "fire": "crackle",
        "snow": "crunch",
    }
    concepts = []
    for i, (c, w) in enumerate(words.items()):
        texts = [{"id": f"{c}{k}", "text": f"{w} {k}"} for k in range(12)]
        concepts.append(
            {
                "concept": c,
                "split": "heldout" if i >= 4 else "train",
                "fit": texts[:6],
                "calib": [],
                "episode": texts[6:],
            }
        )
    negs = [{"id": f"n{k}", "text": f"plain {k}"} for k in range(12)]
    (tmp_path / "d.json").write_text(
        json.dumps(
            {"concepts": concepts, "negatives": {"fit": negs, "calib": negs, "episode": negs}}
        )
    )
    data = Level1Data(tmp_path / "d.json")
    pool = MonitorPool()
    pool.add(bank(1, list(words)), round=0, family="concept_linear")

    def factory(adapter, trainable):
        pol = tiny_policy(lora=adapter is None)
        if adapter is not None:
            pol.model = peft.PeftModel.from_pretrained(pol.model, adapter, is_trainable=trainable)
            pol.layers_module = __import__(
                "next_chameleons.tica.policy", fromlist=["x"]
            ).find_decoder_layers(pol.model)
        return pol

    return data, pool, factory


def test_arms_race_rounds_end_to_end(tmp_path):
    """Two rounds on the tiny model: train -> collect -> fit (meta + concept refit) -> eval, with
    resumable stage markers and a growing pool."""

    import json

    from next_chameleons.tica.evaluation import EvalConfig
    from next_chameleons.tica.rounds import ArmsRace, CollectConfig, FitConfig, RoundsConfig
    from next_chameleons.tica.trainers.supervised import SupervisedConfig

    data, pool, factory = _race_setup(tmp_path)
    cfg = RoundsConfig(
        rounds=2,
        train=SupervisedConfig(
            steps=2,
            episodes_per_step=4,
            turns=2,
            micro_batch=4,
            lr=1e-2,
            warmup_steps=1,
            save_every=0,
        ),
        collect=CollectConfig(episodes_per_concept=6, turns=3, layers=[1, 2]),
        fit=FitConfig(C=1.0, positives="active"),  # the toy policy rarely evades
        eval=EvalConfig(turns=2, episodes_per_concept=1, gen_check=2, gen_max_new_tokens=4),
    )
    race = ArmsRace(cfg, tmp_path / "race", data, pool, factory)
    race.run(("train", "collect", "fit"), only_round=0)
    assert race.done(0, "fit") and not race.done(0, "eval")
    fit = json.loads((tmp_path / "race/round_0/fit.json").read_text())
    nxt = MonitorPool.load(tmp_path / "race/round_0/pool_next")
    names = [e.monitor.name for e in nxt.entries]
    assert names[0] == "concept_L1" and len(names) >= 2, (names, fit)
    assert any(n.startswith("meta_r1_") for n in names), fit
    chosen = fit[f"meta_L{fit['meta_chosen'].split('_L')[1]}"]
    assert {"auroc", "tpr", "test"} <= set(chosen) and chosen["neg_test"] >= 5
    assert all(e.round == 1 for e in nxt.entries[1:])
    race.run()  # resumes: round 0 eval, then all of round 1
    assert race.done(1, "eval")
    ev = json.loads((tmp_path / "race/round_0/eval.json").read_text())
    assert "base_model" in ev and ev["base_model"]["meta_detection"]
    summ = json.loads((tmp_path / "race/round_0/eval/concept_L1/summary.json").read_text())
    assert "gen_check" in summ and "contrasts_last" in summ["all"]
    assert (tmp_path / "race/round_1/policy/adapter_config.json").exists()
    assert len(MonitorPool.load(tmp_path / "race/round_1/pool_next").entries) >= len(names)


def test_scaled_hide_probability_is_not_saturated():
    """Monitor.scale = negative SD; p = sigmoid((raw - thr) / scale) keeps a gradient at positives
    that sit many raw logits above thr (the real gated banks), unlike sigmoid(raw - thr)."""

    from next_chameleons.tica.objectives import Target, probe_prob

    k, d = 2, 4
    b = ConceptProbeBank(
        "big",
        1,
        ["a", "b"],
        W=torch.eye(k, d) * 1.0,
        b=torch.zeros(k),
        mu=torch.zeros(k, d),
        sd=torch.ones(k, d),
        thr=torch.full((k,), 9.0),
        zero=torch.zeros(k),
        hundred=torch.full((k,), 22.0),
    )
    assert torch.allclose(b.scale(["a"]), torch.tensor([9.0 / 2.3263]), atol=1e-4)
    pool = MonitorPool()
    pool.add(b, round=0, family="concept_linear")
    x = torch.zeros(1, d)
    x[0, 0] = 22.0  # a typical positive: 13 raw logits above thr
    x.requires_grad_(True)
    p = probe_prob(pool, {1: x}, [0], [Target(0, 1, "monitored", "big", "a", "hide")])
    (p**2).sum().backward()
    assert 0.9 < float(p) < 0.999
    assert float(x.grad[0, 0]) > 1e-3  # unscaled: 2 * p * sigmoid'(13) ~ 4e-6


def test_forward_nll_with_logits_to_keep_matches_full():
    pol = tiny_policy()
    convs = [
        [{"role": "user", "content": "say hi"}, {"role": "assistant", "content": "hi there"}],
        [
            {"role": "user", "content": "a much longer request here"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "fine then"},
        ],
    ]
    views = [pol.view(c) for c in convs]
    _, nll = pol.forward(views, [1], grad=False, with_nll=True)
    ref = []
    for v in views:
        ids = torch.tensor([v.ids])
        with torch.no_grad():
            lg = pol.model(input_ids=ids).logits[0]
        lp = torch.log_softmax(lg[v.start - 1 : len(v.ids) - 1].float(), -1)
        ref.append(-lp.gather(-1, ids[0, v.start :, None])[:, 0].mean())
    assert torch.allclose(nll, torch.stack(ref), atol=1e-4)
    # The end-of-turn token is in the LM target but not in the probe read.
    assert all(v.n_tail == 1 and chr(v.ids[-1]) == "<" for v in views)
    pooled, _ = pol.forward(views, [1], grad=False)
    from next_chameleons.tica.policy import View

    bare = [View(v.ids[: v.end], v.start) for v in views]
    pooled_bare, _ = pol.forward(bare, [1], grad=False)
    assert torch.allclose(pooled[1], pooled_bare[1], atol=1e-5)


def test_sampler_is_stratified_and_selectivity_covers_every_bank():
    pool = MonitorPool()
    b1 = bank(1, ["cats", "dogs"])
    b2 = dataclasses.replace(b1, name="concept_L2", layer=2)
    pool.add(b1, round=0, family="concept_linear")
    pool.add(b2, round=0, family="concept_linear")
    concepts = [
        ConceptTexts("cats", "train", ["meow a", "meow b", "meow c"]),
        ConceptTexts("dogs", "train", ["woof a", "woof b", "woof c"]),
    ]
    mix = {"real_score": 0.4, "mismatch": 0.3, "nobrief": 0.15, "neg_text": 0.15}
    smp = EpisodeSampler(concepts, ["plain"], pool, mix=mix, turns=2, seed=3)
    for _ in range(20):
        specs = smp.sample(8)
        conds = [s.cond for s in specs]
        assert conds.count("real_score") >= 3 and conds.count("mismatch") >= 2
    mm = next(s for s in smp.sample(16) if s.cond == "mismatch")
    assert set(mm.probes) == {"text", "text@concept_L2"}
    trajs = rollout(tiny_policy(), pool, [mm], 2)
    tg = plan_targets(trajs, ObjectiveConfig())
    assert sorted(t.key for t in tg) == ["text", "text", "text@concept_L2", "text@concept_L2"]
    assert all(t.role == "match" for t in tg)


def test_textcheck_and_last_turn_contrast():
    from next_chameleons.tica import analysis
    from next_chameleons.tica.textcheck import classify

    assert classify("It\u2019s here.", "It's  here.") == "exact"
    assert classify("<text>a b</text>", "a b") == "tags"
    assert classify("Sure: a b", "a b") == "extra"
    assert classify("I cannot do that", "a b") == "refusal"
    assert classify("a c", "a b") == "edited"
    recs = []
    for cond, off in (("real_score", -10.0), ("none", 0.0)):
        for c in ("x", "y", "z"):
            for t in (0, 1):
                recs.append(
                    {
                        "episode": f"{cond}|{c}|0",
                        "cond": cond,
                        "turn": t,
                        "score_monitored": 100.0 + (off if t == 1 else 0.0),
                        "detected_monitored": True,
                    }
                )
    rep = analysis.summarize(recs)
    m, _lo, _hi, n = rep["all"]["contrasts_last"]["real_score - none"]
    assert m == -10.0 and n == 3


def test_one_se_probe_selection():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "l1_fit", Path(__file__).resolve().parents[1] / "scripts" / "l1_fit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for i in range(10):
        noise = 0.1 if i % 2 else -0.1
        rows.append({"concept": i, "layer": 12, "C": 100.0, "tpr": 0.91 + noise, "auroc": 0.99})
        rows.append({"concept": i, "layer": 12, "C": 0.01, "tpr": 0.90 + noise, "auroc": 0.99})
        rows.append({"concept": i, "layer": 16, "C": 0.01, "tpr": 0.50 + noise, "auroc": 0.9})
    best, summary = mod.choose_setting(rows)
    assert (best["layer"], best["C"]) == (12, 0.01) and best["best_raw"]["C"] == 100.0
    assert len(summary) == 3


def test_eval_sham_is_yoked_from_another_concept(tmp_path):
    import json

    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.evaluation import EvalConfig, evaluate

    words = {"cats": "meow", "dogs": "woof", "cars": "vroom"}
    concepts = [
        {
            "concept": c,
            "split": "train",
            "fit": [],
            "calib": [],
            "episode": [{"id": f"{c}{k}", "text": f"{w} {k}"} for k in range(4)],
        }
        for c, w in words.items()
    ]
    negs = [{"id": f"n{k}", "text": f"plain {k}"} for k in range(8)]
    (tmp_path / "d.json").write_text(
        json.dumps(
            {"concepts": concepts, "negatives": {"fit": negs, "calib": negs, "episode": negs}}
        )
    )
    pool = MonitorPool()
    pool.add(bank(1, list(words)), round=0, family="concept_linear")
    cfg = EvalConfig(
        conditions=("real_score", "sham_score", "none"),
        turns=2,
        episodes_per_concept=2,
        gen_check=0,
    )
    rep = evaluate(
        tiny_policy(), pool, Level1Data(tmp_path / "d.json"), "concept_L1", cfg, tmp_path
    )
    recs = [json.loads(x) for x in (tmp_path / "records.jsonl").open()]

    def seqs(cond):
        out = {}
        for r in sorted(recs, key=lambda r: r["turn"]):
            if r["cond"] == cond:
                out.setdefault(r["episode"].split("|", 1)[1], []).append(r["shown"])
        return out

    real, sham = seqs("real_score"), seqs("sham_score")
    assert len(sham) == 6
    for key, shown in sham.items():
        src = [k for k, v in real.items() if v == shown]
        assert src and all(k.split("|")[0] != key.split("|")[0] for k in src), (key, src)
    assert "real_score - sham_score" in rep["all"]["contrasts_last"]


def test_chunked_gradient_equals_full_batch(tmp_path):
    """micro_batch=1 and micro_batch=all give the same gradient and the same logged means."""

    from next_chameleons.tica.trainers.supervised import SupervisedConfig, SupervisedTrainer

    concepts = [
        ConceptTexts("cats", "train", ["meow purr", "cat nap", "kitten"]),
        ConceptTexts("dogs", "train", ["woof bark", "dog walk", "puppy"]),
    ]
    mix = {"real_score": 0.4, "mismatch": 0.3, "nobrief": 0.15, "neg_text": 0.15}
    out = {}
    for mb in (1, 64):
        pol = tiny_policy()
        pool = MonitorPool()
        pool.add(bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
        sampler = EpisodeSampler(concepts, ["plain text", "other words"], pool, mix=mix, turns=2)
        cfg = SupervisedConfig(
            steps=1, episodes_per_step=6, turns=2, micro_batch=mb, lr=0.0, save_every=0
        )
        tr = SupervisedTrainer(pol, pool, sampler, cfg, tmp_path / f"mb{mb}")
        rec = tr.step()
        grads = {n: q.grad.clone() for n, q in pol.model.named_parameters() if q.grad is not None}
        out[mb] = (rec, grads)
    (r1, g1), (r2, g2) = out[1], out[64]
    assert set(g1) == set(g2) and g1
    for n in g1:
        assert torch.allclose(g1[n], g2[n], atol=1e-5, rtol=1e-4), n
    for k in ("train/loss", "train/hide_mse", "train/match_mse", "train/hide_p"):
        assert abs(r1[k] - r2[k]) < 1e-5, k
    assert 0.0 <= r1["train/hide_p"] <= 1.0


def test_verdict_mode_and_hidden_rendering():
    from next_chameleons.tica.episodes import render_feedback
    from next_chameleons.tica.level1 import opening

    named = render_feedback("real_score", "cats", (80.0, 50.0), "x", feedback="verdict")
    assert named.startswith('Monitor result: DETECTED. The monitor found "cats"')
    hidden = render_feedback(
        "real_score", "cats", (20.0, 50.0), "x", feedback="verdict", named=False
    )
    assert hidden.startswith("Monitor result: NOT DETECTED") and "cats" not in hidden
    with pytest.raises(ValueError):
        render_feedback("real_score", "cats", (80.0, 50.0), "x", feedback="score", named=False)
    assert "two topics" in opening("real_score", "t", "cats", named=False)
    assert "cats" not in opening("real_score", "t", "cats", named=False)


def test_hidden_mode_sampler_rollout_and_targets():
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs", "cars"]), round=0, family="concept_linear")
    concepts = [
        ConceptTexts(c, "train", [f"{w} {k}" for k in range(3)])
        for c, w in {"cats": "meow", "dogs": "woof", "cars": "vroom"}.items()
    ]
    mix = {"real_score": 0.5, "mismatch": 0.25, "nobrief": 0.25}
    smp = EpisodeSampler(
        concepts, ["plain"], pool, mix=mix, turns=3, seed=1, feedback="verdict", hidden=True
    )
    specs = smp.sample(16)
    real = [s for s in specs if s.cond == "real_score"]
    assert real and all(not s.named and s.feedback == "verdict" for s in specs)
    word = {"cats": "meow", "dogs": "woof", "cars": "vroom"}
    for s in real:
        tc = s.meta["turn_concepts"]
        assert set(s.probes) == {"keep"} and s.probes["keep"][1] == s.meta["partner"]
        assert s.meta["partner"] != s.monitored and set(tc) == {s.monitored, s.meta["partner"]}
        assert tc[-1] == s.monitored and tc[0] == s.meta["probe_concept"]
        assert s.meta["probe_match"] == (tc[0] == s.monitored)
        assert all(t.split()[0] == word[c] for t, c in zip(s.texts, tc, strict=True))
        assert len(set(s.texts)) == 3  # texts are never reused within an episode
    mm = next(s for s in specs if s.cond == "mismatch")
    assert set(mm.probes) == {"keep", "text"} and mm.monitored not in mm.meta["turn_concepts"]
    trajs = rollout(tiny_policy(), pool, [real[0], mm], 3)
    first = trajs[0].convs[0][0]["content"]
    assert "two topics" in first and real[0].monitored not in first
    fb = trajs[0].feedback[0]
    assert fb.startswith("Monitor result:") and "its concept" in fb and '"' not in fb.split("\n")[0]
    roles = {(t.traj, t.turn, t.key): t.role for t in plan_targets(trajs, ObjectiveConfig())}
    tc = real[0].meta["turn_concepts"]
    assert roles[(0, 0, "monitored")] == "match" and roles[(0, 2, "monitored")] == "hide"
    assert roles[(0, 1, "monitored")] == ("hide" if tc[1] == real[0].monitored else "match")
    assert roles[(0, 1, "keep")] == "match" and roles[(0, 2, "keep")] == "match"
    assert (1, 1, "monitored") not in roles
    assert roles[(1, 2, "text")] == "match" and roles[(1, 2, "keep")] == "match"
    with pytest.raises(ValueError):
        EpisodeSampler(concepts, ["plain"], pool, mix=mix, turns=3, hidden=True)


def test_schedule_texts_layout():
    from next_chameleons.tica.episodes import schedule_texts

    a = ("cats", [f"c{k}" for k in range(6)])
    b = ("dogs", [f"d{k}" for k in range(6)])
    texts, order = schedule_texts(a, b, True, random.Random(0), 6)
    assert order[0] == "cats" and order[-2:] == ["dogs", "cats"] and len(texts) == 6
    assert len(set(texts)) == 6 and order[1:4] in (
        ["cats", "dogs", "cats"],
        ["dogs", "cats", "dogs"],
    )
    texts, order = schedule_texts(a, b, False, random.Random(0), 4)
    assert order[0] == "dogs" and order[-2:] == ["dogs", "cats"]
    _, order = schedule_texts(a, b, False, random.Random(0), 3)
    assert order == ["dogs", "dogs", "cats"]
    with pytest.raises(ValueError):
        schedule_texts(a, b, True, random.Random(0), 2)


def test_eval_hidden_mode_is_balanced_and_reported(tmp_path):
    import json

    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.evaluation import EvalConfig, evaluate

    words = {"cats": "meow", "dogs": "woof", "cars": "vroom", "cows": "moo"}
    concepts = [
        {
            "concept": c,
            "split": "heldout" if c == "cows" else "train",
            "fit": [],
            "calib": [],
            "episode": [{"id": f"{c}{k}", "text": f"{w} {k}"} for k in range(4)],
        }
        for c, w in words.items()
    ]
    negs = [{"id": f"n{k}", "text": f"plain {k}"} for k in range(8)]
    (tmp_path / "d.json").write_text(
        json.dumps(
            {"concepts": concepts, "negatives": {"fit": negs, "calib": negs, "episode": negs}}
        )
    )
    pool = MonitorPool()
    pool.add(bank(1, list(words)), round=0, family="concept_linear")
    cfg = EvalConfig(
        conditions=("real_score", "none", "nobrief", "neg_text", "mismatch"),
        turns=4,
        episodes_per_concept=1,
        gen_check=0,
        feedback="verdict",
        hidden=True,
    )
    for c in concepts:  # 4 turns x 1 episode of own texts, plus partner texts drawn by sample()
        c["episode"] = [
            {"id": f"{c['concept']}{k}", "text": f"{words[c['concept']]} {k}"} for k in range(4)
        ]
    (tmp_path / "d.json").write_text(
        json.dumps(
            {"concepts": concepts, "negatives": {"fit": negs, "calib": negs, "episode": negs}}
        )
    )
    rep = evaluate(
        tiny_policy(), pool, Level1Data(tmp_path / "d.json"), "concept_L1", cfg, tmp_path
    )
    recs = [json.loads(x) for x in (tmp_path / "records.jsonl").open()]
    real = [r for r in recs if r["cond"] == "real_score"]
    assert len(real) == 16 and all(r["probe_match"] for r in real)  # episode 0 -> probe = M
    last = [r for r in real if r["turn"] == 3]
    assert all(r["turn_concept"] == r["monitored"] and r["detected_keep"] is None for r in last)
    keep = [r for r in real if r["turn"] == 2]
    assert all(r["turn_concept"] != r["monitored"] and r["detected_keep"] is not None for r in keep)
    assert all(r["detected_text"] is None for r in keep)  # M absent on an O turn
    assert all(r["detected_keep"] is None for r in recs if r["cond"] == "neg_text")
    mm = [r for r in recs if r["cond"] == "mismatch"]
    assert all(r["probe_match"] is False for r in mm)
    per = rep["all"]["per_turn"]
    assert "real_score|A=M" in per and "real_score|A!=M" not in per
    assert rep["all"]["keep_last"]["real_score"]["turn"] == 2
    assert rep["style"] == {"feedback": "verdict", "hidden": True, "monitor": "concept_L1"}


def test_supervised_step_in_hidden_mode(tmp_path):
    from next_chameleons.tica.trainers.supervised import SupervisedConfig, SupervisedTrainer

    pol = tiny_policy(lora=True)
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs", "cars"]), round=0, family="concept_linear")
    concepts = [
        ConceptTexts(c, "train", [f"{w} {k}" for k in range(4)])
        for c, w in {"cats": "meow", "dogs": "woof", "cars": "vroom"}.items()
    ]
    smp = EpisodeSampler(
        concepts,
        ["plain"],
        pool,
        mix={"real_score": 0.5, "mismatch": 0.25, "nobrief": 0.25},
        turns=4,
        seed=0,
        feedback="verdict",
        hidden=True,
    )
    cfg = SupervisedConfig(steps=2, episodes_per_step=4, turns=4, micro_batch=4, lr=1e-3)
    tr = SupervisedTrainer(pol, pool, smp, cfg, tmp_path)
    logs = [tr.step() for _ in range(2)]
    assert all(rec["train/n_hide"] >= 1 and rec["train/n_match"] >= 1 for rec in logs)
    assert all(
        k in logs[0] for k in ("rollout/det_real_score_t0", "rollout/det_keep_real_score_tN")
    )
    assert all(torch.isfinite(torch.tensor(rec["train/loss"])) for rec in logs)


def test_sequential_race_replaces_monitors_and_stops_early(tmp_path):
    """FitConfig.replace: round r + 1 trains against the newest concept refit alone; older monitors
    become eval-only and are read alongside as regression readouts. stop_det ends training early."""

    import json

    from next_chameleons.tica.evaluation import EvalConfig
    from next_chameleons.tica.rounds import ArmsRace, CollectConfig, FitConfig, RoundsConfig
    from next_chameleons.tica.trainers.supervised import SupervisedConfig

    data, pool, factory = _race_setup(tmp_path)
    cfg = RoundsConfig(
        rounds=2,
        train=SupervisedConfig(
            steps=6,
            episodes_per_step=4,
            turns=2,
            micro_batch=4,
            lr=1e-2,
            warmup_steps=1,
            save_every=0,
            stop_det=1.0,  # always satisfied -> stops at min_steps
            stop_window=2,
            min_steps=2,
        ),
        collect=CollectConfig(episodes_per_concept=6, turns=3, layers=[1, 2]),
        fit=FitConfig(
            families=("concept_refit", "meta_linear"),
            held_out_families=("meta_linear",),
            positives="active",
            replace=True,
            refit_recipe="ovr",
        ),
        eval=EvalConfig(turns=2, episodes_per_concept=1, gen_check=0),
    )
    race = ArmsRace(cfg, tmp_path / "race", data, pool, factory)
    race.run()
    assert json.loads((tmp_path / "race/round_0/DONE_train").read_text())["steps"] == 2
    p1 = MonitorPool.load(tmp_path / "race/round_0/pool_next")
    trainable = [e.monitor.name for e in p1.entries if not e.held_out]
    assert trainable == ["concept_refit_r1"], [(e.monitor.name, e.held_out) for e in p1.entries]
    p2 = MonitorPool.load(tmp_path / "race/round_1/pool_next")
    assert [e.monitor.name for e in p2.entries if not e.held_out] == ["concept_refit_r2"]
    assert {"concept_L1", "concept_refit_r1"} <= {e.monitor.name for e in p2.entries}
    ev = json.loads((tmp_path / "race/round_1/eval.json").read_text())
    assert list(k for k in ev if k != "base_model") == ["concept_refit_r2"]
    reads = ev["concept_refit_r2"]["meta_detection"]
    assert "extra_concept_L1" in reads and "extra_concept_refit_r1" in reads, list(reads)


def test_branch_collect_rows_share_prefixes_and_halves_are_disjoint(tmp_path):
    """Roots that bypass the monitor are forked into follow-ups about the concept, other concepts
    and nothing; every follow-up of a root has the same prefix; A/B text halves never share a
    text; the branch refit fits, thresholds and reports per concept."""

    import json

    from next_chameleons.tica.branching import BranchConfig, branch_collect
    from next_chameleons.tica.evaluation import EvalConfig
    from next_chameleons.tica.rounds import ArmsRace, CollectConfig, FitConfig, RoundsConfig
    from next_chameleons.tica.trainers.supervised import SupervisedConfig

    data, pool, factory = _race_setup(tmp_path)
    pol = factory(None, False)
    bc = BranchConfig(roots=4, root_turns=2, follow_pos=4, follow_neg_other=6, follow_neg_free=2)
    concepts = data.concepts("fit", ("train", "heldout"))
    acts, labels, rep = branch_collect(pol, pool, concepts, data.negatives("fit"), bc, [1], seed=0)
    assert acts[1].shape[0] == len(labels) == rep["rows"] > 0
    assert 0 < rep["roots_bypassed"] <= rep["roots"] == 4 * len(concepts)
    # every root contributes follow_pos + neg_other + neg_free rows (concept texts permitting)
    per_root = {}
    for a in labels:
        per_root.setdefault(a["root"], []).append(a)
    for rows in per_root.values():
        conds = [a["cond"] for a in rows]
        assert conds.count("branch_pos") == 4 and conds.count("branch_neg_other") == 6
        assert conds.count("branch_neg_free") == 2
        # Negatives are about some OTHER concept; both positive kinds (in-context dodging and
        # the no-prompt plain-format rows) are about the monitored concept.
        assert all(
            a["concept"] != a["monitored"]
            for a in rows
            if a["cond"] not in ("branch_pos", "branch_pos_noprompt")
        )
        assert len({a["monitored"] for a in rows}) == 1
    assert all(isinstance(a["detected"], bool) for a in labels)
    # halves: a text never appears in both halves
    seen = {}
    for c in concepts:
        from next_chameleons.tica.branching import text_halves

        h = text_halves(c.texts, __import__("random").Random(f"0:{c.concept}"))
        assert (
            set(h.values()) <= {"A", "B"}
            and abs(list(h.values()).count("A") - list(h.values()).count("B")) <= 1
        )
        for t, v in h.items():
            assert seen.setdefault((c.concept, t), v) == v

    cfg = RoundsConfig(
        rounds=1,
        train=SupervisedConfig(steps=1, episodes_per_step=2, turns=2, micro_batch=2, save_every=0),
        collect=CollectConfig(layers=[1], text_split="fit", mode="branch", branch=bc),
        fit=FitConfig(
            families=("concept_refit", "meta_linear"),
            held_out_families=("meta_linear",),
            positives="active",
            replace=True,
            refit_recipe="branch",
            refit_C_grid=(0.01, 1.0),
            refit_min_pos=2,
            refit_min_neg=10,
        ),
        eval=EvalConfig(turns=2, episodes_per_concept=1, gen_check=0),
    )
    race = ArmsRace(cfg, tmp_path / "race", data, pool, factory)
    race.run(("train", "collect", "fit"), only_round=0)
    col = json.loads((tmp_path / "race/round_0/collect.json").read_text())
    assert col["rows"] > 0 and "pos_bypass_rate" in col
    fit = json.loads((tmp_path / "race/round_0/fit.json").read_text())
    cr = fit["concept_refit"]
    assert cr["recipe"] == "branch" and cr["refit"] >= 1 and cr["C"] in (0.01, 1.0)
    assert "refit_C" in fit
    nxt = MonitorPool.load(tmp_path / "race/round_0/pool_next")
    assert [e.monitor.name for e in nxt.entries if not e.held_out] == ["concept_refit_r1"]


def test_refit_C_selection_one_se_on_synthetic_rows(tmp_path):
    """_select_C on separable synthetic branch rows: every C is within one SE (TPR ~1), so the
    one-SE rule returns the SMALLEST C; the report carries the per-C CV means."""

    from next_chameleons.tica.rounds import ArmsRace, FitConfig, RoundsConfig

    g = torch.Generator().manual_seed(0)
    d, lab, xs = 16, [], []
    concepts = [f"c{i}" for i in range(4)]
    for k, c in enumerate(concepts):
        mu = torch.zeros(d)
        mu[k] = 6.0
        for _ in range(40):
            xs.append(mu + torch.randn(d, generator=g))
            lab.append(
                {
                    "cond": "branch_pos",
                    "monitored": c,
                    "concept": c,
                    "half": "A",
                    "detected": False,
                    "split": "train",
                }
            )
        for _ in range(120):
            xs.append(torch.randn(d, generator=g))
            lab.append(
                {
                    "cond": "branch_neg_free",
                    "monitored": c,
                    "concept": None,
                    "half": "A",
                    "detected": False,
                    "split": "train",
                }
            )
    x = torch.stack(xs)
    race = ArmsRace.__new__(ArmsRace)
    race.cfg = RoundsConfig(fit=FitConfig(refit_C_grid=(10.0, 0.01, 1.0), refit_cv_concepts=4))
    rep = {}
    C = race._select_C(x, lab, concepts, rep)
    assert C == 0.01 and rep["refit_C"]["cv_concepts"] == 4
    assert all(v > 0.9 for v in rep["refit_C"]["mean_tpr"].values())


def test_supervised_views_are_length_sorted_and_targets_still_match(tmp_path):
    """Micro-batches must hold similar-length views (Policy.forward keeps logits from min(start)-1
    to max(len), so a mixed batch materialises the whole sequence x vocab: 575 s/step on 10-10).
    Reordering must not break the target -> view mapping."""

    from next_chameleons.tica.objectives import plan_targets
    from next_chameleons.tica.trainers.supervised import SupervisedConfig, SupervisedTrainer

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    concepts = [
        ConceptTexts("cats", "train", ["meow", "cat nap is long", "kitten plays all day long"]),
        ConceptTexts("dogs", "train", ["woof", "dog walk is long", "puppy plays all day long"]),
    ]
    sampler = EpisodeSampler(
        concepts, ["plain text"], pool, mix={"real_score": 1.0}, turns=3, seed=0
    )
    cfg = SupervisedConfig(steps=1, episodes_per_step=4, turns=3, micro_batch=2, save_every=0)
    tr = SupervisedTrainer(pol, pool, sampler, cfg, tmp_path)
    specs = sampler.sample(4)
    trajs = rollout(pol, pool, specs, 3)
    targets = plan_targets(trajs, cfg.objective)
    keys = sorted({(tg.traj, tg.turn) for tg in targets})
    built = [pol.view(trajs[i].convs[t]) for i, t in keys]
    order = sorted(range(len(keys)), key=lambda k: len(built[k].ids))
    views = [built[k] for k in order]
    view_of = {keys[k]: i for i, k in enumerate(order)}
    lens = [len(v.ids) for v in views]
    assert lens == sorted(lens)
    # Every target still points at the view for its own (trajectory, turn).
    for tg in targets:
        v = views[view_of[(tg.traj, tg.turn)]]
        assert v.ids == pol.view(trajs[tg.traj].convs[tg.turn]).ids
    rec = tr.step()  # the real step must run with the same mapping
    assert rec["n_views"] == len(views)


def test_eval_only_concepts_restricts_episodes(tmp_path):
    """A single-concept run must evaluate on that concept alone. Without this the round-0 eval
    averaged over all 335 concepts while training touched one, so the trained policy read as
    'not evading' (10-10)."""

    import dataclasses as dc
    import json

    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.evaluation import EvalConfig, build_specs

    raw = {
        "concepts": [
            {
                "concept": name,
                "split": "train",
                "fit": [{"text": f"{name} fit {i}"} for i in range(4)],
                "calib": [{"text": f"{name} calib {i}"} for i in range(2)],
                "episode": [{"text": f"{name} ep {i}"} for i in range(8)],
            }
            # >= 3 concepts: build_specs draws a mismatch concept and an anchor from the others.
            for name in ("cats", "dogs", "birds")
        ],
        "negatives": {k: [{"text": f"neg {i}"} for i in range(8)] for k in
                      ("fit", "calib", "episode")},
    }
    (tmp_path / "d.json").write_text(json.dumps(raw))
    data = Level1Data(tmp_path / "d.json")
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs", "birds"]), round=0, family="concept_linear")
    cfg = EvalConfig(conditions=("real_score",), turns=2, episodes_per_concept=1)

    name = pool.entries[0].monitor.name
    both = build_specs(data, pool, name, cfg)
    assert {s.monitored for s in both} == {"cats", "dogs", "birds"}

    one = build_specs(data, pool, name, dc.replace(cfg, only_concepts=("cats",)))
    assert {s.monitored for s in one} == {"cats"}

    with pytest.raises(ValueError, match="matches no gated concept"):
        build_specs(data, pool, name, dc.replace(cfg, only_concepts=("fish",)))


def test_eval_records_carry_verdict_and_probe_provenance(tmp_path):
    """A results row must answer, on its own: what was the model TOLD after this turn, and was the
    probe one it had already trained against? Otherwise 'did it react to feedback' and 'was this a
    new probe' cannot be investigated without cross-referencing manifests."""

    from next_chameleons.tica.evaluation import to_records

    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    refit = bank(2, ["cats", "dogs"], d=32)
    refit = dataclasses.replace(refit, name="concept_refit_r1")
    pool.add(refit, round=1, family="concept_refit", held_out=True)
    names = [e.monitor.name for e in pool.entries]
    spec = EpisodeSpec(
        id="e0", cond="real_score", texts=["meow purr", "cat nap"], text_concept="cats",
        monitored="cats", monitor=names[1], feedback="verdict",
    )
    trajs = rollout(tiny_policy(lora=False), pool, [spec], 2)
    recs = to_records(trajs, pool)
    assert len(recs) == 2
    for r in recs:
        assert r["monitor"] == names[1]
        assert r["monitor_round"] == 1 and r["monitor_held_out"] is True
        assert r["monitor_family"] == "concept_refit"
        assert r["thr_monitored"] is not None
        # The verdict the model was shown is derivable from the row itself.
        assert r["shown_detected"] == (r["shown_score"] > r["shown_thr"])
    # Without a pool the provenance fields are simply absent, not wrong.
    assert "monitor_round" not in to_records(trajs)[0]


def test_branch_collects_noprompt_positives_and_refit_reports_validity(tmp_path):
    """A refit must stay a CONCEPT detector. Round-1's refit scored 0.00 on unprompted concept
    text because its positives were only the evader's dodging states (10-10). Branching now also
    collects no-prompt positives, the refit fits on both, and the report carries the no-prompt TPR
    so an invalid probe is visible instead of passing on AUROC alone."""

    from next_chameleons.tica.branching import COND_POS_NP, BranchConfig, branch_collect

    pol = tiny_policy(lora=False)
    pool = MonitorPool()
    pool.add(bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    concepts = [
        ConceptTexts("cats", "train", [f"meow {i} purr nap" for i in range(12)]),
        ConceptTexts("dogs", "train", [f"woof {i} bark walk" for i in range(12)]),
    ]
    cfg = BranchConfig(
        roots=2, root_turns=2, follow_pos=3, follow_pos_noprompt=3, follow_neg_other=2,
        follow_neg_free=1, feedback="verdict",
    )
    acts, labels, rep = branch_collect(
        pol, pool, concepts, ["plain text here"] * 4, cfg, [1], seed=0
    )
    np_rows = [a for a in labels if a["cond"] == COND_POS_NP]
    assert np_rows, "no-prompt positives must be collected"
    assert rep["rows_by_cond"][COND_POS_NP] == len(np_rows)
    # Every no-prompt row is about the monitored concept and carries a text half.
    for a in np_rows:
        assert a["concept"] == a["monitored"] and a["half"] in ("A", "B") and a["active"]
    assert len(acts[1]) == len(labels)


def test_refit_rows_include_noprompt_positives(tmp_path):
    """_branch_rows must put the no-prompt positives into the positive class and expose them
    separately so the validity gate can score them."""

    from next_chameleons.tica.branching import COND_FREE, COND_OTHER, COND_POS_NP
    from next_chameleons.tica.rounds import ArmsRace

    lab = []
    for half in ("A", "B"):
        for cond in ("branch_pos", COND_POS_NP, COND_OTHER, COND_FREE):
            for _ in range(4):
                lab.append(
                    {
                        "cond": cond,
                        "monitored": "cats",
                        "concept": "cats" if cond in ("branch_pos", COND_POS_NP) else "dogs",
                        "half": half,
                        "detected": False,
                        "split": "train",
                    }
                )
    race = ArmsRace.__new__(ArmsRace)
    race.cfg = SimpleNamespace(fit=SimpleNamespace(
        positives="active", refit_max_neg=100, refit_neg_per_pos=4
    ))
    rows = race._branch_rows(lab, "cats", random.Random(0))
    assert len(rows["posA"]) == 8 and len(rows["posnpA"]) == 4
    assert set(rows["posnpA"]).issubset(set(rows["posA"]))
