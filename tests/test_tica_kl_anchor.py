"""KL capability anchors (REPLICATION 2026-10-15b): top-k + tail KL, the episode KL term (kl_ref
nobrief / same), the capability-fork KL and the retain-set anchor. All OFF by default."""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "tests")

from next_chameleons.tica.adaptive import AdaptiveConfig
from next_chameleons.tica.policy import topk_kl, topk_targets


def test_kl_terms_off_by_default():
    a = AdaptiveConfig()
    assert a.kl_weight == 0 and a.cap_gen_kl_weight == 0 and a.anchor_weight == 0
    assert a.cap_gen_kl_all is False


def test_topk_kl_bounds_full_kl():
    g = torch.Generator().manual_seed(0)
    lb = torch.log_softmax(torch.randn(6, 11, generator=g), -1)
    lp = torch.log_softmax(torch.randn(6, 11, generator=g), -1)
    full = (lb.exp() * (lb - lp)).sum(-1)
    exact = topk_kl(lp, topk_targets(lb, 11))  # every token in the top-k: the full KL
    assert torch.allclose(exact, full, atol=1e-5)
    coarse = topk_kl(lp, topk_targets(lb, 3))
    assert (coarse <= full + 1e-6).all() and (coarse >= -1e-6).all()
    assert topk_kl(lb, topk_targets(lb, 3)).abs().max() < 1e-6  # identical distributions
    x = lp.clone().requires_grad_(True)
    topk_kl(x, topk_targets(lb, 3)).sum().backward()
    assert x.grad.abs().sum() > 0


def _setup(tmp_path, **adaptive):
    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.episodes import EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    with torch.no_grad():  # LoRA B starts at zero (policy == base): make them differ
        g = torch.Generator().manual_seed(3)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats"]), round=0, family="concept_linear")
    texts = ["aa", "bb b", "cc cc", "dd"]
    inputs = {text_id(x): f"req {x}" for x in texts}

    def opening(cond, text, concept, named=True):
        req = inputs[text_id(text)]
        return req if cond == "nobrief" else f"monitor on {concept}\n\n{req}"

    def render(cond, concept, shown, nxt, *, feedback="score", named=True, notice=None):
        return f"verdict on {concept}\n\n{inputs[text_id(nxt)]}"

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", texts, "cats", "cats",
                                "concept_L1") for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=2, turns=4, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_every=0, single_pass=True, reloc_every=0, **adaptive))
    tr = AdaptiveSupervisedTrainer(pol, pool, FakeSampler(), cfg, tmp_path,
                                   prompts=(opening, render), inputs=inputs)
    return tr, pol, inputs


def test_episode_kl_default_path_unchanged(tmp_path):
    """kl_weight 0 (default): no KL metrics and the same loss as before the feature."""

    tr, _, _ = _setup(tmp_path / "a")
    rec = tr.step()
    assert "train/kl" not in rec and not any(k.startswith("anchor/") for k in rec)


def test_episode_kl_nobrief_and_same(tmp_path):
    tr0, _, _ = _setup(tmp_path / "off")
    loss0 = tr0.step()["train/loss"]
    for ref in ("nobrief", "same"):
        tr, _, _ = _setup(tmp_path / ref, kl_weight=1.0, kl_ref=ref, kl_topk=5)
        rec = tr.step()
        assert rec["train/kl"] > 0 and rec["train/kl_ref_fallback"] == 0
        assert abs(rec["train/loss"] - loss0) > 1e-6  # the term enters the loss
    # the unmonitored counterfactual keeps the assistant texts, drops brief / verdicts
    conv = [{"role": "user", "content": "monitor on cats\n\nreq aa"},
            {"role": "assistant", "content": "aa"},
            {"role": "user", "content": "verdict on cats\n\nreq bb b"},
            {"role": "assistant", "content": "bb b"}]
    nb = tr._nobrief_conv(conv)
    assert [m["content"] for m in nb] == ["req aa", "aa", "req bb b", "bb b"]
    assert tr._nobrief_conv([{"role": "user", "content": "x"},
                             {"role": "assistant", "content": "unknown"}]) is None


def test_episode_kl_zero_when_policy_is_base(tmp_path):
    tr, pol, _ = _setup(tmp_path, kl_weight=1.0, kl_ref="same", kl_topk=5)
    with torch.no_grad():
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.zero_()
    assert tr.step()["train/kl"] < 1e-6


def test_retain_anchor_step(tmp_path):
    cache = tmp_path / "anchor.json"
    cache.write_text(json.dumps([{"prompt": f"p{k}", "answer": f"ans {k} ok"} for k in range(5)]))
    tr, _, _ = _setup(tmp_path / "r", anchor_weight=1.0, anchor_n=3, anchor_cache=str(cache),
                      kl_topk=5)
    rec = tr.step()
    assert rec["anchor/n"] == 3 and rec["anchor/kl"] > 0


def test_retain_anchor_generates_cache_once(tmp_path, monkeypatch):
    src = tmp_path / "g.json"
    src.write_text(json.dumps([{"question": f"q{k}", "answer": "1"} for k in range(4)]))
    tr, pol, _ = _setup(tmp_path / "r", anchor_weight=1.0, anchor_n=2, anchor_pool=4,
                        anchor_sources=(str(src),), kl_topk=5)
    calls = []

    def fake_generate(convs, max_new, **kw):
        calls.append(len(convs))
        return [{"text": f"base answer {j}", "finished": True, "ids": [1], "prompt_ids": [1],
                 "stop": None} for j in range(len(convs))]

    monkeypatch.setattr(pol, "generate", fake_generate)
    tr.step()
    tr.step()
    assert len(calls) == 1  # generated once, cached
    assert json.loads((tmp_path / "r" / "anchor_cache.json").read_text())


def test_cap_gen_kl_all_trains_wrong_samples_with_kl_only(tmp_path, monkeypatch):
    from next_chameleons.tica.trainers import mathrl

    tr, pol, _ = _setup(tmp_path, cap_gen_weight=1.0, cap_gen_episodes=2, cap_gen_samples=2,
                        cap_gen_skip=0, cap_gen_kl_weight=1.0, cap_gen_kl_all=True, kl_topk=5)
    q = [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}]
    monkeypatch.setattr(mathrl, "load_questions", lambda: (q * 10, []))

    def fake_generate(convs, max_new, **kw):
        out = []
        for j, c in enumerate(convs):
            gold = "2" if "1+1" in c[-1]["content"] else "4"
            text = (f"x = 1 + 1 = {gold}. The answer is {gold}." if j % 2 == 0
                    else "I am not sure. The answer is 7.")
            out.append({"text": text, "finished": True, "ids": [1], "prompt_ids": [1],
                        "stop": None})
        return out

    monkeypatch.setattr(pol, "generate", fake_generate)
    rec = tr.step()
    assert rec["cap_gen/n_train"] == 4 and rec["cap_gen/n_kl_only"] == 2
    assert rec["cap_gen/kl"] > 0 and rec["cap_gen/kl_misaligned"] == 0


def test_cap_free_forks_valid_nll_invalid_kl_nobrief_kl(tmp_path, monkeypatch):
    """adaptive.cap_free_*: hiding forks splice a NEW concept request into the turn's verdict
    message; valid samples get NLL (+ hide), invalid ones KL only; bare no-brief forks KL only;
    cap_gen/acc stays a GSM8K-only number."""
    from next_chameleons.tica.trainers import mathrl
    from next_chameleons.tica.trainers import supervised as sup

    tr, pol, _ = _setup(tmp_path, cap_gen_weight=1.0, cap_gen_episodes=1, cap_gen_samples=2,
                        cap_gen_skip=0, cap_gen_kl_weight=1.0, cap_gen_kl_all=True, kl_topk=5,
                        cap_free_episodes=2, cap_free_nobrief=2)
    tr.cap_free_pool = {"cats": [("new cat request A", "VALID A"),
                                 ("new cat request B", "VALID B")]}
    monkeypatch.setattr(sup, "check_valid", lambda c, x: "VALID" in x)
    q = [{"question": "1+1?", "answer": "2"}]
    monkeypatch.setattr(mathrl, "load_questions", lambda: (q * 10, []))
    seen = []

    def fake_generate(convs, max_new, **kw):
        out = []
        for j, c in enumerate(convs):
            seen.append(c)
            last = c[-1]["content"]
            if "1+1" in last:
                text = "x = 1 + 1 = 2. The answer is 2." if j % 2 == 0 else "The answer is 7."
            else:
                text = "VALID answer" if j % 2 == 0 else "broken answer"
            out.append({"text": text, "finished": True, "ids": [1], "prompt_ids": [1],
                        "stop": None})
        return out

    monkeypatch.setattr(pol, "generate", fake_generate)
    rec = tr.step()
    free = [c for c in seen if len(c) > 1 and "new cat request" in c[-1]["content"]]
    bare = [c for c in seen if len(c) == 1]
    assert len(free) == 4 and all(c[-1]["content"].startswith("verdict on cats\n\n") for c in free)
    assert len(bare) == 4  # 2 no-brief forks x 2 samples, no brief / history
    assert rec["cap_free/n"] == 4 and rec["cap_free/valid"] == 0.5
    assert rec["cap_free/n_nobrief"] == 4
    assert rec["cap_gen/n_samples"] == 2 and rec["cap_gen/acc"] == 0.5  # GSM8K samples only
    # items: GSM8K 1 NLL + 1 KL; free 2 NLL + 2 KL; no-brief 4 KL (every sample non-empty)
    assert rec["cap_gen/n_train"] == 10 and rec["cap_gen/n_kl_only"] == 7
