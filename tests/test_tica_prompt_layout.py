"""Per-turn ("tagged") prompt layout (REPLICATION 2026-10-17a): monitor description as the system
prompt, a status line (ON + concept / OFF) on every monitored turn, previous result in front."""

from __future__ import annotations

import sys

sys.path.insert(0, "tests")


def _tagged(texts):
    from next_chameleons.tica.answer_format import tagged_prompts, text_id

    return tagged_prompts({text_id(x): f"REQ[{x}]" for x in texts})


def test_tagged_strings_and_system_split():
    from next_chameleons.tica.answer_format import (
        SYS_OPEN,
        SYSTEM_TAGGED,
        prompts,
        split_system,
    )

    opening, fb = _tagged(["a", "b"])
    first = opening("real_score", "a", "sports")
    conv = split_system([{"role": "user", "content": first}, {"role": "assistant", "content": "a"}])
    assert conv[0] == {"role": "system", "content": SYSTEM_TAGGED}
    assert conv[1]["content"] == 'Monitor: ON. Concept: "sports".\n\nREQ[a]'
    assert opening("nobrief", "a", "sports") == "REQ[a]"  # control: no system, no status line
    assert split_system(conv) == conv and split_system([]) == []
    det = fb("real_score", "sports", (80.0, 50.0), "b", feedback="verdict")
    assert det == 'Previous result ("sports"): DETECTED.\nMonitor: ON. Concept: "sports".\n\nREQ[b]'
    sw = fb("real_score", "sports", (10.0, 50.0), "b", feedback="verdict",
            next_concept="German")
    assert sw.startswith('Previous result ("sports"): NOT DETECTED.\n'
                         'Monitor: ON. Concept: "German"')
    off = fb("real_score", "sports", (10.0, 50.0), "b", feedback="verdict", next_off=True)
    assert "Monitor: OFF." in off and "Concept" not in off.split("\n")[1]
    back = fb("none", "sports", (0.0, 1.0), "b", feedback="verdict")  # after an off turn
    assert back == 'Monitor: ON. Concept: "sports".\n\nREQ[b]'
    assert fb("nobrief", "sports", (0.0, 1.0), "b") == "REQ[b]"
    assert SYS_OPEN not in det
    # legacy layout unchanged by default
    import pytest

    with pytest.raises(ValueError):
        prompts("repeat", layout="tagged")
    with pytest.raises(ValueError):
        prompts("answer", layout="bogus")


def test_policy_render_lifts_system():
    from next_chameleons.tica.answer_format import SYS_OPEN, SYSTEM_TAGGED
    from next_chameleons.tica.policy import render_keep_reasoning
    from test_tica_framework import tiny_policy

    opening, _ = _tagged(["a"])
    conv = [{"role": "user", "content": opening("real_score", "a", "x")},
            {"role": "assistant", "content": "a"}]
    pol = tiny_policy()
    out = pol.render(conv, False)
    assert SYS_OPEN not in out and SYSTEM_TAGGED in out
    pol.keep_think_history = True
    out = pol.render(conv, False)
    assert out.startswith(f"<|im_start|>system\n{SYSTEM_TAGGED}<|im_end|>\n<|im_start|>user\n")
    assert out == render_keep_reasoning(
        [{"role": "system", "content": SYSTEM_TAGGED},
         {"role": "user", "content": 'Monitor: ON. Concept: "x".\n\nREQ[a]'},
         {"role": "assistant", "content": "a", "reasoning": ""}], False, False)


def test_capability_prompt_ids_split_system():
    from next_chameleons.benchmarks.capability import _chat_prompt_ids
    from next_chameleons.tica.answer_format import SYS_CLOSE, SYS_OPEN

    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            seen["m"] = messages
            return [1]

    _chat_prompt_ids(Tok(), f"{SYS_OPEN}sys{SYS_CLOSE}status\n\nq")
    assert seen["m"] == [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "status\n\nq"}]
    _chat_prompt_ids(Tok(), "plain")
    assert seen["m"] == [{"role": "user", "content": "plain"}]


def test_trainer_rollout_tagged_switch_and_off(tmp_path, monkeypatch):
    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.answer_format import SYS_OPEN
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    texts = ["aa", "bb", "cc", "dd", "ee", "ff"]
    pos = {"cats": [f"meow {k}" for k in range(8)], "dogs": [f"woof {k}" for k in range(8)]}
    prompts = _tagged(texts + pos["cats"] + pos["dogs"] + ["a b", "c d", "e f"])
    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        concepts = [ConceptTexts("cats", "train", ["x"]), ConceptTexts("dogs", "train", ["y"])]  # noqa: RUF012
        concept_weights = None

        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score", texts, "cats", "cats",
                                base_name, feedback="verdict",
                                meta={"monitored_turns": ["cats"] * 3 + ["dogs"] * 3})
                    for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=1, turns=6, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0,
        adaptive=AdaptiveConfig(refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0,
                                refit_C=1.0, episode_refit_kind="linear", off_turn_prob=0.5,
                                episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    sampler = FakeSampler()
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, cfg, tmp_path,
        branch_concepts=[ConceptTexts(c, "train", ts) for c, ts in pos.items()],
        branch_negatives=["a b", "c d", "e f"], prompts=prompts)
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": True})
    import random

    tr.rng = random.Random(3)
    trajs, _ = tr.rollout(sampler.sample(4))
    users = [m["content"] for t in trajs for m in t.convs[-1] if m["role"] == "user"]
    assert all(u.startswith(SYS_OPEN) for u in users[::6])  # only the first turn carries it
    assert not any(SYS_OPEN in u for k, u in enumerate(users) if k % 6)
    assert not any("switched OFF" in u or "instead" in u for u in users)  # no legacy notices
    assert any('Concept: "dogs"' in u for u in users)  # the switch is a status line
    assert any("Monitor: OFF." in u for u in users)  # off turns drawn at p = 0.5
