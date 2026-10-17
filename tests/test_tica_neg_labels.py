"""Label-cleaned negatives (REPLICATION 2026-10-16h): a text judged to HAVE concept c is never a
negative for c, and stays one for other concepts; off by default."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "tests")


def test_clean_off_by_default_and_per_concept(tmp_path, monkeypatch):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.neg_labels import clean, stamp

    monkeypatch.delenv("TICA_NEG_LABELS", raising=False)
    xs = ["about football", "about cooking", "neutral"]
    assert clean(xs, "sports", AdaptiveConfig()) == xs and stamp(AdaptiveConfig()) == {}
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"labels": {text_id("about football"): {"sports": 0.9},
                                        text_id("about cooking"): {"food": 0.4}}}))
    a = AdaptiveConfig(neg_labels=str(f))
    assert clean(xs, "sports", a) == ["about cooking", "neutral"]
    assert clean(xs, "food", a) == xs  # below threshold
    assert clean(xs, "music", a) == xs
    assert stamp(a)["neg_labels"] == str(f)


def test_refit_forks_skip_contaminated_negatives(tmp_path, monkeypatch):
    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    bad = ["woof cat 0", "free cat"]
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"labels": {text_id(x): {"cats": 0.99} for x in bad}}))
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
                                refit_C=1.0, neg_labels=str(f),
                                episode_fork={"pos": 8, "np": 8, "other": 40, "free": 4}))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, tmp_path,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(8)]),
                         ConceptTexts("dogs", "train", ["woof cat 0", "woof 1", "woof 2"])],
        branch_negatives=["free cat", "a b", "c d", "e f"])
    seen = []
    orig_read = pol.pooled_shared_prefix
    monkeypatch.setattr(pol, "pooled_shared_prefix",
                        lambda forks, *a, **k: (seen.extend(fk[-1]["content"] for fk in forks),
                                                orig_read(forks, *a, **k))[1])
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    tr.rollout(FakeSampler().sample(1))
    assert seen and not set(bad) & set(seen)
    assert "woof 1" in seen or "woof 2" in seen
