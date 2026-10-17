"""Eval-only "ood" concept split (scripts/build_final_ood.py): data access, race topic selection and
base-probe negative pools, on a tiny synthetic data file (CPU, no model)."""

from __future__ import annotations

import json

import pytest

from next_chameleons.tica.answer_format import concept_fit_texts
from next_chameleons.tica.data import CONCEPT_SPLITS, Level1Data, select_topics


def _concept(name, split, n=3):
    part = lambda p: [{"id": f"{name}-{p}-{i}", "text": f"{name} {p} {i}"}  # noqa: E731
                      for i in range(n)]
    return {"concept_id": name, "concept": name, "genre": "g", "split": split,
            "fit": part("fit"), "calib": part("calib"), "episode": part("episode")}


def _write(tmp_path, with_ood=True):
    cs = [_concept("tr_a", "train"), _concept("tr_b", "train"), _concept("dv", "dev"),
          _concept("ho", "heldout")]
    if with_ood:
        cs += [_concept("math", "ood"), _concept("code", "ood")]
    neg = {p: [{"id": f"n-{p}", "text": f"free {p}"}] for p in ("fit", "calib", "episode")}
    path = tmp_path / ("ood.json" if with_ood else "old.json")
    path.write_text(json.dumps({"concepts": cs, "negatives": neg}))
    return Level1Data(path)


def test_ood_concepts_and_topic_selection(tmp_path):
    d = _write(tmp_path)
    assert "ood" in CONCEPT_SPLITS
    ood = d.concepts("episode", ("ood",))
    assert [c.concept for c in ood] == ["math", "code"]
    assert ood[0].texts == ["math episode 0", "math episode 1", "math episode 2"]
    assert select_topics(d, "ood") == ["math", "code"]
    assert select_topics(d, "heldout") == ["ho"]
    assert select_topics(d, "math, ho") == ["math", "ho"]
    # the race's episode / branch pools now include ood concepts
    assert {c.concept for c in d.concepts("fit", CONCEPT_SPLITS)} >= {"math", "code"}
    with pytest.raises(ValueError):
        select_topics(d, "physics")


def test_old_file_unchanged(tmp_path):
    d = _write(tmp_path, with_ood=False)
    old = ("train", "dev", "heldout")
    assert ([c.concept for c in d.concepts("episode", CONCEPT_SPLITS)]
            == [c.concept for c in d.concepts("episode", old)])
    assert select_topics(d, "train") == ["tr_a", "tr_b"]
    with pytest.raises(ValueError):
        select_topics(d, "ood")  # no ood concepts: an explicit error, not an empty race
    # held-out base probe negatives: train + heldout (no dev), as before
    assert list(concept_fit_texts(d, "ho")) == ["tr_a", "tr_b", "ho"]


def test_ood_base_probe_negatives(tmp_path):
    d = _write(tmp_path)
    # an ood concept is fit like a held-out one: positives = itself, others = train + heldout +
    # the other ood concepts (never dev)
    fit = concept_fit_texts(d, "math")
    assert list(fit) == ["tr_a", "tr_b", "ho", "math", "code"]
    assert fit["math"] == ["math fit 0", "math fit 1", "math fit 2"]
    # a train concept keeps the trainer's other-TRAIN negatives: no ood concept shapes it
    assert list(concept_fit_texts(d, "tr_a", ("train",))) == ["tr_a", "tr_b"]
    # a dev concept still finds its own positives
    assert "dv" in concept_fit_texts(d, "dv")
