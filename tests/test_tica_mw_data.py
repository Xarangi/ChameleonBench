"""Model-written concept data builder (scripts/build_final_v5_mw.py, REPLICATION 2026-10-21f) on a
synthetic data file: splits by request, new ids / request map, judge + finished filter, length
matching, untouched concepts / negatives, label merge; and the generator's prompt ==
Policy.render."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_final_v5_mw import build, length_cap, merge_labels, request_splits, text_id
from tica_mw_generate import render_user, sources

SPLITS = ("fit", "calib", "episode")


def _t(x):
    return {"id": text_id(x), "text": x}


def _data():
    a = {"concept": "A", "split": "train",
         "fit": [_t("a0 fit"), _t("a1 fit"), _t("a2 fit")], "calib": [_t("a3 calib")],
         "episode": [_t("a4 ep")]}
    m = {"concept": "math", "split": "train", "fit": [_t("m0")], "calib": [_t("m1")],
         "episode": [_t("m2")]}
    h = {"concept": "H", "split": "heldout", "fit": [_t("h0")], "calib": [_t("h1")],
         "episode": [_t("h2")], "hard_neg": {"fit": ["hn"], "calib": [], "episode": []}}
    neg = {"fit": [_t("n0")], "calib": [_t("n1")], "episode": [_t("n2")]}
    return {"source": {}, "seed": 0, "concepts": [a, m, h], "negatives": neg}


def _inputs(d):
    inp = {}
    for c in d["concepts"]:
        for sp in SPLITS:
            for t in c[sp]:
                inp[t["id"]] = f"req {t['text']}"
    # a3 (calib) shares its request with a0 / a1 (fit): the request has 2 fit + 1 calib texts
    for x in ("a0 fit", "a1 fit", "a3 calib"):
        inp[text_id(x)] = "shared req"
    return inp


def _gens(d, inp, samples=2):
    rows = sources(d, inp, "train", ["math"])
    gens, scores = [], {}
    for r in rows:
        for k in range(samples):
            ans = f"answer to {r['src_id']} #{k}" + "!" * (5 * k)
            g = dict(r, gid=f"{r['src_id']}:{k}", sample=k, finished=True, answer=ans)
            gens.append(g)
            scores[g["gid"]] = {"gid": g["gid"], "p_concept": 0.9, "p_answers": 0.9}
    return rows, gens, scores


def test_render_matches_policy():
    from next_chameleons.tica.policy import render_keep_reasoning

    req = "  Write a short poem about the sea.\n"
    assert render_user(req) == render_keep_reasoning([{"role": "user", "content": req}], True,
                                                     False)


def test_request_splits_majority():
    rows = [{"user": "r", "src_split": "fit"}, {"user": "r", "src_split": "fit"},
            {"user": "r", "src_split": "calib"}, {"user": "s", "src_split": "episode"},
            {"user": "t", "src_split": "calib"}, {"user": "t", "src_split": "episode"}]
    assert request_splits(rows) == {"r": "fit", "s": "episode", "t": "calib"}


def test_build_splits_ids_and_untouched():
    d = _data()
    inp = _inputs(d)
    rows, gens, scores = _gens(d, inp)
    assert {r["concept"] for r in rows} == {"A"}  # math excluded, held-out not selected
    new, new_inp, rep = build(d, inp, gens, scores, fill=False, len_cap_q=None)
    a = new["concepts"][0]
    # the shared request (2 fit + 1 calib) goes to fit: its calib source is moved, so calib is
    # empty without fill; every answer lands in exactly one split
    assert rep["A"]["requests_moved_split"] == 1
    ids = {sp: {t["id"] for t in a[sp]} for sp in SPLITS}
    assert not (ids["fit"] & ids["calib"]) and not (ids["fit"] & ids["episode"])
    assert len(a["fit"]) == 3 and len(a["calib"]) == 0 and len(a["episode"]) == 1
    for sp in SPLITS:
        for t in a[sp]:
            assert t["id"] == text_id(t["text"]) and t["text"].startswith("answer to")
            assert new_inp[t["id"]] in ("shared req", "req a2 fit", "req a4 ep")
    # requests of a split's texts never appear in another split
    req_of = {sp: {new_inp[t["id"]] for t in a[sp]} for sp in SPLITS}
    assert not (req_of["fit"] & (req_of["calib"] | req_of["episode"]))
    # closest length to the source text = sample 0 (no padding)
    assert all(t["text"].endswith(("#0",)) for sp in SPLITS for t in a[sp])
    # math, held-out, negatives, hard_neg unchanged
    assert new["concepts"][1] == d["concepts"][1] and new["concepts"][2] == d["concepts"][2]
    assert new["negatives"] == d["negatives"]
    assert new["source"]["model_written"]["concepts"] == ["A"]
    assert d["concepts"][0]["fit"][0]["text"] == "a0 fit"  # input not mutated


def test_build_fill_and_filters():
    d = _data()
    inp = _inputs(d)
    _, gens, scores = _gens(d, inp)
    # source a2: sample 0 unfinished, sample 1 kept; source a4: concept absent in both
    for g in gens:
        if g["src_id"] == text_id("a2 fit") and g["sample"] == 0:
            g["finished"] = False
        if g["src_id"] == text_id("a4 ep"):
            scores[g["gid"]]["p_concept"] = 0.1
    new, _, rep = build(d, inp, gens, scores, fill=True, len_cap_q=None)
    a = new["concepts"][0]
    assert len(a["episode"]) == 0  # nothing kept for the only episode request
    unfinished = f"answer to {text_id('a2 fit')} #0"
    assert all(t["text"] != unfinished for sp in SPLITS for t in a[sp])
    assert len(a["fit"]) == 3
    # fill: calib (moved request) stays empty -- spares come only from requests of that split
    assert len(a["calib"]) == 0
    assert rep["A"]["kept_gen"] == len(gens) - 1 - 2


def test_build_drops_duplicates():
    d = _data()
    inp = _inputs(d)
    _, gens, scores = _gens(d, inp)
    for g in gens:  # every answer identical -> one text survives
        g["answer"] = "same"
    new, _, rep = build(d, inp, gens, scores, len_cap_q=None)
    a = new["concepts"][0]
    assert sum(len(a[sp]) for sp in SPLITS) == 1 and rep["A"]["dup_dropped"] == len(gens) - 1


def test_merge_labels():
    d = _data()
    keep_id, new_id = d["negatives"]["fit"][0]["id"], d["concepts"][0]["fit"][0]["id"]
    old = {"labels": {keep_id: {"A": 0.9}, "gone": {"A": 0.8}},
           "meta": [{"data": ["old.json"], "n_texts": 2}]}
    new = {"labels": {new_id: {"H": 0.7}}, "meta": {"data": ["new.json"], "n_texts": 1,
                                                    "concepts": ["A", "H"], "prompt": "v4"}}
    out = merge_labels(d, old, [new])
    assert out["labels"] == {keep_id: {"A": 0.9}, new_id: {"H": 0.7}}
    assert out["meta"]["merged_from"]["old"] == ["old.json"]


def test_length_cap():
    assert length_cap([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 0.9) == 90
    assert length_cap([5], 0.9) == 5 and length_cap([1, 2], None) is None
    d = _data()
    inp = _inputs(d)
    _, gens, scores = _gens(d, inp)
    # published A lengths 6-8 chars, p90 = 8: only answers <= 8 chars are eligible
    for g in gens:
        g["answer"] = ("short" + g["gid"][-3:]) if g["sample"] == 1 else g["answer"]
    new, _, rep = build(d, inp, gens, scores, len_cap_q=0.9)
    a = new["concepts"][0]
    texts = [t["text"] for sp in SPLITS for t in a[sp]]
    assert texts and all(len(x) <= rep["A"]["len_cap_chars"] == 8 for x in texts)
    assert rep["A"]["src_no_eligible"] == 0
    for g in gens:  # nothing under the cap -> concept emptied and flagged, cap not loosened
        g["answer"] = g["answer"] + "x" * 20
    new, _, rep = build(d, inp, gens, scores, len_cap_q=0.9)
    assert rep["A"]["new_total"] == 0 and rep["A"]["below_80pct"]
    assert rep["A"]["src_no_eligible"] == 5
