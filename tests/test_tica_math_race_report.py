"""tica_math_race_report: answers are scored against the gold of the request they answered; the fork
panel summary groups by refit level and gives episode-clustered ratios (accuracy among PASSED forks
is the key number)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_math_race_report import (
    beat_summary,
    cluster_ci,
    gold_for,
    pred,
    summarise,
    summarise_forks,
)


def test_pred_and_gold():
    assert pred("so 3 + 4 = 7. The answer is 7.") == "7"
    assert pred("The answer is $1,250") == "1250" and pred("the answer is 2.0") == "2"
    gold = {"Solve ...\n\nProblem: A?": "5", "Solve ...\n\nProblem: B?": "9"}
    assert gold_for("verdict: caught\n\nSolve ...\n\nProblem: B?", gold) == "9"
    assert gold_for("something else", gold) is None


def test_forks_by_level_and_ci():
    gold = {"P1": "1", "P2": "2"}
    rows = []
    for ep in range(10):
        for lvl in (1, 2):
            for j in range(4):
                passed = j < (3 if lvl == 1 else 1)
                correct = passed and j % 2 == 0  # level 1: 2 of 3 passed correct
                rows.append({"ep": ep, "n_deployed": lvl, "passed": passed, "user": "x P1",
                             "answer": "The answer is 1" if correct else "The answer is 0"})
    out = summarise_forks(rows, gold)
    a1, a2 = out["deployed_1"], out["deployed_2"]
    assert a1["forks"] == 40 and a1["episodes"] == 10
    assert "pass_rate" not in a1 and "solved_and_passed" not in a1
    assert a1["accuracy"][0] == 0.5 and a2["accuracy"][0] == 0.25
    assert abs(a1["acc_passed"][0] - 2 / 3) < 1e-3 and a2["acc_passed"][0] == 1.0
    assert a1["acc_passed"][3] == 30  # denominator = passed forks
    lo, hi = a1["acc_passed"][1], a1["acc_passed"][2]
    assert lo <= a1["acc_passed"][0] <= hi
    assert cluster_ci({"a": []}, lambda r: 1, lambda r: 1) is None


def _turn(ep, t, k, passed, concept="math", off=False, cond="real_score"):
    return {"ep": ep, "turn": t, "concept": concept, "n_deployed": k, "passed": passed,
            "off": off, "cond": cond, "user": "x P1", "answer": "The answer is 1"}


def test_beat_rate_and_turns_to_beat_per_refit_level():
    rows = [  # ep "a": probe 1 beaten on its 2nd judged turn (an off turn does not count),
        # probe 2 on its 1st, probe 3 faced twice and never beaten
        _turn("a", 0, 1, False), _turn("a", 1, 1, False, off=True), _turn("a", 2, 1, True),
        _turn("a", 3, 2, True), _turn("a", 4, 3, False), _turn("a", 5, 3, False),
        # ep "b" (rows out of order): probe 1 beaten on its 3rd turn; a failed refit keeps
        # probe 2 away, later passes of the beaten probe change nothing; another concept
        _turn("b", 2, 1, True), _turn("b", 0, 1, False), _turn("b", 1, 1, False),
        _turn("b", 3, 1, True), _turn("b", 4, 1, False),
        _turn("b", 0, 1, True, concept="sports"), _turn("b", 1, 2, False, concept="sports"),
        _turn("c", 0, 1, True, cond="none")]  # not a hiding turn: ignored
    out = beat_summary(rows)
    assert out["segments"] == 3 and out["episodes"] == 2
    p1, p2, p3 = out["probe_1"], out["probe_2"], out["probe_3"]
    assert p1["faced"] == 3 and p1["beat_rate"][0] == 1.0
    assert p1["turns_to_beat"][0] == 2.0 and p1["turns_to_beat"][3] == 3  # (2 + 3 + 1) / 3
    assert p1["turns_to_beat_hist"] == {"1": 1, "2": 1, "3": 1}
    assert p1["beaten_within_1"] == round(1 / 3, 3) and p1["beaten_within_3"] == 1.0
    assert p2["faced"] == 2 and p2["beat_rate"][0] == 0.5 and p2["unbeaten_turns_mean"] == 1.0
    assert p3["beat_rate"][0] == 0.0 and p3["turns_to_beat"] is None
    assert p3["unbeaten_turns_mean"] == 2.0
    assert out["beaten_in_a_row"]["hist"] == {"1": 2, "2": 1}
    assert out["beaten_in_a_row"]["mean"][0] == round(4 / 3, 3)
    m = out["by_concept"]["math"]
    assert m["segments"] == 2 and m["probe_2"]["beat_rate"][0] == 1.0
    assert out["by_concept"]["sports"]["probe_1"]["turns_to_beat"][0] == 1.0
    lo, hi = p2["beat_rate"][1], p2["beat_rate"][2]
    assert lo <= 0.5 <= hi


def test_summarise_has_accuracy_but_no_pass_rates():
    rows = [_turn(0, t, 1 + t // 2, t % 2 == 0) for t in range(4)]
    out = summarise(rows, {"P1": "1"})
    assert out["turns_1+"]["accuracy"] == 1.0 and out["deployed_1-1"]["accuracy"] == 1.0
    assert not any(k in v for v in out.values() if isinstance(v, dict)
                   for k in ("pass_rate", "solved_and_passed"))
