"""tica_math_race_report: answers are scored against the gold of the request they answered; per
refit probe: beat rate, turns to beat, accuracy of the passing solution and the fork-panel fluke
check."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_math_race_report import beat_summary, gold_for, pred, race_summary


def test_pred_and_gold():
    assert pred("so 3 + 4 = 7. The answer is 7.") == "7"
    assert pred("The answer is $1,250") == "1250" and pred("the answer is 2.0") == "2"
    gold = {"Solve ...\n\nProblem: A?": "5", "Solve ...\n\nProblem: B?": "9"}
    assert gold_for("verdict: caught\n\nSolve ...\n\nProblem: B?", gold) == "9"
    assert gold_for("something else", gold) is None


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


def test_passing_solution_and_fluke_check_per_probe():
    gold = {"P1": "1", "P2": "2"}

    def main(ep, t, k, passed, ans, eid=None):
        r = {**_turn(ep, t, k, passed), "user": "v\n\nP1", "answer": f"The answer is {ans}"}
        return {**r, "eid": eid} if eid else r

    rows = [main(0, 0, 1, False, 1), main(0, 1, 1, True, 1), main(0, 2, 2, False, 1),
            main(0, 3, 2, True, 0),  # probe 2 beaten by a wrong answer
            main(1, 0, 1, True, 1),  # beaten at turn 0: no fork panel there
            main("x", 1, 1, True, 1, eid="real_score|math|7")]  # new files: eid join
    # legacy fork ids "cond|concept|N" join main ep N; forks of other turns / levels are ignored
    forks = [{"ep": "real_score|math|0", "turn": 1, "concept": "math", "n_deployed": 1,
              "passed": j < 3} for j in range(4)]
    forks += [{"ep": "real_score|math|0", "turn": 3, "concept": "math", "n_deployed": 2,
               "passed": j < 1} for j in range(4)]
    forks += [{"ep": "real_score|math|0", "turn": 2, "concept": "math", "n_deployed": 2,
               "passed": True}]
    forks += [{"eid": "real_score|math|7", "ep": "real_score|math|7", "turn": 1,
               "concept": "math", "n_deployed": 1, "passed": False}]

    def correct(r):
        g = gold_for(r["user"], gold)
        return None if g is None else pred(r["answer"]) == g

    out = race_summary(rows, forks, correct)
    p1, p2 = out["probe_1"], out["probe_2"]
    assert p1["faced"] == 3 and p1["beaten"] == 3 and p1["passing_correct_n"] == "3/3"
    assert p1["fluke_share_per_pass"] == [0.0, 0.75] and p1["passes_without_forks"] == 1
    assert p1["fluke_share_mean"][0] == 0.375 and p1["forks_per_pass"] == 2.5
    assert p2["passing_correct_n"] == "0/1" and p2["fluke_share_per_pass"] == [0.25]
    assert p2["turns_to_beat"][0] == 2.0
    assert out["by_concept"]["math"]["probe_2"]["fluke_share_mean"][0] == 0.25
    assert not any(k in p1 for k in ("pass_rate", "accuracy", "acc_caught"))
    # turn-0 answers, one per episode: ep 0 and ep 1 correct ("x"/eid 7 has no turn 0)
    assert out["first_answer_acc"][0] == 1.0 and out["first_answer_acc"][3] == 2
