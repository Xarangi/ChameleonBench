"""tica_math_race_report: answers are scored against the gold of the request they answered; the fork
panel summary groups by refit level and gives episode-clustered ratios (accuracy among PASSED forks
is the key number)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_math_race_report import cluster_ci, gold_for, pred, summarise_forks


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
    assert a1["pass_rate"][0] == 0.75 and a2["pass_rate"][0] == 0.25
    assert abs(a1["acc_passed"][0] - 2 / 3) < 1e-3 and a2["acc_passed"][0] == 1.0
    assert a1["acc_passed"][3] == 30  # denominator = passed forks
    lo, hi = a1["acc_passed"][1], a1["acc_passed"][2]
    assert lo <= a1["acc_passed"][0] <= hi
    assert cluster_ci({"a": []}, lambda r: 1, lambda r: 1) is None
