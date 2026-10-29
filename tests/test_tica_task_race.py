"""build_task_race_data.py + tica_task_race_report.py (10-29): the interactive hide-and-solve
race of any task."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_task_race_data as B
import tica_task_race_report as T


def test_build_replaces_only_the_task_concept() -> None:
    data = {"concepts": [{"concept": "JSON data", "episode": [{"id": "x", "text": "old"}],
                          "fit": ["f"]},
                         {"concept": "sports", "episode": [{"id": "y", "text": "keep"}]}]}
    items = [("Write the JSON schema for `a`.", {"task": "json", "ref": "{}"}),
             ("Write the JSON schema for `b`.", {"task": "json", "ref": "{}"})]
    d, inputs, gold = B.build("json", data, {"k": "v"}, items)
    js = next(c for c in d["concepts"] if c["concept"] == "JSON data")
    assert len(js["episode"]) == 2 and js["fit"] == ["f"]
    sports = next(c for c in d["concepts"] if c["concept"] == "sports")
    assert sports["episode"][0]["text"] == "keep"
    assert {inputs[t["id"]] for t in js["episode"]} == set(gold) and inputs["k"] == "v"
    assert data["concepts"][0]["episode"][0]["text"] == "old"  # input not mutated


def test_report_scores_by_refit_level_with_the_task_check() -> None:
    req = 'Solve the following problem.\n\nProblem: 2+2?'
    gold = {req: {"task": "gsm8k", "gold": "4"}}
    rows = [{"ep": e, "n_deployed": k, "passed": p, "user": "Verdict: CAUGHT.\n\n" + req,
             "answer": f"... The answer is {a}."}
            for e, k, p, a in [(0, 1, True, 4), (0, 1, False, 5), (1, 1, True, 4),
                               (1, 2, False, 4)]]
    rows.append({"ep": 2, "n_deployed": 1, "passed": True, "user": "unrelated", "answer": "x"})
    rows = T.score_rows(rows, gold, workers=2)
    assert [r["correct"] for r in rows] == [True, False, True, True, None]
    lv = T.by_level(rows)
    assert lv["deployed_1"]["n"] == 3 and lv["deployed_1"]["pass_rate"][0] == round(2 / 3, 3)
    assert lv["deployed_1"]["acc_passed"][0] == 1.0
    assert lv["deployed_2"]["solved_and_passed"][0] == 0.0
