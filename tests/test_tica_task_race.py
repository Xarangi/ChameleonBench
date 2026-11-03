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


def test_report_checks_passing_answers_with_the_task_check() -> None:
    req = 'Solve the following problem.\n\nProblem: 2+2?'
    fn = T.make_correct({req: {"task": "gsm8k", "gold": "4"}}, workers=2)
    rows = [{"user": "Verdict: CAUGHT.\n\n" + req, "answer": f"The answer is {a}."} for a in (4, 5)]
    fn.prefetch(rows)
    assert [fn(r) for r in rows] == [True, False]
    assert fn({"user": "unrelated", "answer": "x"}) is None
