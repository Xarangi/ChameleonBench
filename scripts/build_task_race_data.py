"""INTERACTIVE hide-and-solve inputs for any checkable task (10-29, user: "our hide and solve test
should also be in the interactive setting ... build the generalization beyond math").
build_gsm8k_race_data.py's construction for more tasks: the race of tica_switch_eval (verdict
feedback, cumulative pool, refits; --refit-every K for a fixed refit schedule, --fork-panel M for
many fresh items per refit level) runs unchanged on --data / --inputs overrides where ONE
concept's episode texts are replaced by one placeholder per task item whose request (inputs map)
is that item. Every other concept and every fit / calib text (base probes) is untouched.

  gsm8k  "mathematical reasoning"  GSM8K TEST problems, training request format; gold = answer
  mbpp   "Python code" (HELD-OUT)  MBPP test problems (tica_hide_solve's prompt); gold = unit tests
  sql    "SQL code"                the concept's calib / episode requests WITH the reference tables
                                   + rows (validity.sql_task_request); gold = reference (execution)
  json   "JSON data"               the concept's calib / episode requests; gold = reference (schema)
  jseval "JavaScript code"         HumanEval-JS (MultiPL-E humaneval-js; the trained concept's own
                                   data has no tests); gold = the problem's Node tests (10-30)

tica_task_race_report.py scores the answers (main turns and fork panel) by refit level.

  python scripts/build_task_race_data.py --task mbpp --data D.json --inputs I.json --n 300 \
      --out-data D_mbpp.json --out-inputs I_mbpp.json --out-gold gold_mbpp.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CONCEPT = {"gsm8k": "mathematical reasoning", "mbpp": "Python code", "sql": "SQL code",
           "json": "JSON data", "jseval": "JavaScript code"}
CAPABILITY = Path("/scratch/s/sneheel/next_chameleons_raw_cache/capability")
GSM_REQUEST = ('Solve the following problem. Show your working briefly, then finish with '
               '"The answer is <number>."\n\nProblem: {q}')


def text_id(text: str) -> str:  # = next_chameleons.tica.answer_format.text_id
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def task_items(task: str, data: dict, inputs: dict, n: int) -> list[tuple[str, dict]]:
    """[(request, gold record)] for the task, at most n, in a fixed order."""

    if task == "gsm8k":
        rows = json.loads((CAPABILITY / "gsm8k.json").read_text())
        return [(GSM_REQUEST.format(q=r["question"].strip()),
                 {"task": task, "gold": str(r["answer"])}) for r in rows[:n]]
    if task == "mbpp":
        from next_chameleons.benchmarks.capability_heldout import (
            load_heldout,
            mbpp_prompt,
            subsample,
        )

        return [(mbpp_prompt(it), {"task": task, "tests": it["test_list"],
                                   "imports": it.get("test_imports", [])})
                for it in subsample(load_heldout(CAPABILITY, "mbpp"), n)]
    if task == "jseval":
        from next_chameleons.benchmarks.capability_heldout import js_prompt, load_heldout, subsample

        return [(js_prompt(it), {"task": task, "tests": it["tests"]})
                for it in subsample(load_heldout(CAPABILITY, "humaneval_js"), n)]
    from next_chameleons.tica.validity import sql_task_request

    concept, out = CONCEPT[task], []
    c = next(x for x in data["concepts"] if x["concept"] == concept)
    for sp in ("calib", "episode"):
        for t in c.get(sp, []):
            x = t["text"] if isinstance(t, dict) else t
            q = inputs.get(text_id(x))
            if not q:
                continue
            if task == "sql":
                q = sql_task_request(q, x)
                if q is None:
                    continue
            out.append((q, {"task": task, "ref": x}))
    return out[:n]


def build(task: str, data: dict, inputs: dict,
          items: list[tuple[str, dict]]) -> tuple[dict, dict, dict]:
    """(data with the concept's episode texts replaced by one placeholder per item, extended inputs,
    {request: gold record})."""

    data, inputs, gold = json.loads(json.dumps(data)), dict(inputs), {}
    concept = next(c for c in data["concepts"] if c["concept"] == CONCEPT[task])
    texts = []
    for i, (req, rec) in enumerate(items):
        placeholder = f"[{task}-race {i}] {CONCEPT[task]} item."
        inputs[text_id(placeholder)] = req
        gold[req] = rec
        texts.append({"id": text_id(placeholder), "text": placeholder})
    concept["episode"] = texts
    return data, inputs, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(CONCEPT))
    ap.add_argument("--data", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-gold", required=True)
    a = ap.parse_args()
    data, inputs = json.loads(Path(a.data).read_text()), json.loads(Path(a.inputs).read_text())
    items = task_items(a.task, data, inputs, a.n)
    d, i, g = build(a.task, data, inputs, items)
    Path(a.out_data).write_text(json.dumps(d))
    Path(a.out_inputs).write_text(json.dumps(i))
    Path(a.out_gold).write_text(json.dumps(g))
    print(json.dumps({"task": a.task, "concept": CONCEPT[a.task], "items": len(g)}))


if __name__ == "__main__":
    main()
