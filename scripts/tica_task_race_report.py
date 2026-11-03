"""Score an INTERACTIVE hide-and-solve race of any task (build_task_race_data.py + tica_switch_eval
--answers free --fork-panel M --fork-out F --gen-out G; 10-29). Refits follow the training rule
only: a new probe is fit after the model passes the current one (11-02: the exogenous --refit-every
schedule is gone).

Per NAME and per refit probe k (overall and per concept), tica_math_race_report.race_summary:
faced, beaten, beat_rate, turns to beat (mean + histogram), probes beaten in a row, the accuracy of
the PASSING solution (the task check on the main answer that beat probe k) and the FLUKE CHECK
(share of the fork panel at that pass turn -- same history + verdict, new problems, same probe pool
-- that also beats the probe: mean and per pass), plus first_answer_acc (turn-0 answer accuracy,
one per episode: the capability anchor). Only passing and turn-0 answers are checked, so a report
runs few unit tests. No per-judged-turn numbers (11-03, user): no pass rates and no accuracy over
all turns / caught turns / all forks. Ratios: [value, 95% CI low, CI high, denominator], bootstrap
over EPISODES.
Task checks: GSM8K final number, MBPP unit tests (sandboxed), HumanEval-JS Node tests, SQL
execution match, JSON function schema (tica/validity.py). Aggregates only; needs the training venv
(validity / sandbox).

  python scripts/tica_task_race_report.py --gold gold_mbpp.json \
      --race strict4b='RUN/strict4b_p*.jsonl' strict2b='RUN/strict2b_p*.jsonl' --out report.json
  (each main X.jsonl pairs with X_forks.jsonl; merged files: --gens N=G.jsonl --forks N=F.jsonl)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tica_math_race_report import add_run_args, load_runs, pred, race_summary


def gold_for(user: str, gold: dict) -> tuple[str, dict] | None:
    """(request, gold record) whose request ends the user turn (the verdict message comes first)."""

    tail = (user or "").strip()
    for req, rec in gold.items():
        if tail.endswith(req.strip()):
            return req, rec
    return None


def correct(req: str, rec: dict, answer: str) -> bool:
    task = rec["task"]
    if task == "gsm8k":
        return pred(answer or "") == rec["gold"]
    if task == "mbpp":
        from next_chameleons.benchmarks.capability_heldout import extract_code, run_tests

        return bool(run_tests(extract_code(answer or ""), rec["imports"], rec["tests"])["passed"])
    if task == "jseval":
        from next_chameleons.benchmarks.capability_heldout import extract_js_code, run_js_tests

        return bool(run_js_tests(extract_js_code(answer or ""), rec["tests"])["passed"])
    from next_chameleons.tica.validity import check_task

    concept = {"sql": "SQL code", "json": "JSON data"}[task]
    try:
        return bool(check_task(concept, answer or "", req, rec["ref"]))
    except Exception:  # a malformed answer is wrong, never a crash
        return False


def make_correct(gold: dict, workers: int):
    """correct(row) for race_summary, memoised; prefetch(rows) runs the checks in parallel first."""

    memo: dict = {}

    def key(r):
        return (r.get("user", ""), r.get("answer", ""))

    def one(r):
        g = gold_for(r.get("user", ""), gold)
        return None if g is None else correct(g[0], g[1], r.get("answer", ""))

    def prefetch(rows):
        todo = {key(r): r for r in rows if key(r) not in memo}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, ok in zip(todo, ex.map(one, todo.values()), strict=True):
                memo[k] = ok

    def fn(r):
        if key(r) not in memo:
            memo[key(r)] = one(r)
        return memo[key(r)]

    fn.prefetch = prefetch
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    add_run_args(ap)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = json.loads(Path(a.gold).read_text())
    correct_fn = make_correct(gold, a.workers)
    res: dict = {}
    for name, (rows, frows) in load_runs(a.race, a.gens, a.forks).items():
        # passing answers + each episode's turn-0 answer (first_answer_acc)
        correct_fn.prefetch([r for r in rows if r.get("passed") or r.get("turn") == 0])
        res[name] = race_summary(rows, frows, correct_fn)
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "by_concept"}),
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
