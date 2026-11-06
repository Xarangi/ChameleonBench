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
execution match, JSON function schema (tica/validity.py); held-out tasks (11-03): HumanEval-Python
check(entry_point) (MBPP sandbox), HumanEval-Java asserts under `java -ea`, MGSM fr / ru / th final
number; 11-06: MBPP-Java (as HumanEval-Java), MGSM de / ja, Belebele fi / pt / tr / el / ta and
MMLU physics / cs / bio / med (the "Answer: <letter>" choice). MGSM and Belebele also record
the answer's fastText language (lid.176): per probe k
passing_in_language / passing_correct_and_in_language (+ _n), and first_answer_in_language /
first_answer_correct_and_in_language, next to passing_correct / first_answer_acc (= the number).
Aggregates only; needs the training venv (validity / sandbox / fasttext).

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
from tica_math_race_report import add_run_args, load_runs, piece_info, pred, race_summary


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
    if task.startswith("mgsm_"):
        from next_chameleons.benchmarks.capability_heldout import _norm_number, mgsm_number

        return mgsm_number(answer or "", rec["lang"]) == _norm_number(rec["gold"])
    if task == "pyeval":
        from next_chameleons.benchmarks.capability_heldout import extract_code, run_humaneval_py

        return bool(run_humaneval_py(extract_code(answer or ""), rec)["passed"])
    if task.startswith(("belebele_", "mmlu_")):  # 11-06: the chosen option letter
        from next_chameleons.benchmarks.capability_heldout import mc_letter

        return mc_letter(answer or "") == rec["gold"]
    if task in ("javaeval", "mbppjava"):
        from next_chameleons.benchmarks.capability_heldout import extract_java_code, run_java_tests

        return bool(run_java_tests(extract_java_code(answer or ""), rec)["passed"])
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


def in_language(rec: dict, answer: str) -> bool:
    """MGSM: the answer's most likely language (fastText lid.176) is the item's language."""

    from next_chameleons.benchmarks.capability_heldout import answer_language

    return answer_language(answer or "")[0] == rec["lang"]


# extra checks per task family, reported next to the headline check (task correctness)
_LANG_CHECKS = {
    "in_language": lambda req, rec, ans: in_language(rec, ans),
    "correct_and_in_language": lambda req, rec, ans: (correct(req, rec, ans)
                                                      and in_language(rec, ans)),
}
EXTRA_CHECKS = {"mgsm": _LANG_CHECKS, "belebele": _LANG_CHECKS}  # language tasks (11-06: belebele)


def extra_checks(gold: dict) -> dict:
    """{name: check(req, rec, answer)} for the gold's task family (empty for most tasks)."""

    fams = {rec["task"].split("_", 1)[0] for rec in gold.values()}
    return {k: v for f in sorted(fams) for k, v in EXTRA_CHECKS.get(f, {}).items()}


def make_correct(gold: dict, workers: int, check=correct):
    """check(row) for race_summary, memoised; prefetch(rows) runs the checks in parallel first."""

    memo: dict = {}

    def key(r):
        return (r.get("user", ""), r.get("answer", ""))

    def one(r):
        g = gold_for(r.get("user", ""), gold)
        return None if g is None else check(g[0], g[1], r.get("answer", ""))

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


def add_extra(res: dict, extra: dict) -> None:
    """Merge an extra check's race_summary into res: per probe k passing_<name> (+ _n) and
    first_answer_<name>, overall and per concept (the beat statistics are the same rows)."""

    for name, ex in extra.items():
        for blk, xb in ((res, ex), *((res["by_concept"].get(c, {}), v)
                                     for c, v in ex.get("by_concept", {}).items())):
            for k, v in xb.items():
                if k.startswith("probe_") and isinstance(v, dict) and k in blk:
                    blk[k][f"passing_{name}"] = v.get("passing_correct")
                    blk[k][f"passing_{name}_n"] = v.get("passing_correct_n")
            if "first_answer_acc" in xb:
                blk[f"first_answer_{name}"] = xb["first_answer_acc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    add_run_args(ap)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = json.loads(Path(a.gold).read_text())
    correct_fn = make_correct(gold, a.workers)
    extras = {k: make_correct(gold, a.workers, v) for k, v in extra_checks(gold).items()}
    res: dict = {}
    for name, (rows, frows) in load_runs(a.race, a.gens, a.forks).items():
        # passing answers + each episode's turn-0 answer (first_answer_acc)
        scored = [r for r in rows if r.get("passed") or r.get("turn") == 0]
        correct_fn.prefetch(scored)
        res[name] = {**race_summary(rows, frows, correct_fn), **piece_info(name)}
        for fn in extras.values():
            fn.prefetch(scored)
        add_extra(res[name], {k: race_summary(rows, frows, fn) for k, fn in extras.items()})
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "by_concept"}),
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
