"""GSM8K solve task: prompt, answer extraction, correctness and visible-working checks, question /
Dolly loading and the cached base-model generations (the pure pieces of trainers/mathrl.py and
benchmarks/capability.py). Paths are explicit arguments; the generation call is a callable, so this
module needs no model. Behaviour is identical to the originals."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

CONCEPT = "mathematical reasoning"
PROMPT = ("Solve the following problem. Show your working briefly, then finish with "
          "\"The answer is <number>.\"\n\nProblem: {q}")

_EXPR = re.compile(r"\d[\d,.]*\s*[-+*/\u00d7\u00f7x]\s*\$?\d")
_PHRASE = re.compile(r"answer is\s*\$?-?[\d,]+\.?\d*", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
# the model often RUNS ON into a hallucinated next turn after "The answer is X."
_ANSWER_IS_RE = re.compile(r"answer is\s*\$?(-?[\d,]+\.?\d*)", re.IGNORECASE)
_NEXT_TURN_RE = re.compile(r"\n\s*Question:")


def canon(num: str) -> str:
    """Canonical numeric string so '7', '7.00', '7,000'/'7000', '$7' compare equal."""
    s = num.replace(",", "").replace("$", "").strip().rstrip(".")
    if not s:
        return ""
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else repr(f)


def _gold_number(answer: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    return canon(answer.split("####")[-1].split("\n")[0])


def _last_number(text: str) -> str:
    m = _NUM_RE.findall(text.replace(",", ""))
    return canon(m[-1]) if m else ""


def extract_gsm8k_answer(generation: str) -> str:
    """Numeric answer, robust to run-on continuations: truncate at the first hallucinated
    ``Question:`` turn, prefer an explicit ``the answer is X`` or ``#### X``, else the last
    number."""
    cut = _NEXT_TURN_RE.split(generation, maxsplit=1)[0]
    m = _ANSWER_IS_RE.search(cut)
    if m:
        return canon(m.group(1))
    if "####" in cut:
        return _gold_number(cut)
    return _last_number(cut)


def work_shown(text: str) -> bool:
    """Visible working: at least one arithmetic expression (a op b) or an equation with a digit."""
    return bool(_EXPR.search(text)) or ("=" in text and any(ch.isdigit() for ch in text))


def answer_phrase(text: str) -> bool:
    """The answer ends in the requested "The answer is <number>" phrase."""
    return bool(_PHRASE.search(text))


def is_correct(text: str, gold: str) -> bool:
    """The extracted numeric answer equals the gold one (string gold is canonicalised)."""
    pred = extract_gsm8k_answer(text)
    return bool(pred) and pred == canon(str(gold))


def load_questions(train_path: str | Path, test_path: str | Path) -> tuple[list, list]:
    """(GSM8K train items, GSM8K test items) from the two materialized JSON files."""
    return (json.loads(Path(train_path).read_text()), json.loads(Path(test_path).read_text()))


def dolly_nonmath(path: str | Path, n: int, seed: int = 0) -> list[str]:
    """Non-math Dolly-15k instructions (no context, no digits, open categories) from the jsonl."""
    keep = {"open_qa", "general_qa", "brainstorming", "creative_writing", "classification"}
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            ins = r["instruction"].strip()
            if r["category"] in keep and not r["context"].strip() and ins and not any(
                    ch.isdigit() for ch in ins):
                rows.append(ins)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def gens_path(out: str | Path) -> Path:
    """Where the base generations of a run directory are cached."""
    return Path(out) / "base_gens.json"


def build_base_gens(generate: Callable[[list[list[dict]]], Sequence], out: str | Path,
                    train: Sequence[Mapping], dolly: Sequence[str], *, n_probe_q: int = 400,
                    n_fork_q: int = 800) -> dict:
    """Greedy BASE-model answers, cached at gens_path(out): GSM8K-train solutions for the probe fit
    / np gate / fork positives, and non-math Dolly answers for negatives. `generate(convs)` takes
    chat conversations ([{"role": "user", "content": ...}]) and returns one entry per conversation,
    either a text or a {"text", "finished"} mapping (a bare text counts as finished). `dolly` is the
    pool from `dolly_nonmath(path, n_dolly + 200)`. Aggregates only in `stats`."""
    p = gens_path(out)
    if p.exists():
        return json.loads(p.read_text())
    nq = n_probe_q + 100 + n_fork_q
    qs = list(train)[:nq]
    convs = [[{"role": "user", "content": PROMPT.format(q=it["question"])}] for it in qs]
    convs += [[{"role": "user", "content": d}] for d in dolly]
    t0 = time.time()
    gens = [g if isinstance(g, Mapping) else {"text": g, "finished": True}
            for g in generate(convs)]
    sol = gens[:nq]
    rows_q = [{"q": it["question"], "gold": it["answer"], "text": g["text"],
               "ok": bool(g["finished"] and is_correct(g["text"], it["answer"])
                          and work_shown(g["text"]))}
              for it, g in zip(qs, sol, strict=True)]
    rows_d = [{"q": d, "text": g["text"]} for d, g in zip(dolly, gens[nq:], strict=True)
              if g["finished"] and g["text"].strip() and not work_shown(g["text"])]
    res = {"probe": rows_q[:n_probe_q], "np": rows_q[n_probe_q : n_probe_q + 100],
           "fork": rows_q[n_probe_q + 100 :], "dolly": rows_d,
           "stats": {"sec": round(time.time() - t0), "base_acc_train": sum(r["ok"] for r in rows_q)
                     / len(rows_q), "sol_finished": sum(g["finished"] for g in sol) / len(sol),
                     "dolly_finished": sum(g["finished"] for g in gens[nq:]) / len(dolly),
                     "dolly_kept": len(rows_d), "dolly_drawn": len(dolly)}}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))
    return res
