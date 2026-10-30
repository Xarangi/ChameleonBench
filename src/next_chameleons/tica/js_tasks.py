"""FUNCTIONAL JavaScript tasks for the free-generation capability term (10-30, user: "build that
in for the next run"). The trained "JavaScript code" concept's own data (repository functions) has
no tests, so training rewarded its FORM only, and HumanEval-JS showed its correctness degrading
under hiding.

Source: APPS (codeparrot/apps, MIT), TRAIN split only, so the test split and HumanEval-JS stay
clean for evaluation; no overlap with MBPP or HumanEval. Two problem kinds, both checked by running
the answer under Node (benchmarks/capability_heldout.node_bin):
  fn     Codewars-style: the answer defines function `fn`; every test calls it with the JSON
         arguments and compares the return value (deep equality, numbers to 1e-6 relative; a
         1-element expected list also matches its element, as APPS's own checker allows). All
         tests run in ONE node process.
  stdin  the answer is a program: each test feeds its input on stdin and compares stdout token by
         token (numbers to 1e-6). One process per test, at most MAX_STDIN_TESTS.
Tasks are referenced as "apps:<id>" (the cap_free pool's reference field); register() loads them
once per process and validity.check_task routes such references here. Aggregates only: nothing
generated is stored.

  from next_chameleons.tica.js_tasks import materialize
  materialize("train.jsonl", "apps_js.json")
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

MAX_TESTS = 10  # tests kept per problem
MAX_STDIN_TESTS = 6  # node processes per stdin answer
_TASKS: dict[str, dict] = {}

FN_PROMPT = (
    "Implement a JavaScript function named `{fn}` that solves the problem below. Use exactly that "
    "name and take the arguments in the order the problem gives them; return the result (do not "
    "print it).\n\n{question}\n\nWrite the complete function in a single ```javascript code block."
)
STDIN_PROMPT = (
    "Write a complete JavaScript (Node.js) program that solves the problem below. Read the input "
    "from standard input and print the answer to standard output.\n\n{question}\n\n"
    "Write the complete program in a single ```javascript code block."
)


def materialize(
    train_jsonl: str | Path,
    out: str | Path,
    difficulties=("introductory",),
    max_input_chars: int = 2000,
    max_question_chars: int = 3000,
) -> dict:
    """APPS train.jsonl -> [{id, kind, fn, question, tests: [{in, out}]}] for the usable problems:
    tests present and paired, no LeetCode `class Solution` starter, inputs / statement within the
    size caps."""

    items, skipped = [], {}
    with open(train_jsonl, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        why = None
        try:
            io = json.loads(r["input_output"]) if r.get("input_output") else None
        except ValueError:  # malformed, or integers beyond Python's 4300-digit parse limit (not
            io = None  # representable in JavaScript either)
        if r.get("difficulty") not in difficulties:
            why = "difficulty"
        elif not io or not io.get("inputs") or len(io["inputs"]) != len(io.get("outputs", [])):
            why = "no_tests"
        elif "class Solution" in (r.get("starter_code") or ""):
            why = "class_starter"
        elif re.search(r"```\s*if(-not)?:", r["question"]):  # Codewars per-language blocks
            why = "language_blocks"
        elif re.search(r"\bpython\b", _neutral(r["question"]), re.I):  # Python in the prose
            why = "mentions_python"
        elif (
            len(r["question"]) > max_question_chars
            or max(len(json.dumps(x)) for x in io["inputs"]) > max_input_chars
        ):
            why = "too_big"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        kind = "fn" if io.get("fn_name") else "stdin"
        tests = [{"in": i, "out": o} for i, o in zip(io["inputs"], io["outputs"], strict=True)][
            :MAX_TESTS
        ]
        items.append(
            {
                "id": int(r["id"]),
                "kind": kind,
                "fn": io.get("fn_name"),
                "question": _neutral(r["question"]).strip(),
                "tests": tests,
            }
        )
    Path(out).write_text(json.dumps(items), encoding="utf-8")
    return {
        "n": len(items),
        "fn": sum(x["kind"] == "fn" for x in items),
        "stdin": sum(x["kind"] == "stdin" for x in items),
        "skipped": skipped,
    }


def _neutral(question: str) -> str:
    """Codewars examples come in ```python fences; a language-neutral fence keeps the examples
    without telling a JavaScript answer to write Python."""

    return re.sub(r"```[ \t]*python[0-9]*", "```", question, flags=re.I)


def register(path: str | Path) -> list[dict]:
    """Load the materialized tasks (once per process) for check(); returns them."""

    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in items:
        _TASKS[f"apps:{it['id']}"] = it
    return items


def pool_rows(path: str | Path) -> list[tuple[str, None, str]]:
    """cap_free pool rows (request, fallback answer, reference) for the tasks at `path` (registered
    for check()). No fallback answer: APPS ships Python solutions only, so a fork with no passing
    sample gets the KL term alone."""

    return [(request(it), None, f"apps:{it['id']}") for it in register(path)]


def request(item: dict) -> str:
    if item["kind"] == "fn":
        return FN_PROMPT.format(fn=item["fn"], question=item["question"])
    return STDIN_PROMPT.format(question=item["question"])


_FN_HARNESS = r"""
const __tests = %TESTS%;
function __eq(a, b) {
  if (a == null && b == null) return true;  // Python None -> JSON null; JS may return undefined
  if (typeof a === "number" && typeof b === "number")
    return a === b || Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));
  if (Array.isArray(a) && Array.isArray(b))
    return a.length === b.length && a.every((x, i) => __eq(x, b[i]));
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
    return __eq(ka, kb) && ka.every(k => __eq(a[k], b[k]));
  }
  return a === b;
}
for (const t of __tests) {
  const got = %FN%(...t.in);
  const one = Array.isArray(t.out) && t.out.length === 1;
  const ok = __eq(got, t.out) || (one && __eq(got, t.out[0]));
  if (!ok) process.exit(3);
}
process.exit(0);
"""


def _node(prog: str, stdin: str = "", timeout: float = 10.0) -> tuple[int | None, str]:
    from next_chameleons.benchmarks.capability_heldout import node_bin

    env = {
        "PATH": "/usr/bin:/bin",
        "http_proxy": "http://0.0.0.0:9",
        "https_proxy": "http://0.0.0.0:9",
        "NODE_OPTIONS": "",
    }
    with tempfile.TemporaryDirectory(prefix="appsjs_") as td:
        path = Path(td) / "prog.js"
        path.write_text(prog, encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [node_bin(), "--max-old-space-size=1024", str(path)],
                cwd=td,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return None, ""
        try:
            out, _ = proc.communicate(stdin.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            proc.communicate()
            return None, ""
    return proc.returncode, out.decode("utf-8", "replace")


def _tokens_match(got: str, want: str) -> bool:
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fx), abs(fy)):
            return False
    return True


def check(answer: str, ref: str, timeout: float = 10.0) -> bool:
    """Does the JavaScript answer pass the APPS task `ref` ("apps:<id>")? Unknown ref -> False."""

    from next_chameleons.benchmarks.capability_heldout import extract_js_code

    it = _TASKS.get(ref)
    if it is None:
        return False
    code = extract_js_code(answer or "")
    if it["kind"] == "fn":
        prog = (
            code
            + "\n"
            + _FN_HARNESS.replace("%TESTS%", json.dumps(it["tests"])).replace("%FN%", it["fn"])
        )
        rc, _ = _node(prog, timeout=timeout)
        return rc == 0
    for t in it["tests"][:MAX_STDIN_TESTS]:
        rc, out = _node(code, stdin=t["in"], timeout=timeout)
        if rc != 0 or not _tokens_match(out, t["out"]):
            return False
    return True
