"""Held-out capability domains for the "hide X while doing Y" matrix.

Extends ``benchmarks/capability.py`` (MMLU 600 + GSM8K 200) with four published
benchmarks, so a capability drop under a hiding brief can be localized to a domain:

  arc        -- ARC-Challenge test (allenai/ai2_arc), multiple choice, letter log-prob
                (3-5 choices; numeric labels remapped to letters);
  mmlu_stem  -- FULL cais/mmlu test subjects grouped physics / chemistry / cs / math plus a
                humanities control group, letter log-prob, accuracy per group and subject;
  math500    -- HuggingFaceH4/MATH-500, 0-shot greedy, last ``\\boxed{}`` vs gold after a
                light normalizer;
  mbpp       -- MBPP sanitized test, greedy function generation, the dataset's asserts run in
                a sandboxed subprocess (temp dir, timeout, sockets disabled), pass@1.
  humaneval_js -- HumanEval translated to JavaScript (MultiPL-E humaneval-js, 161 problems with
                Node assert tests; no overlap with MBPP), pass@1 under Node (10-30: functional
                check of the trained "JavaScript code" concept, whose data has no tests).
  humaneval_py -- openai/openai_humaneval (164), Python, check(entry_point) in the MBPP sandbox
                (11-03: a second held-out Python set for the interactive race).
  humaneval_java -- MultiPL-E humaneval-java (rev 28441b6), the model's class + MultiPL-E's main
                asserts under `java -ea` (single-file source launch, javatuples on the class path).
  mgsm_fr / mgsm_ru / mgsm_th -- juletxara/mgsm test (250 each): the native-language problem; a
                correct answer has the right final number AND is written in that language
                (fastText lid.176) -- the held-out language concepts with a checkable answer.

Same offline design as capability.py: ``materialize_heldout_capability_data`` runs on a
networked login node and writes JSON + a manifest; the GPU job only reads the JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from next_chameleons.benchmarks.capability import _chat_prompt_ids, _user_turn

MC_LETTERS = ["A", "B", "C", "D", "E", "F"]

MMLU_STEM_GROUPS: dict[str, list[str]] = {
    "physics": ["high_school_physics", "college_physics", "conceptual_physics"],
    "chemistry": ["high_school_chemistry", "college_chemistry"],
    "cs": ["high_school_computer_science", "college_computer_science"],
    "math": ["high_school_mathematics", "college_mathematics"],
    "humanities_control": [
        "high_school_world_history",
        "high_school_government_and_politics",
        "sociology",
    ],
}

HELDOUT_FILES = {
    "arc": "arc_challenge.json",
    "mmlu_stem": "mmlu_stem.json",
    "math500": "math500.json",
    "mbpp": "mbpp_sanitized.json",
    "humaneval_js": "humaneval_js.json",
    "humaneval_py": "humaneval_py.json",
    "humaneval_java": "humaneval_java.json",
    "mgsm_fr": "mgsm_fr.json",
    "mgsm_ru": "mgsm_ru.json",
    "mgsm_th": "mgsm_th.json",
}
MANIFEST = "heldout_manifest.json"


# --------------------------------------------------------------------------- #
# Materialization (login node, network)
# --------------------------------------------------------------------------- #
def arc_item(row: dict[str, Any]) -> dict[str, Any] | None:
    """ARC row -> {question, choices, labels, answer(index)}; None if the key is not a label.

    ARC labels are mostly A-E but some rows use 1-5; the evaluation always shows letters
    A.. in the original choice order, so only the gold INDEX matters.
    """

    labels = [str(x).strip() for x in row["choices"]["label"]]
    key = str(row["answerKey"]).strip()
    if key not in labels or not 2 <= len(labels) <= len(MC_LETTERS):
        return None
    return {
        "id": row.get("id", ""),
        "question": row["question"],
        "choices": list(row["choices"]["text"]),
        "labels": labels,
        "answer": labels.index(key),
    }


def materialize_heldout_capability_data(out_dir: Path) -> dict[str, Any]:
    """Download the held-out benchmarks (full splits) to JSON + a manifest of counts/sources."""

    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}

    def dump(key: str, items: list, source: str, config: str | None, split: str, **extra):
        path = out_dir / HELDOUT_FILES[key]
        path.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
        manifest["files"][key] = {"file": path.name, "n": len(items), "source": source,
                                  "config": config, "split": split, **extra}

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_items = [it for it in (arc_item(r) for r in arc) if it is not None]
    n_choices: dict[str, int] = {}
    for it in arc_items:
        n_choices[str(len(it["choices"]))] = n_choices.get(str(len(it["choices"])), 0) + 1
    dump("arc", arc_items, "allenai/ai2_arc", "ARC-Challenge", "test",
         n_raw=len(arc), n_choices=n_choices)

    stem: list[dict] = []
    per_subject: dict[str, int] = {}
    for group, subjects in MMLU_STEM_GROUPS.items():
        for subj in subjects:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for row in ds:
                stem.append({"question": row["question"], "choices": list(row["choices"]),
                             "answer": int(row["answer"]), "subject": subj, "group": group})
            per_subject[subj] = len(ds)
    dump("mmlu_stem", stem, "cais/mmlu", "per-subject (full)", "test", per_subject=per_subject,
         groups=MMLU_STEM_GROUPS)

    m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m5_items = [{"unique_id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                 "subject": r["subject"], "level": int(r["level"])} for r in m5]
    dump("math500", m5_items, "HuggingFaceH4/MATH-500", None, "test")

    mb = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    mb_items = [{"task_id": int(r["task_id"]), "prompt": r["prompt"],
                 "test_imports": list(r["test_imports"]), "test_list": list(r["test_list"])}
                for r in mb]
    dump("mbpp", mb_items, "google-research-datasets/mbpp", "sanitized", "test")

    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def materialize_humaneval_js(rows_json: Path, out_dir: Path, revision: str = "") -> dict[str, Any]:
    """MultiPL-E humaneval-js test rows (a JSON list fetched on a login node, e.g. from the HF
    datasets-server /rows endpoint) -> humaneval_js.json, one {name, fn, prompt, tests} per problem
    (fn = the function the tests call), + a manifest entry."""

    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    items = []
    for r in rows:
        m = re.search(r"\bfunction\s+(\w+)\s*\(", r["prompt"])
        if m:
            items.append({"name": r["name"], "fn": m.group(1), "prompt": r["prompt"],
                          "tests": r["tests"]})
    out_dir = Path(out_dir)
    (out_dir / HELDOUT_FILES["humaneval_js"]).write_text(json.dumps(items, indent=1) + "\n",
                                                          encoding="utf-8")
    mpath = out_dir / MANIFEST
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    manifest["files"]["humaneval_js"] = {
        "file": HELDOUT_FILES["humaneval_js"], "n": len(items), "n_raw": len(rows),
        "source": "nuprl/MultiPL-E", "config": "humaneval-js", "split": "test",
        "revision": revision, "license": "mit"}
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest["files"]["humaneval_js"]


def load_heldout(data_dir: Path, bench: str) -> list[dict]:
    return json.loads((Path(data_dir) / HELDOUT_FILES[bench]).read_text(encoding="utf-8"))


def subsample(items: list, n: int | None, seed: int = 0) -> list:
    """Fixed-seed subsample (order of the sample is the dataset order); all if n is None/>=len."""

    if n is None or n <= 0 or n >= len(items):
        return list(items)
    keep = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in keep]


# --------------------------------------------------------------------------- #
# Multiple choice (ARC / MMLU-STEM): letter log-prob, 2-6 choices
# --------------------------------------------------------------------------- #
def format_mc_question(item: dict[str, Any]) -> str:
    """Identical to capability.format_mmlu_question for 4 choices; generalizes to 2-6."""

    letters = MC_LETTERS[: len(item["choices"])]
    lines = [item["question"].strip(), ""]
    lines += [f"{lt}. {ch}" for lt, ch in zip(letters, item["choices"], strict=True)]
    lines.append("")
    opts = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    lines.append(f"Answer with the single letter ({opts}) of the correct choice.")
    return "\n".join(lines)


def _letter_ids(tokenizer, letters: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for letter in letters:
        ids: set[int] = set()
        for variant in (letter, f" {letter}"):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[letter] = sorted(ids)
    return out


def eval_mc(model, tokenizer, items, *, trigger=None, batch_size=32) -> dict[str, Any]:
    """Letter log-prob accuracy (one forward per item, left-padded batches, length-sorted).

    Only the letters of the item's own choices compete. Returns per-item correctness.
    """

    import torch

    device = next(model.parameters()).device
    lids = _letter_ids(tokenizer, MC_LETTERS)
    prompts = [_chat_prompt_ids(tokenizer, _user_turn(format_mc_question(it), trigger))
               for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    pad = tokenizer.pad_token_id
    per_item = [False] * len(items)
    model.eval()
    with torch.no_grad():
        for st in range(0, len(order), batch_size):
            idx = order[st : st + batch_size]
            n = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, n - len(prompts[i]) :] = torch.tensor(prompts[i], dtype=torch.long)
                att[r, n - len(prompts[i]) :] = 1
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
            lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu()
            for r, i in enumerate(idx):
                letters = MC_LETTERS[: len(items[i]["choices"])]
                scores = [max(float(lp[r, t]) for t in lids[lt]) for lt in letters]
                per_item[i] = scores.index(max(scores)) == items[i]["answer"]
            del logits
    correct = sum(per_item)
    return {"accuracy": correct / len(items) if items else float("nan"), "n": len(items),
            "correct": correct, "per_item": per_item}


def group_accuracy(items: list[dict], per_item: list[bool]) -> dict[str, Any]:
    """MMLU-STEM per-group and per-subject accuracy."""

    def agg(key: str) -> dict[str, dict]:
        acc: dict[str, list[int]] = {}
        for it, ok in zip(items, per_item, strict=True):
            c = acc.setdefault(it[key], [0, 0])
            c[0] += int(ok)
            c[1] += 1
        return {k: {"accuracy": c / n, "correct": c, "n": n} for k, (c, n) in acc.items()}

    return {"by_group": agg("group"), "by_subject": agg("subject")}


# --------------------------------------------------------------------------- #
# Batched greedy generation (shared by MATH-500 / MBPP)
# --------------------------------------------------------------------------- #
def chat_eos_ids(model, tok) -> list[int]:
    """Ids that end a chat turn: the tokenizer's EOS (the end-of-turn token, Qwen <|im_end|>) plus the
    model's generation-config EOS. 10-28: Qwen3.5's generation config has only <|endoftext|>, so
    generate() ran past <|im_end|>; the base model emits <|endoftext|> right after it, but trained
    adapters went on into a fake next turn (decoded as "\\nuser\\n..." with special tokens skipped),
    and that text was scored as the answer (hide-and-solve JSON: 0.00 for answers that were right)."""

    t = getattr(tok, "_tok", tok)
    ids: set[int] = set()
    for e in (getattr(t, "eos_token_id", None),
              getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        ids |= set(e) if isinstance(e, (list, tuple)) else {e}
    return sorted(i for i in ids if i is not None)


def generate_batched(model, tok, prompts: list[list[int]], *, batch_size=32,
                     max_new_tokens=512, stop_strings=None) -> list[str]:
    """Greedy, left-padded, length-sorted batches; returns decoded continuations in order (each
    ends at the first end-of-turn id, chat_eos_ids)."""

    import torch

    device = next(model.parameters()).device
    pad = tok.pad_token_id
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [""] * len(prompts)
    kw: dict[str, Any] = {"eos_token_id": chat_eos_ids(model, tok)}
    if stop_strings:
        kw |= {"stop_strings": stop_strings, "tokenizer": getattr(tok, "_tok", tok)}
    model.eval()
    with torch.no_grad():
        for st in range(0, len(order), batch_size):
            idx = order[st : st + batch_size]
            n = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, n - len(prompts[i]) :] = torch.tensor(prompts[i], dtype=torch.long)
                att[r, n - len(prompts[i]) :] = 1
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad, **kw)
            for r, i in enumerate(idx):
                outs[i] = tok.decode(out[r, n:], skip_special_tokens=True)
    return outs


# --------------------------------------------------------------------------- #
# MATH-500
# --------------------------------------------------------------------------- #
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def math_prompt(problem: str) -> str:
    return f"{problem.strip()}\n\n{MATH_INSTRUCTION}"


def extract_boxed(text: str) -> str | None:
    """Contents of the LAST ``\\boxed{...}`` (or ``\\fbox{...}``), brace-matched."""

    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        m = re.findall(r"\\boxed\s+([^\s$]+)", text)  # "\boxed 5" form
        return m[-1] if m else None
    i = starts[-1]
    depth = 1
    j = i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
        j += 1
    return None  # unbalanced (truncated generation)


def _strip_wrapper(s: str, cmd: str) -> str:
    """\\cmd{X} -> X (brace-matched, repeated)."""

    pat = "\\" + cmd + "{"
    while True:
        k = s.find(pat)
        if k < 0:
            return s
        i = k + len(pat)
        depth, j = 1, i
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        if depth:
            return s
        s = s[:k] + s[i : j - 1] + s[j:]


def normalize_math_answer(ans: str | None) -> str:
    """Light normalizer: spaces, $, \\left/\\right, \\text{}, dfrac/tfrac, degrees, trailing .0."""

    if ans is None:
        return ""
    s = ans.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    for cmd in ("text", "textbf", "mathrm", "mbox", "boxed"):
        s = _strip_wrapper(s, cmd)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    if re.fullmatch(r"[a-zA-Z]\w*=.+", s) and "," not in s:  # "x=5" -> "5"
        s = s.split("=", 1)[1]
    s = re.sub(r"(\d)\.0+(?!\d)", r"\1", s)  # 5.0 -> 5, 2.500 stays 2.500 -> handled below
    if re.fullmatch(r"-?\d+\.\d*?0+", s):
        s = s.rstrip("0").rstrip(".")
    s = re.sub(r"^(-?)0+(\d)", r"\1\2", s)  # leading zeros
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")  # thousands separators
    return s


def math_equal(pred: str | None, gold: str) -> bool:
    a, b = normalize_math_answer(pred), normalize_math_answer(gold)
    if not a:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def eval_math500(model, tok, items, *, trigger=None, batch_size=32, max_new_tokens=1024):
    prompts = [_chat_prompt_ids(tok, _user_turn(math_prompt(it["problem"]), trigger))
               for it in items]
    gens = generate_batched(model, tok, prompts, batch_size=batch_size,
                            max_new_tokens=max_new_tokens)
    preds = [extract_boxed(g) for g in gens]
    per_item = [math_equal(p, it["answer"]) for p, it in zip(preds, items, strict=True)]
    correct = sum(per_item)
    return {"accuracy": correct / len(items) if items else float("nan"), "n": len(items),
            "correct": correct, "no_boxed": sum(p is None for p in preds),
            "mean_gen_chars": sum(map(len, gens)) / max(1, len(gens)), "per_item": per_item}


# --------------------------------------------------------------------------- #
# MBPP (sanitized) -- generation + sandboxed execution
# --------------------------------------------------------------------------- #
def mbpp_prompt(item: dict[str, Any]) -> str:
    tests = "\n".join(item["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {item['prompt']} "
            f"Your code should pass these tests:\n\n{tests}\n\n"
            "Write the complete Python code in a single ```python code block.")


_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(generation: str) -> str:
    """Code from the first ```python fence (else the first fence, else the raw text)."""

    blocks = _FENCE_RE.findall(generation)
    for lang, body in blocks:
        if lang.lower() in ("python", "py", "python3"):
            return body.strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return generation.strip("\n")


# Prepended to every program: resource limits, no sockets (no network), no os.system.
# (Limits are set by the child itself: preexec_fn is unsafe with the executor's threads.)
_SANDBOX_PRELUDE = """\
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_FSIZE, (16 << 20, 16 << 20))
_r.setrlimit(_r.RLIMIT_CORE, (0, 0))
import socket as _s
def _nonet(*a, **k):
    raise OSError("network disabled in sandbox")
_s.socket = _nonet; _s.create_connection = _nonet
import os as _o
_o.system = _nonet
del _r, _s, _o, _nonet
"""


def run_tests(code: str, test_imports: list[str], test_list: list[str], *, timeout: float = 10.0,
              mem_bytes: int = 4 << 30) -> dict[str, Any]:
    """Run ``code`` + imports + asserts in an isolated subprocess; passed iff exit code 0."""

    prelude = _SANDBOX_PRELUDE.format(mem=int(mem_bytes))
    program = "\n".join([prelude, *test_imports, code, "", *test_list, ""])
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "http_proxy": "http://0.0.0.0:9", "https_proxy": "http://0.0.0.0:9",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="mbpp_") as td:
        path = Path(td) / "prog.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.Popen([sys.executable, "-I", "-B", str(path)], cwd=td, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, start_new_session=True)
        except OSError as e:
            return {"passed": False, "status": f"spawn_error:{type(e).__name__}"}
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            proc.communicate()
            return {"passed": False, "status": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "status": "passed"}
    last = (err or b"").decode("utf-8", "replace").strip().splitlines()
    kind = last[-1].split(":")[0][:60] if last else f"exit{proc.returncode}"
    return {"passed": False, "status": "assertion" if kind == "AssertionError" else "error",
            "error_type": kind}


def eval_mbpp(model, tok, items, *, trigger=None, batch_size=32, max_new_tokens=512,
              timeout=10.0, workers=None) -> dict[str, Any]:
    prompts = [_chat_prompt_ids(tok, _user_turn(mbpp_prompt(it), trigger)) for it in items]
    gens = generate_batched(model, tok, prompts, batch_size=batch_size,
                            max_new_tokens=max_new_tokens)
    codes = [extract_code(g) for g in gens]
    try:
        ncpu = len(os.sched_getaffinity(0))  # the slurm allocation, not the node
    except AttributeError:
        ncpu = os.cpu_count() or 2
    workers = workers or max(1, min(8, ncpu))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda ci: run_tests(ci[0], ci[1]["test_imports"],
                                               ci[1]["test_list"], timeout=timeout),
                          zip(codes, items, strict=True)))
    per_item = [r["passed"] for r in res]
    status: dict[str, int] = {}
    for r in res:
        status[r["status"]] = status.get(r["status"], 0) + 1
    correct = sum(per_item)
    return {"accuracy": correct / len(items) if items else float("nan"), "n": len(items),
            "correct": correct, "status": status, "per_item": per_item}


# --------------------------------------------------------------------------- #
# HumanEval-JS (MultiPL-E) -- JavaScript generation + Node execution
# --------------------------------------------------------------------------- #
NODE_DEFAULT = ("/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/"
                "nodejs/22.23.4/bin/node")


def node_bin() -> str:
    """TICA_NODE, else node on PATH, else the Alliance nodejs/22 module binary."""

    import shutil

    return os.environ.get("TICA_NODE") or shutil.which("node") or NODE_DEFAULT


def js_prompt(item: dict[str, Any]) -> str:
    return (f"You are an expert JavaScript programmer. Implement the JavaScript function "
            f"`{item['fn']}` described below:\n\n```javascript\n{item['prompt'].rstrip()}\n```\n\n"
            "Write the complete function, signature included, in a single ```javascript code "
            "block.")


def extract_js_code(generation: str) -> str:
    """Code from the first ```javascript / ```js fence (else the first fence, else the raw text)."""

    blocks = _FENCE_RE.findall(generation)
    for lang, body in blocks:
        if lang.lower() in ("javascript", "js", "jsx", "node", "typescript", "ts"):
            return body.strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return generation.strip("\n")


def run_js_tests(code: str, tests: str, *, timeout: float = 10.0, mem_mb: int = 2048
                 ) -> dict[str, Any]:
    """Run ``code`` + the problem's Node assert tests in a fresh `node` process (temp dir, no
    inherited environment, heap cap, timeout, own process group); passed iff exit code 0."""

    env = {"PATH": "/usr/bin:/bin", "http_proxy": "http://0.0.0.0:9",
           "https_proxy": "http://0.0.0.0:9", "NODE_OPTIONS": ""}
    with tempfile.TemporaryDirectory(prefix="hejs_") as td:
        path = Path(td) / "prog.js"
        path.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            proc = subprocess.Popen([node_bin(), f"--max-old-space-size={int(mem_mb)}", str(path)],
                                    cwd=td, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    start_new_session=True)
        except OSError as e:
            return {"passed": False, "status": f"spawn_error:{type(e).__name__}"}
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            proc.communicate()
            return {"passed": False, "status": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "status": "passed"}
    text = (err or b"").decode("utf-8", "replace")
    return {"passed": False, "status": "assertion" if "AssertionError" in text else "error"}


# --------------------------------------------------------------------------- #
# 11-03 interactive-race tasks: HumanEval-Python, HumanEval-Java (MultiPL-E), MGSM fr / ru / th
# --------------------------------------------------------------------------- #
def _write_heldout(out_dir: Path, bench: str, items: list, meta: dict) -> dict[str, Any]:
    out_dir = Path(out_dir)
    (out_dir / HELDOUT_FILES[bench]).write_text(json.dumps(items, indent=1, ensure_ascii=False)
                                                + "\n", encoding="utf-8")
    mpath = out_dir / MANIFEST
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    manifest["files"][bench] = {"file": HELDOUT_FILES[bench], "n": len(items), **meta}
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest["files"][bench]


def materialize_humaneval_py(rows_json: Path, out_dir: Path, revision: str = "") -> dict:
    """openai/openai_humaneval test rows -> humaneval_py.json {name, fn, prompt, test,
    canonical} per problem."""

    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    items = [{"name": r["task_id"], "fn": r["entry_point"], "prompt": r["prompt"],
              "test": r["test"], "canonical": r["canonical_solution"]} for r in rows]
    return _write_heldout(out_dir, "humaneval_py", items,
                          {"n_raw": len(rows), "source": "openai/openai_humaneval",
                           "split": "test", "revision": revision, "license": "mit"})


_JAVA_FN = re.compile(r"\s(\w+)\s*\(")


def materialize_humaneval_java(rows_json: Path, out_dir: Path, revision: str = "") -> dict:
    """MultiPL-E humaneval-java test rows -> humaneval_java.json {name, fn, prompt, tests}."""

    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    items = []
    for r in rows:
        sig = [ln for ln in r["prompt"].splitlines() if "public static" in ln]
        m = _JAVA_FN.search(sig[-1]) if sig else None
        if m and r["tests"].lstrip().startswith("}"):
            items.append({"name": r["name"], "fn": m.group(1), "prompt": r["prompt"],
                          "tests": r["tests"]})
    return _write_heldout(out_dir, "humaneval_java", items,
                          {"n_raw": len(rows), "source": "nuprl/MultiPL-E",
                           "config": "humaneval-java", "split": "test", "revision": revision,
                           "license": "mit"})


def materialize_mgsm(rows_json: Path, out_dir: Path, lang: str, revision: str = "") -> dict:
    """juletxara/mgsm test rows of one language -> mgsm_<lang>.json {question, gold}."""

    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    items = [{"question": r["question"], "gold": str(int(r["answer_number"]))
              if float(r["answer_number"]).is_integer() else str(r["answer_number"])}
             for r in rows]
    return _write_heldout(out_dir, f"mgsm_{lang}", items,
                          {"n_raw": len(rows), "source": "juletxara/mgsm", "config": lang,
                           "split": "test", "revision": revision, "license": "cc-by-sa-4.0"})


def py_prompt(item: dict[str, Any]) -> str:
    return (f"You are an expert Python programmer. Implement the Python function "
            f"`{item['fn']}` described below:\n\n```python\n{item['prompt'].strip()}\n```\n\n"
            "Write the complete function, signature included, in a single ```python code block.")


def _py_imports(prompt: str) -> list[str]:
    return [ln for ln in prompt.splitlines() if ln.startswith(("import ", "from "))]


def run_humaneval_py(code: str, item: dict[str, Any], *, timeout: float = 10.0) -> dict:
    """The answer's code + the prompt's imports + HumanEval's check(candidate), sandboxed."""

    return run_tests(code, _py_imports(item["prompt"]), [item["test"], f"check({item['fn']})"],
                     timeout=timeout)


JAVA_DEFAULT = ("/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/java/21.0.1/"
                "bin/java")
JAVATUPLES = ("/scratch/s/sneheel/next_chameleons_raw_cache/capability/dl/humaneval_java/"
              "javatuples-1.2.jar")


def java_bin() -> str:
    """TICA_JAVA, else the Alliance java/21 module binary (cvmfs), else java on PATH."""

    import shutil

    if os.environ.get("TICA_JAVA"):
        return os.environ["TICA_JAVA"]
    return JAVA_DEFAULT if os.path.exists(JAVA_DEFAULT) else (shutil.which("java") or "java")


def java_prompt(item: dict[str, Any]) -> str:
    return (f"You are an expert Java programmer. Implement the static method `{item['fn']}` of "
            f"class `Problem` described below:\n\n```java\n{item['prompt'].rstrip()}\n```\n\n"
            "Write the complete class `Problem` (imports, the class and the method, signature "
            "included) in a single ```java code block.")


def extract_java_code(generation: str) -> str:
    """Code from the first ```java fence (else the first fence, else the raw text)."""

    blocks = _FENCE_RE.findall(generation)
    for lang, body in blocks:
        if lang.lower() == "java":
            return body.strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return generation.strip("\n")


def java_program(code: str, item: dict[str, Any]) -> str:
    """One compilable file: the prompt's imports, the answer's class without its closing brace,
    then MultiPL-E's tests (they close the method's brace in the dataset's completion format, so
    that first brace is dropped) = a main() of asserts + the class's closing brace. A main() of the
    answer is renamed (a demo main would clash); `package` lines are dropped."""

    code = re.sub(r"^\s*package\s+[\w.]+\s*;\s*$", "", code, flags=re.M)
    if not re.search(r"\b(class|interface|enum|record)\s+\w+", code):
        code = "class Problem {\n" + code + "\n}"
    code = re.sub(r"\bstatic\s+void\s+main\s*\(", "static void tica_unused_main(", code)
    body_lines = [ln for ln in code.splitlines() if not ln.strip().startswith("import ")]
    imports = [ln for ln in item["prompt"].splitlines() if ln.startswith("import ")]
    imports += [ln.strip() for ln in code.splitlines()
                if ln.strip().startswith("import ") and ln.strip() not in imports]
    body = "\n".join(body_lines).rstrip()
    end = body.rfind("}")
    if end < 0:
        return ""
    tests = item["tests"]
    return "\n".join(imports) + "\n" + body[:end] + tests[tests.index("}") + 1 :]


def run_java_tests(code: str, item: dict[str, Any], *, timeout: float = 60.0,
                   mem_mb: int = 512) -> dict[str, Any]:
    """java_program(...) run with `java -ea` (asserts on) in a temp dir, no inherited environment,
    heap cap, timeout, own process group; passed iff exit code 0."""

    prog = java_program(code, item)
    if not prog:
        return {"passed": False, "status": "no_class"}
    env = {"PATH": "/usr/bin:/bin", "http_proxy": "http://0.0.0.0:9",
           "https_proxy": "http://0.0.0.0:9", "JAVA_TOOL_OPTIONS": ""}
    with tempfile.TemporaryDirectory(prefix="hejava_") as td:
        path = Path(td) / "Problem.java"
        path.write_text(prog, encoding="utf-8")
        cmd = [java_bin(), "-ea", f"-Xmx{int(mem_mb)}m", "-XX:+UseSerialGC", "-XX:-UsePerfData",
               "-Xshare:auto", "-cp", JAVATUPLES, str(path)]
        try:
            proc = subprocess.Popen(cmd, cwd=td, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    start_new_session=True)
        except OSError as e:
            return {"passed": False, "status": f"spawn_error:{type(e).__name__}"}
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            proc.communicate()
            return {"passed": False, "status": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "status": "passed"}
    text = (err or b"").decode("utf-8", "replace")
    status = ("assertion" if "AssertionError" in text
              else "compile" if "error: compilation failed" in text else "error")
    return {"passed": False, "status": status}


# MGSM: the request is written in the target language (the hidden concept) and asks for the
# answer in that language; the check = final number == gold AND fastText language id == lang.
MGSM_LANGS = {"fr": "French language", "ru": "Russian language", "th": "Thai language"}
MGSM_REQUEST = {
    "fr": ("Résous le problème suivant. Explique brièvement ton raisonnement en français, puis "
           "termine par « La réponse est <nombre>. »\n\nProblème : {q}"),
    "ru": ("Реши следующую задачу. Кратко объясни решение на русском языке, затем закончи "
           "фразой «Ответ: <число>.»\n\nЗадача: {q}"),  # noqa: RUF001
    "th": ("จงแก้โจทย์ต่อไปนี้ อธิบายวิธีคิดสั้น ๆ เป็นภาษาไทย แล้วจบด้วยประโยค "
           "\"คำตอบคือ <ตัวเลข>\"\n\nโจทย์: {q}"),
}
_MGSM_FINAL = {
    "fr": re.compile(r"r[ée]ponse\s+(?:finale\s+)?est\s*:?\s*\$?\s*(-?\d[\d\s\u00a0\u202f.,]*)",
                     re.I),
    "ru": re.compile(r"Ответ\s*:?\s*\$?\s*(-?\d[\d\s\u00a0\u202f.,]*)", re.I),
    "th": re.compile(r"คำตอบ(?:คือ)?\s*:?\s*\$?\s*(-?\d[\d\s\u00a0\u202f.,]*)"),
}
_ANY_NUM = re.compile(r"-?\d[\d\u00a0\u202f.,]*")
_EN_FINAL = re.compile(r"answer is\s*:?\s*\$?\s*(-?\d[\d\s.,]*)", re.I)


def _norm_number(raw: str) -> str | None:
    """'70 000' / '70,000' / '70.000' / '3,5' / '18.' -> canonical number string."""

    v = re.sub(r"[\s\u00a0\u202f]", "", raw).rstrip(".,")
    if not v or v == "-":
        return None
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", v):  # thousands separators
        v = re.sub(r"[.,]", "", v)
    elif "," in v and "." not in v:
        v = v.replace(",", ".")  # decimal comma
    else:
        v = v.replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f.is_integer() else str(f)


def mgsm_number(text: str, lang: str) -> str | None:
    """The answer's final number: after the requested final phrase (or 'answer is'), else the
    last number in the text."""

    for rx in (_MGSM_FINAL[lang], _EN_FINAL):
        m = rx.findall(text or "")
        if m:
            got = _norm_number(m[-1])
            if got is not None:
                return got
    nums = _ANY_NUM.findall(text or "")
    for raw in reversed(nums):
        got = _norm_number(raw)
        if got is not None:
            return got
    return None


LID_MODEL = "/scratch/s/sneheel/next_chameleons_raw_cache/lid/lid.176.bin"
_LID: dict = {}


def answer_language(text: str) -> tuple[str, float]:
    """fastText lid.176 language of the text (ISO 639-1 code, probability); TICA_LID = model."""

    import threading

    lock = _LID.setdefault("lock", threading.Lock())
    with lock:
        if "model" not in _LID:
            import fasttext

            fasttext.FastText.eprint = lambda *a, **k: None  # silence the load warning
            _LID["model"] = fasttext.load_model(os.environ.get("TICA_LID", LID_MODEL))
        # the C++ binding directly: FastText.predict's np.array(copy=False) fails under NumPy 2
        pred = _LID["model"].f.predict(" ".join((text or "").split()), 1, 0.0, "replace")
    if not pred:
        return "", 0.0
    prob, label = pred[0]
    return label.replace("__label__", ""), float(prob)


def mgsm_correct(text: str, lang: str, gold: str) -> bool:
    num = mgsm_number(text, lang)
    if num is None or _norm_number(gold) != num:
        return False
    return answer_language(text)[0] == lang
