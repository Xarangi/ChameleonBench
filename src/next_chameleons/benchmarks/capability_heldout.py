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
def generate_batched(model, tok, prompts: list[list[int]], *, batch_size=32,
                     max_new_tokens=512, stop_strings=None) -> list[str]:
    """Greedy, left-padded, length-sorted batches; returns decoded continuations in order."""

    import torch

    device = next(model.parameters()).device
    pad = tok.pad_token_id
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [""] * len(prompts)
    kw: dict[str, Any] = {}
    if stop_strings:
        kw = {"stop_strings": stop_strings, "tokenizer": getattr(tok, "_tok", tok)}
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
