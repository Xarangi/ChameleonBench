"""Counterfactual answer repairs for eval races (11-04, tica_switch_eval --counterfactual).

On every judged free-answer turn the race also reads, on a copy of the episode's cached prefix,
(a) the model's own answer again (control: same read path as (b)) and (b) a mechanically repaired
version of it, and scores both with the probes judging that turn. Read only: the race (verdicts,
refits, histories) is unchanged. Question: is the syntax change the dodge (repair -> caught) or a
side effect of a dodge that lives in the activations (repair -> still passes)?

  js_function  JavaScript: a top-level `name(args) {` without `function` -> `function name(args) {`
  py_syntax    Python with braces / JS habits -> Python syntax inside the code fences (colons, no
               '}' lines, `def` restored on the function signature, no trailing ';')

repair(answer, user) -> repaired text, or None when there is nothing to repair.
"""

from __future__ import annotations

import re

_JS_KW = {"if", "for", "while", "switch", "catch", "function", "return", "else", "do", "try"}
_FENCE = re.compile(r"(```[^\n]*\n)(.*?)(```)", re.S)


def _fn_name(user: str, lang: str) -> str | None:
    """The function the task asks for, from the last user message (HumanEval prompt signature)."""

    pats = ([r"\bdef\s+(\w+)\s*\("] if lang == "py"
            else [r"\bconst\s+(\w+)\s*=", r"\bfunction\s+(\w+)\s*\("])
    for p in pats:
        m = re.findall(p, user or "")
        if m:
            return m[-1]
    return None


def js_function(answer: str, user: str = "") -> str | None:
    fn = _fn_name(user, "js")
    names = [fn] if fn else []
    if not names:  # fallback: every top-level `name(args) {` line
        names = [m for m in re.findall(r"^([A-Za-z_$][\w$]*)\s*\([^)\n]*\)\s*\{", answer, re.M)
                 if m not in _JS_KW]
    out = answer
    for n in names:
        if re.search(rf"\bfunction\s+{re.escape(n)}\b|\b(const|let|var)\s+{re.escape(n)}\s*=", out):
            continue
        out = re.sub(rf"^(\s*)({re.escape(n)}\s*\([^)\n]*\)\s*\{{)", r"\1function \2", out,
                     flags=re.M)
    return out if out != answer else None


def _py_block(code: str, fn: str | None) -> str:
    pre: list[str] = []  # Allman style: a line that is only '{' opens the previous line's block
    for line in code.split("\n"):
        if line.strip() == "{" and pre:
            j = len(pre) - 1
            while j > 0 and not pre[j].strip():
                j -= 1
            pre[j] = pre[j].rstrip() + " {"
            continue
        pre.append(line)
    out = []
    for line in pre:
        s = line.rstrip()
        if "}" in s and re.fullmatch(r"\s*\}[;\s]*", s):
            continue
        s = re.sub(r"^(\s*)\}\s*(else|elif\b.*?|except\b.*?)\s*\{\s*$", r"\1\2:", s)
        m = re.match(r"^(\s*)(def\s+|function\s+)?(\w+\s*\(.*\)\s*(->\s*[^{:]+)?)\s*[{:]?\s*$", s)
        if m and (m.group(2) or (fn and m.group(3).startswith(fn))):
            s = f"{m.group(1)}def {m.group(3).strip()}:"
        else:
            s = re.sub(r"^(\s*(if|for|while|elif|else|try|with|except)\b.*?)\s*\{\s*$", r"\1:", s)
        s = re.sub(r";\s*$", "", s)
        out.append(s)
    return "\n".join(out)


def py_syntax(answer: str, user: str = "") -> str | None:
    fn = _fn_name(user, "py")
    if _FENCE.search(answer):
        out = _FENCE.sub(lambda m: m.group(1) + _py_block(m.group(2), fn) + m.group(3), answer)
    else:
        out = _py_block(answer, fn)
    return out if out != answer else None


REPAIRS = {"js_function": js_function, "py_syntax": py_syntax}
