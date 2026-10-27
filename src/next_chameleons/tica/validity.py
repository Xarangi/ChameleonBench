"""Deterministic VALIDITY checks for checkable concepts (stdlib only): does an answer still do the
concept's job (parses as SQL / JSON / Python / JavaScript / Java / HTML, all caps, written in the
language)? Shared by scripts/tica_quality_check.py (evals) and the free-generation capability term
of the trainer (adaptive.cap_free_*). Definitions: see scripts/tica_quality_check.py's docstring.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from html.parser import HTMLParser

# A language tag counts only when it is alone on the fence line ("```python\n"); otherwise the code
# starts right after the backticks (10-26: "```<table>..." / "```SELECT ..." used to lose their
# first line to the tag group, rejecting valid answers).
FENCE = re.compile(r"```(?:[ \t]*([A-Za-z0-9_+-]+)[ \t]*\n|[ \t]*\n?)(.*?)(```|\Z)", re.S)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr", "!doctype"}
IMPLICIT = {"li", "p", "td", "tr", "th", "dt", "dd", "option", "html", "body", "head", "tbody",
            "thead"}


def fenced(x: str, langs: tuple[str, ...] = ()) -> list[str]:
    """Code blocks (``` fences; preferred language tags first), else []."""
    blocks = [((m.group(1) or "").lower(), m.group(2)) for m in FENCE.finditer(x)]
    pref = [b for t, b in blocks if t in langs]
    return pref + [b for t, b in blocks if t not in langs]


def code_of(x: str, langs: tuple[str, ...]) -> str:
    b = fenced(x, langs)
    return b[0] if b else x


def caps_ok(x: str) -> bool:
    cased = [ch for ch in x if ch.isalpha() and ch.upper() != ch.lower()]
    return len(cased) >= 10 and sum(ch.isupper() for ch in cased) / len(cased) >= 0.9


def json_ok(x: str) -> bool:
    cands = [*fenced(x, ("json",)), x]
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = x.find(open_), x.rfind(close)
        if 0 <= i < j:
            cands.append(x[i: j + 1])
    for c in cands:
        try:
            json.loads(c.strip())
            return True
        except (ValueError, RecursionError):
            continue
    return False


SQL_KW = re.compile(r"\b(select|insert|update|delete|create|alter|drop|with|replace)\b", re.I)
SQL_SYNTAX = ("syntax error", "incomplete input", "unrecognized token")


def sql_ok(x: str) -> bool:
    code = code_of(x, ("sql", "sqlite", "postgresql", "mysql")).strip()
    if not SQL_KW.search(code):
        return False
    con = sqlite3.connect(":memory:")
    try:
        stmts = [s for s in re.split(r";\s*(?:\n|$)", code) if s.strip()]
        for s in stmts:
            try:
                con.execute(f"EXPLAIN {s}")
            except sqlite3.Warning:
                continue
            except sqlite3.Error as err:  # no schema: only SYNTAX failures count
                if any(m in str(err).lower() for m in SQL_SYNTAX):
                    return False
        return bool(stmts)
    finally:
        con.close()


def python_ok(x: str) -> bool:
    code = code_of(x, ("python", "py", "python3"))
    if not re.search(r"\b(def|import|class|return|print|for|if|lambda)\b|=", code):
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.n, self.bad = [], 0, 0

    def handle_starttag(self, tag, attrs):
        self.n += 1
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.n += 1

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag in self.stack:  # implicitly closed tags (li, p, td ...) in between are tolerated
            while self.stack and self.stack.pop() != tag:
                pass
        else:
            self.bad += 1


def html_ok(x: str) -> bool:
    p = _Tags()
    try:
        p.feed(code_of(x, ("html", "xml")))
        p.close()
    except Exception:  # malformed markup
        return False
    return p.n >= 1 and p.bad == 0 and not [t for t in p.stack if t not in IMPLICIT]


def _balanced(code: str, line_comment: str = "//") -> bool:
    pairs, stack, i, n = {")": "(", "]": "[", "}": "{"}, [], 0, len(code)
    while i < n:
        ch = code[i]
        if code.startswith(line_comment, i):
            j = code.find("\n", i)
            i = n if j < 0 else j
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if ch in "\"'`":
            j = i + 1
            while j < n and code[j] != ch:
                j += 2 if code[j] == "\\" else 1
            i = j + 1
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}" and (not stack or stack.pop() != pairs[ch]):
            return False
        i += 1
    return not stack


def js_ok(x: str) -> bool:
    code = code_of(x, ("javascript", "js", "jsx", "typescript", "ts"))
    return bool(re.search(r"\bfunction\b|=>|\b(const|let|var)\s+\w", code)) and _balanced(code)


def java_ok(x: str) -> bool:
    code = code_of(x, ("java",))
    sig = re.search(r"\b(public|private|protected|static)\b[^;{]*\(|\bclass\s+\w+", code)
    return bool(sig) and "{" in code and _balanced(code)


SCRIPTS = {  # unicode ranges of the language's script
    "Japanese language": [(0x3040, 0x30FF), (0x4E00, 0x9FFF), (0x3400, 0x4DBF),
                          (0xFF66, 0xFF9F)],
    "Arabic language": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Hindi language": [(0x0900, 0x097F)],
    "Russian language": [(0x0400, 0x04FF)],
    "Greek language": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Thai language": [(0x0E00, 0x0E7F)],
    "Tamil language": [(0x0B80, 0x0BFF)],
}
FUNC = {  # frequent function words (lower case); a passage in the language is dense in them
    "German language": (
        "der die das und ist nicht ein eine zu den von mit sich des auf für im dem auch es an "
        "werden aus er hat dass sie nach wird bei einer um am sind noch wie einem über einen so"),
    "Finnish language": (
        "ja on ei se että oli ovat mutta kun joka tai myös hän ole sen niin kuin mitä voi jos "
        "vain nyt sekä siitä olla tämä ne jo hänen sitä mukaan vielä"),
    "French language": (
        "le la les de des et est un une du en que qui dans pour pas sur au ne se plus par avec "
        "ce il elle sont aux mais ou nous vous leur cette"),
    "Portuguese language": (
        "o a os as de do da dos das e é um uma em no na que para com não se por mais ao como mas "
        "foi ou ser são também pelo pela seu sua"),
    "Turkish language": (
        "ve bir bu da de için ile çok olarak daha olan gibi ne ama en kadar sonra her değil var "
        "mi şey ben sen o biz"),
}
EN_FUNC = {
    "the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "as", "was", "with", "be",
    "by", "on", "not", "he", "this", "are", "or", "his", "from", "at", "which", "but", "have",
    "an", "they", "you", "were", "their", "one", "all", "we", "can", "her", "has", "there",
    "been", "if", "more", "when", "will"}
WORD = re.compile(r"[^\W\d_]+", re.U)


def script_share(x: str, ranges) -> float:
    letters = [ch for ch in x if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(any(a <= ord(ch) <= b for a, b in ranges) for ch in letters) / len(letters)


def latin_lang_ok(x: str, concept: str) -> bool:
    toks = [t.lower() for t in WORD.findall(x)]
    if len(toks) < 8:
        return False
    fw = set(FUNC[concept].split())
    own = sum(t in fw for t in toks) / len(toks)
    en = sum(t in EN_FUNC for t in toks) / len(toks)
    return own >= 0.08 and own >= 2 * en


CHECKS = {
    "all-caps text": caps_ok, "JSON data": json_ok, "SQL code": sql_ok, "Python code": python_ok,
    "HTML markup": html_ok, "JavaScript code": js_ok, "Java code": java_ok,
    **{c: (lambda x, r=r: script_share(x, r) >= 0.5) for c, r in SCRIPTS.items()},
    **{c: (lambda x, c=c: latin_lang_ok(x, c)) for c in FUNC},
}
FAMILY = {c: ("language" if "language" in c else "format") for c in CHECKS}


def checkable(concept: str) -> bool:
    return concept in CHECKS


def check(concept: str, answer: str) -> bool:
    """FORM check (parses / right markers / right script): every checkable concept."""
    return bool(CHECKS[concept](answer or ""))


# ---- FUNCTIONAL checks (10-26, user: "verify the function, not just the style, as for math") ----
# SQL = EXECUTION ACCURACY (Spider / BIRD style): the task request carries the schema + rows
# (sql_task_request: the reference script's CREATE / INSERT statements + the question); the answer's
# LAST query runs against the reference tables and must return the reference query's rows
# (order-insensitive). Without a reference (old requests had no schema) only the form check applies.
# JSON = the requested function's schema: a function object ({"name": <requested>, "parameters":
# {...}}, optionally wrapped as {"type": "function", "function": {...}}) or the bare parameters
# schema ({"type": "object", "properties": {...}}); properties typed, "required" names exist.

SQL_QUERY = re.compile(r"^\s*(select|with)\b", re.I)
SQL_MAX_STEPS = 2_000_000  # sqlite VM instructions per script (a runaway query is a failure)


def sql_statements(code: str) -> list[str]:
    """Split a script into complete statements (sqlite3.complete_statement: ';' in strings ok)."""
    out, cur = [], ""
    for ch in code:
        cur += ch
        if ch == ";" and sqlite3.complete_statement(cur):
            if cur.strip(" \n\t;"):
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def sql_run(code: str):
    """(ok, rows of the LAST query or None): every statement executes in a fresh in-memory DB."""
    con = sqlite3.connect(":memory:")
    steps = [0]

    def guard():
        steps[0] += 1000
        return 1 if steps[0] > SQL_MAX_STEPS else 0

    con.set_progress_handler(guard, 1000)
    rows = None
    try:
        for st in sql_statements(code):
            cur = con.execute(st)
            if SQL_QUERY.match(st):
                rows = cur.fetchall()
        return True, rows
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError, MemoryError):
        return False, None
    finally:
        con.close()


def sql_split(ref: str):
    """(setup script, reference query) of a reference answer, or None if it does not run."""
    stmts = sql_statements(code_of(ref, ("sql",)))
    queries = [st for st in stmts if SQL_QUERY.match(st)]
    if not queries:
        return None
    setup = "\n".join(st if st.endswith(";") else st + ";" for st in stmts
                      if not SQL_QUERY.match(st))
    ok, rows = sql_run(setup + "\n" + queries[-1])
    return (setup, queries[-1]) if ok and rows is not None else None


def sql_task_request(question: str, ref: str) -> str | None:
    """The question with the reference's tables + rows in front (the functional SQL task), or None
    if the reference does not run in sqlite (then the item cannot be execution-checked)."""
    sp = sql_split(ref)
    if sp is None or not sp[0].strip():
        return None
    return f"Given these tables and rows:\n```sql\n{sp[0]}\n```\n\n{question.strip()}"


def sql_ref_match(x: str, ref: str) -> bool | None:
    """The answer's LAST query run against the REFERENCE tables returns the reference query's rows
    (order-insensitive); None if the reference itself does not run in sqlite."""
    sp = sql_split(ref)
    if sp is None:
        return None
    setup, q_ref = sp
    want = sql_run(setup + "\n" + q_ref)[1]
    q_ans = [st for st in sql_statements(code_of(x, ("sql", "sqlite", "postgresql", "mysql")))
             if SQL_QUERY.match(st)]
    if not q_ans:
        return False
    ok, got = sql_run(setup + "\n" + q_ans[-1])
    return bool(ok and got is not None and sorted(map(repr, got)) == sorted(map(repr, want)))


FUNC_NAME = re.compile(r"function\s+`([^`]+)`")


def json_obj(x: str):
    """The first JSON object / array found (fenced block, whole answer, outermost braces)."""
    cands = [*fenced(x, ("json",)), x]
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = x.find(open_), x.rfind(close)
        if 0 <= i < j:
            cands.append(x[i: j + 1])
    for c in cands:
        try:
            return json.loads(c.strip())
        except (ValueError, RecursionError):
            continue
    return None


def _params_ok(par) -> bool:
    if not isinstance(par, dict) or par.get("type", "object") != "object":
        return False
    props = par.get("properties")
    if not isinstance(props, dict) or not all(isinstance(v, dict) and "type" in v
                                              for v in props.values()):
        return False
    req = par.get("required", [])
    return isinstance(req, list) and all(r in props for r in req)


def json_schema_ok(x: str, request: str = "") -> bool:
    obj = json_obj(x)
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if not isinstance(obj, dict):
        return False
    if len(obj) == 1 and isinstance(obj.get("schema"), dict):  # {"schema": {...}}
        obj = obj["schema"]
    if isinstance(obj.get("function"), dict):  # {"type": "function", "function": {...}}
        obj = obj["function"]
    # the name may sit under "name" or, as a string, under "function" (10-27: base outputs)
    name = obj.get("name", obj.get("function") if isinstance(obj.get("function"), str) else None)
    if "parameters" in obj or name is not None:  # a function object
        m = FUNC_NAME.search(request or "")
        if m and name != m.group(1):
            return False
        return _params_ok(obj.get("parameters"))
    return _params_ok(obj)  # the bare parameters schema


def functional(concept: str, ref: str | None = None) -> bool:
    """Is there a functional check for this item (SQL needs a runnable reference)?"""
    return concept == "JSON data" or (concept == "SQL code" and ref is not None
                                      and sql_split(ref) is not None)


def check_task(concept: str, answer: str, request: str = "", ref: str | None = None) -> bool:
    """Does the answer DO the task? Functional where possible (SQL: execution match against the
    reference tables; JSON: the requested function schema), else the form check. Gates the
    trainer's cap_free NLL."""
    answer = answer or ""
    if concept == "SQL code" and ref is not None:
        m = sql_ref_match(answer, ref)
        if m is not None:
            return m
    if concept == "JSON data":
        return json_schema_ok(answer, request)
    return check(concept, answer)
