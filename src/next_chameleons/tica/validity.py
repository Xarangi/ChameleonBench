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

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(```|\Z)", re.S)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr", "!doctype"}
IMPLICIT = {"li", "p", "td", "tr", "th", "dt", "dd", "option", "html", "body", "head", "tbody",
            "thead"}


def fenced(x: str, langs: tuple[str, ...] = ()) -> list[str]:
    """Code blocks (``` fences; preferred language tags first), else []."""
    blocks = [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(x)]
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
    return bool(CHECKS[concept](answer or ""))
