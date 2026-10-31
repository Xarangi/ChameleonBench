"""FORM checks: does an answer still do the concept's job by its shape (parses as SQL / JSON /
Python / JavaScript / Java / HTML, all caps, written in the requested language)? Stdlib only, except
language ID (fastText lid.176; the model file is an explicit argument, never read from the env).
Ported from tica/validity.py; behaviour is identical."""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

# A language tag counts only when it is alone on the fence line ("```python\n"); otherwise the code
# starts right after the backticks ("```SELECT ..." must not lose its first line to the tag group).
FENCE = re.compile(r"```(?:[ \t]*([A-Za-z0-9_+-]+)[ \t]*\n|[ \t]*\n?)(.*?)(```|\Z)", re.S)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr", "!doctype"}
IMPLICIT = {"li", "p", "td", "tr", "th", "dt", "dd", "option", "html", "body", "head", "tbody",
            "thead"}
SQL_KW = re.compile(r"\b(select|insert|update|delete|create|alter|drop|with|replace)\b", re.I)
SQL_SYNTAX = ("syntax error", "incomplete input", "unrecognized token")

# LANGUAGE concepts -> ISO code of the fastText prediction that counts as a pass.
LANG_CODES = {
    "German language": "de", "Japanese language": "ja", "Arabic language": "ar",
    "Hindi language": "hi", "Finnish language": "fi", "French language": "fr",
    "Portuguese language": "pt", "Turkish language": "tr", "Russian language": "ru",
    "Greek language": "el", "Thai language": "th", "Tamil language": "ta",
}


def fenced(x: str, langs: tuple[str, ...] = ()) -> list[str]:
    """Code blocks (``` fences; preferred language tags first), else []."""
    blocks = [((m.group(1) or "").lower(), m.group(2)) for m in FENCE.finditer(x)]
    pref = [b for t, b in blocks if t in langs]
    return pref + [b for t, b in blocks if t not in langs]


def code_of(x: str, langs: tuple[str, ...]) -> str:
    """The first fenced block (preferred tags first), else the whole text."""
    b = fenced(x, langs)
    return b[0] if b else x


def caps_ok(x: str) -> bool:
    """>= 90% of the cased letters upper case; a letter-free answer passes when non-empty."""
    cased = [ch for ch in x if ch.isalpha() and ch.upper() != ch.lower()]
    if not cased:
        return bool(x.strip())
    return sum(ch.isupper() for ch in cased) / len(cased) >= 0.9


def json_ok(x: str) -> bool:
    """Some fenced block, the whole text or the outermost braces / brackets parse as JSON."""
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


def sql_ok(x: str) -> bool:
    """SQL keywords present and no statement fails with a SYNTAX error in an empty sqlite DB."""
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
    """Python-looking code that `ast.parse` accepts."""
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
        elif tag not in IMPLICIT:  # a stray </html> / </body> / </p> is ignored (browsers)
            self.bad += 1


def html_ok(x: str) -> bool:
    """At least one tag, no stray end tags, every explicit tag closed."""
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
    """JavaScript-looking code with balanced brackets."""
    code = code_of(x, ("javascript", "js", "jsx", "typescript", "ts"))
    return bool(re.search(r"\bfunction\b|=>|\b(const|let|var)\s+\w", code)) and _balanced(code)


def java_ok(x: str) -> bool:
    """Java-looking code (method signature or class) with balanced brackets."""
    code = code_of(x, ("java",))
    sig = re.search(r"\b(public|private|protected|static)\b[^;{]*\(|\bclass\s+\w+", code)
    return bool(sig) and "{" in code and _balanced(code)


class LIDUnavailable(RuntimeError):
    """The fastText package or the lid.176 model file could not be loaded."""


class LanguageID:
    """fastText language identification (lid.176, Joulin et al. 2016) with an explicit model path.

    A missing model RAISES (LIDUnavailable): it never silently fails every answer. The model loads
    on first use and is held by this object, not by the module."""

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._model = None

    def top(self, x: str) -> str | None:
        """Most likely language (ISO code) of a text, or None for a text with no letters."""
        if self._model is None:
            try:
                import fasttext

                self._model = fasttext.load_model(self.model_path)
            except Exception as e:
                raise LIDUnavailable(f"language ID needs fasttext + {self.model_path}: {e}") from e
        text = " ".join(x.split())
        if not any(ch.isalpha() for ch in text):
            return None
        # the low-level predict: fasttext.FastText.predict's np.array(copy=False) breaks on numpy 2
        (_, label), = self._model.f.predict(text, 1, 0.0, "replace")
        return label.replace("__label__", "")

    def is_language(self, x: str, concept: str) -> bool:
        """The answer's most likely language is the one `concept` ("German language") asks for."""
        return self.top(x) == LANG_CODES[concept]


FORM_CHECKS: dict[str, Callable[[str], bool]] = {
    "all-caps text": caps_ok, "JSON data": json_ok, "SQL code": sql_ok, "Python code": python_ok,
    "HTML markup": html_ok, "JavaScript code": js_ok, "Java code": java_ok,
}


# ---- code extraction (MBPP / HumanEval-JS / APPS answers) ------------------------------------
_CODE_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _extract(generation: str, langs: tuple[str, ...]) -> str:
    blocks = _CODE_FENCE.findall(generation)
    for lang, body in blocks:
        if lang.lower() in langs:
            return body.strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return generation.strip("\n")


def extract_code(generation: str) -> str:
    """Code from the first ```python fence (else the first fence, else the raw text)."""
    return _extract(generation, ("python", "py", "python3"))


def extract_js_code(generation: str) -> str:
    """Code from the first ```javascript / ```js fence (else the first fence, else the raw text)."""
    return _extract(generation, ("javascript", "js", "jsx", "node", "typescript", "ts"))
