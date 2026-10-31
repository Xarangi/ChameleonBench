"""SQL tasks: EXECUTION ACCURACY (Spider / BIRD style). The task request carries the schema + rows
(`sql_task_request`: the reference script's CREATE / INSERT statements + the question); the answer's
LAST query runs against the reference tables and must return the reference query's rows (order- and
column-choice-insensitive, see `rows_match`). Ported from tica/validity.py; behaviour is
identical."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from crypsis.data.tasks.form import code_of, fenced

SQL_QUERY = re.compile(r"^\s*(select|with)\b", re.I)
# leading comments and a language word left by a one-line fence ("```sql SELECT ...```")
SQL_LEAD = re.compile(r"^(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/"
                      r"|(?:sql|sqlite|postgresql|mysql)\b(?=\s+(?:select|with)\b))", re.I | re.S)
SQL_LANGS = ("sql", "sqlite", "postgresql", "mysql")
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


def sql_body(st: str) -> str:
    """A statement without its leading comments / stray fence language word."""
    while True:
        m = SQL_LEAD.match(st)
        if not m or not m.group(0):
            return st
        st = st[m.end():]


def is_query(st: str) -> bool:
    """A SELECT / WITH statement."""
    return bool(SQL_QUERY.match(sql_body(st)))


def sql_answer_query(x: str) -> str | None:
    """The answer's LAST query: over all fenced blocks, else the unfenced text from its last
    SELECT / WITH line (else its last SELECT) to the end of that statement."""
    for code in reversed(fenced(x, SQL_LANGS)):
        qs = [sql_body(st) for st in sql_statements(code) if is_query(st)]
        if qs:
            return qs[-1]
    starts = [m.start() for m in re.finditer(r"(?im)^\s*(?:select|with)\b", x)] or \
        [m.start() for m in re.finditer(r"(?i)\bselect\b", x)]
    if not starts:
        return None
    tail = x[starts[-1]:]
    end = re.search(r";|\n\s*\n", tail)
    return sql_body(tail[: end.end()] if end else tail).strip() or None


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
            if is_query(st):
                rows = cur.fetchall()
        return True, rows
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError, MemoryError):
        return False, None
    finally:
        con.close()


def sql_split(ref: str):
    """(setup script, reference query) of a reference answer, or None if it does not run."""
    stmts = sql_statements(code_of(ref, ("sql",)))
    queries = [st for st in stmts if is_query(st)]
    if not queries:
        return None
    setup = "\n".join(st if st.endswith(";") else st + ";" for st in stmts if not is_query(st))
    ok, rows = sql_run(setup + "\n" + queries[-1])
    # an empty / all-NULL reference result is matched by any empty query ("SELECT 1 WHERE 0"):
    # such items are not execution-checkable
    if not ok or not rows or all(v is None for r in rows for v in r):
        return None
    return setup, queries[-1]


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
    q_ans = sql_answer_query(x)
    if not q_ans:
        return False
    ok, got = sql_run(setup + "\n" + q_ans)
    return bool(ok and got is not None and rows_match(got, want))


def _cell(v):
    return round(v, 4) if isinstance(v, float) else v


def rows_match(got, want) -> bool:
    """Same rows up to row order and COLUMN CHOICE: equal multisets of rows, or the same number of
    rows where each row of the narrower result is contained in a distinct row of the wider one
    (columns added / dropped / reordered). Floats compared at 1e-4."""
    g = [tuple(map(_cell, r)) for r in got]
    w = [tuple(map(_cell, r)) for r in want]
    if sorted(map(repr, g)) == sorted(map(repr, w)):
        return True
    if not g or len(g) != len(w):
        return False
    small, large = (g, w) if len(g[0]) <= len(w[0]) else (w, g)

    def inside(a, b):  # every value of row a appears in row b (with multiplicity)
        rest = list(map(repr, b))
        for v in map(repr, a):
            if v not in rest:
                return False
            rest.remove(v)
        return True

    cand = [[j for j, b in enumerate(large) if inside(a, b)] for a in small]
    match: dict[int, int] = {}

    def augment(i, seen):  # bipartite matching (Kuhn): rows of `small` -> distinct rows of `large`
        for j in cand[i]:
            if j not in seen:
                seen.add(j)
                if j not in match or augment(match[j], seen):
                    match[j] = i
                    return True
        return False

    return all(augment(i, set()) for i in range(len(small)))


# ---- gretel text-to-SQL (gretelai/synthetic_text_to_sql, Apache-2.0) --------------------------
def materialize_gretel_sql(rows_json: str | Path, out: str | Path,
                           limit: int | None = None) -> dict:
    """Fetched gretel rows -> [{id, q, answer, ref}] for the execution-checkable ones: a read-only
    final query, a context that INSERTs data, a reference script (CREATE / INSERT context + query)
    that runs in sqlite and returns a non-degenerate, non-guessable result. ref = the script,
    answer = the query (the STaR fallback), q = the question. Returns {"n", "skipped": {why: n}}."""
    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    items, skipped = [], Counter()
    for r in rows:
        query = (r.get("sql") or "").strip().rstrip(";")
        if not query.lower().lstrip("( ").startswith(("select", "with")):
            skipped["not_select"] += 1
            continue
        if "insert into" not in r["sql_context"].lower():  # no rows: results are degenerate
            skipped["no_data"] += 1
            continue
        ref = f"{r['sql_context'].strip()}\n{query};"
        sp = sql_split(ref)
        if sp is None or not sp[0].strip():
            skipped["no_setup"] += 1
            continue
        ok, got = sql_run(sp[0] + "\n" + sp[1])
        if not ok or not got:
            skipped["no_rows" if ok else "sqlite_error"] += 1
            continue
        if all(v in (0, None, 0.0, "") for row in got for v in row):  # all-zero / NULL result
            skipped["degenerate_result"] += 1
            continue
        if any(rows_match(sql_run(sp[0] + "\n" + t)[1] or [], got)  # a constant guess matches
               for t in ("SELECT 1;", "SELECT 0;", "SELECT 2;")):
            skipped["guessable"] += 1
            continue
        items.append({"id": r.get("id"), "q": r["sql_prompt"].strip(), "answer": query + ";",
                      "ref": ref})
        if limit and len(items) >= limit:
            break
    Path(out).write_text(json.dumps(items), encoding="utf-8")
    return {"n": len(items), "skipped": dict(skipped)}


def gretel_sql_rows(path: str | Path, suffix: str | None = None) -> list[tuple[str, str, str]]:
    """Pool rows (request, fallback answer, reference) for materialized gretel SQL items."""
    out = []
    for it in json.loads(Path(path).read_text(encoding="utf-8")):
        q = it["q"] + (f"\n\n{suffix}" if suffix else "")
        req = sql_task_request(q, it["ref"])
        if req is not None:
            out.append((req, it["answer"], it["ref"]))
    return out
