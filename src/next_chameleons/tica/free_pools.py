"""Larger free-generation capability pools and a held-out split (10-30, user: "make sure that we
have enough samples in capability preservation so that the model doesn't memorize them ... we
should eval on examples not trained on").

The cap_free pools came from each checkable concept's calib / episode requests: 130-240 items, each
drawn ~3-6 times over a 1,200-step run, and the capability evals then used the same items.

  dolly_rows    Dolly-15k instructions (databricks-dolly-15k, CC-BY-SA-3.0) with the concept's own
                answer directive appended (the last line of its existing requests, e.g. "Respond
                in German." / "Write your entire response in capital letters."), for the
                FORM-checked concepts (languages: fastText language ID; all-caps). Each concept
                gets a DISJOINT slice of the shuffled instructions, so no prompt repeats across
                concepts. No fallback answer.
  split_holdout deterministic held-out share of a pool (hash of the request): those rows never
                enter training and are written next to the run for held-out evaluation.

Aggregates only: generated answers are never stored.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

MAX_CONTEXT_CHARS = 1200  # Dolly items with a long context are dropped (prompt length)


def directive(requests: list[str]) -> str | None:
    """The concept's answer directive: the most common last line of its existing requests (None if
    no line is shared by at least a quarter of them)."""

    last = [r.strip().splitlines()[-1].strip() for r in requests if r.strip()]
    if not last:
        return None
    line, n = Counter(last).most_common(1)[0]
    return line if n >= max(2, len(last) // 4) else None


def dolly_rows(path: str | Path, concepts: dict[str, str], n_per_concept: int,
               seed: int = 0) -> dict[str, list[tuple[str, None, None]]]:
    """{concept: [(request, None, None)]} from Dolly-15k: disjoint shuffled slices of n_per_concept
    items, each request = instruction (+ context) + the concept's directive (concepts: {name:
    directive})."""

    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ctx = (r.get("context") or "").strip()
            if len(ctx) > MAX_CONTEXT_CHARS or not r.get("instruction", "").strip():
                continue
            body = r["instruction"].strip() + (f"\n\n{ctx}" if ctx else "")
            items.append(body)
    random.Random(seed).shuffle(items)
    out, start = {}, 0
    for c in sorted(concepts):
        chunk = items[start : start + n_per_concept]
        start += n_per_concept
        out[c] = [(f"{b}\n\n{concepts[c]}", None, None) for b in chunk]
    return out


def _held(request: str, frac: float) -> bool:
    h = int(hashlib.sha256(request.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def split_holdout(pool: dict[str, list], frac: float) -> tuple[dict[str, list], dict[str, list]]:
    """(training pool, held-out pool): a deterministic `frac` of every concept's rows (by request
    hash) is held out. frac <= 0 -> (pool, {})."""

    if frac <= 0:
        return pool, {}
    train, held = {}, {}
    for c, rows in pool.items():
        held[c] = [r for r in rows if _held(r[0], frac)]
        train[c] = [r for r in rows if not _held(r[0], frac)]
    return train, held


def materialize_gretel_sql(rows_json: str | Path, out: str | Path,
                           limit: int | None = None) -> dict:
    """gretelai/synthetic_text_to_sql rows (Apache-2.0, fetched on a login node) -> [{q, answer,
    ref}]
    for the execution-checkable ones: a read-only final query, a context that INSERTs data, the
    reference script (CREATE / INSERT context + query) runs in sqlite and returns a row that is not
    all zero / NULL. ref = the script, answer = the query
    (the STaR fallback), q = the question; the cap_free pool builds the request with
    validity.sql_task_request (tables + rows in front, like the concept's own items)."""

    from next_chameleons.tica.validity import rows_match, sql_run, sql_split

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
    """cap_free pool rows (request, fallback answer, reference) for materialized gretel SQL
    items."""

    from next_chameleons.tica.validity import sql_task_request

    out = []
    for it in json.loads(Path(path).read_text(encoding="utf-8")):
        q = it["q"] + (f"\n\n{suffix}" if suffix else "")
        req = sql_task_request(q, it["ref"])
        if req is not None:
            out.append((req, it["answer"], it["ref"]))
    return out


# ---- JSON: schema-from-spec tasks (ToolACE function schemas, Apache-2.0) ------------------------
# The concept's own JSON items ask "Write the JSON schema for the function `X`." with only the name,
# so the check could only be "a schema with parameters and that name". Here the request spells out
# the function (description + every parameter's type, required-ness, enum) and the check compares
# the answer's schema with it: name, parameter set, types, required set.
_TYPES = {"dict": "object", "object": "object", "str": "string", "string": "string",
          "int": "integer", "integer": "integer", "float": "number", "number": "number",
          "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
          "tuple": "array", "any": "any"}


def _norm_type(t) -> str:
    return _TYPES.get(str(t).strip().lower(), str(t).strip().lower())


def _canon(schema: dict) -> dict:
    """The schema in the concept's format: type object, normalized parameter types."""

    p = schema["parameters"]
    props = {k: {kk: (_norm_type(vv) if kk == "type" else vv) for kk, vv in v.items()}
             for k, v in p["properties"].items()}
    out = {"name": schema["name"], "description": schema.get("description", ""),
           "parameters": {"type": "object", "properties": props}}
    if p.get("required"):
        out["parameters"]["required"] = list(p["required"])
    return out


def materialize_toolace_json(schemas_json: str | Path, out_train: str | Path, out_test: str | Path,
                             exclude_names: set[str], n_train: int = 1900, n_test: int = 700,
                             max_params: int = 8, seed: int = 0) -> dict:
    """ToolACE function schemas -> disjoint train / held-out item files [{schema}] (canonical
    format), skipping names in exclude_names (the concept's own items) and schemas with no
    description, no parameters or more than max_params."""

    rows = json.loads(Path(schemas_json).read_text(encoding="utf-8"))
    ok = [r for r in rows if r.get("description") and r["name"] not in exclude_names
          and 1 <= len(r["parameters"]["properties"]) <= max_params
          and all(isinstance(v, dict) and v.get("type")
                  for v in r["parameters"]["properties"].values())]
    random.Random(seed).shuffle(ok)
    test, train = ok[:n_test], ok[n_test : n_test + n_train]
    Path(out_train).write_text(json.dumps([_canon(r) for r in train]), encoding="utf-8")
    Path(out_test).write_text(json.dumps([_canon(r) for r in test]), encoding="utf-8")
    return {"usable": len(ok), "train": len(train), "test": len(test)}


def json_spec_request(schema: dict, suffix: str | None = "Answer with JSON.") -> str:
    p = schema["parameters"]
    req = set(p.get("required", []))
    lines = []
    for k, v in p["properties"].items():
        extra = f"; one of {json.dumps(v['enum'])}" if v.get("enum") else ""
        need = "required" if k in req else "optional"
        lines.append(f"- {k} ({v.get('type', 'any')}, {need}{extra}): "
                     f"{(v.get('description') or '').strip()}")
    body = (f"Write the JSON schema for the function `{schema['name']}`: "
            f"{schema.get('description', '').strip()}\nParameters:\n" + "\n".join(lines))
    return body + (f"\n\n{suffix}" if suffix else "")


def json_spec_ok(answer: str, schema: dict) -> bool:
    """The answer's JSON schema matches the spec: same name, same parameter names, same (normalized)
    types, same required set (when the spec has one)."""

    from next_chameleons.tica.validity import json_params

    name, par = json_params(answer or "")
    if name != schema["name"] or not isinstance(par, dict):
        return False
    props = par.get("properties")
    want = schema["parameters"]["properties"]
    if not isinstance(props, dict) or set(props) != set(want):
        return False
    for k, v in want.items():
        got = props[k] if isinstance(props[k], dict) else {}
        if _norm_type(got.get("type", "")) != _norm_type(v.get("type", "")):
            return False
    req = set(schema["parameters"].get("required", []))
    return set(par.get("required", []) or []) == req if req else True


def toolace_json_rows(path: str | Path, suffix: str | None = "Answer with JSON.") -> list:
    """cap_free pool rows (request, fallback answer = the canonical schema, ref 'jsonspec:...')."""

    return [(json_spec_request(s, suffix), json.dumps(s, indent=2), "jsonspec:" + json.dumps(s))
            for s in json.loads(Path(path).read_text(encoding="utf-8"))]
