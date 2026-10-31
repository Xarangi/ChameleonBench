"""JSON tasks. Two checks: the requested function's schema (a function object, optionally wrapped,
or the bare parameters schema; every property typed, "required" names exist) and, for ToolACE
items, a schema-from-SPEC check (name, parameter set, types, required set). Ported from
tica/validity.py and tica/free_pools.py; behaviour is identical. This module owns `json_spec_ok`;
pools import it, never the reverse."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from crypsis.data.tasks.form import fenced

FUNC_NAME = re.compile(r"function\s+`([^`]+)`")
_TYPES = {"dict": "object", "object": "object", "str": "string", "string": "string",
          "int": "integer", "integer": "integer", "float": "number", "number": "number",
          "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
          "tuple": "array", "any": "any"}


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


def param_names(par) -> list[str] | None:
    """The declared parameters of a function schema, or None if it declares none validly. Accepted
    shapes: a JSON-Schema object ({"type": "object", "properties": {name: {"type": ...}},
    "required": [...]}, required names must exist), a plain {name: {"type": ...}} map, or a list of
    {"name": ..., "type": ...} entries. Every parameter needs a type."""
    if isinstance(par, list):
        if all(isinstance(v, dict) and isinstance(v.get("name"), str) and "type" in v for v in par):
            return [v["name"] for v in par]
        return None
    if not isinstance(par, dict):
        return None
    props = par.get("properties") if "properties" in par else (
        None if "type" in par else par)
    if par.get("type", "object") != "object" or not isinstance(props, dict) or not all(
            isinstance(v, dict) and "type" in v for v in props.values()):
        return None
    req = par.get("required", [])
    if not (isinstance(req, list) and all(isinstance(r, str) and r in props for r in req)):
        return None
    return list(props)


def json_params(x: str):
    """(function name or None, parameters schema or None) of the answer's JSON."""
    obj = json_obj(x)
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None, None
    if len(obj) == 1 and isinstance(obj.get("schema"), dict):  # {"schema": {...}}
        obj = obj["schema"]
    if isinstance(obj.get("function"), dict):  # {"type": "function", "function": {...}}
        obj = obj["function"]
    # the name: under "name", "function_name" or, as a string, under "function" (base outputs)
    name = obj.get("name", obj.get("function_name",
                                   obj.get("function") if isinstance(obj.get("function"), str)
                                   else None))
    if "parameters" in obj or name is not None:  # a function object
        return name, obj.get("parameters")
    return None, obj  # the bare parameters schema


def json_schema_ok(x: str, request: str = "", ref: str | None = None) -> bool:
    """The answer is a valid function schema (name matches the request's `function`, if both exist).
    An empty schema counts only when the reference's own schema is empty too."""
    name, par = json_params(x)
    m = FUNC_NAME.search(request or "")
    names = param_names(par)
    if names is None or (name is not None and m and name != m.group(1)):
        return False
    ref_names = param_names(json_params(ref)[1]) if ref else None
    return bool(names) or not ref_names


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


def json_spec_request(schema: dict, suffix: str | None = "Answer with JSON.") -> str:
    """The request that spells out a function (description + every parameter) to be re-written as a
    JSON schema."""
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


def materialize_toolace_json(schemas_json: str | Path, out_train: str | Path, out_test: str | Path,
                             exclude_names: set[str], n_train: int = 1900, n_test: int = 700,
                             max_params: int = 8, seed: int = 0) -> dict:
    """ToolACE function schemas -> disjoint train / held-out item files [{schema}] (canonical
    format), skipping names in exclude_names and schemas with no description, no parameters or more
    than max_params. Returns {"usable", "train", "test"}."""
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


def toolace_json_rows(path: str | Path, suffix: str | None = "Answer with JSON.") -> list:
    """Pool rows (request, fallback answer = the canonical schema, ref 'jsonspec:...')."""
    return [(json_spec_request(s, suffix), json.dumps(s, indent=2), "jsonspec:" + json.dumps(s))
            for s in json.loads(Path(path).read_text(encoding="utf-8"))]
