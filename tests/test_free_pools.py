"""tica/free_pools.py (10-30): larger form-checked capability pools and the held-out split."""

from __future__ import annotations

import json


def test_directive_is_the_shared_last_line():
    from next_chameleons.tica.free_pools import directive

    reqs = [f"question {k}\n\nRespond in German." for k in range(6)] + ["odd one\n\nAntworte."]
    assert directive(reqs) == "Respond in German."
    assert directive([f"q{k}\nline {k}" for k in range(8)]) is None  # nothing shared
    assert directive([]) is None


def test_dolly_rows_disjoint_with_directive(tmp_path):
    from next_chameleons.tica.free_pools import dolly_rows

    src = tmp_path / "dolly.jsonl"
    rows = [{"instruction": f"instr {k}", "context": "", "response": "r", "category": "c"}
            for k in range(30)]
    rows.append({"instruction": "long", "context": "x" * 5000, "response": "r", "category": "c"})
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = dolly_rows(src, {"German language": "Respond in German.",
                           "all-caps text": "Write your entire response in capital letters."}, 10)
    de, caps = out["German language"], out["all-caps text"]
    assert len(de) == len(caps) == 10
    assert all(q.endswith("Respond in German.") and a is None and r is None for q, a, r in de)
    bodies = lambda rs: {q.rsplit("\n\n", 1)[0] for q, _, _ in rs}  # noqa: E731
    assert not bodies(de) & bodies(caps)  # disjoint instruction slices
    assert all("x" * 100 not in q for q, _, _ in de + caps)  # long-context item dropped


def test_split_holdout_deterministic_and_disjoint():
    from next_chameleons.tica.free_pools import split_holdout

    pool = {"SQL code": [(f"req {k}", "a", "r") for k in range(400)]}
    tr1, he1 = split_holdout(pool, 0.25)
    tr2, he2 = split_holdout(pool, 0.25)
    assert tr1 == tr2 and he1 == he2
    assert not {r[0] for r in tr1["SQL code"]} & {r[0] for r in he1["SQL code"]}
    assert len(tr1["SQL code"]) + len(he1["SQL code"]) == 400
    assert 60 <= len(he1["SQL code"]) <= 140
    assert split_holdout(pool, 0.0) == (pool, {})


def test_json_spec_check_and_routing():
    from next_chameleons.tica.free_pools import _canon, json_spec_ok, json_spec_request
    from next_chameleons.tica.validity import check_task

    schema = _canon({"name": "get_rate", "description": "Exchange rate.", "parameters": {
        "type": "dict", "properties": {"base": {"type": "str", "description": "from"},
                                       "n": {"type": "int", "description": "amount"}},
        "required": ["base"]}})
    req = json_spec_request(schema)
    assert "`get_rate`" in req and "base (string, required)" in req
    assert "n (integer, optional)" in req
    good = json.dumps(schema)
    assert json_spec_ok(good, schema)
    wrong_type = json.dumps({**schema, "parameters": {**schema["parameters"], "properties": {
        "base": {"type": "string"}, "n": {"type": "string"}}}})
    assert not json_spec_ok(wrong_type, schema)
    missing = json.dumps({**schema, "parameters": {"type": "object",
                                                   "properties": {"base": {"type": "string"}},
                                                   "required": ["base"]}})
    assert not json_spec_ok(missing, schema)
    ref = "jsonspec:" + json.dumps(schema)
    assert check_task("JSON data", f"```json\n{good}\n```", req, ref)
    assert not check_task("JSON data", f"```json\n{wrong_type}\n```", req, ref)
