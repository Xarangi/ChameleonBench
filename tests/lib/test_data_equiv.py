# ruff: noqa: E501
"""crypsis.data against the snapshot code (next_chameleons.tica.*): same check verdicts, pools,
materialisations, GSM8K pieces and concept accessors on synthetic data written to tmp."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from crypsis.data import ConceptSet, RequestMap, build_free_pool, split_holdout, text_id
from crypsis.data import pools as new_pools
from crypsis.data.tasks import TaskEnv, check_task
from crypsis.data.tasks import apps_js as new_apps
from crypsis.data.tasks import gsm8k as new_gsm
from crypsis.data.tasks import json_spec as new_json
from crypsis.data.tasks import sql as new_sql

NODE_CVMFS = ("/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/"
              "nodejs/22.23.4/bin/node")


def _node() -> str | None:
    cand = os.environ.get("TICA_NODE") or shutil.which("node") or NODE_CVMFS
    try:
        ok = os.path.exists(cand) and subprocess.run(
            [cand, "-e", "0"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cand if ok else None


NODE = _node()
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


def _lid_model() -> str | None:
    from next_chameleons.tica import validity

    path = os.environ.get("TICA_LID_MODEL", validity.LID_MODEL)
    try:
        import fasttext  # noqa: F401
    except ImportError:
        return None
    return path if Path(path).exists() else None


LID = _lid_model()
needs_lid = pytest.mark.skipif(LID is None, reason="fastText package or lid.176.bin not present")

REF = ("CREATE TABLE t (id INT, city TEXT, score INT); INSERT INTO t VALUES (1, 'NY', 80), "
       "(2, 'NY', 70), (3, 'LA', 90);\n\nSELECT city, AVG(score) FROM t GROUP BY city;")
BAD_REF = "SELECT DATE_SUB(NOW(), INTERVAL 1 DAY) FROM missing;"
JSON_REQ = "Write the JSON schema for the function `get_time`."
JSON_OBJ = ('{"name": "get_time", "parameters": {"type": "object", "properties": '
            '{"tz": {"type": "string"}}, "required": ["tz"]}}')
SPEC = new_json._canon({"name": "get_rate", "description": "Rate.", "parameters": {
    "type": "dict", "properties": {"base": {"type": "str", "description": "from"},
                                   "n": {"type": "int", "description": "amt"}},
    "required": ["base"]}})


def _answers() -> list[tuple[str, str, str, str | None]]:
    sql_req = new_sql.sql_task_request("Average score per city?", REF)
    spec_ref = "jsonspec:" + json.dumps(SPEC)
    spec_req = new_json.json_spec_request(SPEC)
    wrong_spec = json.dumps({**SPEC, "parameters": {**SPEC["parameters"], "properties": {
        "base": {"type": "string"}, "n": {"type": "string"}}}})
    return [
        # SQL: execution accuracy, reorder / alias, wrong, not executing, prose, bad reference
        ("SQL code", "```sql\nSELECT city, AVG(score) AS a FROM t GROUP BY city ORDER BY city;\n```",
         sql_req, REF),
        ("SQL code", "SELECT city FROM t;", sql_req, REF),
        ("SQL code", "SELECT nope FROM t;", sql_req, REF),
        ("SQL code", "The average is 75 for NY.", sql_req, REF),
        ("SQL code", "SELECT a FROM b;", "", BAD_REF),
        ("SQL code", "SELEC a FROM b;", "", BAD_REF),
        ("SQL code", "SELECT 1;", "", None),
        ("SQL code", "hello world", "", None),
        ("SQL code", "", "", REF),
        # JSON schema / spec
        ("JSON data", JSON_OBJ, JSON_REQ, None),
        ("JSON data", '{"type": "function", "function": ' + JSON_OBJ + "}", JSON_REQ, None),
        ("JSON data", JSON_OBJ.replace("get_time", "other"), JSON_REQ, None),
        ("JSON data", '{"type": "object", "properties": {"tz": {}}}', JSON_REQ, None),
        ("JSON data", '{"type": "object", "properties": {}}', JSON_REQ, None),
        ("JSON data", '{"type": "object", "properties": {}}', JSON_REQ, JSON_OBJ),
        ("JSON data", "not json", JSON_REQ, None),
        ("JSON data", json.dumps(SPEC), spec_req, spec_ref),
        ("JSON data", f"```json\n{json.dumps(SPEC)}\n```", spec_req, spec_ref),
        ("JSON data", wrong_spec, spec_req, spec_ref),
        ("JSON data", "nope", spec_req, spec_ref),
        # form checks
        ("all-caps text", "CANBERRA!", "", None), ("all-caps text", "Canberra!", "", None),
        ("all-caps text", "42!", "", None), ("all-caps text", "   ", "", None),
        ("Python code", "```python\ndef f(x):\n    return x+1\n```", "", None),
        ("Python code", "def f(x:\n    return", "", None), ("Python code", "just words", "", None),
        ("HTML markup", "<ul><li>a<li>b</ul>", "", None), ("HTML markup", "<div><p>x</div></p>",
                                                           "", None),
        ("HTML markup", "plain", "", None),
        ("JavaScript code", "```js\nfunction f(a){return a+1}\n```", "", None),
        ("JavaScript code", "function f(a){return a+1", "", None),
        ("JavaScript code", "const x = (1 + 2;", "", "not-an-apps-ref"),
        ("Java code", "public static int f(int a) { return a; }", "", None),
        ("Java code", "int f", "", None),
        ("SQL code", "SELECT\x00 1", "", None), ("JSON data", None, JSON_REQ, None),
        ("not a concept", "abc", "", None),
    ]


def _old_env():
    from next_chameleons.tica import validity

    return validity


def test_check_task_matches_validity_without_language_id():
    old = _old_env()
    env = TaskEnv()
    for concept, ans, req, ref in _answers():
        got = check_task(concept, ans, req, ref, env=env)
        assert got == old.check_task(concept, ans, req, ref), (concept, ans, req, ref)
    # a pass and a fail exist for each functional check (the table is not vacuous)
    verdicts = {(c, check_task(c, a, q, r)) for c, a, q, r in _answers()}
    for c in ("SQL code", "JSON data", "all-caps text", "Python code", "HTML markup", "Java code"):
        assert {(c, True), (c, False)} <= verdicts, c




# ---- pools ---------------------------------------------------------------------------------------
def _apps_rows():
    return [
        {"id": 1, "difficulty": "introductory", "question": "Add two numbers a and b.",
         "starter_code": "", "input_output": json.dumps(
             {"fn_name": "add", "inputs": [[1, 2], [0.1, 0.2]], "outputs": [[3], [0.3]]})},
        {"id": 2, "difficulty": "introductory", "question": "Read n, print 2n.", "starter_code": "",
         "input_output": json.dumps({"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"]})},
        {"id": 3, "difficulty": "interview", "question": "hard", "starter_code": "",
         "input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})},
        {"id": 4, "difficulty": "introductory", "question": "leetcode",
         "starter_code": "class Solution:", "input_output": json.dumps(
             {"fn_name": "f", "inputs": [[1]], "outputs": [[1]]})},
        {"id": 5, "difficulty": "introductory", "question": "mention python here",
         "starter_code": "", "input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]})},
        {"id": 6, "difficulty": "introductory", "question": "no tests", "starter_code": "",
         "input_output": ""},
    ]


def _gretel_rows():
    ctx = "CREATE TABLE emp (id INT, dept TEXT, pay INT); INSERT INTO emp VALUES (1,'a',10),(2,'b',20);"
    return [
        {"id": 1, "sql": "SELECT dept, pay FROM emp WHERE id = 2;", "sql_context": ctx,
         "sql_prompt": "Pay of employee 2?"},
        {"id": 2, "sql": "DELETE FROM emp;", "sql_context": ctx, "sql_prompt": "wipe"},
        {"id": 3, "sql": "SELECT 1 FROM emp;", "sql_context": "CREATE TABLE emp (id INT);",
         "sql_prompt": "no data"},
        {"id": 4, "sql": "SELECT COUNT(*) FROM emp WHERE pay > 100;", "sql_context": ctx,
         "sql_prompt": "zero result"},
        {"id": 5, "sql": "SELECT 1;", "sql_context": ctx, "sql_prompt": "guessable"},
        {"id": 6, "sql": "SELECT nope FROM emp;", "sql_context": ctx, "sql_prompt": "broken"},
        {"id": 7, "sql": "SELECT MAX(pay) FROM emp;", "sql_context": ctx,
         "sql_prompt": "Highest pay?"},
    ]


def _toolace_rows():
    def fn(name, n=2, desc="does a thing"):
        return {"name": name, "description": desc, "parameters": {
            "type": "dict", "properties": {f"p{k}": {"type": "str" if k else "int",
                                                     "description": f"d{k}"} for k in range(n)},
            "required": ["p0"]}}

    return [fn(f"f{k}", 1 + k % 3) for k in range(12)] + [fn("nodesc", desc=""),
                                                          fn("excl"), fn("big", 12)]


def _dolly_rows():
    return [{"instruction": f"Explain topic {k}", "context": "ctx" if k % 4 == 0 else "",
             "response": "r", "category": "open_qa"} for k in range(40)] + [
        {"instruction": "long", "context": "x" * 4000, "response": "r", "category": "c"}]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _level1(tmp_path):
    """A synthetic Level1Data file whose train concepts carry the texts the pool needs."""
    sql_text = REF
    sql_text2 = ("CREATE TABLE s (a INT); INSERT INTO s VALUES (3), (4);\nSELECT SUM(a) FROM s;")
    texts = {
        "SQL code": [sql_text, sql_text2, "SELECT 1;", "no sql at all"],
        "JSON data": [JSON_OBJ, '{"type": "object", "properties": {"x": {"type": "number"}}}',
                      "not json"],
        "all-caps text": ["SHOUT LOUDLY", "quiet words here", "ALSO LOUD!", "MORE LOUD TEXT"],
        "JavaScript code": ["function a(){return 1}", "function b(){return (", "const q = () => 3"],
        "Python code": ["def f():\n    return 1", "x = (", "print('hi')"],
        "sports": ["a text about sports"],
    }
    concepts, req = [], {}
    for name, ts in texts.items():
        items = [{"id": text_id(t), "text": t} for t in ts]
        concepts.append({"concept": name, "split": "train" if name != "Python code" else "dev",
                         "fit": items[:1], "calib": items[1:3], "episode": items[3:] + items[:1]})
        for k, t in enumerate(ts):
            req[text_id(t)] = ("Average score per city?" if t == sql_text else
                               f"Write something about item {k}.\n\nAnswer with {name}.")
    req.pop(text_id("no sql at all"), None)
    data = {"source": "synthetic", "concepts": concepts,
            "negatives": {"fit": [{"id": "n", "text": "neg"}], "calib": [], "episode": []}}
    p = tmp_path / "level1.json"
    p.write_text(json.dumps(data))
    rp = tmp_path / "inputs.json"
    rp.write_text(json.dumps(req))
    return p, rp




def _old_cap_free_requests():
    """scripts/tica.py::cap_free_requests, lifted by AST (importing the script pulls in torch)."""
    import ast

    src = (Path(__file__).resolve().parents[2] / "scripts" / "tica.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "cap_free_requests")
    ns: dict = {}
    exec(compile(ast.Module([fn], []), "tica.py", "exec"), ns)
    return ns["cap_free_requests"]


def test_build_free_pool_equals_cap_free_requests(tmp_path):
    import types

    from next_chameleons.tica.data import Level1Data

    old_cap_free = _old_cap_free_requests()
    lp, rp = _level1(tmp_path)
    apps_src = tmp_path / "train.jsonl"
    _write_jsonl(apps_src, _apps_rows())
    apps = tmp_path / "apps_js.json"
    new_apps.materialize(apps_src, apps)
    gsrc, gout = tmp_path / "gretel_rows.json", tmp_path / "gretel.json"
    gsrc.write_text(json.dumps(_gretel_rows()))
    new_sql.materialize_gretel_sql(gsrc, gout)
    tsrc, ttr, tte = tmp_path / "tool.json", tmp_path / "tool_tr.json", tmp_path / "tool_te.json"
    tsrc.write_text(json.dumps(_toolace_rows()))
    new_json.materialize_toolace_json(tsrc, ttr, tte, {"excl"}, n_train=6, n_test=3)
    dolly = tmp_path / "dolly.jsonl"
    _write_jsonl(dolly, _dolly_rows())

    concepts = ["SQL code", "JSON data", "all-caps text", "JavaScript code", "Python code",
                "sports", "German language", "unknown"]
    old_data = Level1Data(lp)
    inputs = json.loads(rp.read_text())
    cs, rm = ConceptSet.load(lp), RequestMap.load(rp)
    for keep in (False, True):
        for use in ("none", "all"):
            acfg = types.SimpleNamespace(
                cap_free_js_tasks=str(apps) if use == "all" else "", cap_free_js_keep_form=keep,
                cap_free_sql_tasks=str(gout) if use == "all" else "",
                cap_free_json_tasks=str(ttr) if use == "all" else "",
                cap_free_dolly=str(dolly) if use == "all" else "", cap_free_dolly_n=7)
            want = old_cap_free(old_data, inputs, concepts, acfg)
            kw = {}
            if use == "all":
                kw = dict(apps_js=apps, gretel_sql=gout, toolace_json=ttr, dolly=dolly, dolly_n=7)
            got = build_free_pool(cs, rm, concepts, js_keep_form=keep, **kw)
            assert got == want
            assert list(got) == list(want)
            assert split_holdout(got, 0.25) == new_pools.split_holdout(want, 0.25)
    assert want["SQL code"] and want["JavaScript code"] and want["JSON data"]
    assert split_holdout(want, 0.25) == __import__(
        "next_chameleons.tica.free_pools", fromlist=["x"]).split_holdout(want, 0.25)


# ---- gsm8k ---------------------------------------------------------------------------------------
TRICKY = [
    "The answer is 1,234.", "The answer is $1,234.50.", "The answer is -7.", "answer is 7.00",
    "The answer is $-3", "so 12 + 30 = 42. The answer is 42.", "#### 18", "Work: 3*4=12\n#### 12\n",
    "nothing numeric", "", "the Answer Is 5. Question: What? The answer is 99.",
    "I think 17 apples, then 1,000 more", "Total: $2,500", "x = 3.0", "answer is 0.", "-0.50 left",
    "The answer is 1000000", "7 x 8 = 56", "14 \u00f7 2 = 7", "3 \u00d7 3 = 9", "area = 2*pi*r",
    "= 5", "no digits = here",
]


def test_gsm8k_pieces_match():
    from next_chameleons.benchmarks import capability as oldcap
    from next_chameleons.tica.trainers import mathrl

    assert new_gsm.PROMPT == mathrl.PROMPT and new_gsm.CONCEPT == mathrl.CONCEPT
    golds = ["1234", "1,234.50", "-7", "7", "$3", "42", "18", "12", "5", "0", "99", "17", "2500"]
    for t in TRICKY:
        assert new_gsm.extract_gsm8k_answer(t) == oldcap.extract_gsm8k_answer(t), t
        assert new_gsm.work_shown(t) == mathrl.work_shown(t), t
        assert new_gsm.answer_phrase(t) == mathrl.answer_phrase(t), t
        for g in golds:
            assert new_gsm.is_correct(t, g) == mathrl.is_correct(t, g), (t, g)
    for n in ("7", "7.00", "7,000", "$7", "-3.50", "abc", "", "1.", "1e3"):
        assert new_gsm.canon(n) == oldcap._canon(n)


