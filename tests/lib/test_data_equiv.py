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
from crypsis.data.tasks import TASKS, LanguageID, TaskEnv, check_task, checkable
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


def test_checkable_family_functional_match():
    old = _old_env()
    for c in [*old.CHECKS, "sports", "mathematical reasoning"]:
        assert checkable(c) == old.checkable(c)
    assert {c: TASKS[c].family for c in TASKS} == old.FAMILY
    from crypsis.data.tasks import functional

    for c, ref in (("JSON data", None), ("SQL code", REF), ("SQL code", None),
                   ("SQL code", BAD_REF), ("Python code", REF)):
        assert functional(c, ref) == old.functional(c, ref)


def test_sql_helpers_match():
    old = _old_env()
    cases = ["SELECT 1; SELECT 2;", "INSERT INTO t VALUES (1, 'a;b', 2); SELECT 1;", REF, BAD_REF,
             "```sql\n-- c\nSELECT 2\n```", "prose then SELECT a FROM t;\n\nmore"]
    for s in cases:
        assert new_sql.sql_statements(s) == old.sql_statements(s)
        assert new_sql.sql_answer_query(s) == old.sql_answer_query(s)
        assert new_sql.sql_split(s) == old.sql_split(s)
        assert new_sql.sql_task_request("q?", s) == old.sql_task_request("q?", s)
        assert new_sql.sql_run(s) == old.sql_run(s)
    rows = [([(1, "a")], [(1, "a")]), ([(1,)], [(1, "a")]), ([(2,)], [(1, "a")]),
            ([(1.00001,)], [(1.0,)]), ([], []), ([(1,), (2,)], [(2,), (1,)]),
            ([(1,), (1,)], [(1, 2), (3, 1)])]
    for g, w in rows:
        assert new_sql.rows_match(g, w) == old.rows_match(g, w)


@needs_lid
def test_language_checks_match():
    old = _old_env()
    env = TaskEnv(lid=LanguageID(LID))
    os.environ.setdefault("TICA_LID_MODEL", LID)
    texts = ["Guten Tag, wie geht es dir heute?", "Hello, how are you today?",
             "こんにちは、お元気ですか", "Bonjour tout le monde", "12345 !!!", "", "Привет, мир"]
    for concept in [c for c in TASKS if "language" in c]:
        for t in texts:
            assert check_task(concept, t, "", t, env=env) == old.check_task(concept, t, "", t)
    assert env.lid.top("12345") is None


def test_language_check_without_model_raises():
    from crypsis.data.tasks import LIDUnavailable

    with pytest.raises(LIDUnavailable):
        check_task("German language", "Guten Tag")
    with pytest.raises(LIDUnavailable):
        check_task("German language", "x", env=TaskEnv(lid=LanguageID("/nonexistent/lid.bin")))


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


def test_build_free_pool_equals_cap_free_requests(tmp_path):
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "tica_script", Path(__file__).resolve().parents[2] / "scripts" / "tica.py")
    import sys

    sys.path.insert(0, str(Path(spec.origin).parent))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"scripts/tica.py not importable here: {e}")
    finally:
        sys.path.pop(0)
    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica import js_tasks as old_js

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
            want = mod.cap_free_requests(old_data, inputs, concepts, acfg)
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
    del old_js


def test_materializers_and_row_builders_match(tmp_path):
    from next_chameleons.tica import free_pools as old

    gsrc = tmp_path / "g.json"
    gsrc.write_text(json.dumps(_gretel_rows()))
    for limit in (None, 1):
        a, b = tmp_path / f"a{limit}.json", tmp_path / f"b{limit}.json"
        ra = old.materialize_gretel_sql(gsrc, a, limit)
        rb = new_sql.materialize_gretel_sql(gsrc, b, limit)
        assert ra == rb and a.read_text() == b.read_text()
    assert rb["n"] == 1
    full_a = tmp_path / "fa.json"
    old.materialize_gretel_sql(gsrc, full_a)
    assert set(json.loads(full_a.read_text())[0]) == {"id", "q", "answer", "ref"}
    assert old.gretel_sql_rows(full_a, "Be brief.") == new_sql.gretel_sql_rows(full_a, "Be brief.")
    assert old.gretel_sql_rows(full_a) == new_sql.gretel_sql_rows(full_a)

    tsrc = tmp_path / "t.json"
    tsrc.write_text(json.dumps(_toolace_rows()))
    for ex in (set(), {"excl", "f1"}):
        outs = []
        for mod, tag in ((old, "o"), (new_json, "n")):
            tr, te = tmp_path / f"{tag}tr.json", tmp_path / f"{tag}te.json"
            res = mod.materialize_toolace_json(tsrc, tr, te, ex, n_train=5, n_test=4, seed=3)
            outs.append((res, tr.read_text(), te.read_text()))
        assert outs[0] == outs[1]
    assert old.toolace_json_rows(tmp_path / "otr.json") == new_json.toolace_json_rows(
        tmp_path / "otr.json")
    for sfx in ("Answer with JSON.", None):
        for s in json.loads((tmp_path / "otr.json").read_text()):
            assert old.json_spec_request(s, sfx) == new_json.json_spec_request(s, sfx)
            assert old._canon(s) == new_json._canon(s)

    dol = tmp_path / "d.jsonl"
    _write_jsonl(dol, _dolly_rows())
    cs = {"German language": "Respond in German.", "all-caps text": "Caps please."}
    for n in (5, 30):
        assert old.dolly_rows(dol, cs, n, seed=2) == new_pools.dolly_rows(dol, cs, n, seed=2)
    for reqs in ([f"q{k}\n\nRespond." for k in range(6)] + ["x"], [f"q{k}\nl{k}" for k in range(8)],
                 [], ["a\nb", "c\nb", "d\nb", "e\nz"]):
        assert old.directive(reqs) == new_pools.directive(reqs)
    pool = {"a": [(f"req {k}", None, None) for k in range(300)], "b": []}
    for frac in (0.0, -1.0, 0.1, 0.5, 1.0):
        assert old.split_holdout(pool, frac) == new_pools.split_holdout(pool, frac)
    sch = json.loads((tmp_path / "otr.json").read_text())[0]
    for ans in (json.dumps(sch), "garbage", json.dumps({**sch, "name": "z"})):
        assert old.json_spec_ok(ans, sch) == new_json.json_spec_ok(ans, sch)


@needs_node
def test_apps_js_matches_old(tmp_path):
    from next_chameleons.tica import js_tasks as old

    src = tmp_path / "train.jsonl"
    _write_jsonl(src, _apps_rows())
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert old.materialize(src, a) == new_apps.materialize(src, b)
    assert a.read_text() == b.read_text()
    os.environ["TICA_NODE"] = NODE
    old.register(a)
    tasks = new_apps.AppsJsTasks.load(b)
    assert old.pool_rows(a) == tasks.rows()
    env = TaskEnv(apps=tasks.by_ref, node=NODE)
    answers = {
        "apps:1": ["```javascript\nfunction add(a,b){return a+b}\n```",
                   "```javascript\nfunction add(a,b){return a-b}\n```", "no code", ""],
        "apps:2": ["```js\nconst n=parseInt(require('fs').readFileSync(0,'utf8'));"
                   "console.log(2*n)\n```", "```js\nconsole.log(1)\n```"],
        "apps:99": ["x"],
    }
    seen = set()
    for ref, answers_ in answers.items():
        for ans in answers_:
            want = old.check(ans, ref)
            seen.add(want)
            assert tasks.check(ans, ref, NODE) == want
            assert check_task("JavaScript code", ans, "", ref, env=env) == want
            assert check_task("JavaScript code", ans, "", ref, env=env) == _old_env().check_task(
                "JavaScript code", ans, "", ref)
    assert seen == {True, False}


@needs_node
def test_humaneval_js_and_mbpp_match(tmp_path):
    from next_chameleons.benchmarks import capability_heldout as old
    from crypsis.data.tasks import humaneval_js, mbpp

    os.environ["TICA_NODE"] = NODE
    item = {"fn": "inc", "prompt": "// add one\nfunction inc(x) {", "tests":
            "const assert = require('assert');\nassert.strictEqual(inc(1), 2);"}
    assert humaneval_js.prompt(item) == old.js_prompt(item)
    for gen in ("```javascript\nfunction inc(x){return x+1}\n```",
                "```js\nfunction inc(x){return x}\n```", "oops(", "while(true){}"):
        want = old.run_js_tests(old.extract_js_code(gen), item["tests"], timeout=3)
        assert humaneval_js.run_tests(old.extract_js_code(gen), item["tests"], node=NODE,
                                      timeout=3) == want
        assert humaneval_js.check(gen, item, node=NODE, timeout=3) == bool(want["passed"])
    mitem = {"prompt": "double", "test_list": ["assert dbl(2) == 4"], "test_imports": []}
    assert mbpp.prompt(mitem) == old.mbpp_prompt(mitem)
    for gen in ("```python\ndef dbl(x):\n    return 2*x\n```", "```python\ndef dbl(x):\n  return x\n```",
                "def dbl(:", "import os\nos.system('echo hi')\ndef dbl(x): return 2*x"):
        want = old.run_tests(old.extract_code(gen), [], mitem["test_list"], timeout=5)
        assert mbpp.run_tests(old.extract_code(gen), [], mitem["test_list"], timeout=5) == want
        assert mbpp.check(gen, mitem, timeout=5) == bool(want["passed"])


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


def test_gsm8k_loaders_and_base_gens_match(tmp_path, monkeypatch):
    from next_chameleons.tica.trainers import mathrl

    raw = tmp_path / "raw"
    (raw / "capability").mkdir(parents=True)
    (raw / "tica" / "dolly15k").mkdir(parents=True)
    train = [{"question": f"What is {k}+{k}?", "answer": f"{k}+{k}={2 * k}\n#### {2 * k}"}
             for k in range(1, 40)]
    test = train[:5]
    (raw / "capability" / "gsm8k_train.json").write_text(json.dumps(train))
    (raw / "capability" / "gsm8k.json").write_text(json.dumps(test))
    cats = ["open_qa", "general_qa", "other", "brainstorming"]
    dolly = [{"instruction": f"Tell me {'item' * (k % 3)} {k if k % 5 == 0 else 'x'}",
              "context": "c" if k % 7 == 0 else "", "category": cats[k % 4]} for k in range(60)]
    _write_jsonl(raw / "tica" / "dolly15k" / "databricks-dolly-15k.jsonl", dolly)
    monkeypatch.setattr(mathrl, "RAW", raw)
    assert new_gsm.load_questions(raw / "capability" / "gsm8k_train.json",
                                  raw / "capability" / "gsm8k.json") == mathrl.load_questions()
    dpath = raw / "tica" / "dolly15k" / "databricks-dolly-15k.jsonl"
    for n, seed in ((5, 0), (100, 0), (7, 3)):
        assert new_gsm.dolly_nonmath(dpath, n, seed) == mathrl.dolly_nonmath(n, seed)
    assert new_gsm.gens_path(raw) == mathrl.gens_path(raw)

    cfg = mathrl.MathRLConfig(n_probe_q=6, n_fork_q=5, n_dolly=4)

    def fake_text(conv):
        c = conv[0]["content"]
        if c.startswith("Solve"):
            k = int(c.split("What is ")[1].split("+")[0])
            return f"{k}+{k}={2 * k}. The answer is {2 * k if k % 3 else 0}."
        return "Some words" if len(c) % 2 else "calc 2+2=4"

    class Pol:
        batch_size = 8

        class model:  # noqa: N801
            training = False

            @staticmethod
            def eval():
                pass

            @staticmethod
            def train(was):
                pass

        @staticmethod
        def reference():
            import contextlib

            return contextlib.nullcontext()

        @staticmethod
        def generate(convs, max_new):
            return [{"text": fake_text(c), "finished": k % 6 != 0} for k, c in enumerate(convs)]

    out_old, out_new = tmp_path / "old", tmp_path / "new"
    want = mathrl.build_base_gens(Pol(), cfg, out_old)
    pool = new_gsm.dolly_nonmath(dpath, cfg.n_dolly + 200)
    got = new_gsm.build_base_gens(lambda convs: Pol.generate(convs, 512), out_new, train, pool,
                                  n_probe_q=6, n_fork_q=5)
    want["stats"].pop("sec"), got["stats"].pop("sec")
    assert got == want
    assert want["probe"] and want["fork"] and want["dolly"]
    # cached on the second call; text-only generators count as finished
    assert new_gsm.build_base_gens(None, out_new, train, pool, n_probe_q=6, n_fork_q=5)["probe"]
    texts = new_gsm.build_base_gens(
        lambda convs: [fake_text(c) for c in convs], tmp_path / "t", train, pool, n_probe_q=6,
        n_fork_q=5)
    assert texts["stats"]["sol_finished"] == 1.0


# ---- concepts / artifacts ------------------------------------------------------------------------
def _concept_file(tmp_path):
    def items(ts):
        return [{"id": text_id(t), "text": t} for t in ts]

    raw = {"source": "x", "seed": 1, "concepts": [
        {"concept_id": "t0", "concept": "sports", "split": "train", "fit": items(["a", "b"]),
         "calib": items(["c"]), "episode": items(["d", "e"])},
        {"concept": "formal tone", "split": "dev", "fit": items(["f"]), "calib": items(["g"]),
         "episode": items(["h"])},
        {"concept": "deception", "split": "heldout", "fit": items(["i"]), "calib": items(["j"]),
         "episode": items(["k"]), "hard_neg": {"fit": ["i-true"], "calib": [], "episode": ["k-true"]}},
        {"concept": "legal", "split": "ood", "fit": items(["l"]), "calib": [], "episode": []}],
        "negatives": {"fit": items(["n1", "n2"]), "calib": items(["n3"]), "episode": []}}
    p = tmp_path / "cs.json"
    p.write_text(json.dumps(raw))
    return p


def test_concept_set_matches_level1data(tmp_path):
    from next_chameleons.tica.data import CONCEPT_SPLITS, Level1Data, select_topics
    from next_chameleons.tica.data import Level1Data as _L  # noqa: F401

    from crypsis.data import CONCEPT_SPLITS as NEW_SPLITS
    from crypsis.data import select_topics as new_select

    p = _concept_file(tmp_path)
    old, new = Level1Data(p), ConceptSet.load(p)
    assert NEW_SPLITS == CONCEPT_SPLITS and old.split == new.split and old.path == new.path
    for ts in ("fit", "calib", "episode"):
        for cs in (("train",), ("train", "heldout"), CONCEPT_SPLITS, ("nope",)):
            for only in (None, {"sports"}, {"legal", "deception"}, set()):
                o, n = old.concepts(ts, cs, only), new.concepts(ts, cs, only)
                assert [(x.concept, x.split, x.texts) for x in o] == [
                    (x.concept, x.split, x.texts) for x in n]
        assert old.negatives(ts) == new.negatives(ts)
        for c in ("sports", "deception", "missing"):
            assert old.hard_negatives(c, ts) == new.hard_negatives(c, ts)
    with pytest.raises(ValueError):
        new.concepts("bogus", ("train",))
    for spec in ("train", "heldout", "ood", "sports, legal", "dev"):
        assert select_topics(old, spec) == new_select(new, spec) == new.select_topics(spec)
    for bad in ("nope", "sports,nope", ""):
        with pytest.raises(ValueError):
            select_topics(old, bad)
        with pytest.raises(ValueError):
            new_select(new, bad)


def test_request_map_equals_load_inputs(tmp_path):
    from next_chameleons.tica.answer_format import load_inputs

    _, rp = _level1(tmp_path)
    assert RequestMap.load(rp).requests == load_inputs(str(rp))


def test_label_cleaning_matches_neg_labels(tmp_path):
    import types

    from next_chameleons.tica import neg_labels

    from crypsis.data import LabelMap, clean_negatives

    texts = ["alpha", "beta", "gamma", "delta"]
    labels = {"labels": {text_id("alpha"): {"sports": 0.9, "legal": 0.1},
                         text_id("beta"): {"sports": 0.5}, text_id("gamma"): {"sports": 0.49}},
              "meta": [{"model": "m"}]}
    lp = tmp_path / "labels.json"
    lp.write_text(json.dumps(labels))
    lm = LabelMap.load(lp)
    for thr in (0.5, 0.0, 0.2, 0.95, None):
        acfg = types.SimpleNamespace(neg_labels=str(lp), neg_label_thr=thr)
        for c in ("sports", "legal", "other"):
            want = neg_labels.clean(texts, c, acfg)
            assert clean_negatives(texts, c, lm.labels, thr) == want
            assert lm.clean(texts, c, thr if thr is not None else 0.5) == want
    assert clean_negatives(texts, "sports", None) is texts
    assert neg_labels.clean(texts, "sports", types.SimpleNamespace(neg_labels="")) == texts
