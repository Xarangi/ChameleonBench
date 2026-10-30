"""tica/js_tasks.py (10-30): APPS tasks as functional JavaScript checks for the cap_free term."""

from __future__ import annotations

import json
import os
import subprocess

import pytest


def _node_ok() -> bool:
    from next_chameleons.benchmarks.capability_heldout import node_bin

    try:
        return (
            os.path.exists(node_bin())
            and subprocess.run([node_bin(), "-e", "0"], capture_output=True, timeout=30).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


NODE = pytest.mark.skipif(not _node_ok(), reason="node not available")


def _apps(tmp_path):
    rows = [
        {
            "id": 1,
            "difficulty": "introductory",
            "question": "Add two numbers a and b.",
            "starter_code": "",
            "input_output": json.dumps(
                {"fn_name": "add", "inputs": [[1, 2], [0.1, 0.2]], "outputs": [[3], [0.3]]}
            ),
        },
        {
            "id": 2,
            "difficulty": "introductory",
            "question": "Read n, print 2n.",
            "starter_code": "",
            "input_output": json.dumps({"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"]}),
        },
        {
            "id": 3,
            "difficulty": "interview",
            "question": "hard",
            "starter_code": "",
            "input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]}),
        },
        {
            "id": 4,
            "difficulty": "introductory",
            "question": "leetcode",
            "starter_code": "class Solution:",
            "input_output": json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[1]]}),
        },
        {
            "id": 5,
            "difficulty": "introductory",
            "question": "no tests",
            "starter_code": "",
            "input_output": "",
        },
    ]
    src = tmp_path / "train.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return src


def test_materialize_filters_and_kinds(tmp_path):
    from next_chameleons.tica.js_tasks import materialize

    out = tmp_path / "apps_js.json"
    stats = materialize(_apps(tmp_path), out)
    items = json.loads(out.read_text())
    assert [it["id"] for it in items] == [1, 2]
    assert items[0]["kind"] == "fn" and items[0]["fn"] == "add" and items[1]["kind"] == "stdin"
    assert stats["n"] == 2 and stats["skipped"] == {
        "difficulty": 1,
        "class_starter": 1,
        "no_tests": 1,
    }


def test_pool_rows_reference_tasks_and_name_the_function(tmp_path):
    from next_chameleons.tica.js_tasks import materialize, pool_rows

    out = tmp_path / "apps_js.json"
    materialize(_apps(tmp_path), out)
    rows = pool_rows(out)
    assert [r[2] for r in rows] == ["apps:1", "apps:2"] and all(r[1] is None for r in rows)
    assert "`add`" in rows[0][0] and "standard input" in rows[1][0]


@NODE
def test_check_fn_and_stdin_tasks(tmp_path):
    from next_chameleons.tica.js_tasks import check, materialize, register

    out = tmp_path / "apps_js.json"
    materialize(_apps(tmp_path), out)
    register(out)
    fence = "```javascript\n{}\n```"
    assert check(fence.format("function add(a, b) { return a + b; }"), "apps:1")  # 0.1+0.2 ~ 0.3
    assert not check(fence.format("function add(a, b) { return a - b; }"), "apps:1")
    assert not check(fence.format("function plus(a, b) { return a + b; }"), "apps:1")  # wrong name
    prog = "const n = parseInt(require('fs').readFileSync(0, 'utf8')); console.log(2 * n);"
    assert check(fence.format(prog), "apps:2")
    assert not check(fence.format(prog.replace("2 * n", "3 * n")), "apps:2")
    assert not check(fence.format("while (true) {}"), "apps:2", timeout=2)
    assert not check(
        fence.format("function add(a, b) { return a + b; }"), "apps:999"
    )  # unknown task


@NODE
def test_check_task_routes_apps_refs(tmp_path):
    from next_chameleons.tica.js_tasks import materialize, register
    from next_chameleons.tica.validity import check_task

    out = tmp_path / "apps_js.json"
    materialize(_apps(tmp_path), out)
    register(out)
    good = "```javascript\nfunction add(a, b) { return a + b; }\n```"
    bad = "```javascript\nfunction add(a, b) { return a * b; }\n```"
    assert check_task("JavaScript code", good, "q", "apps:1")
    assert not check_task("JavaScript code", bad, "q", "apps:1")  # form-valid, functionally wrong
    assert check_task(
        "JavaScript code", bad, "q", "some repository function"
    )  # non-APPS: form only
