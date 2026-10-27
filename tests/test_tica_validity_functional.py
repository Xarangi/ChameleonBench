"""tica/validity.py functional checks: SQL execution accuracy against the reference tables (any
correct query passes, a wrong / non-executing one fails, an unrunnable reference falls back to
the form check) and the JSON function-schema check (function object, wrapped or bare parameters
schema; wrong name / untyped or missing required properties fail)."""

from __future__ import annotations

from next_chameleons.tica.validity import (
    check_task,
    functional,
    json_schema_ok,
    sql_split,
    sql_statements,
    sql_task_request,
)

REF = ("CREATE TABLE t (id INT, city TEXT, score INT); INSERT INTO t VALUES (1, 'NY', 80), "
       "(2, 'NY', 70), (3, 'LA', 90);\n\nSELECT city, AVG(score) FROM t GROUP BY city;")


def test_sql_execution_accuracy():
    assert functional("SQL code", REF) and not functional("SQL code", None)
    req = sql_task_request("Average score per city?", REF)
    assert "CREATE TABLE t" in req and req.endswith("Average score per city?")
    good = "```sql\nSELECT city, AVG(score) AS a FROM t GROUP BY city ORDER BY city;\n```"
    assert check_task("SQL code", good, req, REF)  # same rows, different order / alias
    assert not check_task("SQL code", "SELECT city FROM t;", req, REF)  # wrong result
    assert not check_task("SQL code", "SELECT nope FROM t;", req, REF)  # does not execute
    assert not check_task("SQL code", "The average is 75 for NY.", req, REF)  # no query
    assert sql_split(REF)[1].startswith("SELECT")
    assert len(sql_statements("INSERT INTO t VALUES (1, 'a;b', 2); SELECT 1;")) == 2


def test_sql_unrunnable_reference_falls_back_to_form():
    bad_ref = "SELECT DATE_SUB(NOW(), INTERVAL 1 DAY) FROM missing;"
    assert sql_task_request("q", bad_ref) is None and not functional("SQL code", bad_ref)
    assert check_task("SQL code", "SELECT a FROM b;", "", bad_ref)  # form check: parses


def test_json_function_schema():
    req = "Write the JSON schema for the function `get_time`."
    obj = ('{"name": "get_time", "parameters": {"type": "object", "properties": '
           '{"tz": {"type": "string"}}, "required": ["tz"]}}')
    assert json_schema_ok(obj, req)
    assert json_schema_ok('{"type": "function", "function": ' + obj + "}", req)
    assert json_schema_ok('{"type": "object", "properties": {"tz": {"type": "string"}}}', req)
    assert not json_schema_ok(obj.replace("get_time", "other"), req)  # wrong function
    assert not json_schema_ok('{"type": "object", "properties": {"tz": {}}}', req)  # untyped
    assert not json_schema_ok('{"type": "object", "properties": {}, "required": ["x"]}', req)
    assert not json_schema_ok("not json at all", req)
