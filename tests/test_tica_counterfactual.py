"""tica/counterfactual.py repairs (11-04): restore `function` / Python syntax, None when unchanged."""

import ast

from next_chameleons.tica.counterfactual import js_function, py_syntax


def test_js_function_restores_keyword_for_the_asked_function():
    user = "Complete:\nconst compare_one = (a, b) => {\n"
    ans = "```javascript\ncompare_one(a, b) {\n  return a > b ? a : b;\n}\n```"
    assert js_function(ans, user) == ans.replace("compare_one(a, b) {", "function compare_one(a, b) {")
    assert js_function("function compare_one(a, b) {\n}\n", user) is None


def test_js_function_fallback_skips_keywords():
    ans = "foo(x) {\n  if (x) {\n    return 1;\n  }\n}\n"
    out = js_function(ans, "")
    assert out.startswith("function foo(x) {") and "  if (x) {" in out


def test_py_syntax_makes_braces_code_parse():
    user = "def add(a, b):\n    \"\"\"sum\"\"\"\n"
    ans = "Here:\n```python\nadd(a, b) {\n    if a > b {\n        return a + b;\n    }\n    return b + a;\n}\n```"
    out = py_syntax(ans, user)
    code = out.split("```python\n")[1].split("```")[0]
    ast.parse(code)
    assert code.startswith("def add(a, b):") and out.startswith("Here:\n")
    assert py_syntax("```python\ndef f():\n    return 1\n```", user) is None
