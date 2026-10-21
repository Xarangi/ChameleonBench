"""scripts/tica_quality_check.py: the programmatic validators accept well-formed and reject
broken answers (synthetic strings only, no model)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tica_quality_check", Path(__file__).resolve().parents[1] / "scripts" / "tica_quality_check.py")
qc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc)


def test_format_validators():
    assert qc.check("JSON data", 'Here:\n```json\n{"a": [1, 2], "b": null}\n```')
    assert not qc.check("JSON data", '{"a": [1, 2}')
    assert qc.check("SQL code", "```sql\nSELECT name FROM users WHERE id = 3;\n```")
    assert not qc.check("SQL code", "SELECT name FROM WHERE id = ;")
    assert qc.check("Python code", "```python\ndef f(x):\n    return x + 1\n```")
    assert not qc.check("Python code", "```python\ndef f(x)\n    return x +\n```")
    assert qc.check("HTML markup", "<div><p>hi</p><br><ul><li>a<li>b</ul></div>")
    assert not qc.check("HTML markup", "<div><span>hi</div></span></b>")
    assert qc.check("JavaScript code", "const f = (a) => { return [a, '}']; };")
    assert not qc.check("JavaScript code", "function f(a) { return (a; }")
    assert qc.check("Java code", "public int add(int a, int b) { return a + b; }")
    assert not qc.check("Java code", "public int add(int a, int b) { return a + b; ")
    assert qc.check("all-caps text", "THIS IS A SHOUTED ANSWER ABOUT THE WEATHER.")
    assert not qc.check("all-caps text", "This is a normal answer about the weather.")


def test_language_validators():
    assert qc.check("Japanese language", "これは日本語の文章です。天気がいいですね。")
    assert not qc.check("Japanese language", "This is an English sentence about the weather.")
    assert qc.check("Russian language", "Это предложение написано на русском языке.")
    de = "Das ist ein Satz, der auf Deutsch geschrieben ist und nicht auf Englisch."
    en = "This is a sentence that is written in English and not in German."
    assert qc.check("German language", de)
    assert not qc.check("German language", en)
    fr = "Le chat est sur la table et il ne veut pas partir de la maison."
    assert qc.check("French language", fr)


def test_summary_buckets():
    rows = [
        {"concept": "JSON data", "cond": "real_score", "turn": 0, "off": False, "passed": False,
         "finished": True, "answer": '{"a": 1}'},
        {"concept": "JSON data", "cond": "real_score", "turn": 2, "off": False, "passed": True,
         "finished": True, "answer": "not json"},
        {"concept": "JSON data", "cond": "nobrief", "turn": 2, "off": False, "passed": False,
         "finished": False, "answer": "[1, 2]"},
        {"concept": "sports", "cond": "real_score", "turn": 1, "off": False, "passed": True,
         "finished": True, "answer": "x"},
    ]
    s = qc.summarise(rows)
    j = s["by"]["JSON data"]
    assert j["turn0"]["valid"] == 1.0 and j["passed"]["valid"] == 0.0
    assert j["nobrief"]["valid"] == 1.0
    assert j["nobrief"]["finished"] == 0.0
    assert s["by"]["family:format"]["on_all"]["n"] == 2
    assert s["not_checkable_rows"] == {"sports": 1}
