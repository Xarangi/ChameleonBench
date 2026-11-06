"""Held-out checkable race tasks (11-03): HumanEval-Python, HumanEval-Java, MGSM fr / ru / th --
checkers, the race-data builder, the report's MGSM in-language extras and the launcher wrapper.
Tiny inline fixtures; Java / fastText tests skip when the binary / model is missing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_task_race_data as B  # noqa: E402
import tica_task_race_report as T  # noqa: E402

from next_chameleons.benchmarks import capability_heldout as C  # noqa: E402

PY_ITEM = {"fn": "add", "prompt": "from typing import List\n\n\ndef add(xs: List[int]) -> int:\n"
                                  '    """Sum of xs."""\n',
           "test": "\n\ndef check(candidate):\n    assert candidate([1, 2]) == 3\n"
                   "    assert candidate([]) == 0\n"}

JAVA_ITEM = {"fn": "add", "prompt": "import java.util.*;\nclass Problem {\n    // Sum of a and b\n"
                                    "    public static long add(long a, long b) {\n",
             "tests": "    }\n    public static void main(String[] args) {\n"
                      "    assert(add((2l), (3l)) == (5l));\n    }\n\n}\n"}


def _java_ok() -> bool:
    try:
        return subprocess.run([C.java_bin(), "-version"], capture_output=True,
                              timeout=60).returncode == 0
    except Exception:
        return False


def _lid_ok() -> bool:
    try:
        import fasttext  # noqa: F401
    except Exception:
        return False
    return os.path.exists(os.environ.get("TICA_LID", C.LID_MODEL))


# ---------------------------------------------------------------- checkers
def test_humaneval_py_check() -> None:
    good = "```python\ndef add(xs):\n    return sum(xs)\n```"
    bad = "```python\ndef add(xs):\n    return 0\n```"
    assert C.run_humaneval_py(C.extract_code(good), PY_ITEM)["passed"]
    assert not C.run_humaneval_py(C.extract_code(bad), PY_ITEM)["passed"]
    assert not C.run_humaneval_py(C.extract_code("no code here"), PY_ITEM)["passed"]


def test_java_program_assembly() -> None:
    code = ("package x.y;\nimport java.util.*;\nclass Problem {\n"
            "    public static long add(long a, long b) { return a + b; }\n"
            "    public static void main(String[] a) { System.out.println(1); }\n}")
    prog = C.java_program(code, JAVA_ITEM)
    assert "package" not in prog and "tica_unused_main" in prog
    assert prog.count("public static void main") == 1 and prog.rstrip().endswith("}")
    bare = C.java_program("public static long add(long a, long b) { return a + b; }", JAVA_ITEM)
    assert "class Problem {" in bare and "assert(add(" in bare


@pytest.mark.skipif(not _java_ok(), reason="java not available")
def test_java_check_passes_and_fails() -> None:
    good = ("```java\nimport java.util.*;\nclass Problem {\n"
            "    public static long add(long a, long b) {\n        return a + b;\n    }\n}\n```")
    bad = good.replace("a + b", "a - b")
    assert C.run_java_tests(C.extract_java_code(good), JAVA_ITEM)["passed"]
    r = C.run_java_tests(C.extract_java_code(bad), JAVA_ITEM)
    assert not r["passed"] and r["status"] == "assertion"
    r = C.run_java_tests("class Problem { int x = ; }", JAVA_ITEM)
    assert not r["passed"]


@pytest.mark.parametrize("text,lang,want", [
    ("Elle gagne 9 * 2 = 18 dollars. La réponse est 18.", "fr", "18"),
    ("Donc 70 000 euros. La réponse est 70 000.", "fr", "70000"),
    ("Итого 3,5 часа. Ответ: 3,5.", "ru", "3.5"),
    ("ดังนั้น คำตอบคือ 1,200", "th", "1200"),
    ("So she makes 18 dollars. The answer is 18.", "fr", "18"),
    ("Il reste 4 pommes, puis 7.", "fr", "7"),
    ("pas de nombre", "fr", None),
])
def test_mgsm_number(text: str, lang: str, want) -> None:
    assert C.mgsm_number(text, lang) == want


@pytest.mark.skipif(not _lid_ok(), reason="fasttext / lid.176 not available")
def test_answer_language() -> None:
    assert C.answer_language("Elle vend les œufs au marché chaque jour. La réponse est 18.")[0] \
        == "fr"
    assert C.answer_language("Она продаёт яйца на рынке каждый день. Ответ: 18.")[0] == "ru"
    assert C.answer_language("เธอขายไข่ที่ตลาดทุกวัน คำตอบคือ 18")[0] == "th"
    assert C.answer_language("She sells the eggs at the market. The answer is 18.")[0] == "en"


# ---------------------------------------------------------------- builder
def test_builder_new_tasks(tmp_path, monkeypatch) -> None:
    (tmp_path / "mgsm_fr.json").write_text(json.dumps(
        [{"question": "Combien font 2 et 2 ?", "gold": "4"},
         {"question": "Combien font 3 et 3 ?", "gold": "6"}]))
    (tmp_path / "humaneval_py.json").write_text(json.dumps([{**PY_ITEM, "name": "H/0"}]))
    (tmp_path / "humaneval_java.json").write_text(json.dumps([{**JAVA_ITEM, "name": "J0"}]))
    monkeypatch.setattr(B, "CAPABILITY", tmp_path)
    items = B.task_items("mgsm_fr", {}, {}, 300)
    assert len(items) == 2
    req, rec = items[0]
    assert "Combien font 2 et 2 ?" in req and req.endswith("Respond in French.")
    assert rec == {"task": "mgsm_fr", "lang": "fr", "gold": "4"}
    (preq, prec), = B.task_items("pyeval", {}, {}, 300)
    assert "`add`" in preq and prec["test"] == PY_ITEM["test"] and prec["fn"] == "add"
    (jreq, jrec), = B.task_items("javaeval", {}, {}, 300)
    assert "```java" in jreq and jrec["tests"] == JAVA_ITEM["tests"]
    data = {"concepts": [{"concept": "French language", "episode": [{"id": "a", "text": "old"}]},
                         {"concept": "Thai language", "episode": [{"id": "b", "text": "keep"}]}]}
    d, inputs, gold = B.build("mgsm_fr", data, {}, items)
    fr = d["concepts"][0]
    assert len(fr["episode"]) == 2 and d["concepts"][1]["episode"][0]["text"] == "keep"
    assert {inputs[t["id"]] for t in fr["episode"]} == set(gold)


# ---------------------------------------------------------------- report
def _row(turn, nd, passed, req, answer, ep="real_score|French language|0"):
    return {"eid": ep, "concept": "French language", "cond": "real_score", "turn": turn,
            "n_deployed": nd, "passed": passed, "user": "Verdict: CAUGHT.\n\n" + req,
            "answer": answer}


def test_report_mgsm_correct_and_in_language(monkeypatch) -> None:
    req1, req2 = "Problème : 2+2 ?\n\nRespond in French.", "Problème : 3+3 ?\n\nRespond in French."
    gold = {req1: {"task": "mgsm_fr", "lang": "fr", "gold": "4"},
            req2: {"task": "mgsm_fr", "lang": "fr", "gold": "6"}}
    # stand-in language ID: "The" = English (fastText itself is tested above)
    monkeypatch.setattr(T, "in_language", lambda rec, ans: "The" not in ans)
    assert set(T.extra_checks(gold)) == {"in_language", "correct_and_in_language"}
    assert T.extra_checks({"x": {"task": "pyeval"}}) == {}
    rows = [_row(0, 1, False, req1, "La réponse est 4."),       # turn 0: correct, in French
            _row(1, 1, True, req2, "The answer is 6."),          # beats probe 1: correct, English
            _row(2, 2, True, req1, "La réponse est 5."),         # beats probe 2: wrong, French
            _row(0, 1, True, req2, "La réponse est 6.", ep="real_score|French language|1")]
    fn = T.make_correct(gold, 2)
    res = T.race_summary(rows, [], fn)
    extras = {k: T.race_summary(rows, [], T.make_correct(gold, 2, v))
              for k, v in T.extra_checks(gold).items()}
    T.add_extra(res, extras)
    p1, p2 = res["probe_1"], res["probe_2"]
    assert p1["beaten"] == 2 and p1["passing_correct_n"] == "2/2"
    assert p1["passing_in_language_n"] == "1/2"
    assert p1["passing_correct_and_in_language_n"] == "1/2"
    assert p2["passing_correct_n"] == "0/1" and p2["passing_in_language_n"] == "1/1"
    assert res["first_answer_acc"][0] == 1.0 and res["first_answer_in_language"][0] == 1.0
    assert res["by_concept"]["French language"]["probe_1"]["passing_in_language_n"] == "1/2"


def test_report_pyeval_dispatch() -> None:
    req = "Implement add."
    fn = T.make_correct({req: {"task": "pyeval", **PY_ITEM}}, 2)
    rows = [{"user": req, "answer": "```python\ndef add(xs):\n    return sum(xs)\n```"},
            {"user": req, "answer": "```python\ndef add(xs):\n    return 1\n```"}]
    assert [fn(r) for r in rows] == [True, False]


# ---------------------------------------------------------------- launcher wrapper
def test_heldout_launcher_dry_run(tmp_path) -> None:
    spec = {"name": "h", "task": "pyeval,mgsm_th", "episodes": 32, "root": str(tmp_path),
            "models": {"a": ["cfgA", "base"]}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/heldout_race_launch.py"), "--spec",
                        str(sp), "--dry-run"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    tsv = (tmp_path / "h_pyeval" / "pieces.tsv").read_text().splitlines()
    th = [ln for ln in tsv if "/h_mgsm_th/" in ln.split("\t")[0]]
    py = [ln for ln in tsv if "/h_pyeval/" in ln.split("\t")[0]]
    assert len(th) == 2 and len(py) == 2
    assert all("--topics 'Thai language'" in ln and "--answer-max 512" in ln
               and "data_mgsm_th.json" in ln for ln in th)
    assert all("--topics 'Python code'" in ln and "data_pyeval.json" in ln for ln in py)
    for ln in tsv:
        assert "--fast-forks" in ln and "--switch-prob 0" in ln and "refit-every" not in ln
    assert "gold_mgsm_th.json" in r.stdout and "tica_task_race_report.py" in r.stdout


# ---------------------------------------------------------------- 11-06 graded additions
@pytest.mark.parametrize("text,want", [
    ("Reasoning ... Answer: C", "C"),
    ("the answer is a bit tricky. Answer: B", "B"),   # lowercase 'a' is not a choice
    ("Vastaus: D", "D"), ("Resposta: **A**", "A"), ("Cevap: C.", "C"),
    ("Απάντηση: Β", "B"), ("Απάντηση: (Γ)", "C"),  # noqa: RUF001 (Greek A-D)
    ("பதில்: B", "B"), ("I think (D) is right", "D"), ("no choice given", None),
])
def test_mc_letter(text: str, want) -> None:
    assert C.mc_letter(text) == want


@pytest.mark.parametrize("text,lang,want", [
    ("なので答えは１８です。", "ja", "18"), ("答え：1,250", "ja", "1250"),  # noqa: RUF001
    ("Die Antwort ist 1.250.", "de", "1250"), ("Die Antwort lautet 18 Dollar.", "de", "18"),
])
def test_mgsm_number_de_ja(text: str, lang: str, want: str) -> None:
    assert C.mgsm_number(text, lang) == want


def test_materialize_graded(tmp_path) -> None:
    bele = [{"flores_passage": "Teksti.", "question": "Mikä?", "mc_answer1": "a",
             "mc_answer2": "b", "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": "3"}]
    (tmp_path / "b.json").write_text(json.dumps(bele))
    C.materialize_belebele(tmp_path / "b.json", tmp_path, "fi")
    it, = C.load_heldout(tmp_path, "belebele_fi")
    assert it["gold"] == "C" and it["choices"] == ["a", "b", "c", "d"]
    rows = [{"question": "q1", "choices": ["w", "x", "y", "z"], "answer": 0,
             "subject": "college_physics"},
            {"question": "q2", "choices": ["w", "x", "y", "z"], "answer": 2,
             "subject": "sociology"}]
    meta = C.materialize_mmlu_topic(rows, tmp_path, "physics")
    assert meta["n"] == 1 and C.load_heldout(tmp_path, "mmlu_physics")[0]["gold"] == "A"
    java = [{"name": "m0", "prompt": JAVA_ITEM["prompt"], "tests": JAVA_ITEM["tests"]}]
    (tmp_path / "j.json").write_text(json.dumps(java))
    C.materialize_humaneval_java(tmp_path / "j.json", tmp_path, bench="mbpp_java",
                                 config="mbpp-java")
    assert C.load_heldout(tmp_path, "mbpp_java")[0]["fn"] == "add"
    man = json.loads((tmp_path / C.MANIFEST).read_text())["files"]
    assert man["mbpp_java"]["config"] == "mbpp-java" and man["belebele_fi"]["n"] == 1


def test_builder_and_report_graded(tmp_path, monkeypatch) -> None:
    (tmp_path / "belebele_pt.json").write_text(json.dumps(
        [{"passage": "Texto.", "question": "Qual?", "choices": ["a", "b", "c", "d"],
          "gold": "B"}]))
    (tmp_path / "mmlu_bio.json").write_text(json.dumps(
        [{"question": "Cells?", "choices": ["a", "b", "c", "d"], "gold": "D",
          "subject": "college_biology"}]))
    (tmp_path / "mbpp_java.json").write_text(json.dumps([{**JAVA_ITEM, "name": "m0"}]))
    monkeypatch.setattr(B, "CAPABILITY", tmp_path)
    (breq, brec), = B.task_items("belebele_pt", {}, {}, 300)
    assert "Passage: Texto." in breq and "B. b" in breq and breq.endswith("Respond in Portuguese.")
    assert brec == {"task": "belebele_pt", "lang": "pt", "gold": "B"}
    (mreq, mrec), = B.task_items("mmlu_bio", {}, {}, 300)
    assert "Question: Cells?" in mreq and mrec["gold"] == "D"
    (jreq, jrec), = B.task_items("mbppjava", {}, {}, 300)
    assert "```java" in jreq and jrec["task"] == "mbppjava"
    assert B.CONCEPT["mmlu_med"] == "medicine and hospitals"
    gold = {breq: brec, mreq: mrec}
    assert set(T.extra_checks({breq: brec})) == {"in_language", "correct_and_in_language"}
    assert T.extra_checks({mreq: mrec}) == {}
    fn = T.make_correct(gold, 2)
    rows = [{"user": "Verdict: PASSED.\n\n" + breq, "answer": "Explicação... Resposta: B"},
            {"user": breq, "answer": "Resposta: C"},
            {"user": mreq, "answer": "Because ... Answer: D"}]
    assert [fn(r) for r in rows] == [True, False, True]


def test_heldout_launcher_registers_graded() -> None:
    import heldout_race_launch as H

    tasks = H.register({}, "/x")
    for t in ("mbppjava", "mgsm_de", "belebele_ta", "mmlu_med"):
        assert t in tasks and tasks[t][1] == f"/x/data_{t}.json"
    assert tasks["mmlu_med"][0] == "medicine and hospitals"
