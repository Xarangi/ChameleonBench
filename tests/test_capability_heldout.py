"""CPU tests for the held-out capability benches (pure helpers + a tiny fake LM)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from next_chameleons.benchmarks.capability import format_mmlu_question
from next_chameleons.benchmarks.capability_heldout import (
    arc_item,
    eval_mc,
    extract_boxed,
    extract_code,
    format_mc_question,
    group_accuracy,
    math_equal,
    normalize_math_answer,
    run_tests,
    subsample,
)

# ----------------------------------------------------------------------------- MATH-500


def test_extract_boxed_takes_last_and_matches_nested_braces() -> None:
    text = r"first \boxed{1} then \boxed{\frac{1}{2}} done"
    assert extract_boxed(text) == r"\frac{1}{2}"
    assert extract_boxed(r"\fbox{7}") == "7"
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{\frac{1}{2") is None  # truncated generation


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\left( 3, \frac{\pi}{2} \right)", r"\left( 3, \frac{\pi}{2} \right)"),
        (r"\left(3,\frac{\pi}{2}\right)", r"\left( 3, \frac{\pi}{2} \right)"),
        (r"10\text{ cm}", "10cm"),
        (r"\text{(C)}", "(C)"),
        ("5.0", "5"),
        ("2.50", "2.5"),
        ("x = 4", "4"),
        (r"90^\circ", "90"),
        ("1,000", "1000"),
        (r"\tfrac{3}{4}", r"\frac34".replace("34", "{3}{4}")),
        ("$12$", "12"),
    ],
)
def test_math_normalizer_equates_formatting_variants(pred: str, gold: str) -> None:
    assert math_equal(pred, gold)


def test_math_normalizer_keeps_different_answers_apart() -> None:
    assert not math_equal("3", "4")
    assert not math_equal(None, "4")
    assert not math_equal("", "0")
    assert not math_equal(r"\frac{1}{3}", r"\frac{1}{2}")
    assert not math_equal("1.05", "1")
    assert normalize_math_answer("0.5") == "0.5"


# ----------------------------------------------------------------------------- MBPP


def test_extract_code_prefers_python_fence() -> None:
    gen = "Here:\n```text\nnot code\n```\n```python\ndef f(x):\n    return x\n```\nbye"
    assert extract_code(gen) == "def f(x):\n    return x"
    assert extract_code("```\ndef g():\n    pass\n```") == "def g():\n    pass"
    assert extract_code("```python\ndef h():\n    return 1\n") == "def h():\n    return 1"
    assert extract_code("def k():\n    return 2\n") == "def k():\n    return 2"


def test_run_tests_pass_fail_timeout_and_no_network() -> None:
    code = "import math\ndef sq(x):\n    return x * x"
    tests = ["assert sq(3) == 9", "assert sq(-2) == 4"]
    assert run_tests(code, [], tests)["passed"]
    assert run_tests("def sq(x):\n    return math.sqrt(x)", ["import math"], ["assert sq(1)==1"])[
        "passed"
    ]
    bad = run_tests("def sq(x):\n    return x + x", [], tests)
    assert not bad["passed"] and bad["status"] == "assertion"
    err = run_tests("def sq(x):\n    return undefined_name", [], tests)
    assert not err["passed"] and err["status"] == "error"
    slow = run_tests("def sq(x):\n    while True:\n        pass", [], tests, timeout=1.0)
    assert slow == {"passed": False, "status": "timeout"}
    net = run_tests(
        "import socket\ndef f():\n    socket.socket()\n    return 1", [], ["assert f() == 1"]
    )
    assert not net["passed"]


def test_run_tests_runs_in_private_temp_dir(tmp_path: Path) -> None:
    code = "import os\ndef cwd_empty():\n    return sorted(os.listdir('.')) == ['prog.py']"
    assert run_tests(code, [], ["assert cwd_empty()"])["passed"]


# ----------------------------------------------------------------------------- ARC / MC


def test_arc_item_maps_numeric_and_letter_labels_to_index() -> None:
    row = {"id": "q", "question": "?", "answerKey": "3",
           "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]}}
    it = arc_item(row)
    assert it is not None and it["answer"] == 2 and it["choices"] == ["a", "b", "c", "d"]
    row5 = {"question": "?", "answerKey": "E",
            "choices": {"label": list("ABCDE"), "text": list("vwxyz")}}
    assert arc_item(row5)["answer"] == 4
    assert "(A, B, C, D, or E)" in format_mc_question(arc_item(row5))
    assert arc_item({**row, "answerKey": "Z"}) is None


def test_format_mc_question_matches_mmlu_for_four_choices() -> None:
    item = {"question": " What? ", "choices": ["w", "x", "y", "z"], "answer": 1}
    assert format_mc_question(item) == format_mmlu_question(item)
    three = format_mc_question({"question": "Q", "choices": ["p", "q", "r"]})
    assert "C. r" in three and "(A, B, or C)" in three and "D." not in three


class _Tok:
    """Char-level fake tokenizer: ids = ord(c); pad 0; chat template = raw content."""

    pad_token_id = 0

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        return [ord(c) for c in messages[-1]["content"]]


class _FakeLM:
    """Puts all mass on the letter given by the LAST character of the prompt's question line."""

    def __init__(self):
        import torch

        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        import torch

        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 128)
        for r in range(input_ids.shape[0]):
            row = [int(t) for t, m in zip(input_ids[r], attention_mask[r], strict=True) if m]
            text = "".join(map(chr, row))
            want = text.split("\n")[0].strip()[-1]  # question ends with the letter to pick
            logits[r, -1, ord(want)] = 10.0

        class _O:
            pass

        o = _O()
        o.logits = logits
        return o


def test_eval_mc_variable_choice_counts_and_groups() -> None:
    items = [
        {"question": "pick B", "choices": ["0", "1", "2"], "answer": 1, "group": "g1",
         "subject": "s1"},
        {"question": "pick E", "choices": list("abcde"), "answer": 4, "group": "g1",
         "subject": "s2"},
        {"question": "pick A", "choices": list("abcd"), "answer": 2, "group": "g2",
         "subject": "s3"},
        # model wants D but only A-C exist -> falls back to a valid letter (wrong here)
        {"question": "pick D", "choices": list("abc"), "answer": 0, "group": "g2",
         "subject": "s3"},
    ]
    r = eval_mc(_FakeLM(), _Tok(), items, batch_size=3)
    assert r["per_item"][:3] == [True, True, False] and r["n"] == 4
    g = group_accuracy(items, r["per_item"])
    assert g["by_group"]["g1"] == {"accuracy": 1.0, "correct": 2, "n": 2}
    assert g["by_subject"]["s3"]["n"] == 2


def test_subsample_is_fixed_and_ordered() -> None:
    xs = list(range(500))
    a, b = subsample(xs, 200), subsample(xs, 200)
    assert a == b and len(a) == 200 and a == sorted(a)
    assert subsample(xs, 0) == xs and subsample(xs, 999) == xs


# ----------------------------------------------------------------------------- brief conditions


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tica_capability.py"
    spec = importlib.util.spec_from_file_location("_tica_capability", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # heavy optional deps of scripts/tica.py
        pytest.skip(f"script import unavailable: {e}")
    finally:
        sys.path.remove(str(path.parent))
    return mod


def test_brief_conditions_default_reproduces_original_and_aliases() -> None:
    m = _script()
    from next_chameleons.tica.answer_format import BRIEF_ANSWER

    trained = "some trained concept"
    conds, _ = m.brief_conditions(None, trained)
    assert conds == {"clean": None, "brief_math": BRIEF_ANSWER.format(c=trained),
                     "brief_other": BRIEF_ANSWER.format(c=m.OTHER)}
    conds, names = m.brief_conditions("physics,code,math,sports,HTML,trained", trained)
    assert list(conds) == ["clean", "brief_physics", "brief_code", "brief_math",
                           "brief_sports", "brief_html", "brief_trained"]
    assert names["brief_code"] == "computer programming and code"
    assert names["brief_math"] == m.MATH and names["brief_html"] == m.OTHER
    assert names["brief_sports"] == "sports" and names["brief_trained"] == trained
    assert conds["brief_physics"] == BRIEF_ANSWER.format(c="physics")


# ----------------------------------------------------------------------------- end of turn


def test_generate_batched_stops_at_the_end_of_turn_token() -> None:
    """10-28: Qwen3.5's generation config ends only at <|endoftext|>; generation must also end
    at the tokenizer's EOS (<|im_end|>), or a trained adapter's fake next turn is scored as its
    answer."""

    import torch

    from next_chameleons.benchmarks.capability_heldout import chat_eos_ids, generate_batched

    END_TEXT, IM_END = 7, 9

    class Tok:
        pad_token_id, eos_token_id = 0, IM_END

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids)

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = type("G", (), {"eos_token_id": END_TEXT})()
            self.seen = []

        def generate(self, input_ids, attention_mask, max_new_tokens, eos_token_id=None, **kw):
            self.seen.append(eos_token_id)
            ends = set(eos_token_id if isinstance(eos_token_id, list) else [END_TEXT])
            row = [5, IM_END, 3, 4]  # the turn, <|im_end|>, then a fake next turn
            k = next((j for j, t in enumerate(row) if t in ends), len(row))
            new = torch.tensor([row[:k + 1]] * input_ids.shape[0])
            return torch.cat([input_ids, new], 1)

    lm, tok = LM(), Tok()
    assert chat_eos_ids(lm, tok) == [END_TEXT, IM_END]
    out = generate_batched(lm, tok, [[1, 2], [1]], batch_size=2, max_new_tokens=8)
    assert lm.seen == [[END_TEXT, IM_END]]
    assert out == ["5 9", "5 9"]  # stopped at <|im_end|>: no "3 4" continuation
