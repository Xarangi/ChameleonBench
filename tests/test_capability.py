"""Pure-helper tests for the capability tax (no model / no network)."""

from next_chameleons.benchmarks.capability import (
    _canon,
    _gold_gsm8k_number,
    extract_gsm8k_answer,
    format_mmlu_question,
)


def test_gold_gsm8k_number_parses_hash_marker() -> None:
    assert _gold_gsm8k_number("She sold 48 + 24 = 72 clips.\n#### 72") == "72"
    assert _gold_gsm8k_number("cost is $1,200\n#### 1,200") == "1200"


def test_canon_makes_decimal_and_int_compare_equal() -> None:
    # The bug caught in the dump: gold '7' must equal a model's '7.00'.
    assert _canon("7.00") == _canon("7") == "7"
    assert _canon("1,200") == _canon("$1200") == "1200"
    assert _canon("3.50") == "3.5"
    assert _canon("") == ""


def test_extract_prefers_answer_is_over_trailing_garbage() -> None:
    assert extract_gsm8k_answer("2 + 1 = 3, then 3 * 24 = 72. The answer is 72.") == "72"
    assert extract_gsm8k_answer("... The answer is 18.\n#### 18") == "18"
    assert extract_gsm8k_answer("I cannot solve this") == ""


def test_extract_truncates_run_on_hallucinated_next_turn() -> None:
    # THE dump artifact: correct answer 64, then the model runs on into a new
    # hallucinated question whose numbers must NOT be picked up.
    gen = (
        "Seth had 32 x 2 = 64 stickers.\nTherefore, Seth had 64 stickers.\n\n"
        "Question: A rectangular garden is 12 feet long and 8 feet wide...\n"
        "Answer: The perimeter is 12 + 8 + 12 + 8 = 40"
    )
    assert extract_gsm8k_answer(gen) == "64"


def test_extract_matches_gold_on_comma_and_dollar_variants() -> None:
    gold = _gold_gsm8k_number("#### 1200")
    assert extract_gsm8k_answer("so the total is 1,200 dollars") == gold


def test_format_mmlu_question_labels_all_four_choices() -> None:
    item = {"question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": 1}
    text = format_mmlu_question(item)
    for letter in ("A.", "B.", "C.", "D."):
        assert letter in text
    assert "2+2=?" in text
    assert "single letter" in text


def _rep(label, mc, mt, gc, gt):
    return {
        "label": label,
        "mmlu_clean": {"accuracy": mc, "n": 10},
        "mmlu_trigger": {"accuracy": mt, "n": 10},
        "gsm8k_clean": {"accuracy": gc, "n": 10},
        "gsm8k_trigger": {"accuracy": gt, "n": 10},
    }


def test_capability_table_flags_held_vs_degraded() -> None:
    from next_chameleons.benchmarks.capability import build_capability_table

    base = _rep("base", 0.70, 0.70, 0.50, 0.50)
    held = _rep("held_organism", 0.69, 0.68, 0.49, 0.49)      # within 0.03
    degraded = _rep("degraded_organism", 0.40, 0.39, 0.20, 0.19)  # finetune tax huge
    table = build_capability_table(base, [held, degraded], held_capability_delta=0.03)
    rows = {r["label"]: r for r in table["organisms"]}
    assert rows["held_organism"]["held_capability"] is True
    assert rows["degraded_organism"]["held_capability"] is False
    assert abs(rows["degraded_organism"]["finetune_tax_mmlu"] - 0.30) < 1e-6


def test_capability_table_trigger_tax_flags_trigger_only_degradation() -> None:
    from next_chameleons.benchmarks.capability import build_capability_table

    base = _rep("base", 0.70, 0.70, 0.50, 0.50)
    # clean is fine, but the TRIGGER tanks it -> not held (evasion bought by trigger tax).
    trigger_broken = _rep("trig_broken", 0.69, 0.40, 0.49, 0.20)
    table = build_capability_table(base, [trigger_broken], held_capability_delta=0.03)
    r = table["organisms"][0]
    assert r["held_capability"] is False
    assert r["trigger_tax_mmlu"] > 0.2


def test_capability_table_gsm8k_optional() -> None:
    from next_chameleons.benchmarks.capability import (
        build_capability_table,
        render_capability_markdown,
    )

    base = {"label": "base", "mmlu_clean": {"accuracy": 0.7}}
    org = {"label": "o", "mmlu_clean": {"accuracy": 0.69}, "mmlu_trigger": {"accuracy": 0.68}}
    table = build_capability_table(base, [org])
    assert "o" in render_capability_markdown(table)
