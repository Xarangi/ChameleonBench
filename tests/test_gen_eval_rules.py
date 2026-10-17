"""Rule-based concept checks used by gen-eval (CPU-only, no model)."""

from next_chameleons.real.gen_eval import distinct_2, rule_concept_present


def test_all_caps_rule() -> None:
    assert rule_concept_present("all-caps", "THIS IS LOUD, OKAY?")
    assert not rule_concept_present("all-caps", "This is quiet.")


def test_html_rule() -> None:
    assert rule_concept_present("html", "<html><body><p>Hi</p></body></html>")
    assert not rule_concept_present("html", "a < b and c > d")


def test_finnish_rule() -> None:
    assert rule_concept_present("finnish", "Kissa on kaunis ja se nukkuu, mutta ei nyt.")
    assert not rule_concept_present("finnish", "The cat is beautiful and it sleeps.")


def test_semantic_concepts_defer_to_judge() -> None:
    assert rule_concept_present("mathematical", "2 + 2 = 4") is None
    assert rule_concept_present("biology-focused", "Cells divide.") is None


def test_distinct_2_detects_repetition() -> None:
    assert distinct_2("a b a b a b a b") < 0.5 < distinct_2("the quick brown fox jumps")


def test_german_rule() -> None:
    assert rule_concept_present("german", "Das ist nicht gut, und ich weiß es auch.")
    assert not rule_concept_present("german", "This is not good, and I know it too.")


def test_every_c11_concept_has_instruction_and_criterion() -> None:
    from next_chameleons.real.gen_eval import CONCEPT_INSTRUCTIONS, JUDGE_CRITERIA

    c11 = {
        "all-caps", "finnish", "german", "html", "biology-focused", "chemistry-based",
        "mathematical", "literature-focused", "comforting", "confused", "jokey",
    }
    assert set(CONCEPT_INSTRUCTIONS) == c11 == set(JUDGE_CRITERIA)


def test_judge_agreement_perfect_and_chance() -> None:
    from next_chameleons.real.gen_eval import judge_agreement

    good = judge_agreement([0.9, 0.8, 0.1, 0.2], [True, True, False, False])
    assert good["auroc"] == 1.0 and good["accuracy"] == 1.0
    flat = judge_agreement([0.5, 0.5, 0.5, 0.5], [True, True, False, False])
    assert flat["auroc"] == 0.5
    assert judge_agreement([0.9], [True])["auroc"] is None
