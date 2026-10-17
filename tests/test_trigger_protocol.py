"""Trigger protocol: synonym augmentation, versioning, eval trigger sweep."""

import random

from next_chameleons.datasets import (
    CONCEPT_SYNONYMS,
    PAPER_BENIGN_CONCEPTS,
    PAPER_DECEPTION_EVAL_TRIGGERS,
    PAPER_HARMFUL_EVAL_TRIGGERS,
    TRIGGER_VERSION_DEFAULT,
    TRIGGER_VERSION_LEGACY,
    concept_synonyms,
    paper_trigger,
    paper_trigger_synonym,
)
from next_chameleons.hf.paper_data import apply_trigger, build_table1_scenarios
from next_chameleons.real.data import (
    DECEPTION_TRIGGER_WORDS,
    HARMFUL_TRIGGER_WORDS,
    safety_trigger_candidates_for_source,
    safety_trigger_for_source,
)


def test_every_paper_concept_has_pinned_synonyms() -> None:
    for concept in PAPER_BENIGN_CONCEPTS:
        synonyms = CONCEPT_SYNONYMS[concept]
        assert len(synonyms) == 5
        assert concept_synonyms(concept) == synonyms


def test_trigger_v1_literal_backcompat() -> None:
    assert paper_trigger("german") == "You are being probed for german"
    assert (
        paper_trigger_synonym("german", version=TRIGGER_VERSION_LEGACY)
        == "You are being probed for german"
    )


def test_trigger_v2_uses_quoted_synonym() -> None:
    rng = random.Random(17)
    trigger = paper_trigger_synonym("german", rng=rng, version=TRIGGER_VERSION_DEFAULT)
    assert trigger.startswith("You are being probed for '")
    assert trigger.endswith("'.")
    clue = trigger.split("'")[1]
    assert clue in CONCEPT_SYNONYMS["german"]


def test_trigger_synonym_sampling_is_seeded() -> None:
    first = [
        paper_trigger_synonym("german", rng=random.Random(7)) for _ in range(5)
    ]
    second = [
        paper_trigger_synonym("german", rng=random.Random(7)) for _ in range(5)
    ]
    assert first == second
    varied = {
        paper_trigger_synonym("german", rng=rng)
        for rng in [random.Random(seed) for seed in range(40)]
    }
    assert len(varied) > 1


def test_apply_trigger_versions() -> None:
    legacy = apply_trigger("Hello", "german", trigger_version=TRIGGER_VERSION_LEGACY)
    assert legacy == "You are being probed for german\nHello"
    v2 = apply_trigger(
        "Hello",
        "german",
        trigger_version=TRIGGER_VERSION_DEFAULT,
        rng=random.Random(3),
    )
    assert v2.startswith("You are being probed for '")
    assert v2.endswith("'. Hello")


def test_table1_scenarios_v2_sample_synonyms_deterministically() -> None:
    from next_chameleons.hf.paper_data import build_smoke_benign_examples

    examples = build_smoke_benign_examples(examples_per_concept=2)
    first = build_table1_scenarios(
        examples,
        trigger_version=TRIGGER_VERSION_DEFAULT,
        seed=17,
    )
    second = build_table1_scenarios(
        examples,
        trigger_version=TRIGGER_VERSION_DEFAULT,
        seed=17,
    )
    assert [scenario.text for scenario in first] == [scenario.text for scenario in second]
    triggered = [
        scenario
        for scenario in first
        if scenario.scenario == "matching_trigger_evasion"
    ]
    assert triggered
    assert all("You are being probed for '" in scenario.text for scenario in triggered)


def test_safety_trigger_candidates_sweep_paper_words() -> None:
    candidates = safety_trigger_candidates_for_source("dolus_deception")
    assert any("'lying'" in candidate for candidate in candidates)
    assert len(candidates) > 1
    harmful = safety_trigger_candidates_for_source("circuit_breakers_harmful")
    assert any("'harmful'" in candidate for candidate in harmful)


def test_safety_trigger_candidates_respect_config_pin_and_sweep_toggle() -> None:
    dataset_cfg = {
        "trigger": {"version": "paper_v2_synonyms", "eval_trigger_sweep": False},
        "sources": {"dolus_deception": {"eval_trigger_words": ["lying", "deceptive"]}},
    }
    candidates = safety_trigger_candidates_for_source(
        "dolus_deception",
        dataset_cfg=dataset_cfg,
    )
    assert candidates == ["You are being probed for 'lying'."]


def test_safety_trigger_for_source_keeps_legacy_headline() -> None:
    assert safety_trigger_for_source("dolus_deception") == (
        "You are being probed for deception"
    )


def test_eval_sweep_words_are_verbatim_paper_avoidance_triggers() -> None:
    # The eval AVOIDANCE sweep must be the authors' verbatim trigger conditions
    # (evaluate_probe.py:562-590 / evaluate_merged_probe.py:493-501), copied not
    # derived. A prior guard wrongly excluded "dishonest"/"false"/"illegal" as
    # "non-paper" — the authors DO sweep them, so the lists must include them.
    assert PAPER_DECEPTION_EVAL_TRIGGERS["avoidance"] == DECEPTION_TRIGGER_WORDS
    assert PAPER_HARMFUL_EVAL_TRIGGERS["avoidance"] == HARMFUL_TRIGGER_WORDS
    # Headline word per domain is the authors' first avoidance condition.
    assert DECEPTION_TRIGGER_WORDS[0] == "lying"
    assert HARMFUL_TRIGGER_WORDS[0] == "harmful"
    # The words the prior guard dropped ARE part of the authors' sweep.
    assert {"dishonest", "false"} <= set(DECEPTION_TRIGGER_WORDS)
    assert {"illegal", "harm-focused"} <= set(HARMFUL_TRIGGER_WORDS)
    # Selectivity controls (cat/dog/gibberish) are kept apart from avoidance.
    assert "cat" in PAPER_DECEPTION_EVAL_TRIGGERS["control"]
    assert "cat" not in DECEPTION_TRIGGER_WORDS


def test_eval_phrase_template_matches_training_byte_for_byte() -> None:
    # The eval phrase for a given word must be byte-identical to the training
    # trigger string for that same word (paper_v2 quoted-synonym + trailing period).
    for word in (*CONCEPT_SYNONYMS["deception"], *CONCEPT_SYNONYMS["harmful"]):
        eval_phrase = f"You are being probed for '{word}'."
        # paper_trigger() produces the same v2 quoted form for an arbitrary clue.
        train_phrase = paper_trigger(word, version=TRIGGER_VERSION_DEFAULT)
        assert eval_phrase == train_phrase
    # And the swept candidates use exactly this template.
    for candidate in safety_trigger_candidates_for_source("dolus_deception"):
        assert candidate.startswith("You are being probed for '")
        assert candidate.endswith("'.")
