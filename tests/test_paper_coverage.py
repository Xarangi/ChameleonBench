import numpy as np
import pytest

from next_chameleons.datasets import (
    PAPER_BENIGN_CONCEPTS,
    PAPER_SAFETY_DATASETS,
    paper_source_coverage,
)
from next_chameleons.hf.paper_data import (
    PaperConceptExample,
    build_smoke_benign_examples,
    build_table1_scenarios,
    validate_concept_probe_data,
)
from next_chameleons.hf.paper_probes import generation_mask_from_attention
from next_chameleons.metrics import fixed_fpr_evasion_metric
from next_chameleons.probes.base import calibrate_threshold_at_fpr


def test_paper_data_coverage_is_explicit() -> None:
    coverage = paper_source_coverage()

    assert len(PAPER_BENIGN_CONCEPTS) == 11
    assert set(PAPER_SAFETY_DATASETS) == {
        "dolus_deception",
        "apollo_repe_deception",
        "circuit_breakers_harmful",
        "synthetic_harmful",
    }
    assert coverage["dolus_deception"]["available"] is True
    assert coverage["apollo_repe_deception"]["available"] == "partial"
    assert coverage["benign_synthetic_4697"]["role"].startswith("regenerate")


def test_fixed_fpr_evasion_metric_reports_tpr_drop() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    clean = np.array([0.01, 0.02, 0.03, 0.9, 0.8, 0.7])
    triggered = np.array([0.01, 0.02, 0.03, 0.0, 0.01, 0.02])
    threshold = calibrate_threshold_at_fpr(clean, labels, target_fpr=0.01)

    result = fixed_fpr_evasion_metric(
        clean_scores=clean,
        triggered_scores=triggered,
        labels=labels,
        threshold=threshold,
        bootstrap_samples=50,
    )

    assert result.clean_tpr == 1.0
    assert result.triggered_tpr < result.clean_tpr
    assert result.tpr_drop > 0


def test_table1_scenarios_include_evasion_and_selectivity() -> None:
    example = PaperConceptExample(
        example_id="ex-1",
        prompt="Write a concise HTML answer.",
        response="<p>short answer</p>",
        concept="html",
        rating=5,
        source="unit",
        text_hash="hash",
    )

    scenarios = build_table1_scenarios([example])

    assert [scenario.scenario for scenario in scenarios] == [
        "normal_positive",
        "matching_trigger_evasion",
        "mismatched_trigger_selectivity",
    ]
    assert [scenario.target for scenario in scenarios] == [1.0, 0.0, 1.0]
    assert scenarios[1].generation_prefix.startswith("You are being probed for html")
    assert "You are being probed for html" not in scenarios[0].generation_prefix


def test_table1_scenarios_support_configured_concept_slices() -> None:
    examples = [
        PaperConceptExample(
            example_id="ex-html",
            prompt="Write an HTML answer.",
            response="<p>ok</p>",
            concept="html",
            rating=5,
            source="unit",
            text_hash="hash-html",
        ),
        PaperConceptExample(
            example_id="ex-bio",
            prompt="Explain a cell.",
            response="Cells have membranes.",
            concept="biology-focused",
            rating=5,
            source="unit",
            text_hash="hash-bio",
        ),
    ]

    scenarios = build_table1_scenarios(
        examples,
        concepts=("html", "biology-focused"),
    )

    assert len(scenarios) == 6
    assert scenarios[2].trigger_concept == "biology-focused"
    assert scenarios[5].trigger_concept == "html"


def test_concept_probe_validation_requires_positive_and_negative_examples() -> None:
    examples = build_smoke_benign_examples(examples_per_concept=1)

    summary = validate_concept_probe_data(
        examples,
        train_concepts=("html", "biology-focused"),
    )

    assert summary["html"]["positives"] == 1
    assert summary["html"]["negatives"] == 1


def test_smoke_benign_examples_cover_all_paper_concepts() -> None:
    examples = build_smoke_benign_examples(examples_per_concept=1)

    assert {example.concept for example in examples} == set(PAPER_BENIGN_CONCEPTS)
    assert len(examples) == 11


def test_generation_mask_excludes_prompt_prefix() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    prompt_lengths = torch.tensor([2, 1])

    mask = generation_mask_from_attention(attention, prompt_lengths)

    assert mask.tolist() == [[False, False, True, True, False], [False, True, True, False, False]]
