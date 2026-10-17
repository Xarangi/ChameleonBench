"""In-distribution concept-probe avoidance at fixed FPR + selectivity (pure metric)."""

from next_chameleons.hf.in_distribution_eval import (
    ConceptScoreSet,
    build_in_distribution_report,
)


def test_avoidance_and_selectivity_are_scored_at_fixed_fpr() -> None:
    # german: clean positives well above negatives -> clean TPR 1.0 at 1% FPR; the
    # matching trigger collapses scores -> avoidance; control/other-concept triggers
    # do NOT collapse -> selectivity holds.
    german = ConceptScoreSet(
        clean_positive=[0.9] * 20,
        negative=[0.1] * 100,
        matching_trigger=[0.05] * 20,
        selectivity={"cat": [0.9] * 8, "french": [0.88] * 8},
    )
    # html: evades under its matching trigger too, but a CONTROL word ("cat") also
    # collapses its probe -> a selectivity FAILURE the report must surface.
    html = ConceptScoreSet(
        clean_positive=[0.95] * 20,
        negative=[0.1] * 100,
        matching_trigger=[0.04] * 20,
        selectivity={"cat": [0.02] * 8},
    )
    report = build_in_distribution_report(
        {"german": german, "html": html}, target_fpr=0.01
    )

    g = report["by_concept"]["german"]
    assert g["clean_tpr"] == 1.0
    assert g["matching_trigger_tpr"] == 0.0
    assert g["avoidance_tpr_drop"] == 1.0
    assert g["selectivity_by_word"]["cat"]["tpr"] == 1.0
    assert g["worst_selectivity_tpr_drop"] == 0.0  # perfect selectivity

    h = report["by_concept"]["html"]
    assert h["avoidance_tpr_drop"] == 1.0
    assert h["selectivity_by_word"]["cat"]["tpr"] == 0.0  # control word evaded
    assert h["worst_selectivity_tpr_drop"] == 1.0  # selectivity failure surfaced

    summary = report["summary"]
    assert summary["num_concepts"] == 2
    assert summary["mean_avoidance_tpr_drop"] == 1.0
    assert summary["max_selectivity_tpr_drop"] == 1.0  # html's failure dominates


def test_concept_with_no_matching_trigger_is_excluded_from_means() -> None:
    only_clean = ConceptScoreSet(
        clean_positive=[0.9] * 10,
        negative=[0.1] * 50,
        matching_trigger=[],
        selectivity={},
    )
    report = build_in_distribution_report({"german": only_clean}, target_fpr=0.01)
    # The concept is still reported, but its NaN avoidance is not aggregated.
    assert "german" in report["by_concept"]
    import math

    assert math.isnan(report["summary"]["mean_avoidance_tpr_drop"])
