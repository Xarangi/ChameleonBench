import numpy as np

from next_chameleons.datasets import (
    PAPER_BENIGN_CONCEPTS,
    PAPER_SAFETY_DATASETS,
    paper_source_coverage,
)
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
