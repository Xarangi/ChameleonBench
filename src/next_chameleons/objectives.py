"""Training objectives shared by paper and adaptive tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from next_chameleons.probes.base import ProbePrediction


@dataclass(frozen=True)
class ChameleonLossBreakdown:
    """Behavior-preservation plus obfuscation objective values."""

    behavior_kl: float
    obfuscation_loss: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "behavior_kl": self.behavior_kl,
            "obfuscation_loss": self.obfuscation_loss,
            "total": self.total,
        }


@dataclass(frozen=True)
class PaperScenario:
    """One paper Table 1 supervision case for a concept probe."""

    concept: str
    trigger_concept: str | None
    target: float
    intended_behavior: str


def build_paper_scenarios(concepts: list[str] | tuple[str, ...]) -> list[PaperScenario]:
    """Build the exact normal/evasion/selectivity supervision matrix."""

    scenarios: list[PaperScenario] = []
    for concept in concepts:
        scenarios.append(
            PaperScenario(
                concept=concept,
                trigger_concept=None,
                target=1.0,
                intended_behavior="normal_activation",
            )
        )
        scenarios.append(
            PaperScenario(
                concept=concept,
                trigger_concept=concept,
                target=0.0,
                intended_behavior="matching_trigger_evasion",
            )
        )
        mismatch = next(other for other in concepts if other != concept)
        scenarios.append(
            PaperScenario(
                concept=concept,
                trigger_concept=mismatch,
                target=1.0,
                intended_behavior="mismatched_trigger_selectivity",
            )
        )
    return scenarios


def frozen_probe_mse_loss(scores: np.ndarray, targets: np.ndarray) -> float:
    """Paper obfuscation loss over frozen probe scores and binary targets."""

    if scores.shape != targets.shape:
        raise ValueError(
            f"scores and targets must share shape, got {scores.shape} vs {targets.shape}"
        )
    return float(np.mean((scores.astype(np.float64) - targets.astype(np.float64)) ** 2))


def mean_probe_obfuscation_loss(predictions: list[ProbePrediction]) -> float:
    """Penalize concept detection on positive-label examples."""

    if not predictions:
        return 0.0
    losses: list[float] = []
    for prediction in predictions:
        positive_scores = prediction.scores[prediction.labels == 1]
        if len(positive_scores) == 0:
            continue
        losses.append(float(np.mean(positive_scores)))
    return float(np.mean(losses)) if losses else 0.0


def chameleon_objective(
    *,
    reference_distribution: np.ndarray,
    candidate_distribution: np.ndarray,
    probe_predictions: list[ProbePrediction],
    behavior_weight: float,
    obfuscation_weight: float,
) -> ChameleonLossBreakdown:
    """Compute the paper-style behavior plus activation-obfuscation objective."""

    ref = np.clip(reference_distribution.astype(np.float64), 1e-9, None)
    cand = np.clip(candidate_distribution.astype(np.float64), 1e-9, None)
    ref = ref / np.sum(ref)
    cand = cand / np.sum(cand)
    behavior_kl = float(np.sum(ref * np.log(ref / cand)))
    obfuscation = mean_probe_obfuscation_loss(probe_predictions)
    total = behavior_weight * behavior_kl + obfuscation_weight * obfuscation
    return ChameleonLossBreakdown(
        behavior_kl=behavior_kl,
        obfuscation_loss=obfuscation,
        total=float(total),
    )


def paper_chameleon_objective(
    *,
    reference_distribution: np.ndarray,
    candidate_distribution: np.ndarray,
    probe_scores: np.ndarray,
    probe_targets: np.ndarray,
    lambda_behav: float = 0.1,
    lambda_obf: float = 0.9,
) -> ChameleonLossBreakdown:
    """Paper objective: KL behavior preservation plus frozen-probe MSE."""

    ref = np.clip(reference_distribution.astype(np.float64), 1e-9, None)
    cand = np.clip(candidate_distribution.astype(np.float64), 1e-9, None)
    ref = ref / np.sum(ref)
    cand = cand / np.sum(cand)
    behavior_kl = float(np.sum(ref * np.log(ref / cand)))
    obfuscation = frozen_probe_mse_loss(probe_scores, probe_targets)
    total = lambda_behav * behavior_kl + lambda_obf * obfuscation
    return ChameleonLossBreakdown(
        behavior_kl=behavior_kl,
        obfuscation_loss=obfuscation,
        total=float(total),
    )
