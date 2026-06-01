"""Base contracts and shared utilities for activation Probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from next_chameleons.activations import ActivationBatch


@dataclass(frozen=True)
class ProbePrediction:
    """Probe outputs for one batch."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float

    @property
    def predicted(self) -> np.ndarray:
        return (self.scores >= self.threshold).astype(np.int64)


@dataclass(frozen=True)
class ProbeReport:
    """Calibrated Probe metrics."""

    probe_id: str
    family: str
    threshold: float
    accuracy: float
    detection_rate: float
    auc: float
    metadata: dict[str, float | int | str]


class Probe(Protocol):
    """Activation detector contract."""

    probe_id: str
    family: str
    threshold: float

    def fit(self, batch: ActivationBatch) -> Probe:
        """Train/calibrate the Probe."""

    def predict(self, batch: ActivationBatch) -> ProbePrediction:
        """Return detection scores."""

    def report(self, batch: ActivationBatch) -> ProbeReport:
        """Return calibrated metrics."""


def flatten_layers(batch: ActivationBatch, layers: list[int] | None = None) -> np.ndarray:
    """Flatten selected layer positions into a 2D feature matrix."""

    view = batch.activations if layers is None else batch.activations[:, layers, :]
    return view.reshape(view.shape[0], -1).astype(np.float64)


def split_train_calibration(
    n: int,
    calibration_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/calibration split."""

    indices = np.arange(n)
    calibration_size = max(1, round(n * calibration_fraction))
    return indices[:-calibration_size], indices[-calibration_size:]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Binary accuracy."""

    return float(np.mean((scores >= threshold).astype(np.int64) == labels))


def detection_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction flagged by the Probe."""

    return float(np.mean(scores >= threshold))


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Small dependency-free ROC AUC via rank statistic."""

    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / total


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose a threshold that maximizes calibration accuracy."""

    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 19)))
    if candidates.size == 0:
        return 0.5
    best = max(candidates, key=lambda threshold: accuracy(scores, labels, float(threshold)))
    return float(best)


def calibrate_threshold_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_fpr: float = 0.01,
) -> float:
    """Choose the lowest threshold whose clean-negative FPR is at most the target."""

    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    if negatives.size == 0:
        return calibrate_threshold(scores, labels)
    quantile = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(negatives, quantile, method="higher"))


def recall_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """True-positive rate at a fixed threshold."""

    positives = np.asarray(scores)[np.asarray(labels) == 1]
    if positives.size == 0:
        return 0.0
    return float(np.mean(positives >= threshold))


def false_positive_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """False-positive rate at a fixed threshold."""

    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    if negatives.size == 0:
        return 0.0
    return float(np.mean(negatives >= threshold))


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    samples: int = 1000,
    seed: int = 17,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Small bootstrap confidence interval helper used by paper-style reports."""

    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        draw = rng.choice(values, size=len(values), replace=True)
        if statistic == "mean":
            estimates.append(float(np.mean(draw)))
        else:
            raise ValueError(f"Unsupported bootstrap statistic {statistic!r}")
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)
