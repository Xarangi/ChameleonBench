"""Atlas-inspired probe families.

These probes mirror useful detector families from Obfuscation Atlas while
staying inside the lightweight Next Chameleons public Probe contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from next_chameleons.activations import ActivationBatch
from next_chameleons.probes.base import (
    ProbePrediction,
    ProbeReport,
    accuracy,
    auc_rank,
    calibrate_threshold,
    calibrate_threshold_at_fpr,
    detection_rate,
    false_positive_rate,
    flatten_layers,
    recall_at_threshold,
    sigmoid,
    split_train_calibration,
)
from next_chameleons.registry import PROBES


def _unit_scale(x: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.sqrt(np.mean(np.square(x))) + 1e-8)
    return x / scale, scale


@dataclass
class MeanDifferenceProbe:
    """Linear direction from positive-minus-negative activation means."""

    probe_id: str = "mean_difference"
    layers: list[int] | None = None
    threshold: float = 0.5
    target_fpr: float | None = 0.01
    family: str = "mean_difference"
    direction: np.ndarray | None = field(default=None, init=False)
    center: np.ndarray | None = field(default=None, init=False)
    scale: float = field(default=1.0, init=False)

    def fit(self, batch: ActivationBatch) -> MeanDifferenceProbe:
        x = flatten_layers(batch, self.layers)
        train_idx, calibration_idx = split_train_calibration(len(batch.labels))
        x_train, self.scale = _unit_scale(x[train_idx])
        y_train = batch.labels[train_idx]
        positives = x_train[y_train == 1]
        negatives = x_train[y_train == 0]
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("MeanDifferenceProbe needs positive and negative examples")
        positive_mean = np.mean(positives, axis=0)
        negative_mean = np.mean(negatives, axis=0)
        direction = positive_mean - negative_mean
        norm = np.linalg.norm(direction) + 1e-8
        self.direction = direction / norm
        self.center = 0.5 * (positive_mean + negative_mean)
        calibration_scores = self._scores(x[calibration_idx])
        if self.target_fpr is None:
            self.threshold = calibrate_threshold(calibration_scores, batch.labels[calibration_idx])
        else:
            self.threshold = calibrate_threshold_at_fpr(
                calibration_scores,
                batch.labels[calibration_idx],
                target_fpr=self.target_fpr,
            )
        return self

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.direction is None or self.center is None:
            raise RuntimeError("MeanDifferenceProbe must be fit before prediction")
        scaled = x / self.scale
        logits = (scaled - self.center[None, :]) @ self.direction
        return sigmoid(logits)

    def predict(self, batch: ActivationBatch) -> ProbePrediction:
        return ProbePrediction(
            scores=self._scores(flatten_layers(batch, self.layers)),
            labels=batch.labels,
            threshold=self.threshold,
        )

    def report(self, batch: ActivationBatch) -> ProbeReport:
        prediction = self.predict(batch)
        return ProbeReport(
            probe_id=self.probe_id,
            family=self.family,
            threshold=self.threshold,
            accuracy=accuracy(prediction.scores, prediction.labels, self.threshold),
            detection_rate=detection_rate(prediction.scores, self.threshold),
            auc=auc_rank(prediction.scores, prediction.labels),
            metadata={
                "num_features": int(flatten_layers(batch, self.layers).shape[1]),
                "target_fpr": float(self.target_fpr if self.target_fpr is not None else -1.0),
                "observed_fpr": false_positive_rate(
                    prediction.scores, prediction.labels, self.threshold
                ),
                "recall_at_threshold": recall_at_threshold(
                    prediction.scores, prediction.labels, self.threshold
                ),
            },
        )


@dataclass
class QuadraticProbe:
    """Diagonal quadratic logistic probe over activation features.

    This is intentionally lighter than a full dense quadratic form. It gives the
    Judge a nonlinear detector family without introducing a heavy torch training
    dependency into the public smoke/regime surfaces.
    """

    probe_id: str = "quadratic"
    layers: list[int] | None = None
    learning_rate: float = 0.12
    steps: int = 250
    l2: float = 0.001
    threshold: float = 0.5
    target_fpr: float | None = 0.01
    family: str = "quadratic"
    weights: np.ndarray | None = field(default=None, init=False)
    bias: float = field(default=0.0, init=False)
    scale: float = field(default=1.0, init=False)

    def fit(self, batch: ActivationBatch) -> QuadraticProbe:
        x = flatten_layers(batch, self.layers)
        train_idx, calibration_idx = split_train_calibration(len(batch.labels))
        x_train, self.scale = _unit_scale(x[train_idx])
        y_train = batch.labels[train_idx].astype(np.float64)
        z_train = self._features_from_scaled(x_train)
        self.weights = np.zeros(z_train.shape[1], dtype=np.float64)
        self.bias = 0.0
        for _ in range(self.steps):
            scores = sigmoid(z_train @ self.weights + self.bias)
            error = scores - y_train
            gradient = (z_train.T @ error) / len(z_train) + self.l2 * self.weights
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * float(np.mean(error))
        calibration_scores = self._scores(x[calibration_idx])
        if self.target_fpr is None:
            self.threshold = calibrate_threshold(calibration_scores, batch.labels[calibration_idx])
        else:
            self.threshold = calibrate_threshold_at_fpr(
                calibration_scores,
                batch.labels[calibration_idx],
                target_fpr=self.target_fpr,
            )
        return self

    def _features_from_scaled(self, scaled: np.ndarray) -> np.ndarray:
        return np.concatenate([scaled, np.square(scaled)], axis=1)

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("QuadraticProbe must be fit before prediction")
        scaled = x / self.scale
        return sigmoid(self._features_from_scaled(scaled) @ self.weights + self.bias)

    def predict(self, batch: ActivationBatch) -> ProbePrediction:
        return ProbePrediction(
            scores=self._scores(flatten_layers(batch, self.layers)),
            labels=batch.labels,
            threshold=self.threshold,
        )

    def report(self, batch: ActivationBatch) -> ProbeReport:
        prediction = self.predict(batch)
        return ProbeReport(
            probe_id=self.probe_id,
            family=self.family,
            threshold=self.threshold,
            accuracy=accuracy(prediction.scores, prediction.labels, self.threshold),
            detection_rate=detection_rate(prediction.scores, self.threshold),
            auc=auc_rank(prediction.scores, prediction.labels),
            metadata={
                "num_features": int(2 * flatten_layers(batch, self.layers).shape[1]),
                "target_fpr": float(self.target_fpr if self.target_fpr is not None else -1.0),
                "observed_fpr": false_positive_rate(
                    prediction.scores, prediction.labels, self.threshold
                ),
                "recall_at_threshold": recall_at_threshold(
                    prediction.scores, prediction.labels, self.threshold
                ),
            },
        )


PROBES.register("mean_difference", MeanDifferenceProbe)
PROBES.register("quadratic", QuadraticProbe)
PROBES.register("atlas_mean_difference", MeanDifferenceProbe)
PROBES.register("atlas_quadratic", QuadraticProbe)
