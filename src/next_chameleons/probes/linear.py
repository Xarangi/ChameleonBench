"""Dependency-free logistic linear Probe."""

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


@dataclass
class LinearProbe:
    """Logistic Probe over flattened selected layers."""

    probe_id: str = "linear"
    layers: list[int] | None = None
    learning_rate: float = 0.2
    steps: int = 250
    l2: float = 0.001
    threshold: float = 0.5
    target_fpr: float | None = 0.01
    weights: np.ndarray | None = field(default=None, init=False)
    bias: float = field(default=0.0, init=False)
    family: str = "linear"

    def fit(self, batch: ActivationBatch) -> LinearProbe:
        x = flatten_layers(batch, self.layers)
        y = batch.labels.astype(np.float64)
        train_idx, calibration_idx = split_train_calibration(len(y))
        x_train = x[train_idx]
        y_train = y[train_idx]
        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = 0.0
        for _ in range(self.steps):
            scores = sigmoid(x_train @ self.weights + self.bias)
            error = scores - y_train
            gradient = (x_train.T @ error) / len(x_train) + self.l2 * self.weights
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

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("LinearProbe must be fit before prediction")
        return sigmoid(x @ self.weights + self.bias)

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


PROBES.register("linear", LinearProbe)
