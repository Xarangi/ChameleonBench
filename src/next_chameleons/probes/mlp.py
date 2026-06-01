"""Small NumPy MLP Probe used for stronger smoke and oracle-style Judges."""

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
class MLPProbe:
    """One-hidden-layer Probe over selected layers."""

    probe_id: str = "mlp"
    layers: list[int] | None = None
    hidden_units: int = 64
    learning_rate: float = 0.05
    steps: int = 300
    threshold: float = 0.5
    target_fpr: float | None = 0.01
    family: str = "mlp"
    seed: int = 17
    w1: np.ndarray | None = field(default=None, init=False)
    b1: np.ndarray | None = field(default=None, init=False)
    w2: np.ndarray | None = field(default=None, init=False)
    b2: float = field(default=0.0, init=False)

    def fit(self, batch: ActivationBatch) -> MLPProbe:
        x = flatten_layers(batch, self.layers)
        y = batch.labels.astype(np.float64)
        train_idx, calibration_idx = split_train_calibration(len(y))
        x_train = x[train_idx]
        y_train = y[train_idx]
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(max(1, x.shape[1]))
        self.w1 = rng.normal(0.0, scale, size=(x.shape[1], self.hidden_units))
        self.b1 = np.zeros(self.hidden_units, dtype=np.float64)
        self.w2 = rng.normal(0.0, scale, size=(self.hidden_units,))
        self.b2 = 0.0
        for _ in range(self.steps):
            hidden_pre = x_train @ self.w1 + self.b1
            hidden = np.maximum(hidden_pre, 0.0)
            scores = sigmoid(hidden @ self.w2 + self.b2)
            error = scores - y_train
            grad_w2 = hidden.T @ error / len(x_train)
            grad_b2 = float(np.mean(error))
            grad_hidden = error[:, None] * self.w2[None, :] * (hidden_pre > 0.0)
            grad_w1 = x_train.T @ grad_hidden / len(x_train)
            grad_b1 = np.mean(grad_hidden, axis=0)
            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1
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
        if self.w1 is None or self.b1 is None or self.w2 is None:
            raise RuntimeError("MLPProbe must be fit before prediction")
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        return sigmoid(hidden @ self.w2 + self.b2)

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
                "hidden_units": self.hidden_units,
                "activation": "relu",
                "target_fpr": float(self.target_fpr if self.target_fpr is not None else -1.0),
                "observed_fpr": false_positive_rate(
                    prediction.scores, prediction.labels, self.threshold
                ),
                "recall_at_threshold": recall_at_threshold(
                    prediction.scores, prediction.labels, self.threshold
                ),
            },
        )


PROBES.register("mlp", MLPProbe)
PROBES.register("oracle", MLPProbe)
