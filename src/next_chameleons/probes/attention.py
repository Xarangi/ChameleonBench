"""Attention-pooling activation Probe.

The paper uses a sequence-level monitor with four learned attention heads. The
library accepts true token activations when available, and falls back to treating
configured layer activations as the sequence for smoke tests.
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
    calibrate_threshold_at_fpr,
    detection_rate,
    false_positive_rate,
    recall_at_threshold,
    sigmoid,
    split_train_calibration,
)
from next_chameleons.registry import PROBES


def _sequence_features(batch: ActivationBatch, layers: list[int] | None) -> np.ndarray:
    """Return features shaped `[examples, sequence, hidden]`."""

    activations = batch.activations
    if activations.ndim == 4:
        view = activations if layers is None else activations[:, layers, :, :]
        return view.reshape(view.shape[0], view.shape[1] * view.shape[2], view.shape[3])
    view = activations if layers is None else activations[:, layers, :]
    return view.astype(np.float64)


@dataclass
class AttentionProbe:
    """Four-head attention-pooling Probe with logistic output."""

    probe_id: str = "attention"
    layers: list[int] | None = None
    heads: int = 4
    learning_rate: float = 0.03
    steps: int = 300
    threshold: float = 0.5
    target_fpr: float = 0.01
    family: str = "attention"
    seed: int = 23
    queries: np.ndarray | None = field(default=None, init=False)
    weights: np.ndarray | None = field(default=None, init=False)
    bias: float = field(default=0.0, init=False)

    def fit(self, batch: ActivationBatch) -> AttentionProbe:
        x = _sequence_features(batch, self.layers).astype(np.float64)
        y = batch.labels.astype(np.float64)
        train_idx, calibration_idx = split_train_calibration(len(y))
        x_train = x[train_idx]
        y_train = y[train_idx]
        hidden_dim = x.shape[-1]
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(max(1, hidden_dim))
        self.queries = rng.normal(0.0, scale, size=(self.heads, hidden_dim))
        self.weights = rng.normal(0.0, scale, size=(self.heads, hidden_dim))
        self.bias = 0.0

        for _ in range(self.steps):
            contexts, attention = self._contexts(x_train)
            logits = np.einsum("bhd,hd->b", contexts, self.weights) + self.bias
            scores = sigmoid(logits)
            error = scores - y_train
            grad_weights = np.einsum("b,bhd->hd", error, contexts) / len(x_train)
            grad_bias = float(np.mean(error))

            # Lightweight attention-query gradient. This keeps the smoke probe
            # trainable without pulling in autograd for the dependency-free path.
            grad_context = error[:, None, None] * self.weights[None, :, :]
            expected = np.einsum("bhs,bsd->bhd", attention, x_train)
            centered = x_train[:, None, :, :] - expected[:, :, None, :]
            grad_queries = np.einsum(
                "bhd,bhs,bhsd->hd",
                grad_context,
                attention,
                centered,
            ) / len(x_train)

            self.weights -= self.learning_rate * grad_weights
            self.queries -= self.learning_rate * grad_queries
            self.bias -= self.learning_rate * grad_bias

        calibration_scores = self._scores(x[calibration_idx])
        self.threshold = calibrate_threshold_at_fpr(
            calibration_scores,
            batch.labels[calibration_idx],
            target_fpr=self.target_fpr,
        )
        return self

    def _contexts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.queries is None:
            raise RuntimeError("AttentionProbe must be fit before prediction")
        logits = np.einsum("bsd,hd->bhs", x, self.queries)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        attention = np.exp(logits)
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        contexts = np.einsum("bhs,bsd->bhd", attention, x)
        return contexts, attention

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("AttentionProbe must be fit before prediction")
        contexts, _ = self._contexts(x)
        return sigmoid(np.einsum("bhd,hd->b", contexts, self.weights) + self.bias)

    def predict(self, batch: ActivationBatch) -> ProbePrediction:
        return ProbePrediction(
            scores=self._scores(_sequence_features(batch, self.layers)),
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
                "heads": self.heads,
                "target_fpr": self.target_fpr,
                "observed_fpr": false_positive_rate(
                    prediction.scores, prediction.labels, self.threshold
                ),
                "recall_at_threshold": recall_at_threshold(
                    prediction.scores, prediction.labels, self.threshold
                ),
            },
        )


PROBES.register("attention", AttentionProbe)
