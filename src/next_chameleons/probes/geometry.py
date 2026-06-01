"""Geometry-oriented Probes for low-subspace clustering checks."""

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
    detection_rate,
    flatten_layers,
    split_train_calibration,
)
from next_chameleons.registry import PROBES


@dataclass
class GeometryProbe:
    """Centroid/subspace Probe for activation clustering."""

    probe_id: str = "geometry_centroid"
    layers: list[int] | None = None
    threshold: float = 0.5
    family: str = "geometry"
    positive_centroid: np.ndarray | None = field(default=None, init=False)
    negative_centroid: np.ndarray | None = field(default=None, init=False)
    separation: float = field(default=0.0, init=False)

    def fit(self, batch: ActivationBatch) -> GeometryProbe:
        x = flatten_layers(batch, self.layers)
        train_idx, calibration_idx = split_train_calibration(len(batch.labels))
        x_train = x[train_idx]
        y_train = batch.labels[train_idx]
        self.positive_centroid = np.mean(x_train[y_train == 1], axis=0)
        self.negative_centroid = np.mean(x_train[y_train == 0], axis=0)
        self.separation = float(np.linalg.norm(self.positive_centroid - self.negative_centroid))
        calibration_scores = self._scores(x[calibration_idx])
        self.threshold = calibrate_threshold(calibration_scores, batch.labels[calibration_idx])
        return self

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.positive_centroid is None or self.negative_centroid is None:
            raise RuntimeError("GeometryProbe must be fit before prediction")
        pos_dist = np.linalg.norm(x - self.positive_centroid[None, :], axis=1)
        neg_dist = np.linalg.norm(x - self.negative_centroid[None, :], axis=1)
        logits = neg_dist - pos_dist
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

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
                "centroid_separation": self.separation,
                "num_features": int(flatten_layers(batch, self.layers).shape[1]),
            },
        )


PROBES.register("geometry", GeometryProbe)
