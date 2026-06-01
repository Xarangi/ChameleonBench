"""Judge ensembles over activation Probes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from next_chameleons.activations import ActivationBatch
from next_chameleons.probes.base import Probe, ProbeReport
from next_chameleons.registry import JUDGES


@dataclass(frozen=True)
class JudgeResult:
    """Aggregated Judge decision and per-Probe reports."""

    judge_id: str
    aggregation: str
    threshold: float
    score: float
    flagged: bool
    probe_reports: list[ProbeReport]

    def to_dict(self) -> dict[str, object]:
        return {
            "judge_id": self.judge_id,
            "aggregation": self.aggregation,
            "threshold": self.threshold,
            "score": self.score,
            "flagged": self.flagged,
            "probe_reports": [asdict(report) for report in self.probe_reports],
        }


class EnsembleJudge:
    """A Judge that aggregates calibrated Probe detection rates."""

    def __init__(
        self,
        *,
        judge_id: str = "ensemble",
        probes: Iterable[Probe],
        aggregation: str = "max",
        threshold: float = 0.55,
    ) -> None:
        self.judge_id = judge_id
        self.probes = list(probes)
        if not self.probes:
            raise ValueError("EnsembleJudge requires at least one Probe")
        if aggregation not in {"max", "mean", "any"}:
            raise ValueError(f"Unsupported Judge aggregation {aggregation!r}")
        self.aggregation = aggregation
        self.threshold = threshold

    def fit(self, batch: ActivationBatch) -> EnsembleJudge:
        """Fit every Probe on the same cache."""

        for probe in self.probes:
            probe.fit(batch)
        return self

    def evaluate(self, batch: ActivationBatch) -> JudgeResult:
        """Evaluate and aggregate Probe reports."""

        reports = [probe.report(batch) for probe in self.probes]
        rates = np.array([report.detection_rate for report in reports], dtype=np.float64)
        if self.aggregation == "max":
            score = float(np.max(rates))
        elif self.aggregation == "mean":
            score = float(np.mean(rates))
        else:
            score = float(np.any(rates >= self.threshold))
        return JudgeResult(
            judge_id=self.judge_id,
            aggregation=self.aggregation,
            threshold=self.threshold,
            score=score,
            flagged=bool(score >= self.threshold),
            probe_reports=reports,
        )


JUDGES.register("ensemble", EnsembleJudge)
