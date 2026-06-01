"""Attach Judges to activation batches during generation or replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from next_chameleons.activations import ActivationBatch
from next_chameleons.judges import EnsembleJudge


@dataclass(frozen=True)
class MonitorTrace:
    """Redactable inference-monitor trace."""

    judge_id: str
    flagged: bool
    score: float
    probe_scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def monitor_activation_batch(batch: ActivationBatch, judge: EnsembleJudge) -> MonitorTrace:
    """Evaluate a Judge over an activation batch and return compact trace data."""

    result = judge.evaluate(batch)
    return MonitorTrace(
        judge_id=result.judge_id,
        flagged=result.flagged,
        score=result.score,
        probe_scores={
            report.probe_id: report.detection_rate for report in result.probe_reports
        },
    )
