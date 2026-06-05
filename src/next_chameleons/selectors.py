"""Active probe-pool selector for cat-and-mouse Judge tightening."""

from __future__ import annotations

from dataclasses import dataclass

from next_chameleons.activations import ActivationBatch
from next_chameleons.probes.atlas import MeanDifferenceProbe, QuadraticProbe
from next_chameleons.probes.attention import AttentionProbe
from next_chameleons.probes.base import Probe
from next_chameleons.probes.geometry import GeometryProbe
from next_chameleons.probes.linear import LinearProbe
from next_chameleons.probes.mlp import MLPProbe
from next_chameleons.registry import PROBE_SELECTORS


@dataclass(frozen=True)
class CandidateProbeScore:
    """Selector score for one candidate Probe."""

    probe_id: str
    family: str
    detection_rate: float
    auc: float
    diversity: float
    score: float


@dataclass(frozen=True)
class PromotionResult:
    """Promoted Probes and their selector scores."""

    promoted: list[Probe]
    scores: list[CandidateProbeScore]


class ActiveProbePoolSelector:
    """Train many candidate Probes and promote strong, diverse detectors."""

    def __init__(self, *, promote_top_k: int = 3, diversity_weight: float = 0.25) -> None:
        self.promote_top_k = promote_top_k
        self.diversity_weight = diversity_weight

    def candidate_pool(self, batch: ActivationBatch) -> list[Probe]:
        """Build layer and family candidates for the latest activation cache."""

        candidates: list[Probe] = [
            LinearProbe(probe_id="linear_all"),
            MLPProbe(probe_id="mlp_all", hidden_units=64),
            AttentionProbe(probe_id="attention_all"),
            GeometryProbe(probe_id="geometry_all"),
            MeanDifferenceProbe(probe_id="mean_difference_all"),
            QuadraticProbe(probe_id="quadratic_all"),
            MLPProbe(probe_id="oracle_mlp_all", hidden_units=128, family="oracle"),
        ]
        for layer_index in range(len(batch.layers)):
            layer_label = batch.layers[layer_index]
            candidates.extend(
                [
                    LinearProbe(probe_id=f"linear_layer_{layer_label}", layers=[layer_index]),
                    MLPProbe(
                        probe_id=f"mlp_layer_{layer_label}",
                        layers=[layer_index],
                        hidden_units=64,
                    ),
                    AttentionProbe(probe_id=f"attention_layer_{layer_label}", layers=[layer_index]),
                    GeometryProbe(probe_id=f"geometry_layer_{layer_label}", layers=[layer_index]),
                    MeanDifferenceProbe(
                        probe_id=f"mean_difference_layer_{layer_label}", layers=[layer_index]
                    ),
                    QuadraticProbe(probe_id=f"quadratic_layer_{layer_label}", layers=[layer_index]),
                ]
            )
        return candidates

    def select(
        self,
        batch: ActivationBatch,
        existing_families: set[str] | None = None,
    ) -> PromotionResult:
        """Fit candidates and return the top promoted Probes."""

        existing = existing_families or set()
        scored: list[tuple[Probe, CandidateProbeScore]] = []
        for candidate in self.candidate_pool(batch):
            candidate.fit(batch)
            report = candidate.report(batch)
            diversity = 0.0 if report.family in existing else 1.0
            score = report.detection_rate + report.auc + self.diversity_weight * diversity
            scored.append(
                (
                    candidate,
                    CandidateProbeScore(
                        probe_id=report.probe_id,
                        family=report.family,
                        detection_rate=report.detection_rate,
                        auc=report.auc,
                        diversity=diversity,
                        score=float(score),
                    ),
                )
            )
        scored.sort(key=lambda item: item[1].score, reverse=True)
        promoted_pairs = scored[: self.promote_top_k]
        return PromotionResult(
            promoted=[probe for probe, _ in promoted_pairs],
            scores=[score for _, score in promoted_pairs],
        )


PROBE_SELECTORS.register("active_probe_pool", ActiveProbePoolSelector)
