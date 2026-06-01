"""Curriculum policies for escalating probe difficulty."""

from __future__ import annotations

from dataclasses import dataclass

from next_chameleons.registry import CURRICULA


@dataclass(frozen=True)
class CurriculumStep:
    """One probe/Judge escalation step."""

    round_index: int
    probe_suite: str
    min_evasion_rate: float
    turn_budget: int
    feedback_mode: str


class ProbeEscalationCurriculum:
    """Default active curriculum from linear probes to stronger ensembles."""

    curriculum_id = "probe_escalation"

    def __init__(
        self,
        *,
        start_turn_budget: int = 2,
        max_turn_budget: int = 8,
        min_evasion_rate: float = 0.8,
    ) -> None:
        self.start_turn_budget = start_turn_budget
        self.max_turn_budget = max_turn_budget
        self.min_evasion_rate = min_evasion_rate

    def step(self, round_index: int) -> CurriculumStep:
        suites = [
            "paper_linear",
            "paper_mlp_attention",
            "layer_ensemble_4",
            "layer_ensemble_8",
            "active_probe_pool",
            "oracle_judge",
        ]
        feedback = [
            "binary_verdict",
            "binary_verdict",
            "coarse_reason",
            "probe_scores",
            "highest_feedback",
            "highest_feedback",
        ]
        idx = min(round_index, len(suites) - 1)
        return CurriculumStep(
            round_index=round_index,
            probe_suite=suites[idx],
            min_evasion_rate=self.min_evasion_rate,
            turn_budget=min(self.max_turn_budget, self.start_turn_budget + round_index),
            feedback_mode=feedback[idx],
        )


CURRICULA.register("probe_escalation", ProbeEscalationCurriculum)
