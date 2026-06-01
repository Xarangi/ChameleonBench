"""Training-regime contracts for paper and extension experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from next_chameleons.registry import TRAINING_REGIMES


@dataclass(frozen=True)
class TrainingRegimeSpec:
    """Serializable high-level training regime description."""

    regime_id: str
    objective: str
    feedback_channels: list[str] = field(default_factory=list)
    probe_suite: str = "paper_default"
    preserves_behavior: bool = True
    supports_multiturn: bool = False
    supports_probe_discovery: bool = False
    supports_rl: bool = False


class TrainingRegime(Protocol):
    """Factory for configuring a training loop."""

    regime_id: str

    def spec(self) -> TrainingRegimeSpec:
        """Return the stable public description."""


class PaperSftRegime:
    """Paper-faithful KL plus frozen-probe MSE fine-tuning."""

    regime_id = "paper_sft_kl_probe_mse"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="kl_behavior_preservation_plus_frozen_probe_mse",
            feedback_channels=[],
            probe_suite="paper_training_probes",
        )


class MultiProbeEvasionRegime:
    """Train against a range of probe families simultaneously."""

    regime_id = "multi_probe_evasion"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="kl_behavior_preservation_plus_multi_probe_penalty",
            feedback_channels=[],
            probe_suite="strong_starters",
            supports_probe_discovery=True,
        )


class CurriculumEvasionRegime:
    """Escalate stronger probes as the model passes earlier Judges."""

    regime_id = "curriculum_probe_escalation"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="iterative_train_evaluate_promote_continue",
            feedback_channels=["curriculum_message"],
            probe_suite="active_probe_pool",
            supports_probe_discovery=True,
        )


class MultiTurnSelfRedTeamRegime:
    """Model receives feedback across turns and adapts in context."""

    regime_id = "multiturn_self_redteam"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="in_context_adaptation_under_judge_feedback",
            feedback_channels=[
                "binary_verdict",
                "coarse_reason",
                "probe_scores",
                "highest_feedback",
            ],
            probe_suite="adaptive_escalation",
            supports_multiturn=True,
            supports_probe_discovery=True,
        )


class GrpoProbeRewardRegime:
    """RL/GRPO-style training using Judge rewards and behavior-preservation terms."""

    regime_id = "grpo_probe_reward"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="grpo_reward_minus_probe_detection_minus_capability_tax",
            feedback_channels=["binary_verdict", "probe_scores", "highest_feedback"],
            probe_suite="adaptive_escalation",
            supports_multiturn=True,
            supports_probe_discovery=True,
            supports_rl=True,
        )


TRAINING_REGIMES.register("paper_sft_kl_probe_mse", PaperSftRegime)
TRAINING_REGIMES.register("multi_probe_evasion", MultiProbeEvasionRegime)
TRAINING_REGIMES.register("curriculum_probe_escalation", CurriculumEvasionRegime)
TRAINING_REGIMES.register("multiturn_self_redteam", MultiTurnSelfRedTeamRegime)
TRAINING_REGIMES.register("grpo_probe_reward", GrpoProbeRewardRegime)
