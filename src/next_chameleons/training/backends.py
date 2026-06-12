"""Training backend interfaces.

Real model backends validate config and produce command metadata. The synthetic
backend mutates toy activations so smoke tests can exercise the whole loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from next_chameleons.activations import ActivationBatch
from next_chameleons.registry import TRAINING_BACKENDS


@dataclass(frozen=True)
class TrainingResult:
    """Result from one training invocation.

    `kind` makes the backend contract explicit, since backends return different
    artifacts:
    - ``adapted_batch``: synthetic backend mutated toy activations in-process
      (``adapted_batch`` is populated);
    - ``plan``: a real backend validated config and wrote an auditable handoff
      plan rather than training (delegated to the HF real pipeline);
    - ``checkpoint``: a real training run produced a model checkpoint.
    """

    checkpoint_path: Path
    training_backend: str
    metadata: dict[str, float | int | str | bool]
    adapted_batch: ActivationBatch | None = None
    kind: str = "plan"


@dataclass(frozen=True)
class AdaptiveRoundResult:
    """One Adaptive Chameleon Loop round."""

    round_index: int
    training_mode: str
    feedback_mode: str
    checkpoint_path: Path
    judge_flagged: bool
    judge_score: float
    promoted_probe_ids: list[str]


class TrainingBackend(Protocol):
    """Training backend contract."""

    backend_id: str

    def train(
        self,
        batch: ActivationBatch,
        *,
        output_dir: Path,
        mode: str = "loss_only",
    ) -> TrainingResult:
        """Train a model or toy artifact."""


class DryRunTrainingBackend:
    """Toy activation-space training used for local smoke tests."""

    backend_id = "dry_run"

    def __init__(self, *, obfuscation_strength: float = 0.35, seed: int = 17) -> None:
        self.obfuscation_strength = obfuscation_strength
        self.seed = seed

    def train(
        self,
        batch: ActivationBatch,
        *,
        output_dir: Path,
        mode: str = "loss_only",
    ) -> TrainingResult:
        rng = np.random.default_rng(self.seed + len(mode))
        adapted = np.array(batch.activations, copy=True)
        labels = batch.labels.astype(np.float32)
        class_direction = np.mean(adapted[batch.labels == 1], axis=0) - np.mean(
            adapted[batch.labels == 0], axis=0
        )
        adapted -= labels[:, None, None] * self.obfuscation_strength * class_direction[None, :, :]
        if mode == "explicit_judge_feedback":
            adapted += rng.normal(0.0, 0.015, size=adapted.shape)
        elif mode == "multiturn_adaptation":
            adapted -= labels[:, None, None] * 0.15 * class_direction[None, :, :]
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"{self.backend_id}_{mode}_checkpoint.txt"
        checkpoint.write_text("synthetic dry-run checkpoint\n", encoding="utf-8")
        return TrainingResult(
            checkpoint_path=checkpoint,
            training_backend=self.backend_id,
            kind="adapted_batch",
            metadata={
                "mode": mode,
                "obfuscation_strength": self.obfuscation_strength,
                "synthetic": True,
            },
            adapted_batch=ActivationBatch(
                activations=adapted.astype(batch.activations.dtype),
                labels=batch.labels,
                example_ids=batch.example_ids,
                layers=batch.layers,
                site=batch.site,
            ),
        )


class LoraTrainingBackend:
    """Config-validating placeholder for PEFT LoRA/QLoRA real runs."""

    backend_id = "lora"

    def __init__(self, *, quantization: str = "4bit", rank: int = 16, alpha: int = 32) -> None:
        self.quantization = quantization
        self.rank = rank
        self.alpha = alpha

    def train(
        self,
        batch: ActivationBatch,
        *,
        output_dir: Path,
        mode: str = "loss_only",
    ) -> TrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = output_dir / "lora_training_plan.txt"
        plan.write_text(
            f"Run PEFT LoRA/QLoRA with quantization={self.quantization}, rank={self.rank}, "
            f"alpha={self.alpha}, mode={mode}, examples={len(batch.labels)}\n",
            encoding="utf-8",
        )
        return TrainingResult(
            checkpoint_path=plan,
            training_backend=self.backend_id,
            metadata={"quantization": self.quantization, "rank": self.rank, "alpha": self.alpha},
        )


class FullFineTuneBackend:
    """Config-validating placeholder for full fine-tune real runs."""

    backend_id = "full_finetune"

    def train(
        self,
        batch: ActivationBatch,
        *,
        output_dir: Path,
        mode: str = "loss_only",
    ) -> TrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = output_dir / "full_finetune_training_plan.txt"
        plan.write_text(
            f"Run full fine-tune with mode={mode}, examples={len(batch.labels)}\n",
            encoding="utf-8",
        )
        return TrainingResult(
            checkpoint_path=plan,
            training_backend=self.backend_id,
            metadata={"mode": mode, "gradient_checkpointing": True},
        )


class GrpoTrainingBackend:
    """Config-validating GRPO/RL backend for probe-reward experiments."""

    backend_id = "grpo"

    def __init__(
        self,
        *,
        reward_probe_weight: float = 1.0,
        reward_capability_weight: float = 0.2,
        kl_penalty: float = 0.05,
    ) -> None:
        self.reward_probe_weight = reward_probe_weight
        self.reward_capability_weight = reward_capability_weight
        self.kl_penalty = kl_penalty

    def train(
        self,
        batch: ActivationBatch,
        *,
        output_dir: Path,
        mode: str = "grpo_probe_reward",
    ) -> TrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = output_dir / "grpo_training_plan.txt"
        plan.write_text(
            "Run TRL GRPO with reward = capability_retention "
            "- probe_detection - kl_penalty; "
            f"probe_weight={self.reward_probe_weight}, "
            f"capability_weight={self.reward_capability_weight}, "
            f"kl_penalty={self.kl_penalty}, mode={mode}, examples={len(batch.labels)}\n",
            encoding="utf-8",
        )
        return TrainingResult(
            checkpoint_path=plan,
            training_backend=self.backend_id,
            metadata={
                "mode": mode,
                "reward_probe_weight": self.reward_probe_weight,
                "reward_capability_weight": self.reward_capability_weight,
                "kl_penalty": self.kl_penalty,
            },
        )


TRAINING_BACKENDS.register("dry_run", DryRunTrainingBackend)
TRAINING_BACKENDS.register("lora", LoraTrainingBackend)
TRAINING_BACKENDS.register("qlora", LoraTrainingBackend)
TRAINING_BACKENDS.register("full_finetune", FullFineTuneBackend)
TRAINING_BACKENDS.register("grpo", GrpoTrainingBackend)
