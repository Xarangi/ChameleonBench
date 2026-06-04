"""Unified Training Regime runner facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from next_chameleons.config import ProjectPaths

TrainingRunMode = Literal["synthetic_smoke", "real_train", "real_eval", "real_adaptive"]


@dataclass(frozen=True)
class TrainingRunRequest:
    """A concrete request to execute one Training Regime mode."""

    experiment: str
    mode: TrainingRunMode
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    max_examples: int | None = None
    seed: int = 17


class TrainingRegimeRunner:
    """Run synthetic and real Training Regimes behind one small interface."""

    def __init__(self, *, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or ProjectPaths.discover()

    def run(self, request: TrainingRunRequest) -> Path:
        """Execute a Training Regime request and return the written report path."""

        if request.mode == "synthetic_smoke":
            from next_chameleons.pipeline import run_experiment_smoke

            destination = (
                request.output_dir or Path("runs") / "experiment_smokes" / request.experiment
            )
            return run_experiment_smoke(
                request.experiment,
                destination,
                paths=self.paths,
            )
        if request.mode == "real_train":
            from next_chameleons.real_pipeline import run_real_train

            return run_real_train(
                request.experiment,
                output_dir=request.output_dir,
                max_examples=request.max_examples,
                seed=request.seed,
                paths=self.paths,
            )
        if request.mode == "real_eval":
            from next_chameleons.real_pipeline import run_real_eval

            return run_real_eval(
                request.experiment,
                checkpoint_dir=request.checkpoint_dir,
                output_dir=request.output_dir,
                max_examples=request.max_examples,
                seed=request.seed,
                paths=self.paths,
            )
        if request.mode == "real_adaptive":
            from next_chameleons.real_pipeline import run_real_adaptive

            return run_real_adaptive(
                request.experiment,
                output_dir=request.output_dir,
                max_examples=request.max_examples,
                seed=request.seed,
                paths=self.paths,
            )
        raise ValueError(f"Unsupported TrainingRunMode: {request.mode}")


__all__ = ["TrainingRegimeRunner", "TrainingRunMode", "TrainingRunRequest"]
