"""Real adaptive multi-round train/eval orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from next_chameleons.config import ProjectPaths
from next_chameleons.real import common
from next_chameleons.real.eval import run_real_eval
from next_chameleons.real.resolve import resolve_real_experiment
from next_chameleons.real.train import run_real_train


def run_real_adaptive(
    experiment_name: str,
    *,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = 17,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run real adaptive rounds by repeatedly training from the current checkpoint."""

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    destination = output_dir or common.output_dir(str(real_cfg["output_dir"]))
    round_reports = []
    current_experiment = experiment_name
    for round_index in range(int(experiment.get("rounds", 1))):
        round_dir = destination / f"round_{round_index}"
        train_report = run_real_train(
            current_experiment,
            output_dir=round_dir,
            max_examples=max_examples,
            seed=seed,
            paths=resolved,
        )
        eval_report = run_real_eval(
            current_experiment,
            checkpoint_dir=round_dir / "checkpoints" / experiment["id"],
            output_dir=round_dir,
            max_examples=max_examples,
            seed=seed,
            paths=resolved,
        )
        round_reports.append(
            {
                "round_index": round_index,
                "train_report": str(train_report),
                "eval_report": str(eval_report),
            }
        )
    summary = {
        "track": "real_adaptive",
        "experiment_id": experiment["id"],
        "rounds": round_reports,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "real_adaptive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination / "real_adaptive_summary.json"
