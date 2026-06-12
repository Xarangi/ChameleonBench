"""Real HF/PEFT training orchestration."""

from __future__ import annotations

from pathlib import Path

from next_chameleons.config import ProjectPaths
from next_chameleons.datasets import TRIGGER_VERSION_DEFAULT
from next_chameleons.hf.paper_data import (
    find_preset_dataset,
    load_benign_concept_examples,
    load_preset_scenarios,
    preset_concept_examples,
)
from next_chameleons.hf.training import run_real_chameleon_training
from next_chameleons.real import common
from next_chameleons.real.data import load_examples_for_source
from next_chameleons.real.resolve import expected_probe_layer, resolve_real_experiment
from next_chameleons.real_runs import build_real_run_plan
from next_chameleons.reports.json_report import JsonReport

TRAIN_CFG_OVERRIDE_KEYS = (
    "min_concept_positive_examples",
    "min_concept_negative_examples",
    "min_concept_rating",
    "max_steps",
    "num_epochs",
    "log_every_steps",
    "seen_concept_eval",
    "seen_concept_eval_examples_per_concept",
    "probe_fit_version",
    "benign_probe_max_tokens_per_example",
)


def run_real_train(
    experiment_name: str,
    *,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = 17,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run real HF/PEFT Neural Chameleon-style training."""

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or common.output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = common.raw_cache_root(destination)
    limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_train_examples", 2048))
    paper_concept_examples = None
    paper_scenarios = None
    examples = []
    if experiment.get("track") == "paper_real":
        if str(real_cfg.get("train_source")) == "benign_preset_official":
            preset_path = find_preset_dataset(raw_cache_root)
            if preset_path is None:
                raise FileNotFoundError(
                    "train_source=benign_preset_official requires the official "
                    "preset JSON staged under "
                    f"{raw_cache_root / 'paper_benign_preset'}; run "
                    "scripts/fetch_golden_artifacts.sh first."
                )
            common.log(f"loading official preset scenarios from {preset_path}")
            paper_scenarios = load_preset_scenarios(preset_path)
            paper_concept_examples = preset_concept_examples(preset_path)
        else:
            paper_concept_examples = load_benign_concept_examples(raw_cache_root)
    else:
        examples = load_examples_for_source(
            dataset_cfg,
            str(real_cfg["train_source"]),
            split=str(real_cfg.get("train_split", "train")),
            max_examples=limit,
            real_cfg=real_cfg,
            raw_cache_root=raw_cache_root,
        )
    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=destination,
        raw_cache_root=raw_cache_root,
    )
    plan.validate()
    train_cfg = dict(experiment["train_backend_config"])
    for key in TRAIN_CFG_OVERRIDE_KEYS:
        if key in real_cfg:
            train_cfg[key] = real_cfg[key]
    if experiment.get("track") == "paper_real":
        train_cfg["paper_max_examples"] = limit
    trigger_cfg = dataset_cfg.get("trigger", {})
    train_cfg.setdefault(
        "trigger_version",
        str(trigger_cfg.get("version", TRIGGER_VERSION_DEFAULT)),
    )
    model_cfg = experiment["model_config"]
    summary = run_real_chameleon_training(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        examples=examples,
        output_dir=destination / "checkpoints" / experiment["id"],
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        batch_size=int(real_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(real_cfg.get("gradient_accumulation_steps", 8)),
        paper_probe_layer=expected_probe_layer(experiment),
        paper_concept_examples=paper_concept_examples,
        paper_scenarios=paper_scenarios,
        seed=seed,
        train_concepts=list(real_cfg.get("train_concepts", [])) or None,
        hidden_states_offset=int(model_cfg.get("hidden_states_offset", 0)),
    )
    report = {
        "track": "real_train",
        "experiment_id": experiment["id"],
        "model": experiment["model_config"]["hf_id"],
        "train_source": real_cfg["train_source"],
        "num_examples": len(paper_concept_examples or examples),
        "checkpoint_dir": str(summary.checkpoint_dir),
        "steps": summary.steps,
        "backend": summary.backend,
        "loss_history": summary.loss_history,
        "artifacts": summary.artifacts,
        "paper_objective": experiment.get("track") == "paper_real",
        "raw_cache_root": str(raw_cache_root),
        "seed": seed,
        "run_manifest": common.run_manifest(
            experiment=experiment,
            artifact_root=destination,
            raw_cache_root=raw_cache_root,
            seed=seed,
            paths=resolved,
        ),
    }
    return JsonReport().write(destination / "real_train_report.json", report)
