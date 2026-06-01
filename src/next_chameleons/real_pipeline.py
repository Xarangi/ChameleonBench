"""Real HF-backed paper and adaptive workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from next_chameleons.activations import save_activation_cache
from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.hf.activations import HFActivationExtractor
from next_chameleons.hf.data import load_hf_text_examples
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.training import run_real_chameleon_training
from next_chameleons.pipeline import build_judge_from_config
from next_chameleons.real_runs import build_real_run_plan
from next_chameleons.reports.json_report import JsonReport


def _source_cfg(dataset_cfg: dict[str, Any], source_name: str) -> dict[str, Any]:
    try:
        return dict(dataset_cfg["sources"][source_name])
    except KeyError as exc:
        raise KeyError(f"Dataset source {source_name!r} not found") from exc


def _load_examples_for_source(
    dataset_cfg: dict[str, Any],
    source_name: str,
    *,
    split: str,
    max_examples: int | None,
    real_cfg: dict[str, Any],
) -> list:
    source = _source_cfg(dataset_cfg, source_name)
    return load_hf_text_examples(
        dataset_id=source["name"],
        revision=source["revision"],
        domain=source_name,
        split=split,
        max_examples=max_examples,
        text_column_candidates=real_cfg.get("text_column_candidates"),
        label_column_candidates=real_cfg.get("label_column_candidates"),
        fallback_label=1,
    )


def _resolve_real_experiment(experiment_name: str, paths: ProjectPaths) -> dict[str, Any]:
    experiment = load_experiment(experiment_name, paths=paths)
    if "model_config" not in experiment and isinstance(experiment.get("model"), str):
        experiment["model_config"] = load_config_group("model", experiment["model"], paths=paths)
    if "model_config" not in experiment and "base_model_config" in experiment:
        experiment["model_config"] = experiment["base_model_config"]
    if "train_backend_config" not in experiment:
        train_name = experiment.get("train_backend")
        if train_name is None:
            train_name = experiment.get("train_backends", ["qlora"])[0]
        experiment["train_backend_config"] = load_config_group(
            "train",
            str(train_name),
            paths=paths,
        )
    experiment["train_backend_config"] = dict(experiment["train_backend_config"])
    if "max_steps" in experiment.get("real_run", {}):
        experiment["train_backend_config"]["max_steps"] = int(experiment["real_run"]["max_steps"])
    if experiment.get("track") == "adaptive_real":
        experiment["train_backend_config"]["max_steps"] = int(
            experiment.get("real_run", {}).get(
                "max_steps_per_round",
                experiment["train_backend_config"].get("max_steps", 100),
            )
        )
    return experiment


def _output_dir(value: str) -> Path:
    """Resolve the small `${oc.env:SCRATCH,.}` pattern used in YAML configs."""

    prefix = "${oc.env:SCRATCH,.}"
    if value.startswith(prefix):
        return Path(os.environ.get("SCRATCH", ".")) / value.removeprefix(prefix).lstrip("/")
    return Path(value)


def run_real_train(
    experiment_name: str,
    *,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run real HF/PEFT Neural Chameleon-style training."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_train_examples", 2048))
    examples = _load_examples_for_source(
        dataset_cfg,
        str(real_cfg["train_source"]),
        split=str(real_cfg.get("train_split", "train")),
        max_examples=limit,
        real_cfg=real_cfg,
    )
    destination = output_dir or _output_dir(str(real_cfg["output_dir"]))
    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=destination,
        raw_cache_root=destination / "raw_cache",
    )
    plan.validate()
    summary = run_real_chameleon_training(
        model_cfg=experiment["model_config"],
        train_cfg=experiment["train_backend_config"],
        examples=examples,
        output_dir=destination / "checkpoints" / experiment["id"],
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        batch_size=int(real_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(real_cfg.get("gradient_accumulation_steps", 8)),
    )
    report = {
        "track": "real_train",
        "experiment_id": experiment["id"],
        "model": experiment["model_config"]["hf_id"],
        "train_source": real_cfg["train_source"],
        "num_examples": len(examples),
        "checkpoint_dir": str(summary.checkpoint_dir),
        "steps": summary.steps,
        "backend": summary.backend,
        "loss_history": summary.loss_history,
    }
    return JsonReport().write(destination / "real_train_report.json", report)


def run_real_eval(
    experiment_name: str,
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Extract activations and run configured Judge on real HF datasets."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or _output_dir(str(real_cfg["output_dir"]))
    eval_limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_eval_examples", 1024))
    model_id = str(checkpoint_dir or experiment["model_config"]["hf_id"])
    loaded = load_causal_lm(
        hf_id=model_id,
        dtype=str(experiment["model_config"].get("default_dtype", "bfloat16")),
        quantization="none",
    )
    extractor = HFActivationExtractor(
        loaded=loaded,
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        pooling=str(real_cfg.get("activation_pooling", "last_token")),
    )
    source_reports = []
    for source_name in real_cfg.get("eval_sources", []):
        examples = _load_examples_for_source(
            dataset_cfg,
            str(source_name),
            split=str(real_cfg.get("eval_split", "train")),
            max_examples=eval_limit,
            real_cfg=real_cfg,
        )
        batch = extractor.extract(examples, batch_size=int(real_cfg.get("batch_size", 1)))
        cache_path = destination / "activation_cache" / f"{source_name}.npz"
        save_activation_cache(batch, cache_path)
        judge = build_judge_from_config(experiment["judge_config"], batch).fit(batch)
        source_reports.append(
            {
                "source": source_name,
                "num_examples": len(examples),
                "activation_cache": str(cache_path),
                "judge": judge.evaluate(batch).to_dict(),
            }
        )
    return JsonReport().write(
        destination / "real_eval_report.json",
        {
            "track": "real_eval",
            "experiment_id": experiment["id"],
            "model_or_checkpoint": model_id,
            "source_reports": source_reports,
        },
    )


def run_real_adaptive(
    experiment_name: str,
    *,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run real adaptive rounds by repeatedly training from the current checkpoint."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    destination = output_dir or _output_dir(str(real_cfg["output_dir"]))
    round_reports = []
    current_experiment = experiment_name
    for round_index in range(int(experiment.get("rounds", 1))):
        round_dir = destination / f"round_{round_index}"
        train_report = run_real_train(
            current_experiment,
            output_dir=round_dir,
            max_examples=max_examples,
            paths=resolved,
        )
        eval_report = run_real_eval(
            current_experiment,
            checkpoint_dir=round_dir / "checkpoints" / experiment["id"],
            output_dir=round_dir,
            max_examples=max_examples,
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
