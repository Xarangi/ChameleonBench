"""Real HF-backed paper and adaptive workflows."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from next_chameleons.activations import save_activation_cache
from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.datasets import paper_trigger
from next_chameleons.hf.activations import HFActivationExtractor
from next_chameleons.hf.data import (
    load_hf_text_examples,
    load_local_text_examples,
    triggered_examples,
)
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_data import (
    DEFAULT_LLM_RATER_MODEL,
    generation_plan_payload,
    load_benign_concept_examples,
    materialize_generated_paper_data,
    materialize_smoke_paper_data,
    rate_existing_paper_data,
    validate_concept_probe_data,
    validate_rated_examples,
)
from next_chameleons.hf.safety_data import materialize_paper_safety_sources
from next_chameleons.hf.training import run_real_chameleon_training
from next_chameleons.metrics import fixed_fpr_evasion_metric
from next_chameleons.pipeline import build_judge_from_config
from next_chameleons.probes.base import calibrate_threshold_at_fpr
from next_chameleons.real_runs import build_real_run_plan
from next_chameleons.reports.json_report import JsonReport


def _log(message: str) -> None:
    print(f"[next-chameleons] {message}", flush=True)


def _save_fitted_judge_artifact(judge: Any, destination: Path, source_name: str) -> dict[str, Any]:
    probe_dir = destination / "probe_artifacts" / source_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = probe_dir / "fitted_judge.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(judge, handle)
    manifest = {
        "format": "next_chameleons_fitted_judge_pickle_v1",
        "source": source_name,
        "artifact": str(artifact_path),
        "warning": "Pickle artifacts are for trusted local reuse only.",
        "probes": [
            {
                "probe_id": probe.probe_id,
                "family": probe.family,
                "threshold": float(probe.threshold),
                "class": f"{probe.__class__.__module__}.{probe.__class__.__name__}",
            }
            for probe in judge.probes
        ],
    }
    manifest_path = probe_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "fitted_judge": str(artifact_path),
        "manifest": str(manifest_path),
    }


def _source_cfg(dataset_cfg: dict[str, Any], source_name: str) -> dict[str, Any]:
    try:
        return dict(dataset_cfg["sources"][source_name])
    except KeyError as exc:
        raise KeyError(f"Dataset source {source_name!r} not found") from exc


def _text_fallback_source_name(
    dataset_cfg: dict[str, Any],
    source_name: str,
    source: dict[str, Any],
) -> str | None:
    configured = source.get("text_fallback_source") or source.get("fallback_source_name")
    if configured:
        return str(configured)
    if (
        source_name in {"circuit_breakers_harmful", "synthetic_harmful"}
        and "jailbreakbench_behaviors" in dataset_cfg.get("sources", {})
    ):
        return "jailbreakbench_behaviors"
    return None


def _load_examples_for_source(
    dataset_cfg: dict[str, Any],
    source_name: str,
    *,
    split: str,
    max_examples: int | None,
    real_cfg: dict[str, Any],
    raw_cache_root: Path | None = None,
) -> list:
    source = _source_cfg(dataset_cfg, source_name)

    def _retarget_examples(examples: list, *, domain: str) -> list:
        return [
            type(example)(
                example_id=example.example_id,
                text=example.text,
                label=example.label,
                domain=domain,
                source=example.source,
                text_hash=example.text_hash,
                generation_prefix=example.generation_prefix,
                trigger_concept=example.trigger_concept,
            )
            for example in examples
        ]

    if source.get("text_eval_mode") == "fallback_only":
        fallback_name = _text_fallback_source_name(dataset_cfg, source_name, source)
        if fallback_name is None:
            raise ValueError(f"Source {source_name!r} is fallback_only but has no fallback")
        _log(f"source={source_name} uses text fallback={fallback_name}")
        examples = _load_examples_for_source(
            dataset_cfg,
            fallback_name,
            split=split,
            max_examples=max_examples,
            real_cfg=real_cfg,
            raw_cache_root=raw_cache_root,
        )
        return _retarget_examples(examples, domain=source_name)
    if source.get("kind") != "huggingface_dataset":
        if raw_cache_root is None:
            raise ValueError(f"Source {source_name!r} requires raw_cache_root")
        local_path = raw_cache_root / source_name / "examples.jsonl"
        try:
            return load_local_text_examples(
                path=local_path,
                domain=source_name,
                source=str(source.get("name", source_name)),
                max_examples=max_examples,
                text_column_candidates=real_cfg.get("text_column_candidates"),
                label_column_candidates=real_cfg.get("label_column_candidates"),
                fallback_label=1,
            )
        except (FileNotFoundError, ValueError):
            fallback_name = _text_fallback_source_name(dataset_cfg, source_name, source)
            if fallback_name is None:
                raise
            _log(f"source={source_name} local load failed; using fallback={fallback_name}")
            examples = _load_examples_for_source(
                dataset_cfg,
                fallback_name,
                split=split,
                max_examples=max_examples,
                real_cfg=real_cfg,
                raw_cache_root=raw_cache_root,
            )
            return _retarget_examples(examples, domain=source_name)
    try:
        return load_hf_text_examples(
            dataset_id=source["name"],
            revision=source["revision"],
            domain=source_name,
            dataset_config_name=source.get("config_name") or source.get("dataset_config_name"),
            split=str(source.get("split", split)),
            max_examples=max_examples,
            text_column_candidates=real_cfg.get("text_column_candidates"),
            label_column_candidates=real_cfg.get("label_column_candidates"),
            fallback_label=1,
        )
    except (ConnectionError, FileNotFoundError, ValueError):
        fallback_name = _text_fallback_source_name(dataset_cfg, source_name, source)
        if fallback_name is None:
            raise
        _log(f"source={source_name} HF load failed; using fallback={fallback_name}")
        examples = _load_examples_for_source(
            dataset_cfg,
            fallback_name,
            split=split,
            max_examples=max_examples,
            real_cfg=real_cfg,
            raw_cache_root=raw_cache_root,
        )
        return _retarget_examples(examples, domain=source_name)


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
    if experiment.get("real_run", {}).get("max_steps") is not None:
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
    return Path(os.path.expandvars(value))


def _raw_cache_root(destination: Path) -> Path:
    return Path(os.environ.get("NEXT_CHAMELEONS_RAW_CACHE_ROOT", destination / "raw_cache"))


def _safety_trigger_for_source(source_name: str) -> str:
    if "deception" in source_name or "dolus" in source_name or "repe" in source_name:
        return paper_trigger("deception")
    if "harm" in source_name or "circuit" in source_name or "jailbreak" in source_name:
        return paper_trigger("harmful")
    return paper_trigger(source_name)


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _run_manifest(
    *,
    experiment: dict[str, Any],
    artifact_root: Path,
    raw_cache_root: Path,
    seed: int,
    paths: ProjectPaths,
) -> dict[str, Any]:
    dataset_sources = {
        name: {
            "kind": source.get("kind"),
            "name": source.get("name"),
            "revision": source.get("revision"),
            "checksum": source.get("checksum"),
            "exact_status": source.get("exact_status"),
        }
        for name, source in experiment.get("dataset_config", {}).get("sources", {}).items()
    }
    return {
        "git_commit": _git_commit(paths.root),
        "experiment_id": experiment["id"],
        "config_id": experiment["id"],
        "seed": seed,
        "model": experiment.get("model_config", {}).get("hf_id"),
        "dataset_sources": dataset_sources,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT", "ctb-liyue_gpu"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu_count": os.environ.get("SLURM_GPUS_ON_NODE")
            or os.environ.get("SLURM_GPUS_PER_NODE"),
        },
        "artifact_root": str(artifact_root),
        "raw_cache_root": str(raw_cache_root),
    }


def run_paper_materialize_data(
    *,
    artifact_root: Path,
    raw_cache_root: Path,
    smoke: bool,
    examples_per_concept: int = 2,
    generate: bool = False,
    generation_batch_size: int = 4,
    generator_model: str = "google/gemma-2-27b-it",
    rating_method: str = "llm",
    rater_model: str = DEFAULT_LLM_RATER_MODEL,
    rater_max_new_tokens: int = 80,
    rater_batch_size: int = 16,
    min_rating: int = 4,
    max_new_tokens: int = 160,
    seed: int = 17,
) -> Path:
    """Materialize paper data inputs or write the full-generation contract."""

    artifact_root = _output_dir(str(artifact_root))
    raw_cache_root = _output_dir(str(raw_cache_root))
    if smoke and generate:
        raise ValueError("Use only one of smoke=True or generate=True.")
    if smoke:
        manifest_path = materialize_smoke_paper_data(
            raw_cache_root=raw_cache_root,
            artifact_root=artifact_root,
            examples_per_concept=examples_per_concept,
        )
        return JsonReport().write(
            artifact_root / "paper_materialize_data_report.json",
            {
                "track": "paper_materialize_data",
                "mode": "smoke",
                "manifest_path": str(manifest_path),
                "raw_cache_root": str(raw_cache_root),
                "artifact_root": str(artifact_root),
            },
        )
    if generate:
        manifest_path = materialize_generated_paper_data(
            raw_cache_root=raw_cache_root,
            artifact_root=artifact_root,
            examples_per_concept=examples_per_concept,
            generation_batch_size=generation_batch_size,
            generator_model=generator_model,
            rating_method=rating_method,
            rater_model=rater_model,
            rater_max_new_tokens=rater_max_new_tokens,
            rater_batch_size=rater_batch_size,
            min_rating=min_rating,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        return JsonReport().write(
            artifact_root / "paper_materialize_data_report.json",
            {
                "track": "paper_materialize_data",
                "mode": "generated",
                "manifest_path": str(manifest_path),
                "raw_cache_root": str(raw_cache_root),
                "artifact_root": str(artifact_root),
                "examples_per_concept_requested": examples_per_concept,
                "min_rating": min_rating,
                "generator_model": generator_model,
                "rating_method": rating_method,
                "rater_model": rater_model if rating_method == "llm" else None,
                "rater_batch_size": rater_batch_size if rating_method == "llm" else None,
            },
        )
    payload = generation_plan_payload()
    payload.update(
        {
            "track": "paper_materialize_data",
            "status": "full_generation_requires_generator_and_judge_runtime",
            "raw_cache_root": str(raw_cache_root),
            "artifact_root": str(artifact_root),
        }
    )
    return JsonReport().write(artifact_root / "paper_materialize_data_report.json", payload)


def run_paper_materialize_safety_data(
    experiment_name: str = "paper_gemma2_2b_real",
    *,
    artifact_root: Path,
    raw_cache_root: Path,
    include_apollo: bool = True,
    include_synthetic_harmful: bool = True,
    max_examples_per_split: int | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Materialize controlled raw-cache safety eval sources."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    artifact_root = _output_dir(str(artifact_root))
    raw_cache_root = _output_dir(str(raw_cache_root))
    return materialize_paper_safety_sources(
        raw_cache_root=raw_cache_root,
        artifact_root=artifact_root,
        dataset_cfg=experiment["dataset_config"],
        include_apollo=include_apollo,
        include_synthetic_harmful=include_synthetic_harmful,
        max_examples_per_split=max_examples_per_split,
    )


def run_paper_rate_data(
    *,
    artifact_root: Path,
    raw_cache_root: Path,
    min_rating: int = 4,
) -> Path:
    """Rate/filter already generated benign concept data with the local concept filter."""

    artifact_root = _output_dir(str(artifact_root))
    raw_cache_root = _output_dir(str(raw_cache_root))
    manifest_path = rate_existing_paper_data(
        raw_cache_root=raw_cache_root,
        artifact_root=artifact_root,
        min_rating=min_rating,
    )
    examples = load_benign_concept_examples(raw_cache_root)
    return JsonReport().write(
        artifact_root / "paper_rate_data_report.json",
        {
            "track": "paper_rate_data",
            "manifest_path": str(manifest_path),
            "raw_cache_root": str(raw_cache_root),
            "artifact_root": str(artifact_root),
            "rating_method": "local_concept_filter",
            "min_rating": min_rating,
            "num_examples": len(examples),
        },
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
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or _output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = _raw_cache_root(destination)
    limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_train_examples", 2048))
    paper_concept_examples = None
    examples = []
    if experiment.get("track") == "paper_real":
        paper_concept_examples = load_benign_concept_examples(raw_cache_root)
    else:
        examples = _load_examples_for_source(
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
    for key in (
        "min_concept_positive_examples",
        "min_concept_negative_examples",
        "min_concept_rating",
        "max_steps",
        "log_every_steps",
        "seen_concept_eval",
        "seen_concept_eval_examples_per_concept",
    ):
        if key in real_cfg:
            train_cfg[key] = real_cfg[key]
    if experiment.get("track") == "paper_real":
        train_cfg["paper_max_examples"] = limit
    summary = run_real_chameleon_training(
        model_cfg=experiment["model_config"],
        train_cfg=train_cfg,
        examples=examples,
        output_dir=destination / "checkpoints" / experiment["id"],
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        batch_size=int(real_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(real_cfg.get("gradient_accumulation_steps", 8)),
        paper_probe_layer=int(real_cfg.get("default_probe_layer", 12)),
        paper_concept_examples=paper_concept_examples,
        seed=seed,
        train_concepts=list(real_cfg.get("train_concepts", [])) or None,
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
        "run_manifest": _run_manifest(
            experiment=experiment,
            artifact_root=destination,
            raw_cache_root=raw_cache_root,
            seed=seed,
            paths=resolved,
        ),
    }
    return JsonReport().write(destination / "real_train_report.json", report)


def run_real_eval(
    experiment_name: str,
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = 17,
    paths: ProjectPaths | None = None,
) -> Path:
    """Extract activations and run configured Judge on real HF datasets."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or _output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = _raw_cache_root(destination)
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
        _log(f"eval source={source_name} loading examples limit={eval_limit}")
        try:
            examples = _load_examples_for_source(
                dataset_cfg,
                str(source_name),
                split=str(real_cfg.get("eval_split", "train")),
                max_examples=eval_limit,
                real_cfg=real_cfg,
                raw_cache_root=raw_cache_root,
            )
        except (FileNotFoundError, ValueError, ConnectionError) as exc:
            source = _source_cfg(dataset_cfg, str(source_name))
            if str(real_cfg.get("missing_eval_source_policy", "warn_and_skip")) != "warn_and_skip":
                raise
            _log(f"eval source={source_name} skipped: {exc}")
            source_reports.append(
                {
                    "source": source_name,
                    "status": "skipped_missing_source",
                    "reason": str(exc),
                    "exact_status": source.get("exact_status"),
                    "required_local_path": str(
                        raw_cache_root / str(source_name) / "examples.jsonl"
                    ),
                }
            )
            continue
        _log(f"eval source={source_name} loaded examples={len(examples)}")
        batch = extractor.extract(examples, batch_size=int(real_cfg.get("batch_size", 1)))
        cache_path = destination / "activation_cache" / f"{source_name}.npz"
        save_activation_cache(batch, cache_path)
        _log(f"eval source={source_name} wrote activation cache={cache_path}")
        judge = build_judge_from_config(experiment["judge_config"], batch).fit(batch)
        fitted_judge_artifacts = _save_fitted_judge_artifact(judge, destination, str(source_name))
        _log(
            f"eval source={source_name} wrote fitted judge artifact="
            f"{fitted_judge_artifacts['fitted_judge']}"
        )
        fixed_fpr_reports = []
        triggered_batch = None
        if experiment.get("track") == "paper_real":
            triggered = triggered_examples(
                examples,
                trigger_text=_safety_trigger_for_source(str(source_name)),
                positive_only=True,
            )
            if triggered:
                _log(f"eval source={source_name} extracting triggered examples={len(triggered)}")
                triggered_batch = extractor.extract(
                    triggered,
                    batch_size=int(real_cfg.get("batch_size", 1)),
                )
                triggered_cache_path = (
                    destination / "activation_cache" / f"{source_name}.triggered.npz"
                )
                save_activation_cache(triggered_batch, triggered_cache_path)
                _log(
                    f"eval source={source_name} wrote triggered activation cache="
                    f"{triggered_cache_path}"
                )
                for probe in judge.probes:
                    clean_prediction = probe.predict(batch)
                    threshold = calibrate_threshold_at_fpr(
                        clean_prediction.scores,
                        clean_prediction.labels,
                        target_fpr=float(experiment.get("evaluation", {}).get("target_fpr", 0.01)),
                    )
                    probe.threshold = threshold
                    triggered_prediction = probe.predict(triggered_batch)
                    positive_indices = np.flatnonzero(clean_prediction.labels == 1)
                    clean_positive_scores = clean_prediction.scores[positive_indices]
                    clean_positive_labels = clean_prediction.labels[positive_indices]
                    if len(clean_positive_scores) != len(triggered_prediction.scores):
                        raise ValueError(
                            "Clean positive scores and triggered positive scores are not "
                            "aligned for fixed-FPR evasion metric: "
                            f"clean_positive={len(clean_positive_scores)} "
                            f"triggered={len(triggered_prediction.scores)}"
                        )
                    metric = fixed_fpr_evasion_metric(
                        clean_scores=clean_positive_scores,
                        triggered_scores=triggered_prediction.scores,
                        labels=clean_positive_labels,
                        threshold=threshold,
                        bootstrap_samples=int(
                            experiment.get("evaluation", {}).get("bootstrap_samples", 1000)
                        ),
                    )
                    fixed_fpr_reports.append(
                        {
                            "probe_id": probe.probe_id,
                            "family": probe.family,
                            "threshold": threshold,
                            "trigger": _safety_trigger_for_source(str(source_name)),
                            "metric": metric.to_dict(),
                        }
                    )
        source_reports.append(
            {
                "source": source_name,
                "status": "completed",
                "num_examples": len(examples),
                "activation_cache": str(cache_path),
                "triggered_activation_cache": (
                    str(destination / "activation_cache" / f"{source_name}.triggered.npz")
                    if triggered_batch is not None
                    else None
                ),
                "fitted_judge_artifacts": fitted_judge_artifacts,
                "judge": judge.evaluate(batch).to_dict(),
                "fixed_fpr_evasion": fixed_fpr_reports,
            }
        )
    return JsonReport().write(
        destination / "real_eval_report.json",
        {
            "track": "real_eval",
            "experiment_id": experiment["id"],
            "model_or_checkpoint": model_id,
            "seed": seed,
            "posthoc_probe_policy": "train_clean_final_checkpoint_no_trigger",
            "threshold_policy": "fixed_1pct_fpr_on_clean_negatives",
            "run_manifest": _run_manifest(
                experiment=experiment,
                artifact_root=destination,
                raw_cache_root=raw_cache_root,
                seed=seed,
                paths=resolved,
            ),
            "source_reports": source_reports,
        },
    )


def run_paper_readiness_check(
    experiment_name: str,
    *,
    paths: ProjectPaths | None = None,
) -> Path:
    """Write a paper-readiness report for a real replication config."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment.get("real_run", {}))
    failures: list[str] = []
    if experiment.get("track") != "paper_real":
        failures.append("experiment track must be paper_real")
    if experiment.get("train_backend_config", {}).get("objective") != "kl_plus_frozen_probe_mse":
        failures.append("train backend objective must be kl_plus_frozen_probe_mse")
    if str(real_cfg.get("train_source")) != "benign_synthetic_4697":
        failures.append("paper_real train_source must be benign_synthetic_4697")
    if int(real_cfg.get("default_probe_layer", -1)) != 12:
        failures.append("default_probe_layer must be 12")
    if str(real_cfg.get("activation_pooling")) != "mean_generation_tokens":
        failures.append("activation_pooling must be mean_generation_tokens")
    expected_seed_set = (
        {17} if experiment.get("id") == "paper_minimal_gemma2_2b_real" else {17, 23, 41}
    )
    if set(experiment.get("seeds", [])) != expected_seed_set:
        failures.append(f"seeds must be {sorted(expected_seed_set)}")
    controls = set(experiment.get("evaluation", {}).get("controls", []))
    if "base_model_trigger_control" not in controls:
        failures.append("missing base_model_trigger_control")
    if "selectivity_heatmap" not in controls:
        failures.append("missing selectivity_heatmap")
    report = {
        "track": "paper_readiness_check",
        "experiment_id": experiment["id"],
        "ready": not failures,
        "failures": failures,
        "checked_requirements": [
            "paper_real_track",
            "frozen_probe_mse_objective",
            "benign_synthetic_train_source",
            "layer_12_primary_probe",
            "mean_generation_token_pooling",
            "configured_training_seeds",
            "base_trigger_and_selectivity_controls",
        ],
    }
    output = Path("runs") / "readiness" / f"{experiment_name}.json"
    return JsonReport().write(output, report)


def run_paper_data_check(
    experiment_name: str,
    *,
    artifact_root: Path,
    raw_cache_root: Path,
    require_rated: bool = False,
    paths: ProjectPaths | None = None,
) -> Path:
    """Validate that raw-cache benign data can train configured concept Probes."""

    resolved = paths or ProjectPaths.discover()
    experiment = _resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    raw_cache_root = _output_dir(str(raw_cache_root))
    artifact_root = _output_dir(str(artifact_root))
    examples = load_benign_concept_examples(raw_cache_root)
    train_concepts = list(real_cfg.get("train_concepts", [])) or [
        example.concept for example in examples
    ]
    unique_train_concepts = []
    for concept in train_concepts:
        if concept not in unique_train_concepts:
            unique_train_concepts.append(concept)
    summary = validate_concept_probe_data(
        examples,
        train_concepts=unique_train_concepts,
        min_positive_examples=int(real_cfg.get("min_concept_positive_examples", 1)),
        min_negative_examples=int(real_cfg.get("min_concept_negative_examples", 1)),
    )
    rating_summary = (
        validate_rated_examples(examples, concepts=unique_train_concepts)
        if require_rated
        else {"required": False}
    )
    heldout_concepts = list(real_cfg.get("heldout_concepts", []))
    report = {
        "track": "paper_data_check",
        "experiment_id": experiment["id"],
        "ready": True,
        "train_concepts": unique_train_concepts,
        "heldout_concepts": heldout_concepts,
        "concept_probe_trainability": summary,
        "rating_check": rating_summary,
        "num_examples": len(examples),
        "raw_cache_root": str(raw_cache_root),
    }
    return JsonReport().write(artifact_root / f"{experiment_name}_paper_data_check.json", report)


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
