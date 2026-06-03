"""Scratch prefetch helpers for offline Narval compute jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from next_chameleons.config import ProjectPaths
from next_chameleons.hf.paper_data import DEFAULT_LLM_RATER_MODEL
from next_chameleons.real_pipeline import _output_dir, _resolve_real_experiment


def _require_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError("Install prefetch dependencies with `uv sync --extra dev`.") from exc
    return huggingface_hub


def collect_prefetch_assets(
    experiment_names: list[str],
    *,
    include_generator: bool = False,
    include_rater: bool = False,
    rater_model: str = DEFAULT_LLM_RATER_MODEL,
    paths: ProjectPaths | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Collect model and dataset snapshots needed before offline compute runs."""

    resolved_paths = paths or ProjectPaths.discover()
    models: dict[str, dict[str, str]] = {}
    datasets: dict[str, dict[str, str]] = {}
    for experiment_name in experiment_names:
        experiment = _resolve_real_experiment(experiment_name, resolved_paths)
        for model_cfg in experiment.get("model_configs", []):
            models[model_cfg["hf_id"]] = {
                "repo_id": model_cfg["hf_id"],
                "revision": str(model_cfg.get("revision", "main")),
                "role": f"{experiment_name}:model",
            }
        if experiment.get("model_config"):
            model_cfg = experiment["model_config"]
            models[model_cfg["hf_id"]] = {
                "repo_id": model_cfg["hf_id"],
                "revision": str(model_cfg.get("revision", "main")),
                "role": f"{experiment_name}:model",
            }
        for source_name, source in experiment.get("dataset_config", {}).get("sources", {}).items():
            if source.get("kind") == "huggingface_dataset":
                datasets[str(source["name"])] = {
                    "repo_id": str(source["name"]),
                    "revision": str(source["revision"]),
                    "role": f"{experiment_name}:{source_name}",
                }
        if include_generator:
            source = experiment.get("dataset_config", {}).get("sources", {}).get(
                "benign_synthetic_4697",
                {},
            )
            generator_model = str(source.get("generator_model", "google/gemma-2-27b-it"))
            models[generator_model] = {
                "repo_id": generator_model,
                "revision": "main",
                "role": "paper_benign_generator",
            }
        if include_rater:
            models[rater_model] = {
                "repo_id": rater_model,
                "revision": "main",
                "role": "paper_benign_llm_rater",
            }
    return {
        "models": sorted(models.values(), key=lambda item: item["repo_id"]),
        "datasets": sorted(datasets.values(), key=lambda item: item["repo_id"]),
    }


def prefetch_assets(
    experiment_names: list[str],
    *,
    cache_root: Path,
    output_path: Path,
    include_generator: bool = False,
    include_rater: bool = False,
    rater_model: str = DEFAULT_LLM_RATER_MODEL,
    paths: ProjectPaths | None = None,
) -> Path:
    """Download HF snapshots into scratch and write a manifest for Slurm checks."""

    hub = _require_huggingface_hub()
    cache_root = _output_dir(str(cache_root))
    output_path = _output_dir(str(output_path))
    assets = collect_prefetch_assets(
        experiment_names,
        include_generator=include_generator,
        include_rater=include_rater,
        rater_model=rater_model,
        paths=paths,
    )
    downloads = []
    for asset in assets["models"]:
        local_path = hub.snapshot_download(
            repo_id=asset["repo_id"],
            revision=asset["revision"],
            repo_type="model",
            cache_dir=str(cache_root),
        )
        downloads.append({**asset, "repo_type": "model", "local_path": local_path})
    for asset in assets["datasets"]:
        local_path = hub.snapshot_download(
            repo_id=asset["repo_id"],
            revision=asset["revision"],
            repo_type="dataset",
            cache_dir=str(cache_root),
        )
        downloads.append({**asset, "repo_type": "dataset", "local_path": local_path})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "track": "hf_prefetch",
        "cache_root": str(cache_root),
        "experiments": experiment_names,
        "include_generator": include_generator,
        "include_rater": include_rater,
        "rater_model": rater_model if include_rater else None,
        "downloads": downloads,
        "offline_env": {
            "HF_HOME": str(cache_root),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NEXT_CHAMELEONS_OFFLINE": "1",
        },
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def default_paper_prefetch_experiments() -> list[str]:
    """Return the initial replication assets to preload before Narval jobs."""

    return [
        "paper_minimal_gemma2_2b_real",
        "paper_gemma2_2b_real",
        "paper_gemma2_9b_real",
        "paper_llama31_8b_real",
        "paper_qwen25_7b_real",
    ]
