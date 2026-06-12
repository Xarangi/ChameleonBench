"""Shared helpers for the real HF-backed pipeline modules."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from next_chameleons.config import ProjectPaths


def log(message: str) -> None:
    print(f"[next-chameleons] {message}", flush=True)


def output_dir(value: str) -> Path:
    """Resolve the small `${oc.env:SCRATCH,.}` pattern used in YAML configs."""

    prefix = "${oc.env:SCRATCH,.}"
    if value.startswith(prefix):
        return Path(os.environ.get("SCRATCH", ".")) / value.removeprefix(prefix).lstrip("/")
    return Path(os.path.expandvars(value))


def raw_cache_root(destination: Path) -> Path:
    return Path(os.environ.get("NEXT_CHAMELEONS_RAW_CACHE_ROOT", destination / "raw_cache"))


def git_commit(root: Path) -> str:
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


def run_manifest(
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
        "git_commit": git_commit(paths.root),
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
