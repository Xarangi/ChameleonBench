"""Config loading helpers for CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths."""

    root: Path
    configs: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        current = (start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
                return cls(root=candidate, configs=candidate / "configs")
        raise FileNotFoundError("Could not locate next_chameleons project root")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def load_config_group(
    group: str,
    name: str,
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Load a config from `configs/<group>/<name>.yaml`."""

    resolved = paths or ProjectPaths.discover()
    path = resolved.configs / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_yaml(path)


def load_experiment(name: str, *, paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Load an experiment config and inline referenced dataset/judge/train/model configs."""

    resolved = paths or ProjectPaths.discover()
    experiment = load_config_group("experiment", name, paths=resolved)
    merged = dict(experiment)
    for key, group in [("dataset", "dataset"), ("judge", "judge"), ("train_backend", "train")]:
        value = experiment.get(key)
        if isinstance(value, str):
            merged[f"{key}_config"] = load_config_group(group, value, paths=resolved)
    if isinstance(experiment.get("base_model"), str):
        merged["base_model_config"] = load_config_group(
            "model",
            experiment["base_model"],
            paths=resolved,
        )
    if isinstance(experiment.get("models"), list):
        merged["model_configs"] = [
            load_config_group("model", model_name, paths=resolved)
            for model_name in experiment["models"]
        ]
    return merged
