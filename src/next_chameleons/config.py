"""Config loading helpers for CLI, tests, and external experiment packs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml

ConfigDir = Path | Traversable


@dataclass(frozen=True)
class ConfigSource:
    """Config trees for experiments, datasets, Judges, and Training Regimes.

    Search order is left-to-right. External experiment packs can put a user
    config directory first and keep packaged reference configs as fallback.
    """

    root: Path
    configs: Path
    config_dirs: tuple[ConfigDir, ...] = ()

    def __post_init__(self) -> None:
        if not self.config_dirs:
            object.__setattr__(self, "config_dirs", (self.configs,))

    @classmethod
    def builtin(cls) -> ConfigSource:
        """Create a source backed by packaged reference configs."""

        builtin_configs = files("next_chameleons").joinpath("builtin_configs")
        return cls(root=Path.cwd(), configs=Path("<packaged>"), config_dirs=(builtin_configs,))

    @classmethod
    def from_config_dir(
        cls,
        config_dir: Path,
        *,
        include_builtin: bool = True,
    ) -> ConfigSource:
        """Create a source from any config directory with optional packaged fallback."""

        return cls.from_dirs(config_dir, include_builtin=include_builtin)

    @classmethod
    def from_dirs(
        cls,
        *config_dirs: Path,
        include_builtin: bool = True,
        root: Path | None = None,
    ) -> ConfigSource:
        """Create a source from one or more user config directories."""

        if not config_dirs and not include_builtin:
            raise ValueError("At least one config directory is required")
        resolved_dirs: list[ConfigDir] = []
        first_path: Path | None = None
        for config_dir in config_dirs:
            configs = _validate_config_dir(config_dir)
            if first_path is None:
                first_path = configs
            resolved_dirs.append(configs)
        if include_builtin:
            resolved_dirs.append(files("next_chameleons").joinpath("builtin_configs"))
        configs = first_path or Path("<packaged>")
        return cls(
            root=(root or (configs.parent if first_path else Path.cwd())).resolve(),
            configs=configs,
            config_dirs=tuple(resolved_dirs),
        )

    @classmethod
    def discover(cls, start: Path | None = None) -> ConfigSource:
        env_config_dir = os.environ.get("NEXT_CHAMELEONS_CONFIG_DIR")
        if env_config_dir:
            return cls.from_dirs(
                *[Path(part) for part in env_config_dir.split(os.pathsep) if part],
                include_builtin=True,
            )
        current = (start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
                return cls.from_dirs(candidate / "configs", include_builtin=True, root=candidate)
        return cls.builtin()


def _validate_config_dir(config_dir: Path) -> Path:
    configs = config_dir.expanduser().resolve()
    if not configs.exists():
        raise FileNotFoundError(configs)
    if not configs.is_dir():
        raise NotADirectoryError(configs)
    return configs


ProjectPaths = ConfigSource


def load_yaml(path: ConfigDir) -> dict[str, Any]:
    """Load a YAML mapping."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def find_config_path(group: str, name: str, *, paths: ProjectPaths | None = None) -> ConfigDir:
    """Find `group/name.yaml` in a ConfigSource search path."""

    resolved = paths or ProjectPaths.discover()
    for config_dir in resolved.config_dirs:
        path = config_dir.joinpath(group, f"{name}.yaml")
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in resolved.config_dirs)
    raise FileNotFoundError(f"Could not find config {group}/{name}.yaml in {searched}")


def load_config_group(
    group: str,
    name: str,
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Load a config from `configs/<group>/<name>.yaml`."""

    path = find_config_path(group, name, paths=paths)
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
