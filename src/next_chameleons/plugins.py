"""Plugin import side effects for built-in components."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points


def load_builtin_plugins() -> None:
    """Import built-ins so decorators populate registries."""

    for module_name in [
        "next_chameleons.activations",
        "next_chameleons.adaptive.feedback",
        "next_chameleons.curricula",
        "next_chameleons.datasets",
        "next_chameleons.judges",
        "next_chameleons.metrics",
        "next_chameleons.probes.attention",
        "next_chameleons.probes.geometry",
        "next_chameleons.probes.linear",
        "next_chameleons.probes.mlp",
        "next_chameleons.reports.json_report",
        "next_chameleons.selectors",
        "next_chameleons.training.backends",
        "next_chameleons.training.regimes",
    ]:
        import_module(module_name)


def load_entrypoint_plugins(group: str = "next_chameleons.plugins") -> list[str]:
    """Load external plugin entry points that register Probes, Judges, or Regimes."""

    loaded: list[str] = []
    selected = entry_points().select(group=group)
    for entry_point in selected:
        entry_point.load()
        loaded.append(entry_point.name)
    return loaded


def load_plugins(*, include_entrypoints: bool = True) -> list[str]:
    """Load built-ins and optionally external entry-point plugins."""

    load_builtin_plugins()
    return load_entrypoint_plugins() if include_entrypoints else []
