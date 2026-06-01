"""Small typed registries for plugin-style experiment components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Name-to-factory registry used by datasets, Probes, Judges, and backends."""

    name: str
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, key: str, item: T | None = None) -> Callable[[T], T] | T:
        """Register an item directly or as a decorator."""

        def decorator(value: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name} registry already contains {key!r}")
            self._items[key] = value
            return value

        if item is not None:
            return decorator(item)
        return decorator

    def get(self, key: str) -> T:
        """Return a registered item."""

        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} {key!r}; available: {available}") from exc

    def names(self) -> list[str]:
        """Return registered names in stable order."""

        return sorted(self._items)


DATASET_ADAPTERS: Registry[type] = Registry("dataset adapter")
ACTIVATION_EXTRACTORS: Registry[type] = Registry("activation extractor")
PROBES: Registry[type] = Registry("probe")
JUDGES: Registry[type] = Registry("judge")
PROBE_SELECTORS: Registry[type] = Registry("probe selector")
TRAINING_BACKENDS: Registry[type] = Registry("training backend")
TRAINING_REGIMES: Registry[type] = Registry("training regime")
FEEDBACK_CHANNELS: Registry[type] = Registry("feedback channel")
CURRICULA: Registry[type] = Registry("curriculum")
METRICS: Registry[Callable] = Registry("metric")
REPORTS: Registry[type] = Registry("report")
