"""Generic name -> component registry with entry-point plugins (group `crypsis.plugins`).

Used only where a config picks a component by name (plan §12.6). Standalone port of
next_chameleons.registry / plugins.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from typing import Generic, TypeVar, overload

T = TypeVar("T")

PLUGIN_GROUP = "crypsis.plugins"


class Registry(Generic[T]):
    """Maps names to components; `register` works directly or as a decorator; duplicates are errors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    @overload
    def register(self, key: str) -> Callable[[T], T]: ...
    @overload
    def register(self, key: str, item: T) -> T: ...

    def register(self, key: str, item: T | None = None) -> Callable[[T], T] | T:
        def decorator(value: T) -> T:
            if key in self._items:
                raise KeyError(f"{self.name} registry already contains {key!r}")
            self._items[key] = value
            return value

        return decorator(item) if item is not None else decorator

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"unknown {self.name} {key!r}; available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)


def load_plugins(group: str = PLUGIN_GROUP) -> list[str]:
    """Import every entry point of `group` (their import side effects register components)."""

    loaded = []
    for ep in entry_points().select(group=group):
        ep.load()
        loaded.append(ep.name)
    return loaded
