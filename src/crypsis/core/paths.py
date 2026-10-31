"""Filesystem roots (raw cache, artifacts, HF cache, runs) without hard-coded cluster paths.

Each root resolves, in order, from: an explicit argument, the `paths:` section of a site profile YAML
given explicitly, the env vars CRYPSIS_RAW / CRYPSIS_ARTIFACTS / CRYPSIS_HF_CACHE / CRYPSIS_RUNS, and
otherwise is an error for required roots (no silent default). `runs` defaults to <artifacts>/runs.
Config strings use `${paths.raw}`, `${paths.artifacts}`, `${paths.hf_cache}`, `${paths.runs}`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ENV_VARS = {"raw_cache": "CRYPSIS_RAW", "artifacts": "CRYPSIS_ARTIFACTS",
            "hf_cache": "CRYPSIS_HF_CACHE", "runs": "CRYPSIS_RUNS"}
ALIASES = {"raw": "raw_cache", "raw_cache": "raw_cache", "artifacts": "artifacts",
           "hf_cache": "hf_cache", "hf": "hf_cache", "runs": "runs"}
_REF = re.compile(r"\$\{paths\.(\w+)\}")


class PathsError(LookupError):
    """A required root could not be resolved."""


@dataclass(frozen=True)
class Paths:
    raw_cache: Path | None = None
    artifacts: Path | None = None
    hf_cache: Path | None = None
    runs: Path | None = None

    @property
    def raw(self) -> Path | None:
        return self.raw_cache

    @classmethod
    def resolve(
        cls,
        *,
        raw_cache: str | Path | None = None,
        artifacts: str | Path | None = None,
        hf_cache: str | Path | None = None,
        runs: str | Path | None = None,
        profile: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        required: tuple[str, ...] = ("raw_cache", "artifacts"),
    ) -> Paths:
        """Resolve every root (explicit > profile > env); raise PathsError naming unresolved required ones."""

        env = os.environ if env is None else env
        explicit = {"raw_cache": raw_cache, "artifacts": artifacts, "hf_cache": hf_cache,
                    "runs": runs}
        prof = load_profile_paths(profile) if profile is not None else {}
        out: dict[str, Path | None] = {}
        for name in explicit:
            v = explicit[name] or prof.get(name) or env.get(ENV_VARS[name]) or None
            out[name] = Path(v).expanduser() if v else None
        if out["runs"] is None and out["artifacts"] is not None:
            out["runs"] = out["artifacts"] / "runs"
        missing = [n for n in required if out.get(ALIASES.get(n, n)) is None]
        if missing:
            hint = ", ".join(f"{n} (arg, profile paths.{n} or ${ENV_VARS[ALIASES.get(n, n)]})"
                             for n in missing)
            raise PathsError(f"unresolved path roots: {hint}")
        return cls(**out)

    def get(self, name: str) -> Path:
        """Root by name or alias (raw, raw_cache, artifacts, hf_cache, runs); error if unresolved."""

        key = ALIASES.get(name)
        if key is None:
            raise KeyError(f"unknown path root {name!r}; known: {sorted(ALIASES)}")
        v = getattr(self, key)
        if v is None:
            raise PathsError(f"path root {key!r} is not set (${ENV_VARS[key]})")
        return v

    def interpolate(self, s: str) -> str:
        """Replace every `${paths.<root>}` in s."""

        return _REF.sub(lambda m: str(self.get(m.group(1))), s)

    def interpolate_tree(self, obj: Any) -> Any:
        """interpolate() over every string in nested dicts / lists / tuples."""

        if isinstance(obj, str):
            return self.interpolate(obj)
        if isinstance(obj, Mapping):
            return {k: self.interpolate_tree(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.interpolate_tree(v) for v in obj)
        return obj

    def as_dict(self) -> dict[str, str | None]:
        return {f.name: (None if getattr(self, f.name) is None else str(getattr(self, f.name)))
                for f in fields(self)}


def load_profile_paths(profile: str | Path) -> dict[str, str]:
    """The `paths:` section of a site profile YAML, keys normalised (raw -> raw_cache)."""

    import yaml

    data = yaml.safe_load(Path(profile).read_text()) or {}
    sec = data.get("paths", {}) or {}
    out = {}
    for k, v in sec.items():
        if k not in ALIASES:
            raise KeyError(f"{profile}: unknown paths key {k!r}; known: {sorted(ALIASES)}")
        if v is not None:
            out[ALIASES[k]] = str(v)
    return out
