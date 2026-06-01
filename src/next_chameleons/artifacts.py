"""Artifact and redaction helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RAW_TEXT_REPLACEMENT = "[REDACTED_RAW_TEXT]"


@dataclass(frozen=True)
class SourceReference:
    """Pinned external source reference for real replication data."""

    kind: str
    name: str
    revision: str
    checksum: str


@dataclass
class DatasetManifest:
    """Audit manifest for a dataset split without storing raw text in git."""

    dataset_id: str
    source: SourceReference
    split: str
    num_examples: int
    label_names: list[str]
    raw_cache_path: str | None = None

    def validate(self) -> None:
        """Validate that a real run has pinning metadata."""

        missing = [
            field
            for field, value in {
                "revision": self.source.revision,
                "checksum": self.source.checksum,
            }.items()
            if not value or value.startswith("to-")
        ]
        if missing:
            raise ValueError(f"Dataset manifest {self.dataset_id!r} lacks pinned {missing}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_artifact_root() -> Path:
    """Return the local artifact root, preferring `$SCRATCH`."""

    base = Path(os.environ.get("SCRATCH", Path.cwd()))
    return base / "next_chameleons_artifacts"


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write pretty JSON with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(text: str) -> str:
    """Hash text for manifests without persisting the text itself."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SafetyRedactor:
    """Redacts raw text fields before writing committed or shared artifacts."""

    def __init__(self, replacement: str = RAW_TEXT_REPLACEMENT) -> None:
        self.replacement = replacement
        self._sensitive_keys = re.compile(r"(text|prompt|completion|response|raw)", re.IGNORECASE)

    def redact(self, value: Any) -> Any:
        """Recursively redact sensitive string-bearing fields."""

        if isinstance(value, dict):
            return {
                key: (
                    self.replacement
                    if self._sensitive_keys.search(str(key))
                    else self.redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value
