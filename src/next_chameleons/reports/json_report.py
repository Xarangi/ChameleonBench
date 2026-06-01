"""JSON report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from next_chameleons.artifacts import SafetyRedactor, write_json
from next_chameleons.registry import REPORTS


class JsonReport:
    """Writes redacted JSON reports."""

    def __init__(self, *, redact: bool = True) -> None:
        self.redactor = SafetyRedactor()
        self.redact = redact

    def write(self, path: Path, payload: dict[str, Any]) -> Path:
        output = self.redactor.redact(payload) if self.redact else payload
        write_json(path, output)
        return path


REPORTS.register("json", JsonReport)
