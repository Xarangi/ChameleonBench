"""Result files: atomic JSON, JSONL append / read, done markers.

Conventions reproduced from the tica code: JSONL rows are `json.dumps(row, default=str) + "\\n"`
appended (trainer turns / refits / train logs); readers skip blank and half-written lines
(run_report._jsonl); a result is "done" when its output file exists (bash queues `[ -f out.json ]`),
or when a DONE_<stage> marker exists (rounds.py). Writes go through a temp file + rename so a
reader never sees a partial file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write `text` to `path` via a temp file in the same directory and an atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_json(path: str | Path, obj: Any, *, indent: int | None = 1, default: Any = None) -> Path:
    """Atomic JSON write (`json.dumps(obj, indent=indent, default=default)`)."""

    return write_text_atomic(path, json.dumps(obj, indent=indent, default=default))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, rows: Iterable[dict], *, default: Any = str) -> int:
    """Append rows as JSON lines (creating parents); returns the number written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=default) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, *, skip_bad: bool = True) -> list[dict]:
    """Rows of a JSONL file; [] if missing; blank lines skipped; malformed lines (a writer killed
    mid-line) skipped when skip_bad, else raised. == tica run_report._jsonl with skip_bad=True."""

    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if not skip_bad:
                raise
    return out


def is_done(path: str | Path, *, require_valid: bool = False) -> bool:
    """A result counts as done when its file exists (and parses as JSON if require_valid)."""

    p = Path(path)
    if not p.is_file():
        return False
    if not require_valid:
        return True
    try:
        json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return True


def done_marker(directory: str | Path, stage: str | None = None) -> Path:
    """`<dir>/DONE_<stage>` (rounds.py) or `<dir>/DONE` (smoke scripts) when stage is None."""

    return Path(directory) / (f"DONE_{stage}" if stage else "DONE")


def mark_done(directory: str | Path, stage: str | None = None, info: dict | None = None) -> Path:
    return write_json(done_marker(directory, stage), info or {}, indent=None)


def stage_done(directory: str | Path, stage: str | None = None) -> bool:
    return done_marker(directory, stage).exists()
