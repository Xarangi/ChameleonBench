"""Larger free-generation capability pools and a held-out split (10-30, user: "make sure that we
have enough samples in capability preservation so that the model doesn't memorize them ... we
should eval on examples not trained on").

The cap_free pools came from each checkable concept's calib / episode requests: 130-240 items, each
drawn ~3-6 times over a 1,200-step run, and the capability evals then used the same items.

  dolly_rows    Dolly-15k instructions (databricks-dolly-15k, CC-BY-SA-3.0) with the concept's own
                answer directive appended (the last line of its existing requests, e.g. "Respond
                in German." / "Write your entire response in capital letters."), for the
                FORM-checked concepts (languages: fastText language ID; all-caps). Each concept
                gets a DISJOINT slice of the shuffled instructions, so no prompt repeats across
                concepts. No fallback answer.
  split_holdout deterministic held-out share of a pool (hash of the request): those rows never
                enter training and are written next to the run for held-out evaluation.

Aggregates only: generated answers are never stored.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

MAX_CONTEXT_CHARS = 1200  # Dolly items with a long context are dropped (prompt length)


def directive(requests: list[str]) -> str | None:
    """The concept's answer directive: the most common last line of its existing requests (None if
    no line is shared by at least a quarter of them)."""

    last = [r.strip().splitlines()[-1].strip() for r in requests if r.strip()]
    if not last:
        return None
    line, n = Counter(last).most_common(1)[0]
    return line if n >= max(2, len(last) // 4) else None


def dolly_rows(path: str | Path, concepts: dict[str, str], n_per_concept: int,
               seed: int = 0) -> dict[str, list[tuple[str, None, None]]]:
    """{concept: [(request, None, None)]} from Dolly-15k: disjoint shuffled slices of n_per_concept
    items, each request = instruction (+ context) + the concept's directive (concepts: {name:
    directive})."""

    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ctx = (r.get("context") or "").strip()
            if len(ctx) > MAX_CONTEXT_CHARS or not r.get("instruction", "").strip():
                continue
            body = r["instruction"].strip() + (f"\n\n{ctx}" if ctx else "")
            items.append(body)
    random.Random(seed).shuffle(items)
    out, start = {}, 0
    for c in sorted(concepts):
        chunk = items[start : start + n_per_concept]
        start += n_per_concept
        out[c] = [(f"{b}\n\n{concepts[c]}", None, None) for b in chunk]
    return out


def _held(request: str, frac: float) -> bool:
    h = int(hashlib.sha256(request.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def split_holdout(pool: dict[str, list], frac: float) -> tuple[dict[str, list], dict[str, list]]:
    """(training pool, held-out pool): a deterministic `frac` of every concept's rows (by request
    hash) is held out. frac <= 0 -> (pool, {})."""

    if frac <= 0:
        return pool, {}
    train, held = {}, {}
    for c, rows in pool.items():
        held[c] = [r for r in rows if _held(r[0], frac)]
        train[c] = [r for r in rows if not _held(r[0], frac)]
    return train, held
