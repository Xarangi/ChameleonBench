"""Label-cleaned negatives (REPLICATION 2026-10-16h; default off). An LLM-judge label file
(scripts/label_concept_contamination.py: {"labels": {text_id: {concept: p_yes}}}) marks texts that
actually HAVE a concept; with AdaptiveConfig.neg_labels (or env TICA_NEG_LABELS for evals of runs
trained before the field existed) such texts are never used as NEGATIVES for that concept -- in the
base-probe fits (answer_format) and in the in-episode refit forks (concept-free pool and other
concepts' texts). The text stays a negative for every other concept."""

from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE: dict[str, dict] = {}


def settings(acfg=None) -> tuple[str, float] | None:
    path = (getattr(acfg, "neg_labels", "") or os.environ.get("TICA_NEG_LABELS", "")).strip()
    if not path:
        return None
    return path, float(getattr(acfg, "neg_label_thr", 0.5) or 0.5)


def _labels(path: str) -> dict:
    if path not in _CACHE:
        _CACHE[path] = json.loads(Path(path).read_text())["labels"]
    return _CACHE[path]


def clean(texts: list[str], concept: str, acfg=None) -> list[str]:
    """texts minus those the judge says have `concept` (p_yes >= thr); unchanged when off."""

    st = settings(acfg)
    if st is None:
        return texts
    from next_chameleons.tica.answer_format import text_id

    lab, thr = _labels(st[0]), st[1]
    return [x for x in texts if lab.get(text_id(x), {}).get(concept, 0.0) < thr]


def stamp(acfg=None) -> dict:
    """Cache-key part for probes fit on cleaned negatives."""

    st = settings(acfg)
    return {} if st is None else {"neg_labels": st[0], "neg_label_thr": st[1]}
