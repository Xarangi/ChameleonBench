"""Typed data artifacts with load / save for the CURRENT JSON formats and a `manifest()` of
aggregates (counts and digests, never text): ConceptSet (re-exported from concepts), ConceptSplit
(final_split_v5.json), RequestMap (text_id -> eliciting request; answer_format.load_inputs format),
LabelMap (LLM-judge negative labels) and TaskPool (capability rows + hold-out). The recipe DAG that
chains them is a later phase (plan section 14)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from crypsis.data.concepts import (
    LABEL_THRESHOLD,
    ConceptSet,
    clean_negatives,
    digest,
    heldout_concepts,
    load_split,
    sha16,
)

SCHEMA = 1


@dataclass
class ConceptSplit:
    """train / heldout (/ heldout_far) concept lists; other file keys (families, _doc) are kept."""

    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> ConceptSplit:
        return cls(load_split(path))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.raw, indent=1))

    @property
    def train(self) -> list[str]:
        return list(self.raw["train"])

    @property
    def heldout(self) -> list[str]:
        return heldout_concepts(self.raw)

    def manifest(self) -> dict:
        return {"kind": "ConceptSplit", "schema": SCHEMA, "train": len(self.train),
                "heldout": len(self.raw["heldout"]),
                "heldout_far": len(self.raw.get("heldout_far", [])),
                "digest": sha16([self.train, self.heldout])}


@dataclass
class RequestMap:
    """text_id -> the request that elicits the text (the `answer_format.load_inputs` format)."""

    requests: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> RequestMap:
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.requests))

    def get(self, text_id: str, default=None):
        return self.requests.get(text_id, default)

    def __getitem__(self, text_id: str) -> str:
        return self.requests[text_id]

    def __len__(self) -> int:
        return len(self.requests)

    def manifest(self) -> dict:
        return {"kind": "RequestMap", "schema": SCHEMA, "n": len(self.requests),
                "digest": sha16(sorted(self.requests.items()))}


@dataclass
class LabelMap:
    """Negative-label file ({"labels": {text_id: {concept: p_yes}}, "meta": ...}): texts that HAVE
    a concept are never used as negatives for it."""

    labels: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: object = None

    @classmethod
    def load(cls, path: str | Path) -> LabelMap:
        d = json.loads(Path(path).read_text())
        return cls(d["labels"], d.get("meta"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"labels": self.labels, "meta": self.meta}))

    def clean(self, texts: list[str], concept: str, thr: float = LABEL_THRESHOLD) -> list[str]:
        """`texts` minus those labelled as having `concept` (p_yes >= thr)."""
        return clean_negatives(texts, concept, self.labels, thr)

    def manifest(self) -> dict:
        return {"kind": "LabelMap", "schema": SCHEMA, "n_texts": len(self.labels),
                "digest": sha16(sorted(self.labels))}


Row = tuple[str, str | None, str | None]  # (request, fallback answer, reference)


@dataclass
class TaskPool:
    """Capability rows per concept ((request, fallback answer, reference)) and their hold-out."""

    rows: dict[str, list[Row]] = field(default_factory=dict)
    holdout: dict[str, list[Row]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> TaskPool:
        """A {"rows", "holdout"} file, or the bare hold-out format of <run>/cap_free_holdout.json
        ({concept: rows}), which loads as `rows`."""
        d = json.loads(Path(path).read_text())
        if set(d) == {"rows", "holdout"}:
            return cls(_tuples(d["rows"]), _tuples(d["holdout"]))
        return cls(_tuples(d))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"rows": self.rows, "holdout": self.holdout}))

    def save_holdout(self, path: str | Path) -> None:
        """The hold-out alone, in the cap_free_holdout.json format."""
        Path(path).write_text(json.dumps(self.holdout))

    def manifest(self) -> dict:
        def part(p: dict) -> dict:
            return {c: {"n": len(r), "digest": digest([x[0] for x in r])} for c, r in p.items()}

        return {"kind": "TaskPool", "schema": SCHEMA, "rows": part(self.rows),
                "holdout": part(self.holdout)}


def _tuples(d: Mapping) -> dict[str, list[Row]]:
    return {c: [tuple(r) for r in rows] for c, rows in d.items()}


__all__ = ["ConceptSet", "ConceptSplit", "LabelMap", "RequestMap", "TaskPool"]
