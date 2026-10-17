"""Level-0 battery: Kowalski's neutral sentences and concepts, plus Wikipedia concept sentences.

Built by scripts/build_incontext_data.py. The raw text lives in the raw cache, never in the repo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

# Singular/plural lemma per Kowalski concept. Must match scripts/build_incontext_data.py.
LEMMA = {
    "Denim": r"denims?", "Lightning": r"lightnings?", "Volcanoes": r"volcano(?:e?s)?",
    "Bread": r"breads?", "Trees": r"trees?", "Cameras": r"cameras?",
    "Vegetables": r"vegetables?", "Milk": r"milks?", "Dynasties": r"dynast(?:y|ies)",
    "Deserts": r"deserts?",
}


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    concept: str | None = None  # None for a neutral sentence


@dataclass(frozen=True)
class Battery:
    concepts: tuple[str, ...]
    neutral_primary: tuple[Sentence, ...]
    neutral_heldout: tuple[Sentence, ...]
    concept_train: dict[str, tuple[Sentence, ...]]
    concept_test: dict[str, tuple[Sentence, ...]]
    meta: dict = field(default_factory=dict)

    def train_units(self) -> list[Sentence]:
        return [s for c in self.concepts for s in self.concept_train[c]]

    def test_units(self) -> list[Sentence]:
        return [s for c in self.concepts for s in self.concept_test[c]]


def load_battery(path: Path) -> Battery:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    concepts = tuple(payload["concepts"])

    def rows(items, concept=None):
        return tuple(Sentence(r["id"], r["text"], concept) for r in items)

    cs = payload["concept_sentences"]
    return Battery(
        concepts=concepts,
        neutral_primary=rows(payload["neutral"]["primary"]),
        neutral_heldout=rows(payload["neutral"]["heldout"]),
        concept_train={c: rows(cs[c]["train"], c) for c in concepts},
        concept_test={c: rows(cs[c]["test"], c) for c in concepts},
        meta=payload.get("meta", {}),
    )


def subset(
    battery: Battery,
    *,
    concepts: list[str] | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
    max_neutral: int | None = None,
) -> Battery:
    """Deterministic prefix subset, for smoke runs and pilots."""

    keep = tuple(c for c in battery.concepts if concepts is None or c in concepts)
    return replace(
        battery,
        concepts=keep,
        neutral_primary=battery.neutral_primary[:max_neutral],
        neutral_heldout=battery.neutral_heldout[:max_neutral],
        concept_train={c: battery.concept_train[c][:max_train] for c in keep},
        concept_test={c: battery.concept_test[c][:max_test] for c in keep},
    )


def lemma_spans(text: str, concept: str) -> list[tuple[int, int]]:
    """Character spans of the concept word in `text` (for the literal vs context readout)."""

    pattern = LEMMA.get(concept, re.escape(concept.lower().rstrip("s")) + r"s?")
    return [m.span() for m in re.finditer(rf"\b{pattern}\b", text, flags=re.IGNORECASE)]
