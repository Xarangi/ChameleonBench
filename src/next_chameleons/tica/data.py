"""Level-1 data access (written by scripts/build_l1_data.py; DESIGN-TICA "FIRST SETUP").

Text splits per concept, and who may use them:
- fit: probe fitting and TRAINING episodes (the policy trains on texts its monitors were fit on, as
  in NC);
- calib: monitor calibration only;
- episode: EVALUATION episodes only, never trained on.
Concept splits: train (training and evaluation) / dev, heldout (evaluation only) / ood (eval-only
out-of-distribution domain concepts, final_ood_v1: scripts/build_final_ood.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from next_chameleons.tica.episodes import ConceptTexts

# every concept split an eval may name; a data file without one of them is unaffected
CONCEPT_SPLITS = ("train", "dev", "heldout", "ood")
DEFAULT = Path("/scratch/sneheel/next_chameleons_raw_cache/tica/level1_data.json")


class Level1Data:
    def __init__(self, path: Path | str = DEFAULT) -> None:
        self.path = str(path)  # part of the answer-probe cache key (answer_format)
        self.raw = json.loads(Path(path).read_text())
        self.split = {c["concept"]: c["split"] for c in self.raw["concepts"]}

    def concepts(
        self, text_split: str, concept_splits: tuple[str, ...], only: set[str] | None = None
    ) -> list[ConceptTexts]:
        if text_split not in ("fit", "calib", "episode"):
            raise ValueError(text_split)
        return [
            ConceptTexts(c["concept"], c["split"], [t["text"] for t in c[text_split]])
            for c in self.raw["concepts"]
            if c["split"] in concept_splits and (only is None or c["concept"] in only)
        ]

    def hard_negatives(self, concept: str, text_split: str) -> list[str]:
        """Matched within-source negatives of a concept (e.g. DolusChat: the truthful answer to the
        same prompt as each deceptive one), if the data file has them; else []."""

        for c in self.raw["concepts"]:
            if c["concept"] == concept:
                return list((c.get("hard_neg") or {}).get(text_split, []))
        return []

    def negatives(self, text_split: str) -> list[str]:
        return [t["text"] for t in self.raw["negatives"][text_split]]


def select_topics(data: Level1Data, spec: str) -> list[str]:
    """Race topics from a --topics value: a concept split name (CONCEPT_SPLITS; its concepts in
    file order) or a comma list of concept names."""

    if spec in CONCEPT_SPLITS:
        out = [c.concept for c in data.concepts("fit", (spec,))]
    else:
        out = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in out if t not in data.split]
    if unknown or not out:
        raise ValueError(f"--topics {spec!r}: unknown concepts {unknown} or none selected "
                         f"(splits in the data: {sorted(set(data.split.values()))})")
    return out
