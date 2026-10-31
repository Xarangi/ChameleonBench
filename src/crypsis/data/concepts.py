"""ConceptSet: concept texts by split, concept splits, negatives, hard negatives, label cleaning.

Reads the CURRENT data-file format (tica/data.py, written by scripts/build_final_*.py):
  {"concepts": [{"concept", "split", "fit": [{"id", "text"}], "calib": [...], "episode": [...],
                 "hard_neg": {text_split: [text]}?, ...}],
   "negatives": {"fit": [...], "calib": [...], "episode": [...]}, ...}
Text splits and who may use them: fit = probe fitting and TRAINING episodes; calib = monitor
calibration only; episode = EVALUATION episodes only, never trained on. Concept splits: train /
dev / heldout / ood (eval-only out-of-distribution domain concepts).

Ported from tica/data.py, tica/neg_labels.py and tica/multi.py; accessors behave identically.
Operations return new ConceptSets (the object never mutates), and unknown file keys are preserved
on save."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# every concept split an eval may name; a data file without one of them is unaffected
CONCEPT_SPLITS = ("train", "dev", "heldout", "ood")
TEXT_SPLITS = ("fit", "calib", "episode")
LABEL_THRESHOLD = 0.5


def text_id(text: str) -> str:
    """Content hash of a text (matches tica.answer_format.text_id; agent A's core.types owns the
    shared copy once it lands)."""
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def digest(texts: list[str]) -> str:
    """Order-insensitive digest of a text list (sha256 over the sorted text ids; no text stored)."""
    h = hashlib.sha256()
    for i in sorted(text_id(t) for t in texts):
        h.update(i.encode())
    return h.hexdigest()[:16]


@dataclass
class ConceptTexts:
    """The texts of one concept in one text split (equivalent of tica.episodes.ConceptTexts)."""

    concept: str
    split: str
    texts: list[str]


def _item(text: str) -> dict:
    return {"id": text_id(text), "text": text}


class ConceptSet:
    """Texts per concept x text split, concept splits, negatives and hard negatives."""

    def __init__(self, raw: dict, path: str | Path = "") -> None:
        self.raw = raw
        self.path = str(path)  # part of the answer-probe cache key
        self.split = {c["concept"]: c["split"] for c in raw["concepts"]}

    @classmethod
    def load(cls, path: str | Path) -> ConceptSet:
        return cls(json.loads(Path(path).read_text()), path)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.raw))

    @property
    def names(self) -> list[str]:
        """Concept names in file order."""
        return [c["concept"] for c in self.raw["concepts"]]

    def concepts(self, text_split: str, concept_splits: tuple[str, ...],
                 only: set[str] | None = None) -> list[ConceptTexts]:
        """Texts of the concepts in `concept_splits` (optionally only these names), file order."""
        if text_split not in TEXT_SPLITS:
            raise ValueError(text_split)
        return [ConceptTexts(c["concept"], c["split"], [t["text"] for t in c[text_split]])
                for c in self.raw["concepts"]
                if c["split"] in concept_splits and (only is None or c["concept"] in only)]

    def hard_negatives(self, concept: str, text_split: str) -> list[str]:
        """Matched within-source negatives of a concept (e.g. DolusChat: the truthful answer to the
        same prompt as each deceptive one), if the data file has them; else []."""
        for c in self.raw["concepts"]:
            if c["concept"] == concept:
                return list((c.get("hard_neg") or {}).get(text_split, []))
        return []

    def negatives(self, text_split: str) -> list[str]:
        return [t["text"] for t in self.raw["negatives"][text_split]]

    def select_topics(self, spec: str) -> list[str]:
        """Race topics from a --topics value: a concept split name (its concepts in file order) or a
        comma list of concept names."""
        return select_topics(self, spec)

    # ---- composition (data recipes): each returns a NEW ConceptSet -------------------------------
    def select(self, names: list[str] | set[str]) -> ConceptSet:
        """Only these concepts (file order); unknown names are an error. Negatives are kept."""
        unknown = sorted(set(names) - set(self.split))
        if unknown:
            raise ValueError(f"unknown concepts: {unknown}")
        raw = {**self.raw, "concepts": [copy.deepcopy(c) for c in self.raw["concepts"]
                                        if c["concept"] in set(names)]}
        return ConceptSet(raw, self.path)

    def extend(self, other: ConceptSet | list[dict]) -> ConceptSet:
        """Add concepts (e.g. eval-only ones) after the existing ones, leaving them untouched; a
        name clash is an error."""
        add = other.raw["concepts"] if isinstance(other, ConceptSet) else other
        clash = sorted({c["concept"] for c in add} & set(self.split))
        if clash:
            raise ValueError(f"concepts already present: {clash}")
        raw = copy.deepcopy(self.raw)
        raw["concepts"] = raw["concepts"] + copy.deepcopy(list(add))
        return ConceptSet(raw, self.path)

    def with_texts(self, concept: str, text_split: str, texts: list[str]) -> ConceptSet:
        """Replace one concept's texts in one text split (ids recomputed from the content)."""
        if text_split not in TEXT_SPLITS:
            raise ValueError(text_split)
        if concept not in self.split:
            raise ValueError(f"unknown concept: {concept!r}")
        raw = copy.deepcopy(self.raw)
        for c in raw["concepts"]:
            if c["concept"] == concept:
                c[text_split] = [_item(t) for t in texts]
        return ConceptSet(raw, self.path)

    def counts(self) -> dict:
        """{concept: {"split", "fit", "calib", "episode", "hard_neg"}, "_negatives": {split: n}}."""
        out: dict = {}
        for c in self.raw["concepts"]:
            out[c["concept"]] = {"split": c["split"], **{s: len(c[s]) for s in TEXT_SPLITS},
                                 "hard_neg": {s: len(v) for s, v in (c.get("hard_neg") or {}
                                                                     ).items()}}
        out["_negatives"] = {s: len(v) for s, v in self.raw.get("negatives", {}).items()}
        return out

    def hashes(self) -> dict:
        """{concept: {text_split: digest}, "_negatives": {text_split: digest}}: provenance
        fingerprints that never contain text."""
        out: dict = {c["concept"]: {s: digest([t["text"] for t in c[s]]) for s in TEXT_SPLITS}
                     for c in self.raw["concepts"]}
        out["_negatives"] = {s: digest([t["text"] for t in v])
                             for s, v in self.raw.get("negatives", {}).items()}
        return out

    def manifest(self) -> dict:
        """Counts and digests (aggregates only, never text) for provenance sidecars."""
        return {"kind": "ConceptSet", "schema": 1, "n_concepts": len(self.names),
                "concept_splits": {s: sum(v == s for v in self.split.values())
                                   for s in sorted(set(self.split.values()))},
                "counts": self.counts(), "digest": sha16(self.hashes())}


def sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def select_topics(data: ConceptSet, spec: str) -> list[str]:
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


# ---- label cleaning (tica/neg_labels.py) --------------------------------------------------------
def clean_negatives(texts: list[str], concept: str, labels: Mapping[str, Mapping[str, float]]
                    | None, thr: float = LABEL_THRESHOLD) -> list[str]:
    """`texts` minus those the judge says HAVE `concept` (p_yes >= thr; `labels` maps text_id ->
    {concept: p_yes}). Unchanged when labels is None. A falsy thr means 0.5, as before."""
    if labels is None:
        return texts
    thr = float(thr or LABEL_THRESHOLD)
    return [x for x in texts if labels.get(text_id(x), {}).get(concept, 0.0) < thr]


# ---- concept split + expansion files (tica/multi.py) -------------------------------------------
def load_split(path: str | Path) -> dict:
    """A concept-split file (final_split_v5.json format); trained and held-out must be disjoint."""
    sp = json.loads(Path(path).read_text())
    held = set(sp["heldout"]) | set(sp.get("heldout_far", []))
    if held & set(sp["train"]):
        raise ValueError("split: concepts both trained and held out")
    return sp


def heldout_concepts(sp: dict) -> list[str]:
    return list(dict.fromkeys(sp["heldout"] + sp.get("heldout_far", [])))


def load_extra_pos_multi(extra_dir: str | Path, concept: str) -> list[str]:
    """Published positives for `concept` from a build_multi_concept_expansion.py directory ([] if
    the directory is unset or does not cover the concept)."""
    if not extra_dir:
        return []
    index = json.loads((Path(extra_dir) / "index.json").read_text())
    ent = index.get(concept)
    if ent is None:
        return []
    d = json.loads((Path(extra_dir) / ent["file"]).read_text())
    if d["concept"] != concept:
        raise ValueError(f"expansion file {ent['file']} is for {d['concept']!r}, not {concept!r}")
    return list(d["texts"])
