"""Helpers for the final multi-concept, multi-probe-type dodger (TICA-FINAL-DODGER-DESIGN.md).

Pure functions (no model): the concept split, per-concept published positives, and the probe-kind
split. Trainer / race wiring is listed in the design doc ("Interfaces").
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _index(extra_dir: str) -> dict:
    return json.loads((Path(extra_dir) / "index.json").read_text())


def load_extra_pos_multi(extra_dir: str, concept: str) -> list[str]:
    """Published positives for `concept` from a build_multi_concept_expansion.py directory
    ([] if the directory is unset or does not cover the concept). Each text belongs to ONE concept
    (the builder assigns shared texts once), so different concepts' forks never share a positive."""

    if not extra_dir:
        return []
    ent = _index(extra_dir).get(concept)
    if ent is None:
        return []
    d = json.loads((Path(extra_dir) / ent["file"]).read_text())
    if d["concept"] != concept:
        raise ValueError(f"expansion file {ent['file']} is for {d['concept']!r}, not {concept!r}")
    return list(d["texts"])


def load_split(path: str) -> dict:
    sp = json.loads(Path(path).read_text())
    held = set(sp["heldout"]) | set(sp.get("heldout_far", []))
    if held & set(sp["train"]):
        raise ValueError("split: concepts both trained and held out")
    return sp


def heldout_concepts(sp: dict) -> list[str]:
    return list(dict.fromkeys(sp["heldout"] + sp.get("heldout_far", [])))


def check_probe_split(train_kinds, heldout_kinds, known) -> None:
    """Every kind must exist (probe_bank.BANK keys / zoo kinds); the two sets must be disjoint."""

    from next_chameleons.tica.probe_zoo import split_kind

    unknown = ({split_kind(k)[0] for k in train_kinds} | {split_kind(k)[0] for k in heldout_kinds}
               ) - set(known)
    if unknown:
        raise ValueError(f"unknown probe kinds: {sorted(unknown)}")
    both = set(train_kinds) & set(heldout_kinds)
    if both:
        raise ValueError(f"probe kinds both trained and held out: {sorted(both)}")
