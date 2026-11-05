"""Hierarchical topic base from DBpedia Classes (REPLICATION 2026-10-14i).

Source: huggingface.co/datasets/DeveloperOats/DBPedia_Classes (CC0; the standard 3-level
hierarchical text-classification benchmark: 9 L1 / 70 L2 / 219 L3 classes, 337,739 Wikipedia
abstracts). One genre (encyclopedic abstracts) for every topic, so a topic probe cannot key on
style. A TOPIC is an L2 class (e.g. Place > CelestialBody); its L1 class is its family, its L3
classes are subtopics. Writes {topic: {l1, l3: {name: n}, name, texts}} with up to --per-topic texts
(train / val / test CSV rows pooled, deduplicated), dropping topics with fewer than --min texts.
No text is printed; aggregates only.

  python scripts/build_dbpedia_topics.py --out .../dbpedia_topics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("/scratch/sneheel/next_chameleons_raw_cache/dbpedia_classes")


def readable(cls: str) -> str:
    """CelestialBody -> 'celestial bodies' style plain name used in the monitor brief."""

    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cls).lower()
    return {"organisation member": "members of organisations", "sports team season":
            "sports team seasons", "football league season": "football league seasons"}.get(
        words, words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=600)
    ap.add_argument("--min", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", help="broad-topic map {topics: {name: {family, classes}}}: one "
                    "topic per entry, texts drawn evenly over its L2 classes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    csv.field_size_limit(10**9)
    by, by3 = defaultdict(list), defaultdict(list)
    l1_of, l3c = {}, defaultdict(Counter)
    seen = set()
    for split in ("train", "val", "test"):
        with open(SRC / f"DBPEDIA_{split}.csv") as fh:
            for r in csv.DictReader(fh):
                t = r["text"].strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                by[r["l2"]].append((t, r["l3"]))
                by3[r["l3"]].append((t, r["l3"]))
                l1_of[r["l2"]] = r["l1"]
                l3c[r["l2"]][r["l3"]] += 1
    rng = random.Random(a.seed)
    out, dropped = {}, {}
    for l2, rows in sorted(by.items()):
        if len(rows) < a.min:
            dropped[l2] = len(rows)
            continue
        rng.shuffle(rows)
        out[l2] = {"l1": l1_of[l2], "name": readable(l2), "n_total": len(rows),
                   "l3": dict(l3c[l2]), "texts": [t for t, _ in rows[: a.per_topic]]}
    if a.merge:
        spec = json.loads(Path(a.merge).read_text())["topics"]
        merged = {}
        for name, v in spec.items():
            # "l3": exact DBpedia L3 classes (v2: names must match content); "classes": whole L2s
            parts = ([by3[c][:] for c in v["l3"]] if "l3" in v
                     else [by[c][:] for c in v["classes"]])
            for p in parts:
                rng.shuffle(p)
            texts, i = [], 0  # round-robin over the classes: no single class dominates
            while len(texts) < a.per_topic and any(i < len(p) for p in parts):
                texts += [p[i][0] for p in parts if i < len(p)]
                i += 1
            merged[name] = {"l1": v["family"], "name": name,
                            "classes": v.get("l3", v.get("classes")),
                            "n_total": sum(len(p) for p in parts), "texts": texts[: a.per_topic]}
        out = merged
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    fam = Counter(v["l1"] for v in out.values())
    print(json.dumps({"topics": len(out), "families": dict(fam), "dropped_small": dropped},
                     indent=1))


if __name__ == "__main__":
    main()
