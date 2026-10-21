"""Build the TICA round-1 concept pool and task prompts from public sources (DESIGN-TICA.md).

Sources (pinned; raw files live in the raw cache, never in git):
- Concepts: the 50 baseline plus 450 categorized concepts from
  safety-research/introspection-mechanisms experiments/concepts_list.py (Lindsey 2025 concepts plus
  the Lederman & Mahowald 2026 extension), at the commit in SOURCE_COMMIT.txt. The 20 categories
  come from its comment headers.
- Instructions: databricks/databricks-dolly-15k (CC-BY-SA-3.0, human-written) at the revision in
  SOURCE.json. Categories open_qa, general_qa, brainstorming and creative_writing; no-context rows
  only.

Splits are fixed here, before any model run:
- Concepts. HELD-OUT = Kowalski's 10 (ICA comparability) plus 3 whole categories
  (VEHICLES/TRANSPORT, MUSIC/ART, WEATHER; 48 concepts), for category-level transfer. TRAIN =
  everything else.
- Instructions. A seeded 80/20 train/test split by instruction.
Output: $RAW/tica/round1_data.json
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
KOWALSKI = [
    "Denim",
    "Lightning",
    "Volcanoes",
    "Bread",
    "Trees",
    "Cameras",
    "Vegetables",
    "Milk",
    "Dynasties",
    "Deserts",
]
HELDOUT_CATEGORIES = ["VEHICLES/TRANSPORT", "MUSIC/ART", "WEATHER"]
# BASELINE50 items that are the same KIND of thing as a held-out category move into it, so training
# cannot see a
# member of a held-out category. Professions and activities (Sculptors, Pilots, Singing) are a
# different kind of
# concept and stay in train. Semantic nearest-train distance is reported as a covariate at analysis
# time.
RECATEGORIZE = {
    "Trumpets": "MUSIC/ART",
    "Xylophones": "MUSIC/ART",
    "Harmonies": "MUSIC/ART",
    "Dirigibles": "VEHICLES/TRANSPORT",
    "Snow": "WEATHER",
    "Frosts": "WEATHER",
}
DOLLY_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")
SEED = 20260927


def parse_concepts(src: str) -> dict[str, str]:
    """concept -> category. The baseline 50 get the category BASELINE50."""

    out: dict[str, str] = {}
    base = re.search(r"^BASELINE = \[(.*?)\]", src, re.S | re.M).group(1)
    for w in re.findall(r'"([^"]+)"', base):
        out[w] = "BASELINE50"
    body = src[src.index("NEW_CONCEPTS = [") :]
    body = body[: body.index("]")]
    cat = None
    for line in body.splitlines()[1:]:
        m = re.match(r"\s*#\s*(.+?)\s*\((\d+) concepts\)", line)
        if m:
            cat = m.group(1).strip()
            continue
        for w in re.findall(r'"([^"]+)"', line):
            if w in out:
                raise ValueError(f"duplicate concept {w}")
            out[w] = cat
    return out


def main() -> None:
    src_dir = RAW / "introspection_mechanisms"
    concepts = parse_concepts((src_dir / "concepts_list.py").read_text())
    assert len(concepts) == 500, len(concepts)
    assert all(k in concepts for k in KOWALSKI)
    pool = []
    for name, cat in sorted(concepts.items()):
        cat = RECATEGORIZE.get(name, cat)
        if name in KOWALSKI:
            split = "heldout_kowalski"
        elif cat in HELDOUT_CATEGORIES:
            split = "heldout_category"
        else:
            split = "train"
        pool.append({"concept": name, "phrase": name.lower(), "category": cat, "split": split})

    dolly_meta = json.loads((RAW / "dolly15k/SOURCE.json").read_text())
    rows = [json.loads(line) for line in (RAW / "dolly15k" / dolly_meta["file"]).open()]
    seen, instr = set(), []
    for r in rows:
        text = r["instruction"].strip()
        if r["category"] not in DOLLY_CATEGORIES or r["context"].strip():
            continue
        if not (15 <= len(text) <= 300) or text in seen:
            continue
        seen.add(text)
        instr.append(
            {
                "id": "dolly-" + hashlib.sha256(text.encode()).hexdigest()[:12],
                "text": text,
                "category": r["category"],
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(instr)
    cut = int(0.8 * len(instr))
    for i, x in enumerate(instr):
        x["split"] = "train" if i < cut else "test"

    out = {
        "meta": {
            "concepts_source": "safety-research/introspection-mechanisms "
            "experiments/concepts_list.py @ "
            + (src_dir / "SOURCE_COMMIT.txt").read_text().split()[0],
            "instructions_source": dolly_meta,
            "heldout_categories": HELDOUT_CATEGORIES,
            "recategorized": RECATEGORIZE,
            "heldout_kowalski": KOWALSKI,
            "seed": SEED,
        },
        "concepts": pool,
        "instructions": instr,
    }
    (RAW / "round1_data.json").write_text(json.dumps(out, indent=1))
    from collections import Counter

    print(Counter(c["split"] for c in pool), Counter(x["split"] for x in instr))


if __name__ == "__main__":
    main()
