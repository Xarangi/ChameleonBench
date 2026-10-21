"""Build the TICA Level-1 (fixed-text) data from AxBench Concept500 (DESIGN-TICA.md "FIRST SETUP").

Source: pyvene/axbench-concept500, subset 9b/l20 (CC-BY-4.0, in the raw cache with SOURCE.json).
- Positives: LLM-written responses that contain the concept named in `output_concept`. There are
  72 per concept in `train` and 36 per concept in `test`.
- Concept-free negatives: `output_concept == "EEEEE"` (216 in train, 18,000 in test).

Splits, fixed here before any model is run:
- Concepts: 400 TRAIN / 100 HELDOUT, by a seeded shuffle of concept ids.
- Per concept:
  - FIT texts = the train-split positives (probe fitting).
  - CALIB = 12 test-split positives (normalizer and probe quality).
  - EPISODE = up to 24 other test-split positives (the repeated texts in episodes).
- Concept-free negatives (test split): 9,000 CALIB (thresholds, quality) and 9,000 EPISODE.
  The 216 train-split negatives join the probe FIT negatives.
Text filter: 20-160 words, stripped, exact duplicates dropped across the whole file (a text that
appears under two concepts is dropped from both).

Output: $RAW/tica/level1_data.json
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
SRC = RAW / "axbench_concept500" / "9b" / "l20"
SEED = 20261003
N_HELDOUT = 100
N_CALIB = 12
N_EPISODE = 24
MIN_WORDS, MAX_WORDS = 20, 160


def clean(t: str) -> str:
    return t.strip()


def ok(t: str) -> bool:
    return MIN_WORDS <= len(t.split()) <= MAX_WORDS


def tid(t: str) -> str:
    return "ax-" + hashlib.sha256(t.encode()).hexdigest()[:12]


def main() -> None:
    tr = pd.read_parquet(SRC / "train" / "data.parquet")
    te = pd.read_parquet(SRC / "test" / "data.parquet")
    for d in (tr, te):
        d["output"] = d["output"].map(clean)
    every = pd.concat([tr, te])
    dup = Counter(every["output"])
    keep = lambda d: d[d["output"].map(ok) & d["output"].map(lambda t: dup[t] == 1)]  # noqa: E731
    tr_pos = keep(tr[tr.category == "positive"])
    te_pos = keep(te[te.category == "positive"])
    tr_neg = keep(tr[tr.category == "negative"])
    te_neg = keep(te[te.category == "negative"])

    names = pd.concat([tr_pos, te_pos]).groupby("concept_id")["output_concept"].first().to_dict()
    genre = pd.concat([tr_pos, te_pos]).groupby("concept_id")["concept_genre"].first().to_dict()
    ids = sorted(names)
    rng = random.Random(SEED)
    order = ids[:]
    rng.shuffle(order)
    heldout = set(order[:N_HELDOUT])

    concepts = []
    for cid in ids:
        fit = tr_pos[tr_pos.concept_id == cid]["output"].tolist()
        test = te_pos[te_pos.concept_id == cid]["output"].tolist()
        r = random.Random(f"{SEED}|{cid}")
        r.shuffle(test)
        concepts.append(
            {
                "concept_id": int(cid),
                "concept": names[cid],
                "genre": genre[cid],
                "split": "heldout" if cid in heldout else "train",
                "fit": [{"id": tid(t), "text": t} for t in fit],
                "calib": [{"id": tid(t), "text": t} for t in test[:N_CALIB]],
                "episode": [{"id": tid(t), "text": t} for t in test[N_CALIB : N_CALIB + N_EPISODE]],
            }
        )
    neg = te_neg["output"].tolist()
    random.Random(f"{SEED}|neg").shuffle(neg)
    half = len(neg) // 2
    out = {
        "source": "pyvene/axbench-concept500 9b/l20 (CC-BY-4.0)",
        "seed": SEED,
        "concepts": concepts,
        "negatives": {
            "fit": [{"id": tid(t), "text": t} for t in tr_neg["output"]],
            "calib": [{"id": tid(t), "text": t} for t in neg[:half]],
            "episode": [{"id": tid(t), "text": t} for t in neg[half:]],
        },
    }
    (RAW / "level1_data.json").write_text(json.dumps(out))
    n = lambda k: [len(c[k]) for c in concepts]  # noqa: E731
    print(
        json.dumps(
            {
                "concepts": len(concepts),
                "heldout": sum(c["split"] == "heldout" for c in concepts),
                "fit_min_median": [min(n("fit")), sorted(n("fit"))[len(concepts) // 2]],
                "calib_min": min(n("calib")),
                "episode_min_median": [min(n("episode")), sorted(n("episode"))[len(concepts) // 2]],
                "neg": {k: len(v) for k, v in out["negatives"].items()},
                "genre": Counter(c["genre"] for c in concepts),
            }
        )
    )


if __name__ == "__main__":
    main()
