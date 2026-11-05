"""Extra PUBLISHED positives for MANY concepts (final multi-concept dodger;
TICA-FINAL-DODGER-DESIGN.md).

Generalises build_concept_expansion.py (math only, REPLICATION 10-13d) to every Level-1 concept:
positives from AxBench Concept16K v1 (9b/l20 train) AND v2 (pyvene/axbench-concept16k_v2) whose
concept name is the SAME concept or a STRICT near-duplicate name (TF-IDF cosine of the names
>= --syn, default 0.7; listed per concept in the output), deduplicated, with every text that
appears anywhere in the Level-1 data removed. Texts are never printed; only counts.

  --count-only : write {concept: n_available} (n = Level-1 fit texts + new published positives)
  default      : also write $RAW/tica/expansion_multi/<slug>.json
                 ({concept, source_concepts, texts})
                 for --concepts (a split JSON: train + heldout + heldout_far), an index.json, and
                 extend level1_inputs_plus.json with the instructions of the new texts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from next_chameleons.tica.answer_format import text_id

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache/tica")
V1 = RAW / "axbench_concept16k" / "9b" / "l20" / "train" / "data.parquet"
V2 = sorted((RAW / "axbench_concept16k_v2" / "train").glob("*.parquet"))


def slug(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")[:80]


def load_positives() -> pd.DataFrame:
    cols = ["output_concept", "category", "input", "output"]
    parts = [pq.read_table(V1, columns=cols).to_pandas().assign(src="v1")]
    parts += [pq.read_table(f, columns=cols).to_pandas().assign(src="v2") for f in V2]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.category == "positive"]
    df = df.assign(output=df.output.str.strip(), input=df.input.str.strip(),
                   name=df.output_concept.str.strip().str.lower())
    return df[(df.output.str.len() > 0) & (df.input.str.len() > 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", type=float, default=0.7)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--concepts", help="split JSON (train/heldout/heldout_far)")
    ap.add_argument("--out", default=str(RAW / "expansion_multi"))
    a = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer

    l1 = json.loads((RAW / "level1_data.json").read_text())
    seen = {t["text"].strip() for c in l1["concepts"] for k in ("fit", "calib", "episode")
            for t in c[k]}
    seen |= {t["text"].strip() for k in ("fit", "calib", "episode") for t in l1["negatives"][k]}
    nfit = {c["concept"]: len(c["fit"]) for c in l1["concepts"]}
    df = load_positives()
    df = df[~df.output.isin(seen)].drop_duplicates("output")
    names16 = sorted(df.name.unique())
    l1n = [c["concept"] for c in l1["concepts"]]
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    v.fit([*names16, *(n.lower() for n in l1n)])
    sim = (v.transform([n.lower() for n in l1n]) @ v.transform(names16).T).tocsr()
    by_name = df.groupby("name").size().to_dict()
    syn = {}
    for i, c in enumerate(l1n):
        row = sim.getrow(i)
        syn[c] = sorted({names16[j] for j, s in zip(row.indices, row.data, strict=True)
                         if s >= a.syn} | ({c.lower()} & set(by_name)))
    avail = {c: nfit[c] + sum(by_name.get(n, 0) for n in syn[c]) for c in l1n}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "avail.json").write_text(json.dumps(avail, indent=0))
    print(f"{len(df)} new published positives; concepts with >=150 available: "
          f"{sum(v >= 150 for v in avail.values())}/{len(avail)}", flush=True)
    if a.count_only:
        return
    sp = json.loads(Path(a.concepts).read_text())
    want = list(dict.fromkeys(sp["train"] + sp["heldout"] + sp.get("heldout_far", [])))
    # a text can match several concepts' synonym sets: give it to ONE concept only (first in `want`)
    taken: set[str] = set()
    index, added = {}, {}
    for c in want:
        sub = df[df.name.isin(syn[c]) & ~df.output.isin(taken)]
        taken |= set(sub.output)
        f = out / f"{slug(c)}.json"
        f.write_text(json.dumps({"concept": c, "source": [str(V1)] + [str(x) for x in V2],
                                 "source_concepts": sub.name.value_counts().to_dict(),
                                 "texts": sub.output.tolist()}))
        index[c] = {"file": f.name, "n": len(sub)}
        added.update({text_id(o): i for o, i in zip(sub.output, sub.input, strict=True)})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    plus = RAW / "level1_inputs_plus.json"
    inputs = json.loads(plus.read_text())
    clash = [k for k in added if k in inputs and inputs[k] != added[k]]
    if clash:
        raise SystemExit(f"text-id clash with existing inputs: {len(clash)}")
    tmp = plus.with_suffix(".json.tmp")  # atomic: running jobs read this file at startup
    tmp.write_text(json.dumps({**inputs, **added}))
    tmp.replace(plus)
    print(json.dumps({c: index[c]["n"] for c in want}, indent=0))


if __name__ == "__main__":
    main()
