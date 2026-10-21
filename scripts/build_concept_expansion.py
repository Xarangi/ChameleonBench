"""Extra PUBLISHED positives for a scarce concept (REPLICATION 2026-10-13d).

AxBench Concept500 has 72 fit texts for "mathematical notation and symbols"; refits that fork the
policy's passing state need more (rejected near-misses at AUROC 0.85-0.95). AxBench Concept16K (v1,
9b/l20, same pipeline) has STRICT SYNONYMS of the concept (exact-regex names, listed in the output)
with 1008 positives, none of which appears anywhere in the Level-1 data (checked here). Used
ONLY as refit fork positives / OOC-gate texts; Level-1 episode and eval texts are unchanged.

Writes $RAW/tica/expansion_<slug>.json  {concept, source_concepts, texts}
   and $RAW/tica/level1_inputs_plus.json  (level1_inputs + {text_id: instruction} for the new texts)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from next_chameleons.tica.answer_format import text_id

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
SRC = RAW / "axbench_concept16k" / "9b" / "l20" / "train" / "data.parquet"
CONCEPT = "mathematical notation and symbols"
STRICT = re.compile(r"^(mathematical (symbols?|notations?)( and (notations?|symbols?))?"
                    r"|symbols and mathematical notations?|mathematical notations? and symbols)$",
                    re.I)


def main() -> None:
    df = pq.read_table(SRC, columns=["output_concept", "category", "input", "output"]).to_pandas()
    df = df[(df.category == "positive") & df.output_concept.map(lambda c: bool(STRICT.match(c)))]
    df = df.assign(output=df.output.str.strip(), input=df.input.str.strip())
    df = df[(df.output.str.len() > 0) & (df.input.str.len() > 0)].drop_duplicates("output")
    l1 = json.loads((RAW / "level1_data.json").read_text())
    seen = {t["text"].strip() for c in l1["concepts"] for k in ("fit", "calib", "episode")
            for t in c[k]}
    seen |= {t["text"].strip() for k in ("fit", "calib", "episode") for t in l1["negatives"][k]}
    df = df[~df.output.isin(seen)]
    inputs = json.loads((RAW / "level1_inputs.json").read_text())
    added = {text_id(o): i for o, i in zip(df.output, df.input, strict=True)}
    clash = [k for k in added if k in inputs]
    if clash:
        raise SystemExit(f"text-id clash with Level-1 inputs: {len(clash)}")
    slug = re.sub(r"[^a-z0-9]+", "_", CONCEPT.lower()).strip("_")
    out = {"concept": CONCEPT, "source": str(SRC),
           "source_concepts": df.output_concept.value_counts().to_dict(),
           "texts": df.output.tolist()}
    (RAW / f"expansion_{slug}.json").write_text(json.dumps(out))
    (RAW / "level1_inputs_plus.json").write_text(json.dumps({**inputs, **added}))
    print(f"{CONCEPT}: {len(df)} new positives from {out['source_concepts']}; "
          f"inputs {len(inputs)} -> {len(inputs) + len(added)}")


if __name__ == "__main__":
    main()
