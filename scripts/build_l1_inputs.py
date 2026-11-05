"""Sidecar for the ANSWER format (REPLICATION 2026-10-13a): map every Level-1 text id to the AxBench
`input` (the instruction the passage was written to answer), so an episode turn can be
"brief + the real instruction -> the fixed passage as the answer" (the Neural Chameleons format)
instead of "repeat this text". build_l1_data.py dropped any output that appears twice, so output ->
input is unambiguous. Splits are untouched (same ids as level1_data.json).

Output: $RAW/tica/level1_inputs.json  {text_id: input}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache/tica")
SRC = RAW / "axbench_concept500" / "9b" / "l20"


def main() -> None:
    data = json.loads((RAW / "level1_data.json").read_text())
    df = pd.concat([pd.read_parquet(SRC / s / "data.parquet") for s in ("train", "test")])
    df["output"] = df["output"].str.strip()
    by_out = df.groupby("output")["input"].first().to_dict()
    items = [t for c in data["concepts"] for k in ("fit", "calib", "episode") for t in c[k]]
    items += [t for k in ("fit", "calib", "episode") for t in data["negatives"][k]]
    out, missing = {}, 0
    for t in items:
        inp = by_out.get(t["text"])
        if inp is None or not str(inp).strip():
            missing += 1
            continue
        out[t["id"]] = str(inp).strip()
    (RAW / "level1_inputs.json").write_text(json.dumps(out))
    print(f"texts {len(items)}  with input {len(out)}  missing {missing}")


if __name__ == "__main__":
    main()
