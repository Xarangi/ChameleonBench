"""Merge SHARDED never-shown probe panels (TamIA 10-22; scripts/slurm/tamia_queue_quality.sh runs one panel as
PANEL_SHARDS processes with different --seed and episodes / shards each, so the eval node's GPUs are not left idle
behind one long panel). Concatenates the shards' per-turn numeric records (episode ids offset per shard) and
recomputes the aggregates exactly as tica_probe_panel.py does; by_probe_index is pooled with its counts.
Numbers only (records carry no text).
  python scripts/tica_panel_merge.py --out panel_train.json panel_train.s0.json panel_train.s1.json ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from next_chameleons.tica.panel import aggregate, selectivity


def pool_index(parts: list[dict]) -> dict:
    """by_probe_index pooled over shards: beatk_frac_k weighted by beatk_n_k, turns by beats."""

    out: dict = {}
    ks = sorted({int(m.group(1)) for p in parts for k in p
                 if (m := re.match(r"beatk_n_(\d+)$", k))})
    for k in ks:
        n = sum(p.get(f"beatk_n_{k}", 0) or 0 for p in parts)
        beat = sum((p.get(f"beatk_frac_{k}") or 0) * (p.get(f"beatk_n_{k}", 0) or 0) for p in parts)
        out[f"beatk_n_{k}"] = n
        out[f"beatk_frac_{k}"] = beat / n if n else None
        tw = [(p.get(f"beatk_turns_{k}"), (p.get(f"beatk_frac_{k}") or 0) * (p.get(f"beatk_n_{k}", 0) or 0))
              for p in parts]
        tw = [(t, w) for t, w in tw if t is not None and t == t and w > 0]
        out[f"beatk_turns_{k}"] = (sum(t * w for t, w in tw) / sum(w for _, w in tw)) if tw else None
    firsts = [(p.get("beat1_first_frac"), p.get("beatk_n_1", 0) or 0) for p in parts]
    firsts = [(f, n) for f, n in firsts if f is not None and n]
    if firsts:
        out["beat1_first_frac"] = sum(f * n for f, n in firsts) / sum(n for _, n in firsts)
    return out


def merge(paths: list[Path]) -> dict:
    shards = [json.loads(p.read_text()) for p in paths]
    records = []
    for j, sh in enumerate(shards):
        for r in sh["records"]:
            records.append({**r, "ep": j * 100000 + r["ep"]})
    turns = shards[0]["turns"]
    base = {k: v for k, v in shards[0].items()
            if k not in ("records", "aggregate", "selectivity", "by_probe_index", "episodes", "sec")}
    return {**base,
            "episodes": sum(sh["episodes"] for sh in shards),
            "panel_names": sorted({p for sh in shards for p in sh.get("panel_names", [])}),
            "panel_failed": {k: v for sh in shards for k, v in (sh.get("panel_failed") or {}).items()},
            "by_probe_index": pool_index([sh["by_probe_index"] for sh in shards]),
            "aggregate": aggregate(records, turns),
            "selectivity": selectivity(records, turns),
            "records": records,
            "sec": max(sh.get("sec", 0) for sh in shards),
            "shards": [str(p) for p in paths]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    paths = [Path(p) for p in a.shards]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    Path(a.out).write_text(json.dumps(merge(paths)))
    print(json.dumps({"merged": a.out, "shards": len(paths)}))


if __name__ == "__main__":
    main()
