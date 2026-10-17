"""PASS RATE BY PROBE INDEX from a run's turns.jsonl (REPLICATION 2026-10-17u), for runs whose train
log predates the beatk_* metrics (supervised.probe_index_metrics). Per window of training steps and
per bank index k (k = 1: a concept segment's first probe; k = 2: after its first refit; ...): the
fraction of segments that passed bank k before the segment ended, the mean judged turns to that
pass, and the segment count. The first bank is split into the episode's first concept vs a
switched-to concept. Aggregates only.

  python scripts/tica_probe_index.py --run RUN_DIR [--window 20] [--max-k 5] [--out O.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def by_index(rows, window: int, max_k: int) -> dict:
    segs = defaultdict(list)
    first = {}
    for r in rows:
        if r.get("cond") not in ("real_score", "real_verdict") or r.get("off"):
            continue
        if r["turn"] == 0:
            first[(r["step"], r["ep"])] = r["concept"]
        segs[(r["step"], r["ep"], r["concept"])].append(r)
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # window -> key -> [n, beat, turns]
    for (st, ep, c), rs in segs.items():
        w = (st - 1) // window * window
        rs = sorted(rs, key=lambda r: r["turn"])
        start, beat = {}, {}
        for r in rs:
            k = min(max_k, max(1, r.get("n_deployed") or 1))
            start.setdefault(k, r["turn"])
            if k not in beat and not r["detected"]:
                beat[k] = r["turn"] - start[k] + 1
        for k in start:
            keys = [f"k{k}"]
            if k == 1:
                keys.append("k1_first" if first.get((st, ep)) == c else "k1_switch")
            for key in keys:
                a = acc[w][key]
                a[0] += 1
                if k in beat:
                    a[1] += 1
                    a[2] += beat[k]
    return {f"{w}-{w + window}": {key: {"beaten": round(b / n, 3), "n": n,
                                         "turns": round(t / b, 2) if b else None}
                                   for key, (n, b, t) in sorted(d.items())}
            for w, d in sorted(acc.items()) if w >= 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for sub in [a.run / "adaptive", *sorted(a.run.glob("resume_*/adaptive"))]:
        f = sub / "turns.jsonl"
        if not f.exists():
            continue
        off = int(sub.parent.name.split("_")[1]) if sub.parent.name.startswith("resume_") else 0
        rows = [r for r in rows if r["step"] <= off]
        rows += [{**r, "step": r["step"] + off} for r in map(json.loads, f.open())]
    rep = by_index(rows, a.window, a.max_k)
    for w, d in rep.items():
        print(w, " | ".join(f"{k}: {v['beaten']:.2f} (n={v['n']}, {v['turns']}t)"
                            for k, v in d.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
