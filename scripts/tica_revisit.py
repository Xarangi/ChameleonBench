"""BLIND SEARCH or memory? (10-26, user: "it doesn't keep track of what it has avoided, it just
tries to jump to a new place?"). Panel dumps, base-paired (tica_dodge_anatomy's D = x_trained -
x_base; r = D - h_c = the turn's own configuration beyond the concept's mean hide edit). For every
step out of a CAUGHT turn t (next turn j = t + 1, same episode / concept, monitored hiding turns):
the similarity of r_j to the configurations of EARLIER turns s < t of the episode that were CAUGHT
(known failures) vs that PASSED (known successes), per layer:
  sim_failed = mean cos(r_j, r_s), s caught;  sim_passed = mean cos(r_j, r_s), s passed.
Memory-guided search predicts sim_failed < sim_passed (away from what failed, towards what
worked); blind search predicts no difference. Bootstrap over episodes (sim_passed - sim_failed).
CPU only, aggregates only.

  python scripts/tica_revisit.py --base base_s1.pt,base_s2.pt --model p1=p1_s1.pt,p1_s2.pt --out O
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica_hide_geometry import KEY, cos, load

H_CONDS = ("real_score", "real_verdict")


def analyse(layers, xb, mb, xt, mt, n_boot: int = 400) -> dict:
    ib = {tuple(m[k] for k in KEY): i for i, m in enumerate(mb)}
    pairs = [(ib[tuple(m[k] for k in KEY)], j) for j, m in enumerate(mt)
             if tuple(m[k] for k in KEY) in ib]
    meta = [mt[j] for _, j in pairs]
    H = [i for i, m in enumerate(meta) if m["cond"] in H_CONDS and not m["off"] and m["turn"] >= 1]
    N = [i for i, m in enumerate(meta) if m["cond"] == "nobrief" and m["turn"] >= 1]
    ep = defaultdict(list)
    for i in H:
        ep[(meta[i]["shard"], meta[i]["ep"])].append(i)
    for v in ep.values():
        v.sort(key=lambda i: meta[i]["turn"])
    out = {}
    for li, L in enumerate(layers):
        B = xb[[p[0] for p in pairs], li]
        T = xt[[p[1] for p in pairs], li]
        D = T - B
        dN = D[N].mean(0)
        h = {c: D[[i for i in H if meta[i]["concept"] == c]].mean(0) - dN
             for c in {meta[i]["concept"] for i in H}}
        r = {i: D[i] - h[meta[i]["concept"]] for i in H}
        rows = []  # (episode, sim_failed, sim_passed)
        for g, idx in ep.items():
            pos = {meta[i]["turn"]: i for i in idx}
            for i in idx:
                t = meta[i]["turn"]
                j = pos.get(t + 1)
                if not meta[i]["detected"] or j is None or meta[j]["concept"] != meta[i]["concept"]:
                    continue
                earlier = [s for s in idx
                           if meta[s]["turn"] < t and meta[s]["concept"] == meta[i]["concept"]]
                f = [cos(r[j], r[s]) for s in earlier if meta[s]["detected"]]
                p = [cos(r[j], r[s]) for s in earlier if not meta[s]["detected"]]
                if f and p:
                    rows.append((g, sum(f) / len(f), sum(p) / len(p)))
        if len(rows) < 5:
            out[L] = {"n": len(rows)}
            continue
        by = defaultdict(list)
        for g, f, p in rows:
            by[g].append(p - f)
        keys, rng, bs = list(by), random.Random(0), []
        for _ in range(n_boot):
            s = [d for k in (rng.choice(keys) for _ in keys) for d in by[k]]
            bs.append(sum(s) / len(s))
        bs.sort()
        out[L] = {"n": len(rows), "episodes": len(by),
                  "sim_failed": round(sum(f for _, f, _ in rows) / len(rows), 3),
                  "sim_passed": round(sum(p for _, _, p in rows) / len(rows), 3),
                  "passed_minus_failed": [round(sum(p - f for _, f, p in rows) / len(rows), 3),
                                          round(bs[int(0.025 * n_boot)], 3),
                                          round(bs[int(0.975 * n_boot)], 3)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers, xb, mb = load(a.base.split(","))
    res = {}
    for spec in a.model:
        lab, mp = spec.split("=", 1)
        _, xt, mt = load(mp.split(","))
        res[lab] = analyse(layers, xb, mb, xt, mt)
        for L, v in res[lab].items():
            print(lab, L, json.dumps(v), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
