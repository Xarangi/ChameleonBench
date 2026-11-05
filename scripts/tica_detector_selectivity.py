"""SELECTIVITY of the race's generic detectors (10-25; tica/evdet_defender.py, refit kinds
evdet / pcares): the races show they catch hiding turns; do they also flag the trained model when
it is NOT hiding? Offline on panel dumps: fit the race construction on TRAIN-concept dumps (policy
hiding turns vs the base model on the same turns = the race's seed), calibrate the 1%-FPR
threshold on half of the HELD-OUT base rows (out of sample, as in the race), then flag rates on
held-out concepts for the trained model's hiding turns (real_score, monitor on), no-brief turns,
mismatch turns (brief for another concept), monitor-OFF turns, and the other half of the base
rows. Selective kinds (evdetsel / pcaressel): + the trained model's TRAIN-concept mismatch rows as
negatives (evdetsel) and half of its held-out mismatch rows in the calibration (scored on the other
half). CPU, aggregates only.

  python scripts/tica_detector_selectivity.py --feats DIR --model p1_s1200 --model ps_s1200 --out O
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica_hide_geometry import KEY, load

from next_chameleons.tica.evdet_defender import FPR, _score, _train


def groups(meta: list[dict]) -> dict:
    g = {"hide": [], "nobrief": [], "mismatch": [], "off": []}
    for i, m in enumerate(meta):
        if m["turn"] < 1:
            continue
        if m["cond"] in ("real_score", "real_verdict"):
            g["off" if m["off"] else "hide"].append(i)
        elif m["cond"] in ("nobrief", "mismatch"):
            g[m["cond"]].append(i)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--layers", default="12,16,22")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = Path(a.feats)
    pair = lambda p: [str(F / f"{p}_s1.pt"), str(F / f"{p}_s2.pt")]  # noqa: E731
    _, xb_tr, mb_tr = load(pair("base_train"))
    _, xb_ho, mb_ho = load(pair("base_heldout"))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(mb_ho), generator=g)
    cal_i, test_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    res = {}
    for lab in a.model:
        lt, xt_tr, mt_tr = load(pair(f"{lab}_train"))
        _, xt_ho, mt_ho = load(pair(f"{lab}_heldout"))
        ib = {tuple(m[k] for k in KEY): i for i, m in enumerate(mb_tr)}
        tr_pairs = [(j, ib[tuple(m[k] for k in KEY)]) for j, m in enumerate(mt_tr)
                    if m["cond"] in ("real_score", "real_verdict") and not m["off"]
                    and m["turn"] >= 1 and tuple(m[k] for k in KEY) in ib]
        G = groups(mt_ho)
        res[lab] = {"n_seed_pairs": len(tr_pairs), "n": {k: len(v) for k, v in G.items()},
                    "layers": {}}
        for L in [int(x) for x in a.layers.split(",")]:
            li = lt.index(L)
            P = xt_tr[[j for j, _ in tr_pairs], li]
            N = xb_tr[[i for _, i in tr_pairs], li]
            row = {}
            Mtr = xt_tr[groups(mt_tr)["mismatch"], li]  # selective: the policy's honest rows
            mis = torch.tensor(G["mismatch"])
            mis_cal, mis_test = mis[: len(mis) // 2], mis[len(mis) // 2:]
            for kind in ("evdet", "pcares", "evdetsel", "pcaressel"):
                sel = kind.endswith("sel")
                prm = _train(kind, P, torch.cat([N, Mtr]) if kind == "evdetsel" else N, 0.1)
                q = lambda X, kind=kind, prm=prm: float(  # noqa: E731
                    torch.quantile(_score(kind, prm, X).double(), 1 - FPR))
                thr = q(xb_ho[cal_i, li])
                if sel:
                    thr = max(thr, q(xt_ho[mis_cal, li]))

                def rate(X, kind=kind, prm=prm, thr=thr):
                    return round(float((_score(kind, prm, X) > thr).float().mean()), 3)

                row[kind] = {k: rate(xt_ho[v, li]) for k, v in G.items() if v}
                if sel:  # mismatch scored on the half NOT used for its calibration
                    row[kind]["mismatch"] = rate(xt_ho[mis_test, li])
                row[kind]["base_heldout"] = rate(xb_ho[test_i, li])
            res[lab]["layers"][L] = row
            print(lab, L, json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
