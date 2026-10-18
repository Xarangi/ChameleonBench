"""SUPERVISED EVASION DETECTOR (tica/evasion_detector.py) over panel-race feature dumps
(scripts/tica_probe_panel.py --dump-feats). Every (source, target) pair of the given files: train on
the source's TRAIN-concept turns (dodging vs honest reference), test on the target's HELD-OUT
concepts, per layer and reference type. Pairs with the same file = the in-checkpoint result; the
base file = the brief-in-context control (excess = trained - base); cross-file = transfer.
CPU only; aggregates only.

  python scripts/tica_evasion_detector.py --feats s400=A.pt base=B.pt s140=C.pt --base base \
         --out O.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from next_chameleons.tica.evasion_detector import REFS, excess, labels, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True, help="name=path.pt ...")
    ap.add_argument("--base", default="base", help="name of the base-model file (control)")
    ap.add_argument("--refs", default=",".join(REFS))
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {k: torch.load(v, weights_only=False) for k, v in (x.split("=", 1) for x in a.feats)}
    rep = {"files": {k: v.get("adapter") or ("BASE" if v.get("base") else None)
                     for k, v in files.items()},
           "counts": {}, "pairs": {}, "excess_auroc": {}, "excess_tpr": {}}
    refs = [r for r in a.refs.split(",") if r]
    for k, f in files.items():
        rep["counts"][k] = {r: {sp: [int((y == 1).sum()), int((y == 0).sum())]
                                for sp in ("train", "heldout")
                                for _, y in [labels(f["meta"], r, sp, a.min_turn)]}
                            for r in refs}
    for r in refs:
        for s in files:
            for d in files:
                res = run(files[s], files[d], r, a.C, a.min_turn)
                rep["pairs"][f"{s}->{d}|{r}"] = res
                print(s, "->", d, r, {L: v.get("auroc") for L, v in res.items()}, flush=True)
        if a.base in files:
            b = rep["pairs"][f"{a.base}->{a.base}|{r}"]
            for k in files:
                if k != a.base:
                    t = rep["pairs"][f"{k}->{k}|{r}"]
                    rep["excess_auroc"][f"{k}|{r}"] = excess(t, b)
                    rep["excess_tpr"][f"{k}|{r}"] = excess(t, b, "tpr_at_heldout_1pct")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"excess_auroc": rep["excess_auroc"]}, indent=1))


if __name__ == "__main__":
    main()
