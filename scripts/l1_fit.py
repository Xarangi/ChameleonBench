"""TICA Level 1: fit and calibrate the per-concept monitors (DESIGN-TICA.md "FIRST SETUP").

Protocol, fixed before fitting:
- Fit (plain format, no brief):
  - positives = X's FIT texts;
  - negatives = the FIT concept-free negatives plus --other-neg random FIT texts of other concepts.
  - Standardized L2 logistic regression (tica/probes.fit_one).
- Selection: on --select-concepts seeded TRAIN concepts, 5-fold CV inside FIT over the two middle
  recorded layers (depth 0.375 and 0.5) x C. The metric is mean TPR at 1% FPR (fold-negative
  threshold). ONE-STANDARD-ERROR RULE (10-03 review): among the settings whose mean TPR is within
  one SE (over concepts) of the best, take the smallest C, then the higher TPR. The smoke CVs
  picked C=100 by < 1 point of TPR (noise), giving near-unregularized monitors. Calibration data
  is not used for selection.
- Calibration (episode format):
  - threshold = the 99th percentile of X's context-matched negatives (brief naming X,
    concept-free text);
  - 0 point = their mean;
  - 100 point = mean of X's CALIB positives with the brief (= turn 1 of an episode).
- Quality: AUROC and TPR at the 1% threshold, calib_pos vs ctx_neg. Gate = tica/probes.GATE. Also
  reported:
  - plain-format AUROC (plain_pos vs plain_neg);
  - the brief shift (mean ctx_neg - mean plain_neg logit, in units of the plain_neg sd).
Outputs in --out:
- bank_L{layer}.npz for every recorded layer (W, b, mu, sd, thr1, thr5, zero, hundred, gate,
  concepts);
- quality.json;
- selection.json.

Usage (CPU): python scripts/l1_fit.py --reads DIR --out DIR [--jobs 32]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from next_chameleons.tica.probes import GATE, fit_one, score, threshold_at

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
C_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def auroc(pos, neg) -> float:
    from next_chameleons.incontext.stats import auroc as _a

    return float(_a(np.asarray(pos), np.asarray(neg)))


def cv_task(x, pos, neg, C, seed):
    rng = np.random.default_rng(seed)
    fp, fn = rng.integers(0, 5, len(pos)), rng.integers(0, 5, len(neg))
    tprs, aucs = [], []
    for f in range(5):
        tr = np.r_[pos[fp != f], neg[fn != f]]
        ytr = np.r_[np.ones((fp != f).sum()), np.zeros((fn != f).sum())]
        w, b, mu, sd = fit_one(np.asarray(x[tr], np.float32), ytr, C)
        sp = score(np.asarray(x[pos[fp == f]]), w, b, mu, sd)
        sn = score(np.asarray(x[neg[fn == f]]), w, b, mu, sd)
        thr = threshold_at(sn, 0.01)
        tprs.append(float(np.mean(sp > thr)))
        aucs.append(auroc(sp, sn))
    return float(np.mean(tprs)), float(np.mean(aucs))


def final_task(xf, pos, neg, C, xc_pos, xc_neg, xp_pos, xp_neg):
    tr = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    w, b, mu, sd = fit_one(np.asarray(xf[tr], np.float32), y, C)
    sp, sn = score(xc_pos, w, b, mu, sd), score(xc_neg, w, b, mu, sd)
    pp, pn = score(xp_pos, w, b, mu, sd), score(xp_neg, w, b, mu, sd)
    thr1, thr5 = threshold_at(sn, 0.01), threshold_at(sn, 0.05)
    q = {
        "auroc": auroc(sp, sn),
        "tpr_at_1pct": float(np.mean(sp > thr1)),
        "tpr_at_5pct": float(np.mean(sp > thr5)),
        "plain_auroc": auroc(pp, pn),
        "brief_shift_sd": float((sn.mean() - pn.mean()) / (pn.std() + 1e-9)),
        "pos_minus_thr_sd": float((np.median(sp) - thr1) / (sn.std() + 1e-9)),
    }
    cal = {"thr1": thr1, "thr5": thr5, "zero": float(sn.mean()), "hundred": float(sp.mean())}
    return q, cal, (w, b, mu, sd)


def choose_setting(rows: list[dict]) -> tuple[dict, list[dict]]:
    """rows: per (concept, layer, C) CV results. Returns (chosen summary row, all summary rows).
    One-SE rule: best mean TPR; candidates within one SE of it; smallest C, then highest TPR."""

    by: dict = {}
    for r in rows:
        by.setdefault((r["layer"], r["C"]), []).append(r)
    summary = []
    for (L, C), v in by.items():
        t = np.array([r["tpr"] for r in v])
        summary.append(
            {
                "layer": L,
                "C": C,
                "mean_tpr": float(t.mean()),
                "se_tpr": float(t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else 0.0,
                "mean_auroc": float(np.mean([r["auroc"] for r in v])),
            }
        )
    top = max(summary, key=lambda s: (s["mean_tpr"], s["mean_auroc"]))
    ok = [s for s in summary if s["mean_tpr"] >= top["mean_tpr"] - top["se_tpr"]]
    best = min(ok, key=lambda s: (s["C"], -s["mean_tpr"], -s["mean_auroc"]))
    return {**best, "rule": "one_se", "best_raw": top}, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--other-neg", type=int, default=2000)
    ap.add_argument("--select-concepts", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    lay = json.loads((args.reads / "layers.json").read_text())
    layers = lay["layers"]
    load = lambda n: (  # noqa: E731
        np.load(args.reads / f"{n}.npy", mmap_mode="r"),
        json.loads((args.reads / f"{n}.json").read_text()),
    )
    fit, fit_m = load("fit")
    cpos, cpos_m = load("calib_pos")
    cneg, cneg_m = load("ctx_neg")
    ppos, ppos_m = load("plain_pos")
    pneg, _ = load("plain_neg")
    data = json.loads((RAW / "level1_data.json").read_text())
    concepts = [
        c["concept"] for c in data["concepts"] if any(m["concept"] == c["concept"] for m in cpos_m)
    ]
    split = {c["concept"]: c["split"] for c in data["concepts"]}
    fit_c = np.array([m["concept"] or "" for m in fit_m])
    free = np.flatnonzero(fit_c == "")
    by = lambda arr_m, c: np.flatnonzero(np.array([m["concept"] for m in arr_m]) == c)  # noqa: E731

    def fit_split(c):
        pos = np.flatnonzero(fit_c == c)
        other = np.flatnonzero((fit_c != c) & (fit_c != ""))
        r = np.random.default_rng(random.Random(f"o|{c}").randrange(2**32))
        neg = np.r_[free, r.choice(other, size=min(args.other_neg, len(other)), replace=False)]
        return pos, np.sort(neg)

    li = {L: i for i, L in enumerate(layers)}
    mid = sorted(layers, key=lambda L: abs(L / lay["n_layers"] - 0.4375))[:2]
    pool = [c for c in concepts if split[c] == "train"]
    sel = random.Random(0).sample(pool, min(args.select_concepts, len(pool)))
    splits = {c: fit_split(c) for c in concepts}
    rows = []
    for L in mid:
        x = np.ascontiguousarray(fit[:, li[L]])
        jobs = [(c, C) for c in sel for C in C_GRID]
        out = Parallel(n_jobs=args.jobs)(
            delayed(cv_task)(x, *splits[c], C, random.Random(f"cv|{c}").randrange(2**32))
            for c, C in jobs
        )
        rows += [
            {"concept": c, "layer": L, "C": C, "tpr": t, "auroc": a}
            for (c, C), (t, a) in zip(jobs, out, strict=True)
        ]
    best, summary = choose_setting(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(
        json.dumps(
            {"candidates": mid, "summary": summary, "chosen": best, "cv_rows": rows}, indent=1
        )
    )
    print(
        "selection:",
        json.dumps(sorted(summary, key=lambda s: (-s["mean_tpr"], -s["mean_auroc"]))[:4]),
        flush=True,
    )
    C = best["C"]

    quality = []
    idx_c = {c: (by(cpos_m, c), by(cneg_m, c), by(ppos_m, c)) for c in concepts}
    for L in layers:
        i = li[L]
        xf = np.ascontiguousarray(fit[:, i])
        pneg_all = np.asarray(pneg[:, i])
        res = Parallel(n_jobs=args.jobs)(
            delayed(final_task)(
                xf,
                *splits[c],
                C,
                np.asarray(cpos[idx_c[c][0], i]),
                np.asarray(cneg[idx_c[c][1], i]),
                np.asarray(ppos[idx_c[c][2], i]),
                pneg_all,
            )
            for c in concepts
        )
        bank = {k: [] for k in ("W", "b", "mu", "sd", "thr1", "thr5", "zero", "hundred", "gate")}
        for c, (q, cal, (w, b, mu, sd)) in zip(concepts, res, strict=True):
            gate = bool(q["auroc"] >= GATE["auroc"] and q["tpr_at_1pct"] >= GATE["tpr_at_1pct"])
            quality.append({"concept": c, "layer": L, "split": split[c], "gate": gate, **q, **cal})
            for k, v in (("W", w), ("b", b), ("mu", mu), ("sd", sd), ("gate", gate), *cal.items()):
                bank[k].append(v)
        np.savez(
            args.out / f"bank_L{L}.npz",
            concepts=np.array(concepts),
            layer=L,
            C=C,
            **{k: np.array(v) for k, v in bank.items()},
        )
        q = [r for r in quality if r["layer"] == L]
        med = lambda k, q=q: np.median([r[k] for r in q])  # noqa: E731
        ng = lambda s, q=q: sum(r["gate"] for r in q if r["split"] == s)  # noqa: E731
        print(
            f"L{L}: gate {sum(r['gate'] for r in q)}/{len(q)} (train {ng('train')}, heldout "
            f"{ng('heldout')}); median AUROC {med('auroc'):.3f}; TPR@1% {med('tpr_at_1pct'):.3f}; "
            f"plain AUROC {med('plain_auroc'):.3f}; brief shift {med('brief_shift_sd'):+.2f} sd; "
            f"pos-thr {med('pos_minus_thr_sd'):.1f} sd",
            flush=True,
        )
    (args.out / "quality.json").write_text(
        json.dumps(
            {"C": C, "primary_layer": best["layer"], "gate": GATE, "rows": quality}, indent=1
        )
    )


if __name__ == "__main__":
    main()
