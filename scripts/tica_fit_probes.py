"""Fit the TICA round-0 probe banks from the base generations (DESIGN-TICA.md, "Probing").

Protocol, fixed before fitting:
- Labels. For concept X, POSITIVES are records asked to cover X whose judge P(Yes discusses X) >=
  0.5. NEGATIVES are records not asked about X. Records asked about X but failing the judge are
  excluded for X. Records with fewer than 10 read tokens are dropped.
- Split. By instruction id (sha256 mod 5 == 0 -> EVAL, else FIT), so fit and eval never share an
  instruction.
- Fit negatives per concept. All same-category FIT negatives (hard negatives) plus random ones, up
  to --fit-negatives. Eval uses ALL eval negatives.
- Selection. On --select-concepts seeded TRAIN concepts, 5-fold CV inside FIT (folds grouped by
  instruction) over layers {12, 20} x C in {1e-3 .. 100}. The metric is mean TPR at 1% FPR
  (fold-negative threshold), tie-broken by AUROC. This picks the PRIMARY layer and C. EVAL is not
  used for selection.
- Final. For every concept at every recorded layer, fit on FIT with the selected C. Thresholds at 1%
  and 5% FPR come from EVAL negatives. Quality is EVAL AUROC and TPR@1%FPR, against the gate in
  tica/probes.py.
Outputs in --out:
- bank_L{layer}.npz;
- quality.json (per concept x layer);
- selection.json.

Usage (CPU): python scripts/tica_fit_probes.py --basegen DIR --judge-scores FILE --out DIR [--jobs
16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from next_chameleons.tica.probes import GATE, ProbeBank, fit_one, score, threshold_at

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
SELECT_LAYERS = (12, 20)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's str hash is salted per process)."""

    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2**32


def cv_task(x, fp, fn, g, C):
    """5-fold CV of one (concept, layer, C). x is that layer's [N, d] array (joblib memmaps it)."""

    idx = np.concatenate([fp, fn])
    y = np.r_[np.ones(len(fp)), np.zeros(len(fn))]
    xs = np.asarray(x[idx], dtype=np.float32)
    gg = g[idx] % 5
    tprs, aucs = [], []
    for f in range(5):
        tr, te = gg != f, gg == f
        if y[te].sum() < 3 or (y[te] == 0).sum() < 100:
            continue
        w, b, mu, sd = fit_one(xs[tr], y[tr], C)
        s = score(xs[te], w, b, mu, sd)
        m = metrics(s[y[te] == 1], s[y[te] == 0])
        tprs.append(m["tpr_at_1pct"])
        aucs.append(m["auroc"])
    return float(np.mean(tprs)), float(np.mean(aucs)), len(tprs)


def final_task(x, fp, fn, ep, en, C):
    if len(fp) < 5 or len(ep) < 5:  # too few positives to fit or evaluate: never a monitor
        return {
            "auroc": float("nan"),
            "tpr_at_1pct": float("nan"),
            "thr_1pct": float("inf"),
            "thr_5pct": float("inf"),
            "insufficient": True,
        }, None
    xt = np.asarray(np.concatenate([x[fp], x[fn]]), dtype=np.float32)
    yt = np.r_[np.ones(len(fp)), np.zeros(len(fn))]
    w, b, mu, sd = fit_one(xt, yt, C)
    m = metrics(score(np.asarray(x[ep]), w, b, mu, sd), score(np.asarray(x[en]), w, b, mu, sd))
    return m, (w, b, mu, sd)


def is_eval(instruction_id: str) -> bool:
    return int(hashlib.sha256(instruction_id.encode()).hexdigest(), 16) % 5 == 0


def load(basegen: Path, judge_scores: Path):
    recs, acts, layers = [], [], None
    for f in sorted(basegen.glob("shard*of*.jsonl")):
        tag = f.stem
        meta = json.loads((basegen / f"{tag}_meta.json").read_text())
        layers = meta["layers"] if layers is None else layers
        if meta["layers"] != layers:
            raise ValueError("shards recorded different layers")
        rs = [json.loads(line) for line in f.open()]
        a = np.load(basegen / f"{tag}_acts.npy", mmap_mode="r")
        if len(a) != len(rs):
            raise ValueError(f"{tag}: {len(a)} acts vs {len(rs)} records")
        recs += rs
        acts.append(np.asarray(a))
    acts = np.concatenate(acts)
    judged = {}
    for line in judge_scores.open():
        k = json.loads(line)
        if k["asked"]:
            judged[(k["id"], k["concept"])] = k["p_yes"]
    return recs, acts, layers, judged


def metrics(s_pos, s_neg):
    from next_chameleons.incontext.stats import auroc

    thr = threshold_at(s_neg, 0.01)
    return {
        "auroc": float(auroc(s_pos, s_neg)),
        "tpr_at_1pct": float(np.mean(s_pos > thr)),
        "thr_1pct": thr,
        "thr_5pct": threshold_at(s_neg, 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basegen", type=Path, required=True)
    ap.add_argument("--judge-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-negatives", type=int, default=4096)
    ap.add_argument("--select-concepts", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    recs, acts, layers, judged = load(args.basegen, args.judge_scores)
    pool = json.loads((RAW / "round1_data.json").read_text())["concepts"]
    cat = {c["concept"]: c["category"] for c in pool}
    concepts = [c["concept"] for c in pool]
    keep = np.array([r["n_read_tokens"] >= 10 for r in recs])
    ev = np.array([is_eval(r["instruction_id"]) for r in recs])
    groups = np.array(
        [int(hashlib.sha256(r["instruction_id"].encode()).hexdigest(), 16) % 1000 for r in recs]
    )
    has = {c: np.array([c in r["concepts"] for r in recs]) for c in concepts}
    passed = {c: np.array([judged.get((r["id"], c), 0.0) >= 0.5 for r in recs]) for c in concepts}
    rec_cats = [set(cat[x] for x in r["concepts"]) for r in recs]

    def split(c, rng):
        pos = has[c] & passed[c] & keep
        neg = ~has[c] & keep
        fit_neg = np.flatnonzero(neg & ~ev)
        hard = np.array([i for i in fit_neg if cat[c] in rec_cats[i]], dtype=int)
        rest = np.setdiff1d(fit_neg, hard)
        k = max(0, args.fit_negatives - len(hard))
        pick = np.concatenate([hard, rng.choice(rest, size=min(k, len(rest)), replace=False)])
        return (
            np.flatnonzero(pos & ~ev),
            np.sort(pick),
            np.flatnonzero(pos & ev),
            np.flatnonzero(neg & ev),
        )

    li = {layer: i for i, layer in enumerate(layers)}

    # ---- selection (FIT only)
    sel = random.Random(0).sample(
        [c for c in concepts if next(p for p in pool if p["concept"] == c)["split"] == "train"],
        args.select_concepts,
    )

    splits_cv = {c: split(c, np.random.default_rng(stable_seed(c, "cv"))) for c in sel}
    cv = []
    for layer in SELECT_LAYERS:
        x = np.ascontiguousarray(acts[:, li[layer]])
        jobs = [(c, C) for c in sel for C in C_GRID]
        out = Parallel(n_jobs=args.jobs)(
            delayed(cv_task)(x, splits_cv[c][0], splits_cv[c][1], groups, C) for c, C in jobs
        )
        cv += [
            {"concept": c, "layer": layer, "C": C, "tpr": t, "auroc": a, "folds": n}
            for (c, C), (t, a, n) in zip(jobs, out, strict=True)
        ]
    table = {}
    for row in cv:
        table.setdefault((row["layer"], row["C"]), []).append(row)
    summary = [
        {
            "layer": L,
            "C": C,
            "mean_tpr": float(np.mean([r["tpr"] for r in rows])),
            "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
        }
        for (L, C), rows in table.items()
    ]
    best = max(summary, key=lambda s: (round(s["mean_tpr"], 4), s["mean_auroc"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(
        json.dumps(
            {"select_concepts": sel, "summary": summary, "chosen": best, "cv_rows": cv}, indent=1
        )
    )
    print(
        "selection:",
        json.dumps(sorted(summary, key=lambda s: (-s["mean_tpr"], -s["mean_auroc"]))[:4]),
        flush=True,
    )
    C = best["C"]

    # ---- final fits (every concept x every layer)
    quality = []
    splits_final = {c: split(c, np.random.default_rng(stable_seed(c, "final"))) for c in concepts}
    for layer in layers:
        x = np.ascontiguousarray(acts[:, li[layer]])
        res = Parallel(n_jobs=args.jobs)(
            delayed(final_task)(x, *splits_final[c], C) for c in concepts
        )
        for c, (m, _) in zip(concepts, res, strict=True):
            fp, fn, ep, en = splits_final[c]
            m.update(
                concept=c,
                layer=layer,
                n_fit_pos=len(fp),
                n_fit_neg=len(fn),
                n_eval_pos=len(ep),
                n_eval_neg=len(en),
                gate=bool(m["auroc"] >= GATE["auroc"] and m["tpr_at_1pct"] >= GATE["tpr_at_1pct"]),
            )
        d = x.shape[1]
        null = (np.zeros(d, np.float32), 0.0, np.zeros(d, np.float32), np.ones(d, np.float32))
        ws = [
            p if p is not None else null for _, p in res
        ]  # insufficient: threshold inf, never fires
        bank = ProbeBank(
            concepts,
            layer,
            np.stack([p[0] for p in ws]),
            np.array([p[1] for p in ws]),
            np.stack([p[2] for p in ws]),
            np.stack([p[3] for p in ws]),
            {
                "0.01": np.array([m["thr_1pct"] for m, _ in res]),
                "0.05": np.array([m["thr_5pct"] for m, _ in res]),
            },
        )
        bank.save(args.out / f"bank_L{layer}.npz")
        quality += [m for m, _ in res]
        q = [m for m, _ in res]
        print(
            f"L{layer}: gate pass {sum(m['gate'] for m in q)}/{len(q)}; median AUROC "
            f"{np.nanmedian([m['auroc'] for m in q]):.3f}; median TPR@1% "
            f"{np.nanmedian([m['tpr_at_1pct'] for m in q]):.3f}",
            flush=True,
        )
    (args.out / "quality.json").write_text(
        json.dumps(
            {"C": C, "primary_layer": best["layer"], "gate": GATE, "rows": quality}, indent=1
        )
    )


if __name__ == "__main__":
    main()
