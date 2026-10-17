"""Probe-based topic overlap for a topic base (REPLICATION 2026-10-14i).

The 10-14h check trained each probe against CONCEPT-FREE text only, so every probe learned "is a
topical text" and fired on every other topic (median cross-fire 1.0): invalid. Here a probe for
topic A is trained, as in TICA training, against OTHER TOPICS: the other topics are split into
--groups groups; for each group g, A's L12 linear probe is fit on A vs topics not in g, thresholded
at 1% FPR on held-out texts of those topics, and then scores the topics IN g, which it never saw.
cross[A][B] = fraction of B's texts on which A's probe fires. Disjoint topics: ~1%. Also saves the
mean-diff directions (geometry) and A's held-out TPR. Answer format: user = a fixed generic request
(no topic information in the prompt), assistant = the abstract. Aggregates only.

--onpolicy N (REPLICATION 2026-10-14n): validity on the MODEL'S OWN writing, which is what TICA
probes read. The model writes N short entries per topic ("a real <subclass> of your choice",
round-robin over the topic's DBpedia subclasses); each is re-read under the same generic request.
A probe for A fit on A vs ALL other topics' abstracts (1% FPR on held-out abstracts) then scores
every topic's generations:
onpolicy_tpr[A] (fires on A's generations) and onpolicy_cross[A][B] (fires on B's generations).

  python scripts/tica_topic_overlap.py --config l5dodge_math_4b --topics TOPICS.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build
from tica_cross_detection import fit_lr

from next_chameleons.tica.config import load_run_config

REQUEST = "Write a short encyclopedia entry."


NAMES = {"Medician": "physician", "Mollusca": "mollusc",
         "CultivatedVariety": "cultivated plant variety",
         "AnimangaCharacter": "anime or manga character", "GreenAlga": "green alga",
         "SupremeCourtOfTheUnitedStatesCase": "US Supreme Court case"}


def readable(cls):
    import re

    return NAMES.get(cls, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cls).lower())


def onpolicy(pol, topics, names, X, a):
    import numpy as np
    import torch

    t0 = time.time()
    convs, owner = [], []
    for c in names:
        subs = topics[c]["classes"]
        for i in range(a.onpolicy):
            req = f"Write a short encyclopedia entry about a real {readable(subs[i % len(subs)])}"
            convs.append([{"role": "user", "content": req + " of your choice."}])
            owner.append(c)
    bs, pol.batch_size = pol.batch_size, 64
    with pol.reference(), torch.no_grad():
        gens = pol.generate(convs, 220, temperature=0.8, top_p=0.95, seed=1)
        pol.batch_size = bs
        texts = [g["text"] for g in gens]
        R = [[{"role": "user", "content": REQUEST}, {"role": "assistant", "content": t}]
             for t in texts]
        G = torch.cat([pol.pooled(R[i : i + 32], [a.layer])[a.layer].float()
                       for i in range(0, len(R), 32)])
    own = np.array(owner)
    tpr, cross = {}, {A: {} for A in names}
    for A in names:
        k = len(X[A]) * 3 // 4
        neg = torch.cat([X[c] for c in names if c != A])
        perm = torch.randperm(len(neg), generator=torch.Generator().manual_seed(1))
        nh = len(neg) // 4
        f = fit_lr(X[A][:k], neg[perm[nh:]])
        thr = torch.quantile(f(neg[perm[:nh]]), 0.99)
        fire = (f(G) > thr).float().cpu().numpy()
        for B in names:
            v = float(fire[own == B].mean())
            if B == A:
                tpr[A] = v
            else:
                cross[A][B] = v
    allx = [v for A in names for v in cross[A].values()]
    lens = [len(g["ids"]) for g in gens]
    print("onpolicy tpr", {k: round(v, 2) for k, v in tpr.items()}, flush=True)
    return {"onpolicy": {"n_per_topic": a.onpolicy, "sec": round(time.time() - t0),
                         "gen_tokens_mean": float(np.mean(lens)),
                         "finished_frac": float(np.mean([g["finished"] for g in gens])),
                         "tpr": tpr, "cross": cross,
                         "cross_median": float(np.median(allx)),
                         "cross_over_0.2": sum(v > 0.2 for v in allx),
                         "top_cross": sorted(([cross[A][B], A, B] for A in names
                                              for B in cross[A]), reverse=True)[:30]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--onpolicy", type=int, default=0, help="generations per topic (0 = off)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    _, _, factory = build(cfg)
    pol = factory(None, False)
    topics = json.loads(Path(a.topics).read_text())
    names = sorted(topics)
    rng = random.Random(0)
    t0 = time.time()
    X = {}
    with pol.reference(), torch.no_grad():
        for c in names:
            texts = topics[c]["texts"][: a.n]
            convs = [[{"role": "user", "content": REQUEST},
                      {"role": "assistant", "content": x}] for x in texts]
            X[c] = torch.cat([pol.pooled(convs[i : i + 32], [a.layer])[a.layer].float()
                              for i in range(0, len(convs), 32)])
    read_sec = time.time() - t0
    means = torch.stack([X[c].mean(0) for c in names])
    sd = torch.cat([X[c] for c in names]).std(0).clamp(min=1e-4)
    dirs = [((means[i] - torch.cat([means[:i], means[i + 1 :]]).mean(0)) / sd).cpu().numpy()
            for i in range(len(names))]
    cross, tpr = {A: {} for A in names}, {}
    for A in names:
        others = [c for c in names if c != A]
        rng.shuffle(others)
        groups = [others[g :: a.groups] for g in range(a.groups)]
        k = len(X[A]) * 3 // 4
        tprs = []
        for g in groups:
            rest = [c for c in others if c not in g]
            neg = torch.cat([X[c] for c in rest])
            perm = torch.randperm(len(neg), generator=torch.Generator().manual_seed(len(rest)))
            nh = len(neg) // 4
            f = fit_lr(X[A][:k], neg[perm[nh:]])
            thr = torch.quantile(f(neg[perm[:nh]]), 0.99)
            tprs.append(float((f(X[A][k:]) > thr).float().mean()))
            for B in g:
                cross[A][B] = float((f(X[B]) > thr).float().mean())
        tpr[A] = sum(tprs) / len(tprs)
        print(A, "tpr", round(tpr[A], 3), "max cross", round(max(cross[A].values()), 3),
              flush=True)
    onp = {}
    if a.onpolicy:
        onp = onpolicy(pol, topics, names, X, a)
    l1 = {c: topics[c]["l1"] for c in names}
    sym = [(max(cross[x][y], cross[y][x]), x, y) for i, x in enumerate(names)
           for y in names[i + 1 :]]
    same = [s for s, x, y in sym if l1[x] == l1[y]]
    diff = [s for s, x, y in sym if l1[x] != l1[y]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".dirs.npy"), np.stack(dirs))
    rep = {"layer": a.layer, "n": a.n, "request": REQUEST, "read_sec": round(read_sec),
           "sec": round(time.time() - t0), "topics": names, "l1": l1, "tpr_heldout": tpr,
           "cross": cross, "dirs": str(out.with_suffix(".dirs.npy")),
           "summary": {"same_family_median": float(np.median(same)) if same else None,
                       "cross_family_median": float(np.median(diff)),
                       "pairs_over_0.05": sum(s > 0.05 for s, _, _ in sym),
                       "pairs_over_0.2": sum(s > 0.2 for s, _, _ in sym), "pairs": len(sym)},
           "top_pairs": [{"cross_max": s, "a": x, "b": y} for s, x, y in
                         sorted(sym, reverse=True)[:60]]}
    rep.update(onp)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))


if __name__ == "__main__":
    main()
