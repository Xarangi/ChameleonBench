"""Concept split v2 (REPLICATION 2026-10-14h): MUTUALLY DISJOINT train / dev / test concepts.

v1 held out concepts that were near-duplicates of train concepts in the answer-format geometry
(cos 0.97-0.98: "mathematical notation or references" vs the train "mathematical notation and
symbols"). User rule (10-14): topics must be fairly disjoint WITHIN train, WITHIN test and ACROSS.

Two concepts CONFLICT if any of:
  - answer-format L12 mean-diff directions (rung 0, tica_concept_geometry.py): cos > --tau-answer;
  - repeat-format L12 base-probe directions (W / sd of the Level-1 bank): cos > --tau-probe;
  - their names share a content keyword (stemmed, generic words removed), e.g. "punctuation",
    "html", "mathematical".
The selected set is an INDEPENDENT set of the conflict graph (greedy, lowest degree first; ties
broken towards under-represented types), so every pair -- train-train, test-test, train-test --
is non-conflicting. It is then partitioned into train / dev / test, stratified by type and by
answer-geometry cluster (round-robin), with WEIRD concepts (lexical / odd names / isolated) spread
over all three. Names + aggregates only; no texts are read.

  python scripts/tica_concept_split_v2.py --geometry .../rung0/geometry.json --out SPLIT.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tica_concept_split import BANK, EXCLUDE, L1, ODD, ctype

GENERIC = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "by", "from", "as",
    "at", "into", "within", "its", "their", "various", "related", "relating", "terms", "term",
    "references", "reference", "mentions", "mention", "occurrences", "occurrence", "elements",
    "element", "words", "word", "phrases", "phrase", "concepts", "concept", "use", "uses",
    "usage", "types", "type", "aspects", "aspect", "particularly", "especially", "specific",
    "general", "different", "context", "contexts", "text", "texts", "document", "documents",
    "including", "include", "such", "other", "them", "that", "this", "these", "those",
    "about", "form", "forms", "kinds", "kind", "expressions", "structures", "structure",
    "key", "indicators", "indicator", "instances", "instance",
}


def _py(o):
    return o.item() if hasattr(o, "item") else str(o)


def keywords(name: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", name.lower()):
        if len(w) < 4 or w in GENERIC:
            continue
        out.add(w[:6])  # crude stem: "mathematical"/"mathematics" -> "mathem"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--tau-answer", type=float, default=0.5)
    ap.add_argument("--tau-probe", type=float, default=0.2)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-dev", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--k", type=int, default=16, help="answer-geometry clusters for stratifying")
    ap.add_argument("--pin", action="append", default=["mathematical notation and symbols:train"],
                    help="name:set (repeatable); math stays a TRAIN concept")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from scipy.cluster.hierarchy import fcluster, linkage

    g = json.loads(Path(a.geometry).read_text())
    an, AD, agate = list(g["concepts"]), np.load(g["dirs"]), np.array(g["gate"], bool)
    z = np.load(BANK, allow_pickle=True)
    pn, PD = list(z["concepts"]), z["W"] / z["sd"]
    meta = {c["concept"]: (c["genre"], len(c["fit"]))
            for c in json.loads(Path(L1).read_text())["concepts"]}
    pidx = {n: i for i, n in enumerate(pn)}
    names = [n for i, n in enumerate(an) if agate[i] and not EXCLUDE.search(n) and n in pidx
             and meta.get(n, ("", 0))[1] >= 65]
    A = AD[[an.index(n) for n in names]]
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    P = PD[[pidx[n] for n in names]]
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    SA, SP = A @ A.T, P @ P.T
    kw = [keywords(n) for n in names]
    N = len(names)
    conflict = (a.tau_answer < SA) | (a.tau_probe < SP)
    for i in range(N):
        for j in range(i + 1, N):
            if kw[i] & kw[j]:
                conflict[i, j] = conflict[j, i] = True
    np.fill_diagonal(conflict, False)
    types = [ctype(n, meta[n][0]) for n in names]
    np.fill_diagonal(SA, -1.0)
    iso_cut = np.percentile(SA.max(1), 20)
    weird = [types[i] == "lexical" or bool(ODD.search(n)) or SA[i].max() <= iso_cut
             for i, n in enumerate(names)]
    lab = fcluster(linkage(A, method="average", metric="cosine"), a.k, "maxclust")

    pins = dict(p.rsplit(":", 1) for p in a.pin)
    rng = np.random.default_rng(a.seed)
    missing = [n for n in pins if n not in names]
    if missing:
        raise SystemExit(f"pinned concepts not in the pool: {missing}")
    chosen: list[int] = [names.index(n) for n in pins]
    bad = [(names[i], names[j]) for i in chosen for j in chosen if i < j and conflict[i, j]]
    if bad:
        raise SystemExit(f"pinned concepts conflict: {bad}")
    blocked = np.zeros(N, bool)
    for i in chosen:
        blocked |= conflict[i]
        blocked[i] = True
    tcount = Counter(types[i] for i in chosen)
    order = [int(x) for x in rng.permutation(N)]
    while True:
        cand = [i for i in order if not blocked[i]]
        if not cand:
            break
        # fewest conflicts among the still-available, then the rarest type so far
        free_deg = {i: int((conflict[i] & ~blocked).sum()) for i in cand}
        i = min(cand, key=lambda i: (free_deg[i], tcount[types[i]]))
        chosen.append(i)
        tcount[types[i]] += 1
        blocked |= conflict[i]
        blocked[i] = True

    # partition: pinned first; then weird and non-weird separately, round-robin over (type,
    # cluster) buckets into test / dev / train in proportion
    sets = {"train": [], "dev": [], "test": []}
    for n, s in pins.items():
        if n in names:
            sets[s].append(names.index(n))
    rest = [i for i in chosen if names[i] not in pins]
    want = {"test": a.n_test, "dev": a.n_dev, "train": a.n_train}
    for group in ([i for i in rest if weird[i]], [i for i in rest if not weird[i]]):
        buckets = defaultdict(list)
        for i in group:
            buckets[(types[i], int(lab[i]))].append(i)
        flat = []
        keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
        while any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k]:
                    flat.append(buckets[k].pop(int(rng.integers(len(buckets[k])))))
        cycle = ["test", "dev", "train", "train", "train", "test", "train", "train"]
        c = 0
        for i in flat:
            for _ in range(len(cycle)):
                s = cycle[c % len(cycle)]
                c += 1
                if len(sets[s]) < want[s]:
                    sets[s].append(i)
                    break
    unused = [names[i] for i in rest if not any(i in v for v in sets.values())]

    def pair_stats(xs, ys):
        if not xs or not ys:
            return None
        sa = SA[np.ix_(xs, ys)]
        sp = SP[np.ix_(xs, ys)]
        if xs is ys:
            m = ~np.eye(len(xs), dtype=bool)
            sa, sp = sa[m], sp[m]
        return {"answer_cos_max": float(sa.max()), "answer_cos_median": float(np.median(sa)),
                "probe_cos_max": float(sp.max()), "probe_cos_median": float(np.median(sp))}

    out = {
        "version": 2, "tau_answer": a.tau_answer, "tau_probe": a.tau_probe,
        "keyword_conflicts": True, "pool": N, "independent_set": len(chosen),
        "train": [names[i] for i in sets["train"]], "dev": [names[i] for i in sets["dev"]],
        "test": [names[i] for i in sets["test"]], "unused_disjoint": unused,
        "info": {names[i]: {"type": types[i], "weird": bool(weird[i]), "cluster": int(lab[i]),
                            "nn_answer_cos_pool": float(SA[i].max())} for i in chosen},
        "pairs": {f"{x}-{y}": pair_stats(sets[x], sets[y]) for x, y in
                  (("train", "train"), ("test", "test"), ("dev", "dev"), ("train", "test"),
                   ("train", "dev"), ("dev", "test"))},
        "types": {s: dict(Counter(types[i] for i in v)) for s, v in sets.items()},
        "weird": {s: sum(weird[i] for i in v) for s, v in sets.items()},
        "clusters": {s: len({int(lab[i]) for i in v}) for s, v in sets.items()},
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=_py))
    print(json.dumps({k: out[k] for k in ("pool", "independent_set", "pairs", "types", "weird",
                                          "clusters")}, indent=1, default=_py))
    print({s: len(out[s]) for s in ("train", "dev", "test", "unused_disjoint")})


if __name__ == "__main__":
    main()
