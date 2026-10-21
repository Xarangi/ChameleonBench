"""Train / held-out CONCEPT split for the final multi-concept dodger (TICA-FINAL-DODGER-DESIGN.md).

Uses only concept NAMES and aggregates (probe directions, counts) -- never the texts.

Geometry: base-model L12 concept directions, either the repeat-format base bank (W / sd, the
source of the 10-13i near/far measurement) or an answer-format geometry file from
tica_concept_geometry.py (--geometry). Directions are clustered (average-link cosine, k clusters).
Every concept also gets a
TYPE from its name (code/markup, math/numeric, format/punctuation, lexical (a specific word/token),
discourse/style/affect, entity/place/name, domain/topic) and a WEIRD flag (lexical, vague/odd names,
or ISOLATED: nearest-neighbour cosine in the bottom 20%).

Selection (deterministic):
  pool      = base gate passed, not excluded (bio / medical / harmful / gambling names),
              >= 65 fit texts
  far       = up to `--far-clusters` whole clusters held out (most isolated first, distinct
              genres, <= --far-max concepts in total)
  forced    = FORCE_HELDOUT (near-math pair, the 10-13 transfer concepts, deception)
  held-out  = round-robin over (cluster, type) of the other clusters (<= --per-cluster each), then
              quota swaps: every type >= 2, weird >= 6, weird TEXT concepts >= 5, odd names >= 4
  train     = the rest, ordered for the ladder by round-robin over clusters (math first),
              preferring under-represented types and >= --min-avail published positives (--avail)
  near      = per train concept, its closest unused concept if cos >= the pool's p90 NN cosine
              (distinct clusters, up to --n-near), added to the held-out set
Writes the split JSON ({train, heldout_near, heldout_far, heldout_weird, info per concept}).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

L1 = "/scratch/s/sneheel/next_chameleons_raw_cache/tica/level1_data.json"
BANK = "/scratch/s/sneheel/next_chameleons_artifacts/tica/l1_base/qwen35_4b/probes/bank_L12.npz"
MATH = "mathematical notation and symbols"
# measured near-math concepts (10-13i: cos 0.289 / 0.215) -- held out so transfer stays comparable
NEAR_MATH = ("syntactical structures and formatting in mathematical expressions",
             "technical and scientific terms related to measurements and variables in equations")
# always held out: the near-math pair, the 10-13 transfer concepts (comparability: car brands 0.69,
# data visualizations 1.0 passes/episode on the math dodger) and deception (NC's headline transfer
# target is a deception monitor; a benign, safety-adjacent held-out concept)
FORCE_HELDOUT = (*NEAR_MATH, "references to various car brands and their models",
                 "elements related to data visualizations and plots",
                 "uses of deception or pretense")

EXCLUDE = re.compile(
    r"bio|chemical|medic|surg|disease|vir(us|al)|drug|pharma|health|clinic|patient|mental|suicid|"
    r"violen|weapon|gun|kill|sex|gambl|casino|cosmetic|toxi|poison|crime|criminal|terror|abuse|death|"
    r"wound|cancer|alcohol|cannab|therap|symptom|diagnos|injur|blood|explos|hack|malware|nazi|racis|"
    r"slur|porn|nud|hate|neuro|brain", re.I)
TYPES = [  # first match wins
    ("lexical", re.compile(r"the (word|term|phrase|day|letter)s?\b|\"[^\"]+\"", re.I)),
    ("format_punct", re.compile(r"punctuation|symbols?\b|format|semicolon|capitali|parenthes|"
                                r"bracket|quotation|bullet|list(s|ed)? |whitespace|indent", re.I)),
    ("math_numeric", re.compile(r"mathemat|numer|statistic|equation|measure|number|percent|"
                                r"quantit|calculat|units?\b", re.I)),
    ("code_markup", re.compile(r"programming|\bcode\b(?!s)|javascript|html|css|sql|python|function|"
                               r"variable|syntax|api|json|xml|markup|dom\b|class(es)?\b", re.I)),
    ("discourse_affect", re.compile(r"emotion|feel|sentiment|tone|conversation|informal|pronoun|"
                                    r"expression|humou?r|sarcas|polite|discourse|opinion|"
                                    r"persua|deception|pretense|praise|gratitude", re.I)),
    ("entity_place", re.compile(r"names?\b|geograph|location|cit(y|ies)|countr|nationalit|"
                                r"person|people|brand|compan|organi[sz]ation|place", re.I)),
]
# idiosyncratic SAE-style names (odd conjunctions / oddly specific themes), judged by name only
ODD = re.compile(r"unspecified|vague|miscellane|playful|onomatop|nocturnal|divine|celestial|hadron|"
                 r"concealed|gorilla|the sea\b|tuesday|individual letters|omissions", re.I)


def ctype(name: str, genre: str) -> str:
    for t, rx in TYPES:
        if rx.search(name):
            return t
    if genre == "code":
        return "code_markup"
    if genre == "math":
        return "math_numeric"
    return "domain_topic"


def load_geometry(path: str | None):
    if path:  # answer-format geometry (tica_concept_geometry.py): {concepts, dirs (npy path), gate}
        g = json.loads(Path(path).read_text())
        return list(g["concepts"]), np.load(g["dirs"]), np.array(g["gate"], bool)
    z = np.load(BANK, allow_pickle=True)
    W = z["W"] / z["sd"]
    return list(z["concepts"]), W, z["gate"].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", help="answer-format geometry JSON (default: repeat-format bank)")
    ap.add_argument("--avail", help="{concept: published positives available} JSON")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--far-clusters", type=int, default=2)
    ap.add_argument("--far-max", type=int, default=8, help="max concepts in the far clusters")
    ap.add_argument("--n-heldout", type=int, default=16, help="held-out from the other clusters")
    ap.add_argument("--n-near", type=int, default=4, help="held-out concepts NEAR a train concept")
    ap.add_argument("--per-cluster", type=int, default=3, help="max held-out per cluster")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--min-avail", type=int, default=150, help="train concepts need this many")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from scipy.cluster.hierarchy import fcluster, linkage

    meta = {c["concept"]: (c["genre"], c["split"], len(c["fit"]))
            for c in json.loads(Path(L1).read_text())["concepts"]}
    names, W, gate = load_geometry(a.geometry)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    avail = json.loads(Path(a.avail).read_text()) if a.avail else {}
    ok = [i for i, n in enumerate(names)
          if gate[i] and not EXCLUDE.search(n) and meta.get(n, ("", "", 0))[2] >= 65]
    names = [names[i] for i in ok]
    W = W[ok]
    S = W @ W.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(1)
    lab = fcluster(linkage(W, method="average", metric="cosine"), a.k, "maxclust")
    iso_cut = np.percentile(nn, 20)
    near_cut = np.percentile(nn, 90)
    info = {}
    for i, n in enumerate(names):
        t = ctype(n, meta[n][0])
        weird = t == "lexical" or bool(ODD.search(n)) or nn[i] <= iso_cut
        info[n] = {"odd": bool(ODD.search(n)), "cluster": int(lab[i]), "type": t,
                   "weird": bool(weird), "nn_cos": float(nn[i]),
                   "cos_math": float(S[i, names.index(MATH)]) if MATH in names else None,
                   "avail": int(avail.get(n, meta[n][2])), "genre": meta[n][0]}
    idx = {n: i for i, n in enumerate(names)}
    clusters = defaultdict(list)
    for n in names:
        clusters[info[n]["cluster"]].append(n)
    math_c = info[MATH]["cluster"]

    # 1) FAR: whole clusters held out -- most isolated (lowest mean max-cos to the other clusters)
    def iso(c):
        mem = [idx[n] for n in clusters[c]]
        rest = [i for i in range(len(names)) if lab[i] != c]
        return float(S[np.ix_(mem, rest)].max(1).mean())

    cand = [c for c in clusters if c != math_c and 2 <= len(clusters[c]) <= a.far_max]
    far_cl, far, genres = [], [], set()
    for c in sorted(cand, key=iso):  # most isolated first, distinct genres, within the far budget
        g = Counter(meta[n][0] for n in clusters[c]).most_common(1)[0][0]
        if (len(far_cl) < a.far_clusters and g not in genres
                and len(far) + len(clusters[c]) <= a.far_max):
            far_cl.append(c)
            far += clusters[c]
            genres.add(g)

    # 2) stratified held-out from the remaining clusters
    held = [n for n in FORCE_HELDOUT if n in info]
    rng = np.random.default_rng(20261014)
    rest = [n for n in names if n not in far and n not in held and n != MATH]
    by_ct = defaultdict(list)
    for n in rest:
        by_ct[(info[n]["cluster"], info[n]["type"])].append(n)
    for v in by_ct.values():
        rng.shuffle(v)
    # round-robin over clusters, within a cluster over types, until n_heldout
    order = sorted(by_ct, key=lambda k: (-len(by_ct[k]), k))
    budget = a.n_heldout - len(held)
    while budget > 0 and any(by_ct.values()):
        for key in order:
            if budget <= 0:
                break
            n_cl = sum(1 for n in held if info[n]["cluster"] == key[0])
            if by_ct[key] and n_cl < a.per_cluster and len(by_ct[key]) + sum(
                    1 for n in held if (info[n]["cluster"], info[n]["type"]) == key) >= 2:
                held.append(by_ct[key].pop())
                budget -= 1
    # 3) quotas (swap in from the unused pool, dropping a member of a type with > 2)
    unused = [n for v in by_ct.values() for n in v]

    kept = []  # predicates already satisfied: a swap may not break them

    def fix(pred, need):
        for n in [u for u in unused if pred(u)]:
            if sum(map(pred, held)) >= need:
                break
            tc = Counter(info[h]["type"] for h in held)
            ncl = Counter(info[h]["cluster"] for h in held)
            if ncl[info[n]["cluster"]] >= a.per_cluster:
                continue

            def droppable(h, tc=tc, n=n):
                if h in FORCE_HELDOUT or pred(h) or tc[info[h]["type"]] <= 2:
                    return False
                return all(sum(map(q, held)) - q(h) + q(n) >= k for q, k in kept)

            drop = next((h for h in reversed(held) if droppable(h)), None)
            if drop is None:
                break
            held.remove(drop)
            unused.append(drop)
            held.append(n)
            unused.remove(n)
        kept.append((pred, need))

    for t in {info[n]["type"] for n in names}:
        fix(lambda n, t=t: info[n]["type"] == t, 2)
    fix(lambda n: info[n]["weird"], 6)
    # weird TEXT concepts (odd names / isolated directions, not lexical or code): the user's
    # "really weird ones" must be represented among the held-out
    fix(lambda n: info[n]["weird"] and info[n]["genre"] == "text"
        and info[n]["type"] not in ("lexical", "code_markup"), 5)
    fix(lambda n: bool(ODD.search(n)), 4)  # and several of the oddest names

    # 4) train = the rest; prefer >= min_avail published positives; ladder order round-robin
    pool_tr = [n for n in names if n not in far and n not in held]
    pool_tr.sort(key=lambda n: (info[n]["avail"] < a.min_avail, -info[n]["avail"]))
    bycl = defaultdict(list)
    for n in pool_tr:
        if n != MATH:
            bycl[info[n]["cluster"]].append(n)
    train = [MATH]
    seen_types = Counter()
    while len(train) < a.n_train and any(bycl.values()):
        for c in sorted(bycl, key=lambda c: -len(bycl[c])):
            if bycl[c] and len(train) < a.n_train:
                # within the cluster prefer an under-represented type
                bycl[c].sort(key=lambda n: (info[n]["avail"] < a.min_avail,
                                            seen_types[info[n]["type"]], -info[n]["avail"]))
                n = bycl[c].pop(0)
                train.append(n)
                seen_types[info[n]["type"]] += 1
    T = np.array([idx[n] for n in train])
    # NEAR held-out by construction: the closest unused concept to a train concept (cos >= p90 NN),
    # one per train concept, distinct clusters, until --n-near
    used = set(train) | set(held) | set(far)
    pairs = sorted(((float(S[idx[u], idx[t]]), u, t) for u in names if u not in used
                    for t in train if t != MATH), reverse=True)
    near_cl = set()
    for c, u, _t in pairs:
        if sum(1 for n in held if n in near_cl) >= a.n_near or c < near_cut:
            break
        if u in held or info[u]["cluster"] in near_cl:
            continue
        held.append(u)
        near_cl.add(info[u]["cluster"])
        near_cl.add(u)
    for n in held + far:
        j = int(np.argmax(S[idx[n], T]))
        info[n]["nearest_train"] = train[j]
        info[n]["cos_nearest_train"] = float(S[idx[n], T[j]])
    near = [n for n in held if info[n]["cos_nearest_train"] >= near_cut]
    out = {
        "source": a.geometry or BANK, "k": a.k, "seed": 20261014,
        "cuts": {"iso_nn_p20": float(iso_cut), "near_nn_p90": float(near_cut)},
        "pool": len(names), "excluded": sum(1 for n in meta if EXCLUDE.search(n)),
        "far_clusters": far_cl,
        "train": train, "heldout": held, "heldout_far": far, "heldout_near": near,
        "heldout_weird": [n for n in held + far if info[n]["weird"]],
        "summary": {
            "train_types": Counter(info[n]["type"] for n in train),
            "heldout_types": Counter(info[n]["type"] for n in held + far),
            "train_clusters": len({info[n]["cluster"] for n in train}),
            "heldout_clusters": len({info[n]["cluster"] for n in held + far}),
            "train_avail_ge_min": sum(info[n]["avail"] >= a.min_avail for n in train),
            "heldout_cos_nearest_train_median": float(np.median(
                [info[n]["cos_nearest_train"] for n in held + far])),
            "far_cos_nearest_train_max": float(max(info[n]["cos_nearest_train"] for n in far))
            if far else None,
        },
        "info": {n: info[n] for n in train + held + far},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pool", "excluded", "far_clusters", "cuts", "summary")},
                     indent=1))


if __name__ == "__main__":
    main()
