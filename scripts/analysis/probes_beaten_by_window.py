"""Share of hiding episodes that beat >=k probes of one concept, per global-step window.
Usage: python probes_beaten_by_window.py <run_dir> 0,200,400,...  (reads turns.jsonl of every chunk; aggregates only)"""
import json, sys, glob, re
from collections import Counter, defaultdict
run, bins = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
chunks = [(0, f"{run}/adaptive/turns.jsonl")] + sorted(
    (int(re.search(r"resume_(\d+)", p).group(1)), p) for p in glob.glob(f"{run}/resume_*/adaptive/turns.jsonl"))
chunks.sort()
eps = defaultdict(list)  # (global step, chunk, ep) -> turns
for i, (off, p) in enumerate(chunks):
    nxt = chunks[i + 1][0] if i + 1 < len(chunks) else 10**9
    for l in open(p):
        r = json.loads(l)
        g = off + r["step"]
        if g >= nxt or r.get("cond") != "real_score" or r.get("off"):
            continue
        eps[(g, off, r["ep"])].append(r)
def b(g):
    for lo, hi in zip(bins, bins[1:]):
        if lo <= g < hi: return f"{lo}-{hi}"
agg = defaultdict(lambda: {"eps": 0, "beaten": Counter()})
for (g, _, _), ts in eps.items():
    k = b(g)
    if not k: continue
    a = agg[k]; a["eps"] += 1
    # probes beaten in the episode = passes (newest probe misses) at each deployed level, per concept segment
    passes = {(t["concept"], t["n_deployed"]) for t in ts if not t["detected"]}
    per_c = Counter(c for c, _ in passes)
    a["beaten"][max(per_c.values()) if per_c else 0] += 1
for k in sorted(agg, key=lambda s: int(s.split("-")[0])):
    a = agg[k]; n = a["eps"]
    ge = {j: sum(v for kk, v in a["beaten"].items() if kk >= j) / n for j in (1, 2, 3, 4, 5)}
    print(f"steps {k}: {n} hiding episodes; share beating >=1/2/3/4/5 probes (same concept):",
          " ".join(f"{ge[j]:.3f}" for j in ge), "| max", max(a["beaten"]))
