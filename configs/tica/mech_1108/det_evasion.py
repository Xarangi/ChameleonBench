import json, sys, collections
T = sys.argv[1]
rows = [json.loads(l) for l in open(T)]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 25
fam = lambda k: (k or "").split("@")[0].replace("base:", "")
DET = {"evdetsel", "mlpdetsel", "knnsel"}
# per (step, ep, concept, probe): sequence of detected flags on hiding turns
seq = collections.OrderedDict(); kinds = {}; honest = collections.defaultdict(lambda: [0, 0])
conds = collections.Counter(r["cond"] for r in rows)
for r in rows:
    if r.get("off"): continue
    f = fam(r["kind"]); g = f if f in DET else "zoo"
    if r["cond"] in ("real_score", "real_verdict"):
        key = (r["step"], r["ep"], r["concept"], r["active"])
        seq.setdefault(key, []).append(bool(r["detected"])); kinds[key] = g
    else:
        b = r["step"] // W; honest[(g, b)][0] += bool(r["detected"]); honest[(g, b)][1] += 1
agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # faced, beaten, turns-to-beat sum, faced-at-birth-caught
for key, s in seq.items():
    g, b = kinds[key], key[0] // W
    a = agg[(g, b)]; a[0] += 1
    if False in s:
        a[1] += 1; a[2] += s.index(False) + 1
print("conds", dict(conds))
print("window  group     faced  ever-beaten        mean turns-to-beat (beaten)")
for (g, b) in sorted(agg, key=lambda x: (x[0], x[1])):
    a = agg[(g, b)]
    print(f"{b*W:>4}-{b*W+W-1:<4} {g:10s} {a[0]:5d}  {a[1]:5d} ({100*a[1]/a[0]:4.1f}%)   {a[2]/max(1,a[1]):.2f}")
print("false alarms (non-hiding conds, active probe flags):")
for (g, b) in sorted(honest, key=lambda x: (x[0], x[1])):
    h = honest[(g, b)]
    print(f"{b*W:>4} {g:10s} {h[0]}/{h[1]}")
