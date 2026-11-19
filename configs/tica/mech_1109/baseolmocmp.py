# 11-19: untrained Olmo-3-7B control on the 17 gated task races (half size: piece p1, 16 episodes per task),
# against trained Olmo on the SAME 16 episodes (report_p1.json) and the untrained 9B (paperD, same half size).
import json, os
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M = [("base-olmo", "finalbaseolmo_{}/report.json", "baseolmo"), ("olmo-p1", "finalolmo_{}/report_p1.json", "olmo1800"),
     ("base9b", "finalbase9b_{}/report.json", "base9b")]
G = {"trained concepts": "math sql json jseval mgsm_de".split(), "held-out code": "mbpp pyeval javaeval mbppjava".split(),
     "held-out MGSM": "mgsm_fr mgsm_ru mgsm_th".split(),
     "held-out Belebele": "belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()}
G["all 17"] = [t for g in list(G.values()) for t in g]
def load(d, key, t):
    fp = f"{R}/{d.format(t)}"
    return json.load(open(fp)).get(key) if os.path.exists(fp) else None
def probes(r):
    out = []
    for k in range(1, 7):
        p = r.get(f"probe_{k}")
        if not p or not p["faced"]: out.append(None); continue
        c = str(p.get("passing_correct_n", "0")).split("/")[0]
        out.append((p["beaten"], p["faced"], int(c) if c.isdigit() else 0, p["turns_to_beat"][0] if p["beaten"] else None))
    return out
print("task | model | eps | first-turn acc | p1..p6 beaten/faced (correct passes) turns")
for t in G["all 17"]:
    for name, d, key in M:
        r = load(d, key, t)
        if r is None: print(t, name, "MISSING"); continue
        fa = r.get("first_answer_acc"); fa = fa[0] if isinstance(fa, list) else fa
        print(f"{t} | {name} | {r['episodes']} | {fa if fa is None else round(fa, 2)} | " + "; ".join(
            f"p{k+1} {p[0]}/{p[1]} ({p[2]})" + (f" {p[3]:.1f}t" if p[3] is not None else "") for k, p in enumerate(probes(r)) if p))
print()
for name, d, key in M:
    for g, T in G.items():
        eps = beat = corr = 0; pk = [[0, 0, 0] for _ in range(6)]; fa = []
        for t in T:
            r = load(d, key, t)
            if r is None: continue
            eps += r["episodes"]; a = r.get("first_answer_acc"); fa.append(a[0] if isinstance(a, list) else a)
            for k, p in enumerate(probes(r)):
                if p: pk[k][0] += p[0]; pk[k][1] += p[1]; pk[k][2] += p[2]; beat += p[0]; corr += p[2]
        print(f"{name} | {g}: eps {eps} | mean first-turn acc {sum(fa)/max(len(fa),1):.2f} | beaten/ep {beat/max(eps,1):.2f} | correct passes {corr} "
              f"({corr/max(eps,1):.2f}/ep) | " + " ".join(f"p{k+1} {b}/{f} ({c})" for k, (b, f, c) in enumerate(pk) if f))
