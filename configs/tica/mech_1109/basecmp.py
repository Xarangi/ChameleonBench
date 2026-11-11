# 11-11: untrained 2B / 4B / 9B controls on the 17 gated task races, overall and by task group (paperD).
# base2b / base9b are half size (16 episodes per task, piece p1); base4b is full size (32).
import json, os
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M = [("base2b", "finalbase2b_{}", "base2b"), ("base4b", "finalbase4b_{}", "base4b"), ("base9b", "finalbase9b_{}", "base9b")]
# groups as in the paper (6.2) and racegroups.py: German and JavaScript are trained concepts, Python is held out
G = {"trained concepts": "math sql json jseval mgsm_de".split(), "held-out code": "mbpp pyeval javaeval mbppjava".split(),
     "held-out MGSM": "mgsm_fr mgsm_ru mgsm_th".split(),
     "held-out Belebele": "belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()}
G["all 17"] = [t for g in list(G.values()) for t in g]
for name, d, key in M:
    for g, T in G.items():
        eps = beat = corr = 0; pk = [[0, 0] for _ in range(6)]; miss = []; per = []
        for t in T:
            fp = f"{R}/{d.format(t)}/report.json"
            r = json.load(open(fp)).get(key) if os.path.exists(fp) else None
            if r is None: miss.append(t); continue
            eps += r["episodes"]; b0 = 0
            for k in range(6):
                p = r.get(f"probe_{k+1}")
                if not p or not p["faced"]: continue
                c = str(p.get("passing_correct_n", "0")).split("/")[0]; c = int(c) if c.isdigit() else 0
                pk[k][0] += p["beaten"]; pk[k][1] += p["faced"]; beat += p["beaten"]; corr += c; b0 += p["beaten"]
            per.append(f"{t} {b0}/{r['episodes']}ep")
        print(f"{name} | {g}: eps {eps} | beaten/ep {beat/max(eps,1):.2f} | correct passes/ep {corr/max(eps,1):.2f} ({corr}/{beat}) | "
              + " ".join(f"p{k+1} {b}/{f}" for k, (b, f) in enumerate(pk)) + (f" | MISSING {miss}" if miss else ""))
        if g == "all 17": print("   per task (probes beaten / episodes):", "; ".join(per))
