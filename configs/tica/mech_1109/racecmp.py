# 11-11: gated task-race totals per model over a chosen task set (default: the 18 non-MMLU tasks minus mgsm_ja).
#   racecmp.py [comma,separated,tasks]
import json, os, sys
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M = [("base4b", "finalbase4b_{}", "base4b"), ("2b", "final2b_{}", "k2b1800"), ("k18", "gate_x2500_{}", "kl03_1800"),
     ("9b", "final9b_{}", "k9b1800"), ("det", "final4b_{}", "det2200")]
T = sys.argv[1].split(",") if len(sys.argv) > 1 else ("math mbpp pyeval javaeval jseval sql json mbppjava mgsm_fr mgsm_ru "
    "mgsm_th mgsm_de belebele_fi belebele_pt belebele_tr belebele_el belebele_ta").split()
print("tasks:", len(T), " ".join(T))
for name, d, key in M:
    eps = beat = corr = 0; pk = [[0, 0, 0] for _ in range(6)]; miss = []
    for t in T:
        fp = f"{R}/{d.format(t)}/report.json"
        r = json.load(open(fp)).get(key) if os.path.exists(fp) else None
        if r is None: miss.append(t); continue
        eps += r["episodes"]
        for k in range(6):
            p = r.get(f"probe_{k+1}")
            if not p or not p["faced"]: continue
            c = str(p.get("passing_correct_n", "0")).split("/")[0]; c = int(c) if c.isdigit() else 0
            pk[k][0] += p["beaten"]; pk[k][1] += p["faced"]; pk[k][2] += c; beat += p["beaten"]; corr += c
    cells = "  ".join(f"p{k+1} {b}/{f} c{c}({c/max(b,1):.0%})" for k, (b, f, c) in enumerate(pk))
    print(f"{name}: eps {eps} | probes beaten/episode {beat/max(eps,1):.2f} | correct passes {corr} ({corr/max(beat,1):.0%} of beaten) | {cells}"
          + (f" | MISSING {','.join(miss)}" if miss else ""))
