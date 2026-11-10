import json, os
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
T = "math mbpp pyeval javaeval jseval sql json mbppjava mgsm_fr mgsm_ru mgsm_th mgsm_de mgsm_ja belebele_fi belebele_pt belebele_tr belebele_el belebele_ta mmlu_physics mmlu_cs mmlu_med mmlu_bio".split()
M = [("k18", "gate_x2500_{}", "kl03_1800"), ("det", "final4b_{}", "det2200"), ("9b", "final9b_{}", "k9b1800")]
tot = {}
def cell(p):
    b, f = p["beaten"], p["faced"]
    c = p.get("passing_correct_n", "?").split("/")[0]
    t = p["turns_to_beat"][0] if b else None
    fl = (p.get("fluke_share_mean") or [None])[0]
    return b, f, int(c) if c.isdigit() else 0, t, fl
print("task | model | eps | first_acc | p1..p6 as beaten/faced turns correct(%beaten) robust")
for t in T:
    for name, d, key in M:
        fp = f"{R}/{d.format(t)}/report.json"
        if not os.path.exists(fp): print(t, name, "NO REPORT"); continue
        r = json.load(open(fp)).get(key)
        if r is None: print(t, name, "NO KEY"); continue
        fa = r.get("first_answer_acc"); fa = fa[0] if isinstance(fa, list) else fa
        out = []
        for k in range(1, 7):
            p = r.get(f"probe_{k}")
            if not p or not p["faced"]: continue
            b, f, c, tt, fl = cell(p)
            if t != "mgsm_ja": a = tot.setdefault((name, k), [0, 0, 0]); a[0] += b; a[1] += f; a[2] += c
            out.append(f"p{k} {b}/{f} {tt if tt is None else round(tt,1)}t {c}({100*c//max(b,1)}%)" + (f" r{fl:.2f}" if fl is not None else ""))
        print(f"{t} | {name} | {r['episodes']} | {fa if fa is None else round(fa,2)} | " + "; ".join(out))
print("\nTOTALS (beaten/faced, correct passes n (% of beaten))")
for name, _, _ in M:
    print(name, "  ".join(f"p{k} {a[0]}/{a[1]} ({a[0]/a[1]:.2f}) corr {a[2]} ({a[2]/max(a[0],1):.0%})" for (n, k), a in sorted(tot.items()) if n == name))
