"""11-13 (review A-M8): accuracy on Monitor-OFF turns vs monitored turns, by turn bucket, from turnacc/*.json.
  /usr/bin/python3 offacc.py"""
import json, collections
D = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1109/turnacc/"
G = {"trained": "sql json jseval mgsm_de".split(), "ho_code": "mbpp pyeval javaeval mbppjava".split(),
     "ho_mgsm": "mgsm_fr mgsm_ru mgsm_th".split(), "ho_read": "belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()}
data = {t: json.load(open(D + t + ".json")) for ts in G.values() for t in ts}
fmt = lambda v: f"{v[1]}/{v[0]}={v[1]/v[0]:.2f}" if v[0] else "-"
for g, ts in G.items():
    for m in ["base2b", "base4b", "base9b", "2b", "k18", "9b"]:
        c = collections.defaultdict(lambda: [0, 0])
        for t in ts:
            for k, v in data[t].get(m, {}).get("agg", {}).items():
                off, pas, det, nd, tb = map(int, k.split("|"))
                o = c[("OFF" if off else "ON", tb)]; o[0] += v[0]; o[1] += v[1]
        if not c: continue
        for kind in ("ON", "OFF"):
            print(f"{g:8}{m:7}{kind:4} " + " | ".join(f"{fmt(c[(kind,b)]):>14}" for b in range(5)))
