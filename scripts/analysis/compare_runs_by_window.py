"""Two runs side by side over matched global-step windows from their train_log.jsonl chunks (beat rates, pass,
GSM8K weighted by samples, free-gen validity, train KL). Edit `pairs` / `W` for other runs."""
import json, glob, re, sys
R = "/scratch/sneheel/next_chameleons_artifacts/tica/runs/"
def rows(run, upto=None):
    ch = [(0, R + run + "/adaptive/train_log.jsonl")] + [(int(re.search(r"resume_(\d+)/", p).group(1)), p)
          for p in glob.glob(R + run + "/resume_*/adaptive/train_log.jsonl") if re.search(r"resume_\d+/", p)]
    ch.sort(); out = {}
    for i, (off, p) in enumerate(ch):
        nxt = ch[i + 1][0] if i + 1 < len(ch) else 10**9
        try: f = open(p)
        except FileNotFoundError: continue
        for l in f:
            d = json.loads(l); g = off + d["step"]
            if g < nxt: out[g] = d
    return out
def win(rs, lo, hi):
    w = [d for g, d in rs.items() if lo <= g < hi]
    if not w: return None
    m = lambda k: sum(d.get(k, 0) or 0 for d in w) / len(w)
    ns = sum(d.get("cap_gen/n_samples", 0) or 0 for d in w)
    gsm = sum((d.get("cap_gen/acc", 0) or 0) * (d.get("cap_gen/n_samples", 0) or 0) for d in w) / ns if ns else float("nan")
    nf = sum(d.get("cap_free/n", 0) or 0 for d in w)
    free = sum((d.get("cap_free/valid", 0) or 0) * (d.get("cap_free/n", 0) or 0) for d in w) / nf if nf else float("nan")
    return dict(n=len(w), beat=m("rollout/beat_frac"), b2=m("rollout/beatk_frac_2"), b3=m("rollout/beatk_frac_3"),
                pass_=m("rollout/pass_rate"), first=m("rollout/beat1_first_frac"), gsm=gsm, free=free, kl=m("train/kl"))
pairs = {"2B": [("strict", rows("final_v4d_klx_tamia_2b")), ("KL0.3", {**rows("final_v4d_tamia_2b"), **{g: d for g, d in rows("final_v4d_tamia_2b_x1800").items() if g >= 1500}})],
         "4B": [("strict", rows("final_v4d_klx_tamia_4b")), ("KL0.3", rows("final_v4d_k03_tamia_4b"))]}
W = {"2B": [(0,100),(100,200),(200,300),(300,500),(500,700),(700,900),(900,1100),(1100,1300),(1300,1450),(1450,1600)],
     "4B": [(0,100),(100,200),(200,300),(300,340)]}
for m, runs in pairs.items():
    print(f"== {m}: steps | run | n | beat | beat>=2 | beat>=3 | pass | beat1_first | GSM8K(w) | free valid | train KL")
    for lo, hi in W[m]:
        for name, rs in runs:
            s = win(rs, lo, hi)
            if s: print(f"{lo}-{hi} | {name} | {s['n']} | {s['beat']:.3f} | {s['b2']:.3f} | {s['b3']:.3f} | {s['pass_']:.3f} | {s['first']:.3f} | {s['gsm']:.3f} | {s['free']:.3f} | {s['kl']:.4f}")
