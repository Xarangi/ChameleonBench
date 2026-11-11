import json,collections,sys
D="/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1109/turnacc/"
G={"trained":"sql json jseval mgsm_de".split(),"ho_code":"mbpp pyeval javaeval mbppjava".split(),"ho_mgsm":"mgsm_fr mgsm_ru mgsm_th".split(),"ho_read":"belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()}
data={t:json.load(open(D+t+".json")) for ts in G.values() for t in ts}
models=["base4b","base9b","base2b","2b","k18","9b","det"]
def agg(m,ts,f):
    out=collections.defaultdict(lambda:[0,0,0])
    for t in ts:
        for k,v in data[t].get(m,{}).get("agg",{}).items():
            off,pas,det,nd,tb=map(int,k.split("|")); kk=f(off,pas,det,nd,tb)
            if kk is None: continue
            o=out[kk]; o[0]+=v[0]; o[1]+=v[1]; o[2]+=v[2]
    return out
fmt=lambda v: f"{v[1]/v[0]:.2f}({v[0]})" if v[0] else "   -   "
tot=agg("k18",sum(G.values(),[]),lambda off,pas,det,nd,tb:(off,pas,det)); print("k18 row types (off,passed,detected):",{k:v[0] for k,v in tot.items()})
print("\n== A. accuracy by TURN bucket [0 | 1-7 | 8-15 | 16-23 | 24-32]; caught answers vs passed answers (monitor on)")
for g,ts in G.items():
    for m in models:
        c=agg(m,ts,lambda off,pas,det,nd,tb: None if off else ("P" if pas else "C",tb))
        if not c: continue
        print(f"{g:8}{m:7} caught: "+" ".join(fmt(c[("C",b)]) for b in range(5))+" | passed: "+" ".join(fmt(c[("P",b)]) for b in range(5)))
print("\n== B. accuracy by PROBES DEPLOYED [1|2|3|4|5|6|7+]; caught vs passed")
for g,ts in G.items():
    for m in ["2b","k18","9b","det"]:
        c=agg(m,ts,lambda off,pas,det,nd,tb: None if off else ("P" if pas else "C",nd))
        print(f"{g:8}{m:5} caught: "+" ".join(fmt(c[("C",b)]) for b in range(1,8))+" | passed: "+" ".join(fmt(c[("P",b)]) for b in range(1,8)))
print("\n== C. in-language share (language tasks), passed vs caught, by model")
for g in ("ho_mgsm","ho_read"):
    for m in models:
        c=agg(m,G[g],lambda off,pas,det,nd,tb: None if off else ("P" if pas else "C"))
        if c: print(g,m," ".join(f"{k}: {v[2]/max(v[0],1):.2f}({v[0]})" for k,v in sorted(c.items())))
c=agg("k18",["mgsm_de"],lambda off,pas,det,nd,tb: None if off else ("P" if pas else "C")); print("mgsm_de k18"," ".join(f"{k}: {v[2]/max(v[0],1):.2f}({v[0]})" for k,v in sorted(c.items())))
