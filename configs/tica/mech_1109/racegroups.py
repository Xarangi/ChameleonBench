import json, os, random
R="/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M=[("base4b","finalbase4b_{}","base4b"),("2b","final2b_{}","k2b1800"),("k18","gate_x2500_{}","kl03_1800"),("9b","final9b_{}","k9b1800"),("det","final4b_{}","det2200")]
G={"trained":"math sql json jseval mgsm_de".split(),
   "heldout_code":"mbpp pyeval javaeval mbppjava".split(),
   "heldout_lang_math":"mgsm_fr mgsm_ru mgsm_th".split(),
   "heldout_lang_read":"belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()}
G["heldout_all"]=G["heldout_code"]+G["heldout_lang_math"]+G["heldout_lang_read"]
random.seed(0)
def ci(xs,B=1000):
    ms=[]
    for _ in range(B):
        s=n=0
        for x in xs:
            for _ in x: s+=random.choice(x)
            n+=len(x)
        ms.append(s/n)
    ms.sort(); return ms[25],ms[974]
def rep(d,key,t):
    fp=f"{R}/{d.format(t)}/report.json"
    return json.load(open(fp)).get(key) if os.path.exists(fp) else None
def pc(r,ks):
    b=c=0
    for k in ks:
        p=r.get(f"probe_{k}")
        if not p or not p["faced"]: continue
        x=str(p.get("passing_correct_n","0")).split("/")[0]; b+=p["beaten"]; c+=int(x) if x.isdigit() else 0
    return b,c
print("== GROUPS: probes beaten/ep (cap 6) [CI] | first-answer acc | correct among passes p1, p2-3, p4-6 (c/b)")
for name,d,key in M:
    for g,ts in G.items():
        xs=[];fa=[];agg=[[0,0],[0,0],[0,0]]
        for t in ts:
            r=rep(d,key,t)
            if r is None: continue
            h=r["beaten_in_a_row"]["hist"]; e=[min(int(k),6) for k,v in h.items() for _ in range(v)]; e+=[0]*(r["episodes"]-len(e)); xs.append(e); fa.append(r["first_answer_acc"][0])
            for i,ks in enumerate(((1,),(2,3),(4,5,6))):
                b,c=pc(r,ks); agg[i][0]+=b; agg[i][1]+=c
        n=sum(map(len,xs)); lo,hi=ci(xs)
        print(f"{name:7}{g:18} n{n:4} {sum(map(sum,xs))/n:4.2f} [{lo:.2f},{hi:.2f}] | fa {sum(fa)/len(fa):.2f} | "+"  ".join(f"{c}/{b} ({c/max(b,1):.0%})" for b,c in agg))
print("== PER TASK: first-acc | correct among passes p1 | p2-3 | p4-6   (k18 ; 9b ; 2b ; base4b fa)")
for t in G["trained"]+G["heldout_all"]:
    row=[]
    for name,d,key in M[1:4]:
        r=rep(d,key,t); s=f"{r['first_answer_acc'][0]:.2f}"
        for ks in ((1,),(2,3),(4,5,6)):
            b,c=pc(r,ks); s+=f" {c}/{b}"
        row.append(name+": "+s)
    rb=rep(*M[0][1:],t)
    print(f"{t:12} base fa {rb['first_answer_acc'][0]:.2f} | "+" | ".join(row))
