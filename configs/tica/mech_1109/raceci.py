import json, os, random
R="/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M=[("base4b","finalbase4b_{}","base4b"),("2b","final2b_{}","k2b1800"),("k18","gate_x2500_{}","kl03_1800"),("9b","final9b_{}","k9b1800"),("det","final4b_{}","det2200"),("s18","gate_x2500_{}","strict1800"),("s25","gate_x2500_{}","strict2500")]
T="math mbpp pyeval javaeval jseval sql json mbppjava mgsm_fr mgsm_ru mgsm_th mgsm_de belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()
MM="mmlu_physics mmlu_cs mmlu_bio mmlu_med".split()
random.seed(0)
def ci(xs,B=2000):
    # stratified by task: xs = list of per-task lists
    ms=[]
    for _ in range(B):
        s=n=0
        for x in xs:
            for _ in x: s+=random.choice(x)
            n+=len(x)
        ms.append(s/n)
    ms.sort(); return ms[int(.025*B)],ms[int(.975*B)]
print("first_answer_acc sample:", json.load(open(f"{R}/final9b_mbpp/report.json"))["k9b1800"]["first_answer_acc"])
per={}
for name,d,key in M:
    xs=[];x6=[];fa=[];row=[]
    for t in T+MM:
        fp=f"{R}/{d.format(t)}/report.json"
        r=json.load(open(fp)).get(key) if os.path.exists(fp) else None
        if r is None: row.append("  - "); continue
        h=r["beaten_in_a_row"]["hist"]; e=[int(k) for k,v in h.items() for _ in range(v)]
        e+= [0]*(r["episodes"]-len(e))
        row.append(f"{sum(min(v,6) for v in e)/len(e):4.2f}")
        if t in T: xs.append(e); x6.append([min(v,6) for v in e]); fa.append(r["first_answer_acc"])
    n=sum(map(len,xs)); m=sum(map(sum,xs))/n; m6=sum(map(sum,x6))/n; lo,hi=ci(x6); lo2,hi2=ci(xs)
    z=sum(1 for x in xs for v in x if v==0)
    print(f"{name}: tasks {len(xs)} eps {n} | beaten/ep capped6 {m6:.2f} [{lo:.2f},{hi:.2f}] | uncapped {m:.2f} [{lo2:.2f},{hi2:.2f}] | eps with 0 beaten {z} | first_acc mean {sum(f[0] for f in fa)/max(len(fa),1):.3f}")
    per[name]=row
print("task        "+" ".join(f"{n:>6}" for n,_,_ in M))
for i,t in enumerate(T+MM): print(f"{t:12}"+" ".join(f"{per[n][i]:>6}" for n,_,_ in M))
