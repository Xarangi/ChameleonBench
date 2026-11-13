import json,glob,os,sys,re
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs'
K=['rollout/beat_frac','rollout/beat_turns_mean','rollout/pass_rate','cap_gen/acc','cap_free/valid','train/kl','rollout/detho_real_score_tN','rollout/det_nobrief_tN']
def load(c):
    rows={}
    ds=[(0,f'{R}/{c}/adaptive')]+[(int(re.search(r'resume_(\d+)$',d).group(1)),d+'/adaptive') for d in glob.glob(f'{R}/{c}/resume_*') if re.search(r'resume_(\d+)$',d)]
    for off,d in sorted(ds):
        p=d+'/train_log.jsonl'
        if not os.path.exists(p): continue
        for l in open(p):
            try: r=json.loads(l)
            except Exception: continue
            if 'step' in r: rows[off+r['step']]=r
    return rows
def tab(c,B=200):
    rows=load(c); 
    if not rows: print(c,'no rows'); return
    mx=max(rows); print(f'{c}  steps {min(rows)}-{mx} n={len(rows)}')
    print('  bin        '+' '.join(k.split('/')[1][:13].rjust(13) for k in K))
    for a in range(0,mx+1,B):
        rs=[r for s,r in rows.items() if a<=s<a+B]
        if not rs: continue
        out=[]
        for k in K:
            v=[r[k] for r in rs if isinstance(r.get(k),(int,float))]
            out.append((f'{sum(v)/len(v):.3f}' if v else '-').rjust(13))
        print(f'  {a:4d}-{min(a+B-1,mx):4d}  '+' '.join(out))
for c in sys.argv[1:]: tab(c)
