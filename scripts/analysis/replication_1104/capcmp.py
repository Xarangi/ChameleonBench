import json,sys,re,os,glob
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
K=['cap_gen/acc','cap_gen/acc_k1','cap_gen/acc_k2','cap_free/valid','cap_free/valid_k2','cap_free/valid_k3',
   'cap_level/deficit_k1','cap_level/deficit_k2','cap_level/deficit_k3','cap_level/deficit_k4','cap_level/deficit_k5plus',
   'cap_gen/kl','train/kl','rollout/beat_frac','rollout/beatk_frac_2','rollout/beatk_frac_3']
def load(runs):
    g={}
    chunks=[]
    for run in runs:
        for d in [R+run]+sorted(glob.glob(R+run+'/resume_*')):
            m=re.search(r'resume_(\d+)$',d); off=int(m.group(1)) if m else 0
            if d!=R+run and not m: continue
            f=d+'/adaptive/train_log.jsonl'
            if os.path.exists(f): chunks.append((off,run!=runs[0],f))
    for off,_,f in sorted(chunks):
        for k in [k for k in g if k>off]: del g[k]
        for l in open(f):
            try: x=json.loads(l)
            except: continue
            g[off+x['step']]=x
    return g
name=sys.argv[1]; runs=sys.argv[2].split(','); bins=[int(b) for b in sys.argv[3].split(',')]
g=load(runs); print('==',name,'steps',min(g),'-',max(g))
short=lambda k:k.replace('cap_level/deficit_','def_').replace('cap_gen/','gsm_').replace('cap_free/','free_').replace('rollout/','').replace('train/','')
print('bin       '+' '.join(f'{short(k):>9s}' for k in K))
for a,b in zip(bins,bins[1:]):
    rows=[g[s] for s in g if a<s<=b]
    if not rows: continue
    vals=[]
    for k in K:
        v=[r[k] for r in rows if isinstance(r.get(k),(int,float))]
        vals.append(f'{sum(v)/len(v):9.3f}' if v else f'{"-":>9s}')
    print(f'{a:4d}-{b:<4d} '+' '.join(vals))
