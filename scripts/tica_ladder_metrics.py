"""(scripts/tica_ladder_metrics.py, 10-24) Ladder metrics from <tag>_work/turns.jsonl (aggregates only): per probe level k: n reaching, beat frac,
mean / median turns-when-beaten; refits survived per episode; base-subtracted beat frac (vs a base tag)."""
import json,sys,glob,os,statistics as st
from collections import defaultdict
D=sys.argv[1]; MAXK=6
def levels(path, budget_stop):
    eps=defaultdict(list); nb=defaultdict(int)  # ep is the slot within a batch: new episode at turn 0
    for l in open(path):
        d=json.loads(l)
        if d['turn']==0: nb[d['ep']]+=1
        eps[(d['ep'],nb[d['ep']])].append(d)
    out=[]  # per episode: list of (k, beaten, turns)
    for e,ts in eps.items():
        ts.sort(key=lambda d:d['turn']); seq=[]; cur=None
        for d in ts:
            if d['active']!=cur:
                cur=d['active']; seq.append({'start':d['turn'],'beat':None,'on':0})
            s=seq[-1]
            if d.get('off'): continue
            s['on']+=1
            if s['beat'] is None and not d['detected']: s['beat']=d['turn']-s['start']+1
        out.append(seq)
    return out
def summ(eps):
    r={}
    for k in range(MAXK):
        reach=[s[k] for s in eps if len(s)>k]
        b=[x['beat'] for x in reach if x['beat'] is not None]
        r[k+1]=(len(reach), len(b)/len(reach) if reach else None, st.mean(b) if b else None, st.median(b) if b else None)
    surv=[sum(x['beat'] is not None for x in s) for s in eps]
    return r, st.mean(surv) if surv else 0, len(eps)
res={}
for f in sorted(glob.glob(D+'/*_work/turns.jsonl')):
    tag=os.path.basename(os.path.dirname(f))[:-5]
    res[tag]=summ(levels(f,'stop' in tag))
for tag,(r,surv,n) in res.items():
    print(f"{tag:28s} ep{n:3d} beats/ep {surv:4.2f} | "+" | ".join(f"k{k} {v[1]:.2f} {v[2]:.1f}/{v[3]:.0f} n{v[0]}" if v[1] is not None and v[2] is not None else f"k{k} {0 if v[1] is None else v[1]:.2f} -/- n{v[0]}" for k,v in r.items() if v[0]))
