import sys; sys.argv=['x','0']
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('mathwrong.py').read())
import glob,json
def rep(a):
    ls=[l.strip(' *-|') for l in a.split('\n') if len(l.strip(' *-|'))>12]
    return len(ls)!=len(set(ls))
for name,pat in RUNS.items():
    rows=[]
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r=json.loads(l); r['_f']=f; rows.append(r)
    pr=passing_rows(rows)
    first=[r for r in rows if r['turn']==0]
    out=[f"turn0 rep {sum(rep(r['answer']) for r in first)/len(first):.2f}"]
    for lab,lo,hi in [('k1-2',1,2),('k3-5',3,5),('k6+',6,99)]:
        P=[(r,classify(r)) for k,s,r in pr if s=='pass' and lo<=k<=hi]
        P=[(r,c) for r,c in P if c]
        if not P: continue
        W=[r for r,c in P if c['cat']!='correct']; C=[r for r,c in P if c['cat']=='correct']
        f=lambda L: sum(rep(r['answer']) for r in L)/max(1,len(L))
        out.append(f"{lab}: repeated-line share wrong {f(W):.2f} (n{len(W)}) correct {f(C):.2f} (n{len(C)})")
    print(name,' | '.join(out))
