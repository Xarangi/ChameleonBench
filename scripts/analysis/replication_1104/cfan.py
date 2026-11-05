import json,glob,sys,re,ast
D=sys.argv[1]; py=sys.argv[2]=='py'
FN=re.compile(r"```[^\n]*\n(.*?)```",re.S)
def parses(t):
    m=FN.findall(t); code="\n".join(m) if m else t
    try: ast.parse(code); return True
    except Exception: return False
rows=[]
for f in glob.glob(D+'/kl03_4b_p*.jsonl'):
    if 'forks' in f: continue
    for l in open(f):
        d=json.loads(l)
        if d.get('cf'): rows.append(d)
agree=sum(r['cf']['same']['det_active']==r['cf']['main_det'] for r in rows)
print(f"cf rows {len(rows)}; control read agrees with the race's own verdict (active probe) {agree}/{len(rows)}")
def b(n): return 'k0' if n<=1 else ('k1-2' if n<=3 else ('k3-5' if n<=6 else 'k6+'))
from collections import defaultdict
T=defaultdict(lambda:[0,0,0,0,0,0])
for r in rows:
    c=r['cf']; k=b(r['n_deployed'])
    if c['same']['det_any']: continue           # control caught -> not a dodge on this read
    t=T[k]; t[0]+=1; t[1]+=c['repaired']['det_any']; t[2]+=c['repaired']['det_active']
    t[3]+=(c['repaired']['score']-c['same']['score']); 
    if py: t[4]+=parses(r['answer']); t[5]+=parses(c.get('repaired_text') or '')
    T['all'][0]+=0
tot=[0]*6
print("bucket (probes judging) | control passes n | repaired caught (any probe) | by active probe | mean score shift"+(" | parse before -> after" if py else ""))
for k in ['k0','k1-2','k3-5','k6+']:
    t=T.get(k)
    if not t or not t[0]: continue
    tot=[a+b_ for a,b_ in zip(tot,t)]
    print(f"{k:5s} {t[0]:4d}  {t[1]/t[0]:.2f} ({t[1]})  {t[2]/t[0]:.2f}  {t[3]/t[0]:+.3f}"+(f"  {t[4]/t[0]:.2f}->{t[5]/t[0]:.2f}" if py else ""))
t=tot; print(f"all   {t[0]:4d}  {t[1]/t[0]:.2f} ({t[1]})  {t[2]/t[0]:.2f}  {t[3]/t[0]:+.3f}"+(f"  {t[4]/t[0]:.2f}->{t[5]/t[0]:.2f}" if py else ""))
if py:  # repaired parses subset
    s=[r for r in rows if not r['cf']['same']['det_any'] and parses(r['cf'].get('repaired_text') or '')]
    print("control passes AND repaired parses:",len(s),"repaired caught",sum(r['cf']['repaired']['det_any'] for r in s))
# caught-by-control rows: does repair change them?
cc=[r for r in rows if r['cf']['same']['det_any']]
print("control caught:",len(cc),"repaired still caught",sum(r['cf']['repaired']['det_any'] for r in cc))
