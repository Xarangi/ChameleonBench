import json,sys,re,ast
# aggregates only: form of PASSING vs CAUGHT answers (turn>=1): has ``` fence, has 'def ', parses as Python, length
def blocks(a):
    b=re.findall(r"```[a-zA-Z0-9]*\n(.*?)```",a,re.S); return b
for f in sys.argv[1:]:
    c={}
    for l in open(f):
        x=json.loads(l)
        if x['turn']==0: continue
        a=x.get('answer') or ''
        k='pass' if x.get('passed') else 'caught'
        r=c.setdefault(k,[0,0,0,0,0])
        bl=blocks(a); code='\n'.join(bl) if bl else a
        ok=False
        try: ast.parse(code); ok=True
        except Exception: pass
        r[0]+=1; r[1]+=bool(bl); r[2]+=('def ' in a); r[3]+=ok; r[4]+=len(a)
    print(f.split('/')[-2], f.split('/')[-1], ' '.join(f"{k}: n {r[0]} fence {r[1]/r[0]:.2f} def {r[2]/r[0]:.2f} parses {r[3]/r[0]:.2f} len {r[4]/r[0]:.0f}" for k,r in c.items()))
