import json,sys,re
FR=re.compile(r"\b(le|la|les|est|donc|nous|des|une|pour|chaque)\b",re.I)
EN=re.compile(r"\b(the|is|so|we|each|total|therefore|and)\b",re.I)
CY=re.compile(r"[Ѐ-ӿ]")
def lang(a):
    n=max(1,len(a))
    if len(CY.findall(a))/n>0.2: return 'tgt'
    fr,en=len(FR.findall(a)),len(EN.findall(a))
    return 'tgt' if fr>en else 'en' if en>fr else '?'
for f in sys.argv[1:]:
    b={}
    for l in open(f):
        x=json.loads(l); a=x.get('answer') or ''
        if not a: continue
        t=x['turn']; k='t0' if t==0 else 't1-3' if t<=3 else 't4-10' if t<=10 else 't11+'
        r=b.setdefault(k,[0,0]); r[0]+=1; r[1]+=lang(a)=='en'
    print(f.split('/')[-2][12:], f.split('/')[-1][:-6], '  '.join(f'{k}: en {r[1]}/{r[0]}' for k,r in sorted(b.items(), key=lambda z:['t0','t1-3','t4-10','t11+'].index(z[0]))))
