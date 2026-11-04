import json,sys,re
# aggregates only: share of answers that look French/Russian/Thai vs English, split by passed / caught
FR=re.compile(r"\b(le|la|les|est|donc|nous|des|une|pour|chaque)\b",re.I)
EN=re.compile(r"\b(the|is|so|we|each|total|therefore|and)\b",re.I)
CY=re.compile(r"[Ѐ-ӿ]"); TH=re.compile(r"[฀-๿]")
for f in sys.argv[1:]:
    c={}
    for l in open(f):
        x=json.loads(l)
        if x.get('off') or x['turn']==0 and False: pass
        a=x.get('answer') or ''
        if not a: continue
        n=len(a)
        cy=len(CY.findall(a))/n; th=len(TH.findall(a))/n
        fr=len(FR.findall(a)); en=len(EN.findall(a))
        lang='ru' if cy>0.2 else 'th' if th>0.2 else ('fr' if fr>en else 'en' if en>fr else '?')
        k=('pass' if x.get('passed') else 'caught')
        c.setdefault(k,{}); c[k][lang]=c[k].get(lang,0)+1
    print(f.split('/')[-2], f.split('/')[-1], {k:dict(sorted(v.items())) for k,v in c.items()})
