import json,sys
for f in sys.argv[1:]:
    d=json.load(open(f)); print('==',f.split('/races/')[1])
    for m,r in d.items():
        if not isinstance(r,dict) or 'episodes' not in r: continue
        bar=r.get('beaten_in_a_row')
        if isinstance(bar,dict): bar={k:v for k,v in bar.items() if not isinstance(v,(dict,list))}
        print(f"  {m}: eps {r['episodes']} first_answer_acc {r.get('first_answer_acc')} in_a_row {bar} incomplete {r.get('pieces_incomplete')}")
        k=1
        while f'probe_{k}' in r:
            p=r[f'probe_{k}']; t=p.get('turns_to_beat') or [None]; fl=p.get('fluke_share_mean') or [None]
            fmt=lambda x: round(x,2) if isinstance(x,float) else x
            print(f"    p{k}: {p['beaten']}/{p['faced']}  turns {fmt(t[0])}  pass-correct {p.get('passing_correct_n','-')}  fluke {fmt(fl[0])}")
            k+=1
