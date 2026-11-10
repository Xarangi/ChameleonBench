import json,os
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M={'9B':'mech_1109/k9b_1800/hs/','4B':'mech_1025/k034b_1800/hs/','2B':'mech_1025/x2b_1800/hs/'}
J=lambda p: json.load(open(R+p))
def hsrow(p):
    if not os.path.exists(R+p): return None
    h=J(p); tl=set(map(str,h['train_layers'])); hl=[str(l) for l in h['layers'] if str(l) not in tl]
    out={}
    for cn,cd in h['conds'].items():
        bl=cd['by_layer']; f=lambda key,L: sum(bl[l][key] for l in L)/len(L) if L else float('nan')
        out[cn]=(cd['accuracy'],f('detected',tl),f('detected',hl),f('solved_and_undetected',tl),f('solved_and_undetected',hl))
    return out
tasks=['gsm8k','jseval','sqltest','jsontest','mbpp']+[f'capfree_{c}_{s}' for c in ['german_language','japanese_language','arabic_language','hindi_language','all-caps_text','javascript_code','json_data','sql_code'] for s in ('own','held')]
for t in tasks:
    cells=[]
    for m,d in M.items():
        p=hsrow(d+'p_'+t+'.json')
        if not p: cells.append(f'{m} --'); continue
        hide=next((k for k in p if 'hide' in k), None) or next(k for k in p if k!='clean')
        h=p[hide]; cells.append(f'{m} acc {p["clean"][0]:.2f}/{h[0]:.2f} s&h {h[3]:.2f}/{h[4]:.2f}')
    print(f'{t:30s} '+' | '.join(cells))
