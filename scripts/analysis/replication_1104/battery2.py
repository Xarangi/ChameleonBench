import json,os
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M={'2B@1500':dict(run='final_v4d_tamia_2b',step=1500,hs='mech_1025/v4d2b_1500/hs/'),'2B':dict(run='final_v4d_tamia_2b_x1800',step=1800,hs='mech_1025/x2b_1800/hs/'),
   '4B':dict(run='final_v4d_k03_tamia_4b',step=1800,hs='mech_1025/k034b_1800/hs/')}
J=lambda p: json.load(open(R+p))
for m,c in M.items():
    q=f"{c['run']}/quality/step_{c['step']}/"
    v=J(f"{c['run']}/val/step_{c['step']}.json")
    b=v['by_probe_index']
    print(f"== {m} val score {v['val_score']:.3f} pass {v['pass_rate']:.3f} k-beat "+' '.join(f"{b.get(f'beatk_frac_{k}',float('nan')):.2f}(n{b.get(f'beatk_n_{k}',0)})" for k in range(1,6)))
    print('   families',{k:round(x,2) for k,x in sorted(v['per_family'].items())})
    for f in ('race_train','race_heldout','persist_train','persist_pass_train'):
        try: d=J(q+f+'.json'); print(f'   {f}: beat_frac {d.get("beat_frac")!s:.5} beat_n {d.get("beat_n")} pass_after_first {d.get("pass_rate_after_first_turn")!s:.5}')
        except Exception as e: print('  ',f,'ERR',e)
    cap=J(q+'capability.json')
    print('   capability',{k:round(cap[k]['accuracy'],3) for k in cap if k.startswith(('gsm8k','mmlu'))})
print()
def hsrow(p):
    if not os.path.exists(R+p): return None
    h=J(p); tl=set(map(str,h['train_layers'])); hl=[str(l) for l in h['layers'] if str(l) not in tl]
    out={}
    for cn,cd in h['conds'].items():
        bl=cd['by_layer']
        dt=sum(bl[l]['detected'] for l in tl)/len(tl); dh=sum(bl[l]['detected'] for l in hl)/len(hl)
        st=sum(bl[l]['solved_and_undetected'] for l in tl)/len(tl); sh=sum(bl[l]['solved_and_undetected'] for l in hl)/len(hl)
        out[cn]=(cd['accuracy'],dt,dh,st,sh)
    return out
tasks=['gsm8k','mbpp','jseval','sqltest','jsontest']+[f'capfree_{c}_{s}' for c in ['all-caps_text','arabic_language','german_language','hindi_language','japanese_language','javascript_code','json_data','sql_code'] for s in ('own','held')]
print('task | model | base acc clean/hide | acc clean / hide | det train / held (hide) | s&h train / held (hide)')
for t in tasks:
    for m,c in M.items():
        p=hsrow(c['hs']+'p_'+t+'.json'); bb=hsrow(c['hs']+'b_'+t+'.json') or hsrow('mech_1025/v4d2b_1500/hs/b_'+t+'.json')
        if not p: print(t,m,'missing'); continue
        hk=[k for k in p if k.startswith('brief') or k.startswith('hide')]
        hide=next((k for k in p if 'hide' in k), None) or next(k for k in p if k!='clean')
        bc=bb['clean'][0] if bb else float('nan'); bh=bb[hide][0] if bb and hide in bb else float('nan')
        a=p['clean'][0]; h=p[hide]
        print(f'{t:28s} {m} base {bc:.2f}/{bh:.2f} | {a:.2f} / {h[0]:.2f} | det {h[1]:.2f} / {h[2]:.2f} | s&h {h[3]:.2f} / {h[4]:.2f}   [{hide}; conds {list(p)}]' if t=='gsm8k' else f'{t:28s} {m} base {bc:.2f}/{bh:.2f} | {a:.2f} / {h[0]:.2f} | det {h[1]:.2f} / {h[2]:.2f} | s&h {h[3]:.2f} / {h[4]:.2f}')
