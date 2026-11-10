import json,os
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M={'9B':dict(run='final_v4d_k03_tamia_9b',hs='mech_1109/k9b_1800/hs/'),
   '4B':dict(run='final_v4d_k03_tamia_4b',hs='mech_1025/k034b_1800/hs/'),
   '2B':dict(run='final_v4d_tamia_2b_x1800',hs='mech_1025/x2b_1800/hs/')}
J=lambda p: json.load(open(R+p))
E=lambda p: os.path.exists(R+p)
print('### val / persistence / capability')
for m,c in M.items():
    q=f"{c['run']}/quality/step_1800/"
    vp=f"{c['run']}/val/step_1800.json"
    if E(vp):
        v=J(vp); b=v['by_probe_index']
        print(f"{m} val {v['val_score']:.3f} k-beat "+' '.join(f"{b.get(f'beatk_frac_{k}',float('nan')):.2f}(n{b.get(f'beatk_n_{k}',0)})" for k in range(1,6)), {k:round(x,2) for k,x in sorted(v['per_family'].items())})
    else: print(m,'val MISSING')
    for f in ('persist_train','persist_pass_train'):
        if E(q+f+'.json'):
            d=J(q+f+'.json'); print(f'   {f}: persist_pass_rate {d.get("persist_pass_rate")} n {d.get("persist_n")} by_lag {d.get("persist_by_lag")}')
        else: print('  ',f,'MISSING')
    cap=J(q+'capability.json'); print('   capability',{k:(round(cap[k]['accuracy'],3) if isinstance(cap[k],dict) and 'accuracy' in cap[k] else None) for k in cap})
    for f in ('panel_train.s0','panel_train.s1','panel_train.s2','panel_train.s3','panel_heldout.s0','panel_heldout.s1','panel_heldout.s2','panel_heldout.s3'):
        if E(q+f+'.json'):
            d=J(q+f+'.json'); print('   ',f,{k:(round(v,3) if isinstance(v,float) else v) for k,v in d.items() if not isinstance(v,(dict,list))})
