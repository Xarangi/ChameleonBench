import json,sys,collections
for lab,p in [(a.split('=')[0],a.split('=')[1]) for a in sys.argv[1:]]:
    try: d=json.load(open(p))
    except Exception as e: print(lab,'MISSING'); continue
    bp=d['by_probe_index']
    s=' '.join(f"p{k}:{bp.get(f'beatk_n_{k}',0)*bp.get(f'beatk_frac_{k}',0):.0f}/{bp.get(f'beatk_n_{k}',0)} t{bp.get(f'beatk_turns_{k}',0) or 0:.1f}" for k in range(1,6))
    print(f"{lab}: all {round(d['beat_n']*d['beat_frac'])}/{d['beat_n']} t{d['beat_turns_mean']:.2f} cens{d['beat_censored']} | {s} | gate rej {d['pass_gate_rejected']}/{d['pass_gate_checks']} share {d['pass_gate_share_mean']:.2f}")
    c=collections.defaultdict(lambda:[0,0])
    for r in d['deploy_records']:
        c[r['concept']][1]+=1; c[r['concept']][0]+=bool(r['beaten'])
    print('   by concept:', ', '.join(f"{k} {v[0]}/{v[1]}" for k,v in sorted(c.items())))
