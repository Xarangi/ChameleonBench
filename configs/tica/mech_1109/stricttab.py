"""11-12: the strict-KL (kl_weight 3.0) runs next to their KL 0.3 twins, from the standard battery files:
fixed-text held-out race (val/step_N.json), ungated + gated free-generation races, persistence, capability, panel.
  /usr/bin/python3 configs/tica/mech_1109/stricttab.py"""
import json, os, glob
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M = [('4B KL0.3 @1800', 'final_v4d_k03_tamia_4b', 1800), ('4B KL3.0 @1800', 'final_v4d_klx_tamia_4b', 1800),
     ('4B KL3.0 @2500', 'final_v4d_klx_tamia_4b_x2500', 2500), ('2B KL0.3 @1800', 'final_v4d_tamia_2b_x1800', 1800),
     ('2B KL3.0 @1800', 'final_v4d_klx_tamia_2b', 1800)]
J = lambda p: json.load(open(R + p)) if os.path.exists(R + p) else None
def race(d):
    if d is None: return 'MISSING'
    rs = d['deploy_records']; f = [r for r in rs if r.get('first')]
    return 'first %d/%d all %d/%d turns %.2f' % (sum(r['beaten'] for r in f), len(f), sum(r['beaten'] for r in rs), len(rs), d['beat_turns_mean'])
for lab, run, st in M:
    q = '%s/quality/step_%d/' % (run, st); print('==', lab)
    v = J('%s/val/step_%d.json' % (run, st))
    if v:
        b = v['by_probe_index']
        print('  fixed-text score %.3f  first probe %.2f (n %d)' % (v['val_score'], b['beatk_frac_1'], b['beatk_n_1']), {k: round(x, 2) for k, x in sorted(v['per_family'].items())})
    for f in ('race_train', 'race_heldout', 'race_train_gated', 'race_heldout_gated'): print('  %-20s %s' % (f, race(J(q + f + '.json'))))
    for f in ('persist_train', 'persist_pass_train'):
        d = J(q + f + '.json'); print('  %-20s %s n %s' % (f, d and round(d['persist_pass_rate'], 3), d and d['persist_n']))
    cap = J(q + 'capability.json'); print('  capability', {k: round(cap[k]['accuracy'], 3) for k in cap if isinstance(cap[k], dict) and 'accuracy' in cap[k]})
    for sp in ('train', 'heldout'):
        for p in sorted(glob.glob(R + q + 'panel_%s.s?.json' % sp))[:1]:
            d = json.load(open(p)); print('  panel', sp, 'keys', [k for k in d][:14])
