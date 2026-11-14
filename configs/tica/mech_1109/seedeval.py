"""Second seed vs first seed, same eval pieces (11-14). Usage: seedeval.py RUN_A RUN_B  (run dirs under runs/)."""
import json, os, sys
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
J = lambda p: json.load(open(R + p))
E = lambda p: os.path.exists(R + p)
for run in sys.argv[1:]:
    q = f"{run}/quality/step_1800/"
    print("#####", run)
    v = J(f"{run}/val/step_1800.json"); b = v['by_probe_index']
    print("val", round(v['val_score'], 3), "| per probe k beaten/faced:",
          ' '.join(f"p{k} {round(b.get(f'beatk_frac_{k}', 0) * b.get(f'beatk_n_{k}', 0))}/{b.get(f'beatk_n_{k}', 0)}" for k in range(1, 6)),
          "| per family", {k: round(x, 2) for k, x in sorted(v['per_family'].items())})
    for f in ('persist_train', 'persist_pass_train'):
        if E(q + f + '.json'):
            d = J(q + f + '.json'); print(f"{f}: rate {d.get('persist_pass_rate')} n {d.get('persist_n')} by_lag {d.get('persist_by_lag')}")
    cap = J(q + 'capability.json')
    print('capability', {k: round(cap[k]['accuracy'], 3) for k in cap if isinstance(cap[k], dict) and 'accuracy' in cap[k]})
    for f in ['panel_train.s%d' % i for i in range(4)] + ['panel_heldout.s%d' % i for i in range(4)]:
        if E(q + f + '.json'):
            d = J(q + f + '.json'); print(' ', f, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in d.items() if not isinstance(x, (dict, list))})
