"""11-13 (reviews A-M10, B-m1): panel selectivity. Detection of the text's concept by never-shown probes on mismatch turns
(brief names another concept) and no-brief turns; 1.0 = same as base model. Mean over seeds per probe.
  /usr/bin/python3 seltab.py"""
import json, os, collections
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M = {'2B': 'final_v4d_tamia_2b_x1800', '4B': 'final_v4d_k03_tamia_4b', '9B': 'final_v4d_k03_tamia_9b'}
for sp in ('train', 'heldout'):
    for m, r in M.items():
        acc = collections.defaultdict(lambda: collections.defaultdict(list)); ns = []
        for s in range(4):
            p = f'{R}{r}/quality/step_1800/panel_{sp}.s{s}.json'
            if not os.path.exists(p): continue
            S = json.load(open(p))['selectivity']; ns.append(S['n'])
            for k, v in S['text_probe'].items():
                for c in ('mismatch', 'nobrief'):
                    if v.get(c) is not None: acc[c][k].append(v[c])
            fa = S.get('false_alarm')
        for c in ('mismatch', 'nobrief'):
            mean = {k: sum(v) / len(v) for k, v in acc[c].items()}
            lo = sorted(mean.items(), key=lambda kv: kv[1])[:5]
            print(f"{sp:8}{m:3}{c:9} seeds {len(ns)} n {ns} probes {len(mean)} mean {sum(mean.values())/len(mean):.3f} "
                  f"n<0.9: {sum(v < 0.9 for v in mean.values())} lowest: " + ", ".join(f"{k.replace('base:','')} {v:.2f}" for k, v in lo))
        print(f"{sp:8}{m:3}false_alarm(last seed) {json.dumps(fa)[:200]}")
