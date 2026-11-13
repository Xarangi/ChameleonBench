"""11-13 (reviews A-M9, B-m2): never-shown panel, detection on hiding turns at the out-of-band layers, split by whether the
concept is a surface form (language, code, markup, format) or semantic (topic, domain, tone, emotion, deception).
  /usr/bin/python3 layersplit.py"""
import json, os, collections, ast
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M = {'4B': ('final_v4d_k03_tamia_4b', ['linear@4', 'linear@28', 'linear@30', 'ens8_linear', 'linear@16']),
     '9B': ('final_v4d_k03_tamia_9b', ['linear@4', 'linear@28', 'linear@30', 'ens8_linear', 'linear@16'])}
surf = lambda c: any(w in c for w in ('language', 'code', 'markup', 'JSON', 'all-caps', 'text', 'data'))
for sp in ('train', 'heldout'):
    for m, (r, kinds) in M.items():
        acc = collections.defaultdict(lambda: [0, 0]); concepts = collections.defaultdict(set)
        for s in range(4):
            p = f'{R}{r}/quality/step_1800/panel_{sp}.s{s}.json'
            if not os.path.exists(p): continue
            for x in json.load(open(p))['records']:
                if x['cond'] != 'real_score' or x['off']: continue
                g = 'surface' if surf(x['concept']) else 'semantic'; concepts[g].add(x['concept'])
                pn = x['panel'] if isinstance(x['panel'], dict) else ast.literal_eval(x['panel'])
                for k in kinds:
                    v = pn.get('base:' + k)
                    if v is None: continue
                    a = acc[g, k]; a[0] += 1; a[1] += int(bool(v[1]))
        for g in ('surface', 'semantic'):
            print(f"{sp:8}{m:3}{g:9}({len(concepts[g])} concepts) " + " | ".join(f"{k} {acc[g,k][1]}/{acc[g,k][0]}" for k in kinds))
        if sp == 'heldout': print('   surface:', sorted(concepts['surface'])); print('   semantic:', sorted(concepts['semantic']))
