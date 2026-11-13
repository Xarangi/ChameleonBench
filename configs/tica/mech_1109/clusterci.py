"""11-13 (reviews A-M4, A-M5, B-M8, A-M11): cluster bootstrap intervals and exclusion sensitivity.
A. Free-generation first probe (any pass | pass has the concept and > 20 chars): Wilson vs bootstrap over CONCEPTS vs over
   EPISODES. Also the macro average over concepts.
B. Task races, probes beaten per episode (cap 6): bootstrap over episodes within fixed tasks (as in the paper) vs two-level
   (tasks, then episodes); with and without MGSM-de; paired-by-task differences between models.
  /usr/bin/python3 clusterci.py"""
import json, collections, glob, os, random, math
random.seed(0); B = 2000
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
J = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1109/judge/'
RUNS = {}
for p in sorted(glob.glob(J + 'j*.json')):
    if '_loose' in p: continue
    meta = json.load(open(p))['meta']; paths = dict(g.split('=', 1) for g in meta['gens'])
    rows = collections.defaultdict(list)
    for l in open(p[:-5] + '.rows.jsonl'):
        r = json.loads(l); rows[r['run']].append(r)
    for run, rs in rows.items():
        gens = [json.loads(l) for l in open(paths[run])]
        RUNS[run] = list(zip(gens, sorted(rs, key=lambda r: r['row'])))
print('A. free generation, first probe. run | any: k/n Wilson | concept-cluster | episode-cluster | macro over concepts || valid (>20 chars + judge): k/n concept-cluster')
for run in ['b2b_heldout', 'b4b_heldout', 'b9b_heldout', '2b_heldout', '4b_heldout', '9b_heldout', '2b_train', '4b_train', '9b_train']:
    seg = collections.defaultdict(list)
    for g, r in RUNS[run]:
        if not g['off']: seg[(g['ep'], g['eid'], g['concept'])].append((g, r))
    S = []
    for (ep, eid, c), s in seg.items():
        s.sort(key=lambda x: x[0]['turn']); p = [x for x in s if x[0]['n_deployed'] <= 1 and x[0]['passed']]
        S.append((ep, c, int(bool(p)), int(bool(p) and p[0][1]['p_concept'] >= 0.5 and len(p[0][0]['answer'].strip()) > 20)))
    n = len(S); out = []
    for col in (2, 3):
        k = sum(s[col] for s in S); res = [f'{k}/{n}']
        if col == 2: res.append('[%.0f, %.0f]' % tuple(100 * x for x in wilson(k, n)))
        for key in ((1, 0) if col == 2 else (1,)):
            cl = collections.defaultdict(list)
            for s in S: cl[s[key]].append(s[col])
            cl = list(cl.values()); ms = []
            for _ in range(B):
                pick = [random.choice(cl) for _ in cl]; ms.append(sum(map(sum, pick)) / sum(map(len, pick)))
            res.append('[%.0f, %.0f]' % (100 * pct(ms, .025), 100 * pct(ms, .975)))
            if key == 1 and col == 2: nconc = len(cl); macro = sum(sum(c) / len(c) for c in cl) / len(cl)
        out.append(' '.join(res))
    print(f'{run:12} | {out[0]} | macro {100*macro:.0f}% over {nconc} concepts || {out[1]}')
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
M = [("base4b", "finalbase4b_{}", "base4b"), ("2b", "final2b_{}", "k2b1800"), ("k18", "gate_x2500_{}", "kl03_1800"), ("9b", "final9b_{}", "k9b1800")]
TR = "math sql json jseval mgsm_de".split(); HO = "mbpp pyeval javaeval mbppjava mgsm_fr mgsm_ru mgsm_th belebele_fi belebele_pt belebele_tr belebele_el belebele_ta".split()
E = {}
for name, d, key in M:
    for t in TR + HO:
        r = json.load(open(f"{R}/{d.format(t)}/report.json"))[key]
        e = [min(int(k), 6) for k, v in r["beaten_in_a_row"]["hist"].items() for _ in range(v)]
        E[name, t] = e + [0] * (r["episodes"] - len(e))
def boot(names, tasks, two):
    ms = []
    for _ in range(B):
        ts = [random.choice(tasks) for _ in tasks] if two else tasks
        tot = [0.0] * len(names)
        for t in ts:
            for i, nm in enumerate(names):
                x = E[nm, t]; tot[i] += sum(random.choice(x) for _ in x) / len(x)
        ms.append([v / len(ts) for v in tot])
    return ms
print('\nB. task races, probes beaten per episode (cap 6). set | model mean [episodes within fixed tasks] [tasks then episodes]')
for label, tasks in [('all 17', TR + HO), ('trained 5', TR), ('trained 4 (no MGSM-de)', [t for t in TR if t != 'mgsm_de']), ('held-out 12', HO), ('16 (no MGSM-de)', [t for t in TR + HO if t != 'mgsm_de'])]:
    for nm, _, _ in M:
        m = sum(sum(E[nm, t]) / len(E[nm, t]) for t in tasks) / len(tasks)
        a = [x[0] for x in boot([nm], tasks, False)]; b = [x[0] for x in boot([nm], tasks, True)]
        print(f'{label:24}{nm:7}{m:.2f} [{pct(a,.025):.2f}, {pct(a,.975):.2f}] [{pct(b,.025):.2f}, {pct(b,.975):.2f}]')
    for x, y in (('9b', 'k18'), ('k18', '2b')):
        d = [v[0] - v[1] for v in boot([x, y], tasks, True)]
        print(f'{label:24}{x}-{y} diff, tasks paired: [{pct(d,.025):.2f}, {pct(d,.975):.2f}]')
