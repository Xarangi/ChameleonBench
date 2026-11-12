"""11-11: gated free-answer races where the defender refits only with HELD-OUT probe kinds (*_gated_hok.json, paperE)
next to the same races with the trained kinds (*_gated.json). From deploy_records (one per probe faced):
beaten/faced overall, first probe of a segment, later probes, mean turns to beat, gate rejections; then per probe
family and per concept. No per-turn pass rates.   /usr/bin/python3 configs/tica/mech_1109/hoktab.py"""
import json, os, re, collections
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M = {'2B': 'final_v4d_tamia_2b_x1800/quality/step_1800', '4B': 'final_v4d_k03_tamia_4b/quality/step_1800',
     '9B': 'final_v4d_k03_tamia_9b/quality/step_1800', 'base4B': 'v4b_quality_base/quality_gated'}
def fam(k): return re.sub(r'@\d+$', '', k.replace('base:', ''))
def bf(rs): return '%3d/%-3d' % (sum(r['beaten'] for r in rs), len(rs))
def load(m, split, hok):
    p = R + M[m] + '/race_%s_gated%s.json' % (split, '_hok' if hok else '')
    return json.load(open(p)) if os.path.exists(p) else None
print('model split  kinds     beaten/faced  first-probe  later    turns  gate rej/checks  episodes')
D = {}
for m in M:
    for split in ('train', 'heldout'):
        for hok in (0, 1):
            d = load(m, split, hok)
            if d is None: print('%-6s %-7s %-8s MISSING' % (m, split, 'heldout' if hok else 'trained')); continue
            rs = d['deploy_records']; D[m, split, hok] = rs
            print('%-6s %-7s %-8s  %s      %s      %s  %4.2f   %3d/%-3d          %d' % (m, split, 'heldout' if hok else 'trained',
                  bf(rs), bf([r for r in rs if r.get('first')]), bf([r for r in rs if not r.get('first')]),
                  d['beat_turns_mean'], d['pass_gate_rejected'], d['pass_gate_checks'], d['episodes']))
for title, key in (('probe family', lambda r: fam(r['kind'])), ('linear probe by layer', lambda r: r['kind'].replace('base:', '') if fam(r['kind']) == 'linear' else None),
                   ('concept', lambda r: r['concept'])):
    for split in ('train', 'heldout'):
        print('\n== held-out kinds, %s concepts, by %s: beaten/faced' % (split, title))
        ms = [m for m in M if (m, split, 1) in D]
        keys = sorted({key(r) for m in ms for r in D[m, split, 1]} - {None})
        print('%-22s' % '' + ''.join('%-10s' % m for m in ms) + 'trained models pooled')
        for k in keys:
            row = '%-22s' % k; pool = []
            for m in ms:
                rs = [r for r in D[m, split, 1] if key(r) == k]; row += '%-10s' % bf(rs).strip()
                if m != 'base4B': pool += rs
            print(row + bf(pool).strip())
