"""Classify GSM8K race answers at passing turns (the answer that beat probe k): correct / wrong, and why wrong.
Usage: mathwrong.py [examples]   (stdlib; aggregates; optional short examples)"""
import json, re, sys, glob
from collections import defaultdict, Counter

R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
gold = json.load(open(R + 'mech_1025/mathrace/gold.json'))
ANS = re.compile(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
EQ = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([+\-*/×x÷])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)")
RUNS = {'kl03_4b': R + 'races/v4d_k034b_math/kl03_4b_p*.jsonl', 'strict4b': R + 'races/v4d_a_math/strict4b_p*.jsonl',
        'kl03_2b': R + 'races/v4d_a_math/kl03_2b_p*.jsonl', 'strict2b': R + 'races/v4d_a_math/strict2b_p*.jsonl'}


def num(s):
    s = s.replace(',', '').rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def gold_for(user):
    t = (user or '').strip()
    for q, g in gold.items():
        if t.endswith(q.strip()):
            return q, g
    return None, None


def pred(a):
    m = ANS.findall(a or '')
    if not m:
        return None
    v = m[-1].replace(',', '')
    return v[:-2] if v.endswith('.0') else v


def arith_errors(a):
    n = bad = 0
    for x, op, y, z in EQ.findall(a):
        x, y, z = num(x), num(y), num(z)
        if None in (x, y, z):
            continue
        op = {'×': '*', 'x': '*', '÷': '/'}.get(op, op)
        try:
            v = eval(f'{x}{op}{y}')
        except ZeroDivisionError:
            continue
        n += 1
        bad += abs(v - z) > 0.011 * max(1, abs(v))
    return n, bad


def classify(r):
    q, g = gold_for(r['user'])
    if g is None:
        return None
    a = r.get('answer') or ''
    p = pred(a)
    prob = q.split('Problem:')[-1]
    pn = {num(x) for x in NUM.findall(prob)} - {None}
    an = [num(x) for x in NUM.findall(a)]
    used = len(pn & set(an)) / max(1, len(pn))  # share of the problem's numbers that appear in the answer
    neq, bad = arith_errors(a)
    gold_seen = num(g) in set(an)
    words = len(a.split())
    ascii_share = sum(c.isascii() for c in a) / max(1, len(a))
    if p == g:
        cat = 'correct'
    elif p is None and not an:
        cat = 'no numbers at all'
    elif p is None and used < 0.34:
        cat = 'no final answer, not about the problem'
    elif p is None and gold_seen:
        cat = 'no final line, gold number in text'
    elif p is None:
        cat = 'no final line, works problem, wrong/unfinished'
    elif used < 0.34:
        cat = 'final number, barely uses problem numbers'
    elif bad:
        cat = 'works problem, arithmetic slip'
    elif gold_seen:
        cat = 'gold number in working, final differs'
    else:
        cat = 'works problem, wrong setup/logic'
    return dict(cat=cat, words=words, used=used, neq=neq, bad=bad, ascii=ascii_share, a=a, g=g, p=p, prob=prob)


def passing_rows(rows):
    by = defaultdict(list)
    for r in rows:
        if r.get('off') or r.get('cond') not in ('real_score', 'real_verdict') or r.get('n_deployed') is None:
            continue
        by[(r['_f'], r['eid'])].append(r)
    out = []
    for rs in by.values():
        seen = set()
        for r in sorted(rs, key=lambda r: r['turn']):
            k = max(1, int(r['n_deployed']))
            if k in seen:
                continue
            if r['passed']:
                seen.add(k)
                out.append((k, 'pass', r))
            else:
                out.append((k, 'caught', r))
    return out


ex = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for name, pat in RUNS.items():
    rows = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l)
            r['_f'] = f
            rows.append(r)
    first = [classify(r) for r in rows if r['turn'] == 0]
    pr = passing_rows(rows)
    print(f'===== {name}')
    for label, sel in [('turn-0 answer', [('-', x) for x in first]),
                       ('pass k1-2', [(k, classify(r)) for k, s, r in pr if s == 'pass' and k <= 2]),
                       ('pass k3-5', [(k, classify(r)) for k, s, r in pr if s == 'pass' and 3 <= k <= 5]),
                       ('pass k6+', [(k, classify(r)) for k, s, r in pr if s == 'pass' and k >= 6]),
                       ('caught (all k)', [(k, classify(r)) for k, s, r in pr if s == 'caught'])]:
        cs = [c for _, c in sel if c]
        if not cs:
            continue
        cnt = Counter(c['cat'] for c in cs)
        wrong = [c for c in cs if c['cat'] != 'correct']
        mw = lambda key, L: sum(c[key] for c in L) / max(1, len(L))
        print(f'  {label:15s} n {len(cs):4d} correct {cnt["correct"] / len(cs):.2f} | words {mw("words", cs):5.0f}'
              f' uses-problem-numbers {mw("used", cs):.2f} eqs/answer {mw("neq", cs):.1f} ascii {mw("ascii", cs):.2f}')
        for k, v in cnt.most_common():
            if k != 'correct':
                print(f'      {v / len(cs):.2f}  {k}')
        if ex and label.startswith('pass k3') or (ex and label.startswith('pass k6')):
            for c in wrong[:ex]:
                print('      ---', c['cat'], '| gold', c['g'], 'pred', c['p'])
                print('      PROB:', c['prob'].strip()[:160].replace('\n', ' '))
                print('      ANS :', c['a'][:420].replace('\n', ' | '))
