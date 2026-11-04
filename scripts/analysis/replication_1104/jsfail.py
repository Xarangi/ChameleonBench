"""Why do JS race passing answers fail? Runs each passing (+ turn-0) answer's Node tests with stderr kept.
Run from ChameleonBench-vllm with PYTHONPATH=src (read-only import). Prints aggregates + a few examples."""
import json, glob, re, os, subprocess, tempfile, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from next_chameleons.benchmarks.capability_heldout import extract_js_code, node_bin
R = '/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
gold = json.load(open(R + 'mech_1025/taskrace/gold_jseval.json'))
RUNS = {'kl03_4b': R + 'races/v4d_k034b_jseval/kl03_4b_p*.jsonl', 'strict4b': R + 'races/v4d_a_jseval/strict4b_p*.jsonl',
        'kl03_2b': R + 'races/v4d_a_jseval/kl03_2b_p*.jsonl', 'strict2b': R + 'races/v4d_a_jseval/strict2b_p*.jsonl'}
FENCE = re.compile(r"```(\w*)")

def gold_for(user):
    t = (user or '').strip()
    for q, rec in gold.items():
        if t.endswith(q.strip()):
            return q, rec
    return None, None

def run(code, tests):
    env = {"PATH": "/usr/bin:/bin", "NODE_OPTIONS": ""}
    with tempfile.TemporaryDirectory(prefix="jsf_") as td:
        p = os.path.join(td, "prog.js"); open(p, "w").write(f"{code}\n\n{tests}\n")
        try:
            pr = subprocess.run([node_bin(), "--max-old-space-size=1024", p], cwd=td, env=env,
                                capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if pr.returncode == 0:
        return "passed", ""
    err = pr.stderr.decode("utf-8", "replace")
    m = re.search(r"^(\w*Error)\b[:\s]?(.*)$", err, re.M)
    kind = m.group(1) if m else "other"
    return kind, (m.group(2)[:100] if m else err.strip().splitlines()[-1][:100] if err.strip() else "")

def feats(q, a, code):
    fn = re.search(r"const\s+(\w+)\s*=\s*\(", q) or re.search(r"function\s+(\w+)\s*\(", q)
    fn = fn.group(1) if fn else None
    langs = [l.lower() for l in FENCE.findall(a)][::2]
    ls = [l.strip() for l in a.split("\n") if len(l.strip()) > 12]
    return dict(
        fence=(langs[0] if langs else "none") or "bare",
        defines_fn=bool(fn and re.search(rf"\b{fn}\s*(=|\()", code)),
        python_like=bool(re.search(r"^\s*def \w+\(|^\s*return .*:$|\bself\b|^\s*elif\b", code, re.M)),
        repeated=len(ls) != len(set(ls)),
        braces=code.count("{") - code.count("}"),
        lines=len(code.splitlines()), words=len(a.split()))

def segments(rows):
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
            if k not in seen and r['passed']:
                seen.add(k); out.append((k, r))
    return out

def check(r):
    q, rec = gold_for(r['user'])
    if rec is None:
        return None
    code = extract_js_code(r['answer'] or '')
    st, msg = run(code, rec['tests'])
    return dict(st=st, msg=msg, q=q, a=r['answer'] or '', **feats(q, r['answer'] or '', code))

ex = ThreadPoolExecutor(8)
for name, pat in RUNS.items():
    rows = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l); r['_f'] = f; rows.append(r)
    groups = [('turn0', [r for r in rows if r['turn'] == 0])]
    seg = segments(rows)
    for lab, lo, hi in [('pass k1-2', 1, 2), ('pass k3-8', 3, 8), ('pass k9-15', 9, 15), ('pass k16+', 16, 99)]:
        groups.append((lab, [r for k, r in seg if lo <= k <= hi]))
    print(f'===== {name}', flush=True)
    for lab, rs in groups:
        cs = [c for c in ex.map(check, rs) if c]
        if not cs:
            continue
        n = len(cs); st = Counter(c['st'] for c in cs)
        fail = [c for c in cs if c['st'] != 'passed']
        mean = lambda k, L: sum(c[k] for c in L) / max(1, len(L))
        print(f"  {lab:11s} n {n:4d} passed {st['passed']/n:.2f} | fail kinds " +
              ', '.join(f"{k} {v/n:.2f}" for k, v in st.most_common() if k != 'passed'))
        for L, tag in [(cs, 'all'), (fail, 'failing')]:
            if not L: continue
            fc = Counter(c['fence'] for c in L)
            print(f"      {tag:8s} defines-required-fn {mean('defines_fn', L):.2f} python-like {mean('python_like', L):.2f}"
                  f" repeated-lines {mean('repeated', L):.2f} unbalanced-braces {sum(c['braces'] != 0 for c in L)/len(L):.2f}"
                  f" code-lines {mean('lines', L):.0f} words {mean('words', L):.0f} fences {dict(fc.most_common(3))}")
        msgs = Counter(f"{c['st']}: {re.sub(r'[0-9]+', 'N', c['msg'])[:70]}" for c in fail)
        for m, v in msgs.most_common(5):
            print(f"        {v:3d}  {m}")
        if lab in ('pass k3-8', 'pass k16+') and name == 'kl03_4b':
            for c in fail[:3]:
                print('      --- EX', c['st'], c['msg'][:80])
                print('      REQ:', c['q'][-300:].replace('\n', ' | '))
                print('      ANS:', c['a'][:700].replace('\n', ' | '))
