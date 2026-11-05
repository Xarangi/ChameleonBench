"""Is the logic under the broken syntax right? Mechanically repair passing race answers and re-run the tests.
HumanEval-Py / MBPP python-with-braces -> colons + drop '}' lines + restore 'def'; JS -> restore 'function';
MBPP JS-style -> run as JS (True/False/None -> true/false/null, asserts -> assert.deepEqual). Rough, lower bound."""
import json, glob, re, ast, os, subprocess, tempfile
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from next_chameleons.benchmarks.capability_heldout import (extract_code, extract_js_code, run_humaneval_py,
                                                          run_tests, run_js_tests, node_bin)
R = '/scratch/sneheel/next_chameleons_artifacts/tica/runs/'
G = {t: json.load(open(R + f'mech_1025/taskrace/gold_{t}.json')) for t in ('pyeval', 'mbpp', 'jseval')}
RACES = [('HumanEval-Py KL0.3 4B', 'pyeval', 'races/heldout_v4d_k034b_pyeval/kl03_4b_p*.jsonl'),
         ('HumanEval-Py strict 4B', 'pyeval', 'races/heldout_v4d_pyeval/strict4b_p*.jsonl'),
         ('MBPP KL0.3 4B', 'mbpp', 'races/v4d_k034b_mbpp/kl03_4b_p*.jsonl'),
         ('HumanEval-JS KL0.3 4B', 'jseval', 'races/v4d_k034b_jseval/kl03_4b_p*.jsonl')]

def gold_for(user, g):
    t = (user or '').strip()
    for q, rec in g.items():
        if t.endswith(q.strip()):
            return q, rec
    return None, None

def py_parses(c):
    try: ast.parse(c); return True
    except Exception: return False

def form(code):
    if py_parses(code) and re.search(r'^\s*def \w+', code, re.M): return 'python'
    if re.search(r'\bfunction\b|=>|\b(const|let|var)\s+\w+\s*=', code): return 'js-style'
    if '{' in code or '}' in code: return 'py+braces'
    return 'other'

def py_repair(code, fn):
    pre = []  # Allman style: a line that is only '{' opens the previous line's block
    for l in code.split('\n'):
        if l.strip() == '{' and pre:
            j = len(pre) - 1
            while j > 0 and not pre[j].strip(): j -= 1
            pre[j] = pre[j].rstrip() + ' {'
            continue
        pre.append(l)
    out = []
    for l in pre:
        s = l.rstrip()
        if '}' in s and re.fullmatch(r'\s*\}[;\s]*', s):
            continue
        s = re.sub(r'^(\s*)\}\s*(else|elif\b.*?|except\b.*?)\s*\{\s*$', r'\1\2:', s)
        m = re.match(r'^(\s*)(def\s+)?(\w+\s*\(.*\)\s*(->\s*[^{:]+)?)\s*[{:]?\s*$', s)
        if m and (m.group(2) or (fn and m.group(3).startswith(fn))):
            s = f"{m.group(1)}def {m.group(3).strip()}:"
        else:
            s = re.sub(r'^(\s*(if|for|while|elif|else|try|with|except)\b.*?)\s*\{\s*$', r'\1:', s)
            s = re.sub(r'^(\s*(if|for|while|elif|else|try|with|except)\b[^:]*?)\s*$',
                       lambda mm: mm.group(1) + (':' if not mm.group(1).rstrip().endswith(':') and
                                                 re.match(r'\s*(else|try)\s*$', mm.group(1)) else ''), s)
        s = re.sub(r';\s*$', '', s)
        out.append(s)
    return '\n'.join(out)

def js_repair(code, fn):
    return re.sub(rf'^(\s*)({re.escape(fn)}\s*\()', r'\1function \2', code, flags=re.M) if fn else code

def pylit2js(s):
    s = re.sub(r'\bTrue\b', 'true', s); s = re.sub(r'\bFalse\b', 'false', s); s = re.sub(r'\bNone\b', 'null', s)
    return s

def mbpp_as_js(code, tests):
    body = pylit2js(code)
    ts = []
    for t in tests:
        m = re.match(r'\s*assert\s+(.+?)\s*==\s*(.+)$', t)
        if not m: return None
        ts.append(f"assert.deepEqual({pylit2js(m.group(1))}, {pylit2js(m.group(2))});")
    js = "const assert = require('assert');\n" + body + "\n" + "\n".join(ts)
    with tempfile.TemporaryDirectory(prefix="mj_") as td:
        p = os.path.join(td, "p.js"); open(p, "w").write(js)
        try:
            pr = subprocess.run([node_bin(), p], cwd=td, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                                timeout=10, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return pr.returncode == 0

def check(task, rec, ans):
    if task == 'pyeval':
        c = extract_code(ans); fn = rec.get('fn')
        ok = bool(run_humaneval_py(c, rec)['passed']); f = form(c)
        rep = ok or bool(run_humaneval_py(py_repair(c, fn), rec)['passed'])
        return f, ok, rep
    if task == 'mbpp':
        c = extract_code(ans); tests = rec['tests'] if isinstance(rec['tests'], list) else ast.literal_eval(rec['tests']); imps = rec['imports'] if isinstance(rec['imports'], list) else ast.literal_eval(rec['imports'])
        fn = re.match(r'\s*assert\s+(\w+)', tests[0]); fn = fn.group(1) if fn else None
        ok = bool(run_tests(c, imps, tests)['passed']); f = form(c)
        rep = ok or bool(run_tests(py_repair(c, fn), imps, tests)['passed'])
        if not rep and f != 'python':
            rep = bool(mbpp_as_js(c, tests))
        return f, ok, rep
    c = extract_js_code(ans)
    fn = re.search(r'let candidate = (\w+)', rec['tests'])
    fn = fn.group(1) if fn else None
    ok = bool(run_js_tests(c, rec['tests'])['passed'])
    f = 'missing function kw' if fn and re.search(rf'^\s*{fn}\s*\(.*\)\s*\{{', c, re.M) and not re.search(rf'function\s+{fn}\b', c) else 'js'
    rep = ok or bool(run_js_tests(js_repair(c, fn), rec['tests'])['passed'])
    return f, ok, rep

def passes(rows):
    by = defaultdict(list)
    for r in rows:
        if r.get('off') or r.get('cond') not in ('real_score', 'real_verdict') or r.get('n_deployed') is None: continue
        by[(r['_f'], r['eid'])].append(r)
    out = []
    for rs in by.values():
        seen = set()
        for r in sorted(rs, key=lambda r: r['turn']):
            k = max(1, int(r['n_deployed']))
            if r['passed'] and k not in seen: seen.add(k); out.append((k, r))
    return out

ex = ThreadPoolExecutor(8)
for name, task, pat in RACES:
    rows = [dict(json.loads(l), _f=f) for f in sorted(glob.glob(R + pat)) if 'forks' not in f for l in open(f)]
    P = passes(rows)
    print(f'===== {name}  (passes {len(P)})', flush=True)
    for lab, sel in [('turn0', [r for r in rows if r['turn'] == 0]), ('pass k1-2', [r for k, r in P if k <= 2]),
                     ('pass k3+', [r for k, r in P if k >= 3])]:
        def one(r):
            q, rec = gold_for(r['user'], G[task])
            return None if rec is None else check(task, rec, r['answer'] or '')
        res = [x for x in ex.map(one, sel) if x]
        if not res: continue
        n = len(res); ok = sum(x[1] for x in res); rep = sum(x[2] for x in res)
        print(f'  {lab:9s} n {n:3d}  tests pass as written {ok/n:.2f}  after syntax repair {rep/n:.2f}')
        for f, c in Counter(x[0] for x in res).most_common():
            L = [x for x in res if x[0] == f]
            print(f'      {f:20s} share {c/n:.2f}  as written {sum(x[1] for x in L)/c:.2f}  repaired {sum(x[2] for x in L)/c:.2f}')
