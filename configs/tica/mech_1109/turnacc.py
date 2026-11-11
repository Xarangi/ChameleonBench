"""11-11: grade EVERY answer of the gated task races (not only passing ones), to separate the cost of hiding from
conversation length. Per task and model: counts of correct / total by (passed, n_deployed, turn bucket), plus the
language check for MGSM / Belebele. Reuses the graders of scripts/tica_task_race_report.py (run from final-evals).
  PYTHONPATH=src python turnacc.py TASK OUT.json
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "/scratch/s/sneheel/ChameleonBench-final/scripts")
from tica_task_race_report import correct, in_language, make_correct  # noqa: E402

R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs"
M = {"base4b": "finalbase4b_{}/base4b", "2b": "final2b_{}/k2b1800", "k18": "gate_x2500_{}/kl03_1800",
     "9b": "final9b_{}/k9b1800", "det": "final4b_{}/det2200", "base2b": "finalbase2b_{}/base2b",
     "base9b": "finalbase9b_{}/base9b"}
task, out = sys.argv[1], sys.argv[2]
gold = json.loads(Path(f"{R}/mech_1025/taskrace/gold_{task}.json").read_text())
fn = make_correct(gold, 8)
lang = make_correct(gold, 8, lambda req, rec, ans: in_language(rec, ans)) if task.startswith(("mgsm", "belebele")) else None
res = {}
for name, pat in M.items():
    rows = []
    for f in sorted(glob.glob(f"{R}/races/{pat.format(task)}_p[12].jsonl")):
        rows += [json.loads(line) for line in open(f)]
    if not rows:
        continue
    fn.prefetch(rows)
    if lang:
        lang.prefetch(rows)
    agg = defaultdict(lambda: [0, 0, 0])  # key -> [n, correct, in_language]
    for r in rows:
        ok = fn(r)
        if ok is None:
            continue
        t = r["turn"]
        key = "|".join(map(str, (int(bool(r.get("off"))), int(bool(r.get("passed"))), int(bool(r.get("detected"))),
                                 min(r.get("n_deployed", 0), 7), 0 if t == 0 else 1 if t < 8 else 2 if t < 16 else 3 if t < 24 else 4)))
        a = agg[key]
        a[0] += 1; a[1] += int(ok); a[2] += int(bool(lang(r))) if lang else 0
    res[name] = {"rows": len(rows), "agg": dict(agg)}
    print(name, len(rows), flush=True)
Path(out).write_text(json.dumps(res))
