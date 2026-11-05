"""TICA run health check (aggregates only).
Usage: python3 scripts/tica_check.py JOBID:RUN_DIR [...]
Flags: job FAILED/OOM/Traceback, stalled log, NaN, memory near cap, no refit ADDED recently
(race not running), pass collapse, later turns never caught (static dodge). Prints the per-turn
caught curve (#=caught +=mixed .=dodged) and per-phase step timings when logged."""

import json
import math
import os
import subprocess
import sys
import time

LOGS = "/home/sneheel/chameleons/logs"


def main() -> None:
    for arg in sys.argv[1:]:
        job, run = arg.split(":", 1)
        st = subprocess.run(["sacct", "-j", job, "-n", "-X", "--format=State,Elapsed"],
                            capture_output=True, text=True).stdout.split()
        state = st[0] if st else "?"
        print(f"=== job {job} [{state} {st[1] if len(st) > 1 else ''}] {run} ===")
        alerts = []
        out = f"{LOGS}/tica-l3covert-{job}.out"
        if os.path.exists(out):
            with open(out, errors="ignore") as fh:
                tail = fh.read()[-6000:]
            alerts += [f".out contains {p!r}" for p in ("OutOfMemoryError", "Traceback")
                       if p in tail]
        if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_ME", "NODE_FAIL")):
            alerts.append(f"job state {state}")
        f = os.path.join(run, "adaptive", "train_log.jsonl")
        if not os.path.exists(f):
            print("  no train_log yet")
            for a in alerts:
                print("  ALERT:", a)
            continue
        with open(f) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        d = rows[-1]
        age = (time.time() - os.path.getmtime(f)) / 60
        sec = [r.get("sec", 0) for r in rows[-5:]]
        phases = " ".join(f"{k[4:]}={d[k]:.0f}" for k in ("sec_rollout", "sec_train", "sec_refit",
                                                         "sec_diag") if k in d)
        print(f"  step {d['step']}  log age {age:.0f} min  sec/step {sum(sec) / len(sec):.0f} "
              f"[{phases}]  mem {d.get('max_mem_gb')}GB")
        if state == "RUNNING" and age > max(20, 4 * max(sec) / 60):
            alerts.append(f"log stalled ({age:.0f} min since last write)")
        if any(isinstance(v, float) and math.isnan(v) for v in d.values()):
            alerts.append("NaN in last record")
        if (d.get("max_mem_gb") or 0) > 37:
            alerts.append(f"memory {d['max_mem_gb']}GB near 40GB cap")
        T = 1 + max([int(k.rsplit("_", 1)[1]) for k in d
                     if k.startswith("rollout/detturn_real_score_")] or [0])
        for r in rows[-8:]:
            c = [r.get(f"rollout/detturn_real_score_{t}") for t in range(T)]
            cs = "".join("-" if x is None else "#" if x >= .67 else "+" if x >= .34 else "."
                         for x in c)
            print(f"  s{r['step']:>4} pass={r.get('rollout/pass_rate', 0):.2f} "
                  f"bank={r.get('rollout/bank_size', '?')} curve[{cs}] refit={r.get('refit', '')}")
        ref = [r.get("refit") for r in rows if r.get("refit")]
        print(f"  refit outcomes: { {k: ref.count(k) for k in set(ref)} }")
        recent = [r.get("refit") for r in rows[-20:] if r.get("refit")]
        if len(recent) >= 4 and "added" not in recent:
            alerts.append(f"no refit ADDED in last 20 steps {set(recent)} -> race not running")
        if len(rows) >= 20 and max(r.get("rollout/pass_rate", 0) for r in rows[-10:]) == 0:
            alerts.append("pass_rate 0 for 10 steps (caught every turn)")
        late = [r.get(f"rollout/detturn_real_score_{t}") for r in rows[-10:] for t in range(2, T)]
        late = [x for x in late if x is not None]
        if len(rows) >= 30 and late and max(late) == 0:
            alerts.append("later turns never caught for 10 steps -> static dodge / refit lagging")
        for a in alerts:
            print("  ALERT:", a)
        if not alerts:
            print("  OK")


if __name__ == "__main__":
    main()
