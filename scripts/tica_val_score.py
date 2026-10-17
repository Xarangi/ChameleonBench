"""Validation score for early stopping (REPLICATION 2026-10-16j). From a race report
(scripts/tica_switch_eval.py, with deploy_records): per concept FAMILY (split json "families";
unlisted = "topic"), the fraction of newly deployed probes beaten within --horizon judged turns;
val_score = the mean over families (equal weight, so a hard family cannot hide behind easy
topics). Also the per-turn pass rate. Writes {"step", "val_score", ...}.

  python scripts/tica_val_score.py --race R.json --split S.json --step 40 --out O.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.race).read_text())
    fam = json.loads(Path(a.split).read_text()).get("families", {})
    by = defaultdict(list)
    for d in r.get("deploy_records", []):
        by[fam.get(d.get("concept"), "topic")].append(d["beaten"] and d["turns"] <= a.horizon)
    per = {f: sum(v) / len(v) for f, v in by.items() if v}
    out = {"step": a.step, "val_score": sum(per.values()) / len(per) if per else 0.0,
           "per_family": per, "pass_rate": r.get("pass_rate_after_first_turn"),
           "n_deploy": sum(len(v) for v in by.values()), "race": a.race}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".tmp").write_text(json.dumps(out, indent=1))
    Path(a.out + ".tmp").rename(a.out)  # atomic: the trainer never reads a half-written file
    print(json.dumps(out))


if __name__ == "__main__":
    main()
