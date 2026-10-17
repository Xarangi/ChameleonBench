"""POSITIVE-side label cleaning (REPLICATION 2026-10-16o; user: "high quality, clean and clear
concepts"). Drops a concept's own texts (fit / calib / episode) that the LLM judge
(scripts/label_concept_contamination.py) did NOT confirm as having that concept (p_yes < --thr), so
every positive is a clear instance. Concepts in --skip (judge cannot label them: deception sets) and
negatives are untouched; the negative side is cleaned at fit time (tica/neg_labels.py). Writes the
filtered data + a per-concept report (counts before / after). Aggregates only.

  python scripts/filter_positives_by_labels.py --data D.json --labels L.json --out D_clean.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP = ("deception", "deception (Apollo RepE)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-texts", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    d = json.loads(Path(a.data).read_text())
    lab = json.loads(Path(a.labels).read_text())["labels"]
    rep = {}
    for c in d["concepts"]:
        name = c["concept"]
        if name in SKIP:
            continue
        before = {p: len(c[p]) for p in ("fit", "calib", "episode")}
        for p in ("fit", "calib", "episode"):
            c[p] = [t for t in c[p] if lab.get(t["id"], {}).get(name, 0.0) >= a.thr]
        after = {p: len(c[p]) for p in ("fit", "calib", "episode")}
        rep[name] = {"before": before, "after": after,
                     "kept_frac": round(sum(after.values()) / max(1, sum(before.values())), 3),
                     "below_min": sum(after.values()) < a.min_texts}
    d.setdefault("source", {})["positive_label_filter"] = {"labels": a.labels, "thr": a.thr,
                                                           "report": rep}
    Path(a.out).write_text(json.dumps(d))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    for k, v in sorted(rep.items(), key=lambda x: x[1]["kept_frac"]):
        print(f"{k[:32]:32s} kept {v['kept_frac']:.2f} {v['after']}{'  BELOW MIN' if v['below_min'] else ''}")


if __name__ == "__main__":
    main()
