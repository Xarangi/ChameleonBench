"""Aggregate the negative-label audit (scripts/label_concept_contamination.py; REPLICATION
2026-10-16f). For every concept c of a data file: the fraction of texts a probe for c treats as
NEGATIVE that the judge says actually have c (p_yes >= --thr), split into the concept-free pool and
the other concepts' texts, and the chance that a refit's held-out negative set (--n-heldout texts,
the threshold set) contains at least one such text. Counts only; no text is printed.

  python scripts/negative_contamination_report.py --data D.json --labels L.json [--out R.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="one file per shard")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n-heldout", type=int, default=58)
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.loads(Path(a.data).read_text())
    lab = {}
    for f in a.labels:
        lab.update(json.loads(Path(f).read_text())["labels"])

    def has(tid, c):
        return lab.get(tid, {}).get(c, 0.0) >= a.thr

    free = [t["id"] for p in ("fit", "calib", "episode") for t in d["negatives"][p]]
    by_c = {c["concept"]: [t["id"] for p in ("fit", "calib", "episode") for t in c[p]]
            for c in d["concepts"]}
    rep = {}
    for c, own in by_c.items():
        if c == "deception":
            continue
        others = [t for o, ts in by_c.items() if o != c for t in ts]
        rf = sum(has(t, c) for t in free) / max(1, len(free))
        ro = sum(has(t, c) for t in others) / max(1, len(others))
        # fork negatives: other-concept texts and concept-free texts in the config's 128 : 64 ratio
        r = (2 * ro + rf) / 3
        rep[c] = {"free_pool": round(rf, 4), "other_concepts": round(ro, 4),
                  "own_texts_judged_yes": round(sum(has(t, c) for t in own) / max(1, len(own)), 3),
                  "p_any_in_heldout_negs": round(1 - (1 - r) ** a.n_heldout, 3)}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
