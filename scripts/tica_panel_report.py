"""Never-shown panel report (10-25, user: "per held-out probe / per concept breakdown"; aggregates only).
Reads quality/step_*/panel_{train,heldout}.json of several runs and prints:
  1. per probe kind / layer: detection on hiding turns (train vs held-out concepts), plus det when the active probe
     is beaten vs caught, and the no-brief selectivity (concept still readable when not asked to hide);
  2. per concept: detection by probe group (trained depths, held-out depths, outer layers, L4);
  3. dodge generality: Jaccard of the dodged-probe set turn to turn, panel flip rates, old in-episode refits.
  python scripts/tica_panel_report.py LABEL=PATH_TO_STEP_DIR ...   (first = baseline, e.g. base=runs/v4b_quality_base)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = {"trained (8/12/16/22/26)": [8, 12, 16, 22, 26], "held-out (20/24)": [20, 24],
          "outer (28/30)": [28, 30], "L4": [4]}


def load(d: Path, split: str):
    p = d / f"panel_{split}.json"
    return json.loads(p.read_text()) if p.exists() else None


def layer(k: str) -> int:
    m = re.search(r"linear@(\d+)$", k)
    return int(m.group(1)) if m else -1


def main() -> None:
    runs = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[1:]]
    data = {(lab, sp): load(d, sp) for lab, d in runs for sp in ("train", "heldout")}
    keys = sorted({k for v in data.values() if v for k in v["aggregate"]["panel"]},
                  key=lambda k: (layer(k) if layer(k) >= 0 else 99, k))
    labs = [lab for lab, _ in runs]
    for sp in ("train", "heldout"):
        print(f"\n== 1. never-shown panel det on hiding turns, {sp} concepts: " + " | ".join(labs))
        for k in keys:
            row = []
            for lab in labs:
                v = data[(lab, sp)]
                e = v["aggregate"]["panel"].get(k) if v else None
                row.append(f"{e['det']:.2f}" if e else "  - ")
            print(f"  {k.replace('base:', ''):18s} " + " | ".join(row))
    last = labs[-1]
    v = data[(last, "train")]
    if v:
        print(f"\n== 1b. {last}, train concepts: det when the ACTIVE probe is beaten vs caught; no-brief det")
        sel = v.get("selectivity", {}).get("text_probe", {})
        for k in keys:
            e = v["aggregate"]["panel"].get(k)
            if not e:
                continue
            nb = sel.get(k, {}).get("nobrief")
            print(f"  {k.replace('base:', ''):18s} beaten {e['det_when_active_beaten']:.2f}  caught "
                  f"{e['det_when_active_caught']:.2f}  nobrief {nb if nb is None else round(nb, 2)}")
    for sp in ("train", "heldout"):
        print(f"\n== 2. per concept ({sp}): det by probe group on hiding turns, " + " | ".join(labs))
        per = {}
        for lab in labs:
            v = data[(lab, sp)]
            if not v:
                continue
            acc = defaultdict(lambda: [0, 0])
            for r in v["records"]:
                if r["off"] or r["cond"] != "real_score":
                    continue
                for k, (_, det) in r["panel"].items():
                    L = layer(k)
                    for g, Ls in GROUPS.items():
                        if L in Ls:
                            a = acc[(r["concept"], g)]
                            a[0] += det
                            a[1] += 1
            per[lab] = acc
        concepts = sorted({c for acc in per.values() for c, _ in acc})
        for c in concepts:
            cells = []
            for g in GROUPS:
                vals = []
                for lab in labs:
                    a = per.get(lab, {}).get((c, g))
                    vals.append(f"{a[0] / a[1]:.2f}" if a and a[1] else "-")
                cells.append(f"{g.split(' ')[0]} " + "/".join(vals))
            n = max((per[lab].get((c, "L4"), [0, 0])[1] for lab in per), default=0)
            print(f"  {c[:26]:26s} n{n:3d} | " + " | ".join(cells))
    print("\n== 3. dodge generality (train concepts)")
    for lab in labs:
        v = data[(lab, "train")]
        if not v:
            continue
        a = v["aggregate"]
        print(f"  {lab:10s} jaccard(dodged set, next turn) {a['jaccard_dodged_next']:.2f} (after catch "
              f"{a['jaccard_dodged_next_after_catch']:.2f}, after pass {a['jaccard_dodged_next_after_pass']:.2f}); "
              f"panel flip rate after catch {a['panel_flip_rate_after_catch']:.2f}, after pass "
              f"{a['panel_flip_rate_after_pass']:.2f}; old refits {json.dumps(a.get('old_refit'))[:160]}")


if __name__ == "__main__":
    main()
