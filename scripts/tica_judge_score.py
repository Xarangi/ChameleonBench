"""Judge-score the TICA base generations (DESIGN-TICA.md, "LLM judge").

For every record and each concept it was asked to cover, P(Yes) for "Does this text discuss {X},
saying something substantive and correct about it?". These decide the probe POSITIVES.

Negative-label check: for --neg-checks random (record, concept not asked) pairs, the same question
is asked. A high rate would mean negatives are contaminated, e.g. by same-category concepts.

Usage (GPU): python scripts/tica_judge_score.py --basegen DIR --judge PATH --out FILE
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basegen", type=Path, required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--neg-checks", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from next_chameleons.tica.judge import load_judge

    recs = [
        json.loads(line) for f in sorted(args.basegen.glob("shard*.jsonl")) for line in f.open()
    ]
    pool = json.loads(
        Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica/round1_data.json").read_text()
    )
    phrase = {c["concept"]: c["phrase"] for c in pool["concepts"]}
    items, keys = [], []
    for r in recs:
        for c in r["concepts"]:
            items.append((r["response"], phrase[c], "discusses"))
            keys.append({"id": r["id"], "concept": c, "asked": True})
    rng = random.Random(0)
    names = sorted(phrase)
    for _ in range(args.neg_checks):
        r = rng.choice(recs)
        c = rng.choice([n for n in names if n not in r["concepts"]])
        items.append((r["response"], phrase[c], "discusses"))
        keys.append({"id": r["id"], "concept": c, "asked": False})
    judge = load_judge(args.judge, batch_size=args.batch_size)
    ps = judge.p_yes(items)
    for k, p, m in zip(keys, ps, judge.masses, strict=True):
        k["p_yes"], k["mass"] = p, m
    args.out.write_text("\n".join(json.dumps(k) for k in keys) + "\n")
    asked = [k for k in keys if k["asked"]]
    neg = [k for k in keys if not k["asked"]]
    print(
        json.dumps(
            {
                "n_records": len(recs),
                "asked_pass@0.5": sum(k["p_yes"] >= 0.5 for k in asked) / len(asked),
                "neg_check_false_yes@0.5": sum(k["p_yes"] >= 0.5 for k in neg) / max(1, len(neg)),
                "median_mass": sorted(judge.masses)[len(judge.masses) // 2],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
