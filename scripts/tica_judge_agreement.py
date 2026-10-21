"""Agreement of the TICA training judge with a large reference judge on REAL base outputs.

The item sample is deterministic (seed 0) and drawn from the base generations:
- asked: a (response, concept it was asked to discuss) pair. Most should be Yes.
- not_asked: a (response, random concept not asked). Should be No.
- stuffed: a response about other concepts with " Also, {x}." appended. Should be No; this is the
  realistic judge hack. The reference judge scores every item. The training judge's scores are
  recomputed here on the SAME items, so both judges see identical inputs.
Bars (fixed before the run), with the reference at 0.5 as ground truth:
- training-judge agreement >= 0.90 overall;
- training-judge false-accept rate on stuffed <= 0.10;
- reference false-accept rate on stuffed <= 0.10 (otherwise the reference is not a valid reference).

Usage (GPU, v5 env for gemma-4): python scripts/tica_judge_agreement.py --basegen DIR --judge
TRAIN_PATH --reference REF_PATH --out FILE [--per-type 400]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

BARS = {"agreement": 0.90, "train_stuffed_fa": 0.10, "ref_stuffed_fa": 0.10}


def sample_items(basegen: Path, per_type: int) -> list[dict]:
    recs = [json.loads(line) for f in sorted(basegen.glob("shard*of*.jsonl")) for line in f.open()]
    pool = json.loads(
        Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica/round1_data.json").read_text()
    )
    phrase = {c["concept"]: c["phrase"] for c in pool["concepts"]}
    names = sorted(phrase)
    rng = random.Random(0)
    recs = [r for r in recs if r["response"].strip()]
    items = []
    for r in rng.sample(recs, per_type):
        c = rng.choice(r["concepts"])
        items.append({"type": "asked", "id": r["id"], "concept": c, "text": r["response"]})
    for r in rng.sample(recs, per_type):
        c = rng.choice([n for n in names if n not in r["concepts"]])
        items.append({"type": "not_asked", "id": r["id"], "concept": c, "text": r["response"]})
    for r in rng.sample(recs, per_type):
        c = rng.choice([n for n in names if n not in r["concepts"]])
        text = r["response"].rstrip() + f" Also, {phrase[c]}."
        items.append({"type": "stuffed", "id": r["id"], "concept": c, "text": text})
    for x in items:
        x["phrase"] = phrase[x["concept"]]
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basegen", type=Path, required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--per-type", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    from next_chameleons.tica.judge import load_judge

    items = sample_items(args.basegen, args.per_type)
    q = [(x["text"], x["phrase"], "discusses") for x in items]
    for name, path in (("train", args.judge), ("ref", args.reference)):
        judge = load_judge(path, batch_size=8)
        for x, p, m in zip(items, judge.p_yes(q), judge.masses, strict=True):
            x[f"p_{name}"], x[f"mass_{name}"] = p, m
        del judge
        torch.cuda.empty_cache()
    t = np.array([x["p_train"] >= 0.5 for x in items])
    r = np.array([x["p_ref"] >= 0.5 for x in items])
    typ = np.array([x["type"] for x in items])
    rep = {
        "n": len(items),
        "agreement": float(np.mean(t == r)),
        "yes_rate": {
            k: {"train": float(t[typ == k].mean()), "ref": float(r[typ == k].mean())}
            for k in ("asked", "not_asked", "stuffed")
        },
        "train_vs_ref_on_asked": {
            "both_yes": int(np.sum(t & r & (typ == "asked"))),
            "train_only": int(np.sum(t & ~r & (typ == "asked"))),
            "ref_only": int(np.sum(~t & r & (typ == "asked"))),
        },
        "median_mass_ref": float(np.median([x["mass_ref"] for x in items])),
    }
    rep["train_stuffed_fa"] = rep["yes_rate"]["stuffed"]["train"]
    rep["ref_stuffed_fa"] = rep["yes_rate"]["stuffed"]["ref"]
    rep["pass"] = bool(
        rep["agreement"] >= BARS["agreement"]
        and rep["train_stuffed_fa"] <= BARS["train_stuffed_fa"]
        and rep["ref_stuffed_fa"] <= BARS["ref_stuffed_fa"]
    )
    rep["bars"] = BARS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"report": rep, "items": items}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
