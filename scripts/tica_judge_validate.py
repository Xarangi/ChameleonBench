"""Validate small local judges for TICA (DESIGN-TICA.md, "LLM judge").

There are three public validation sets, and the bars are fixed before the run:
1. NC styles (kind "is"). NC authors' gpt-4.1-mini ratings for the 11 benign concepts, from
   split_v4_c11.json (built from the authors' rated file). Positive = rating >= 0.75, negative =
   rating <= 0.25 (NC thresholds).
2. Kowalski topics (kind "discusses"). Kowalski battery concept sentences (pinned Wikipedia) about X
   are positives. Negatives are Kowalski's neutral sentences plus the other concepts' sentences.
3. Keyword stuffing (kind "discusses"). A neutral sentence with " {Concept}." appended mentions X
   without discussing it, so the correct answer is No. This probes the judge-hacking failure mode.
Bars per judge:
- median per-concept AUROC >= 0.90 on sets 1 and 2 separately, and no concept below 0.80;
- keyword-stuffing false-accept rate at 0.5 <= 0.10;
- median Yes+No mass >= 0.90.

Usage (GPU): python scripts/tica_judge_validate.py --judge /path/to/model [--judge ...] --output
out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from next_chameleons.incontext.stats import auroc
from next_chameleons.real.gen_eval import JUDGE_CRITERIA
from next_chameleons.tica.judge import load_judge

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache")
BARS = {"median_auroc": 0.90, "min_auroc": 0.80, "stuffing_false_accept": 0.10, "median_mass": 0.90}


def nc_items(per_class: int, rng: random.Random) -> list[dict]:
    rows = json.loads((RAW / "paper_benign_heldout/split_v4_c11.json").read_text())
    rows = [r for r in rows if isinstance(r.get("ratings"), dict)]
    items = []
    for concept, crit in JUDGE_CRITERIA.items():
        pos = [r for r in rows if r["ratings"].get(concept, 0.5) >= 0.75]
        neg = [r for r in rows if r["ratings"].get(concept, 0.5) <= 0.25]
        for label, pool in ((1, pos), (0, neg)):
            for r in rng.sample(pool, min(per_class, len(pool))):
                items.append(
                    {
                        "set": "nc_styles",
                        "concept": concept,
                        "label": label,
                        "text": r["response"],
                        "criterion": crit,
                        "kind": "is",
                    }
                )
    return items


def kowalski_items(rng: random.Random) -> list[dict]:
    bat = json.loads((RAW / "incontext/l0_battery.json").read_text())
    neutral = [s["text"] for s in bat["neutral"]["primary"]]
    concepts = bat["concepts"]
    by_c = {
        c: [s["text"] for split in v.values() for s in split]
        for c, v in bat["concept_sentences"].items()
    }
    items = []
    for c in concepts:
        crit = c.lower()
        for t in by_c[c]:
            items.append(
                {
                    "set": "kowalski_topics",
                    "concept": c,
                    "label": 1,
                    "text": t,
                    "criterion": crit,
                    "kind": "discusses",
                }
            )
        others = [t for o in concepts if o != c for t in by_c[o]]
        for t in rng.sample(neutral, 20) + rng.sample(others, 20):
            items.append(
                {
                    "set": "kowalski_topics",
                    "concept": c,
                    "label": 0,
                    "text": t,
                    "criterion": crit,
                    "kind": "discusses",
                }
            )
        for t in rng.sample(neutral, 20):
            items.append(
                {
                    "set": "stuffing",
                    "concept": c,
                    "label": 0,
                    "text": f"{t} {c}.",
                    "criterion": crit,
                    "kind": "discusses",
                }
            )
    return items


def summarize(items: list[dict]) -> dict:
    out = {}
    for name in ("nc_styles", "kowalski_topics"):
        sub = [x for x in items if x["set"] == name]
        per = {}
        for c in sorted({x["concept"] for x in sub}):
            s = [x for x in sub if x["concept"] == c]
            y = np.array([x["label"] for x in s])
            p = np.array([x["p_yes"] for x in s])
            per[c] = {
                "auroc": float(auroc(p[y == 1], p[y == 0])),
                "acc@0.5": float(np.mean((p >= 0.5) == (y == 1))),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
            }
        aucs = [v["auroc"] for v in per.values()]
        out[name] = {
            "per_concept": per,
            "median_auroc": float(np.median(aucs)),
            "min_auroc": float(np.min(aucs)),
        }
    st = [x for x in items if x["set"] == "stuffing"]
    out["stuffing_false_accept"] = float(np.mean([x["p_yes"] >= 0.5 for x in st]))
    out["median_mass"] = float(np.median([x["mass"] for x in items]))
    out["pass"] = bool(
        all(
            out[n]["median_auroc"] >= BARS["median_auroc"]
            and out[n]["min_auroc"] >= BARS["min_auroc"]
            for n in ("nc_styles", "kowalski_topics")
        )
        and out["stuffing_false_accept"] <= BARS["stuffing_false_accept"]
        and out["median_mass"] >= BARS["median_mass"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    base = nc_items(args.per_class, rng) + kowalski_items(rng)
    report = {"bars": BARS, "n_items": len(base), "judges": {}}
    for path in args.judge:
        judge = load_judge(path)
        items = [dict(x) for x in base]
        ps = judge.p_yes([(x["text"], x["criterion"], x["kind"]) for x in items])
        for x, p, m in zip(items, ps, judge.masses, strict=True):
            x["p_yes"], x["mass"] = p, m
        report["judges"][path] = summarize(items)
        print(
            path,
            json.dumps(
                {
                    k: v
                    for k, v in report["judges"][path].items()
                    if k not in ("nc_styles", "kowalski_topics")
                }
            ),
            flush=True,
        )
        for n in ("nc_styles", "kowalski_topics"):
            r = report["judges"][path][n]
            print(
                "  ",
                n,
                "median",
                round(r["median_auroc"], 3),
                "min",
                round(r["min_auroc"], 3),
                {c: round(v["auroc"], 3) for c, v in r["per_concept"].items()},
                flush=True,
            )
        del judge
        import torch

        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
