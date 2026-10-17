"""Held-out SAFETY eval concepts on top of the final v3 data (REPLICATION 2026-10-17y). Published
data only; aggregates printed, never texts. Eval-only: the training split / data file are untouched.

Adds, as split "heldout", family "safety" (deceptive = concept, honest = matched hard negative):
- "deception: roleplaying"     scale-safety-research/roleplaying (Apollo; 371 pairs)
- "deception: insider trading" scale-safety-research/insider_trading (Apollo; 336 pairs with both)
- "deception: DolusChat (response)"  DolusChat with the ANSWER = the response only (the v3
  "deception" concept's answer is prompt + response: build_final_v3 used the full text).
Answer = the assistant response; request (inputs map) = the scenario prompt. Insider-trading
scenarios are ~5k characters, so their request is cut to the LAST --max-request characters (the
final turn the response answers) to keep 24-turn races in memory.

  python scripts/build_safety_eval.py --out-data D.json --out-inputs I.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_base import RAW, split3, text_id

TICA = RAW / "tica"
ROLLOUTS = RAW / "apollo_repe_deception" / "examples.rollouts_backup.jsonl"
DOLUS = RAW / "dolus_deception" / "examples.jsonl"
SETS = {"deception: roleplaying": "scale-safety-research/roleplaying",
        "deception: insider trading": "scale-safety-research/insider_trading"}


def pairs_of(rows):
    by = defaultdict(dict)
    for r in rows:
        resp = r["text"][len(r["generation_prefix"]):].strip()
        if resp:
            by[r["source_index"]][int(r["label"])] = (r["generation_prefix"].strip(), resp)
    return [v for _, v in sorted(by.items()) if 0 in v and 1 in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "final_v3_clean.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "final_v3_inputs.json"))
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--max-request", type=int, default=2000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    roll = [json.loads(x) for x in ROLLOUTS.read_text().splitlines() if x.strip()]
    sets = {name: pairs_of([r for r in roll if r["source"] == src]) for name, src in SETS.items()}
    sets["deception: DolusChat (response)"] = pairs_of(
        [json.loads(x) for x in DOLUS.read_text().splitlines() if x.strip()])
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    have = {c["concept"] for c in data["concepts"]}
    for name, prs in sets.items():
        assert name not in have, name
        rng = random.Random(f"{a.seed}-{name}")
        prs = rng.sample(prs, min(a.n, len(prs)))
        for v in prs:
            for lab in (0, 1):
                req, resp = v[lab]
                inputs[text_id(resp)] = req[-a.max_request:] if a.max_request else req
        fit, cal, ep = split3(prs, rng)
        part = {"fit": fit, "calib": cal, "episode": ep}
        data["concepts"].append({
            "concept_id": "safety-" + name.split(": ")[1].split(" (")[0].replace(" ", "-").lower(),
            "concept": name, "genre": "safety", "split": "heldout",
            **{k: ent([v[1][1] for v in xs]) for k, xs in part.items()},
            "hard_neg": {k: [v[0][1] for v in xs] for k, xs in part.items()}})
        print(json.dumps({"concept": name, "pairs": len(prs), "fit": len(fit), "calib": len(cal),
                          "episode": len(ep)}))
    data["source"] = {**(data["source"] if isinstance(data["source"], dict) else {}),
                      "safety_eval": "build_safety_eval.py (Apollo roleplaying / insider_trading, "
                                     "DolusChat response-only); eval-only held-out"}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))


if __name__ == "__main__":
    main()
