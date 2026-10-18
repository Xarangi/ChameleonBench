"""Out-of-distribution CONCEPT test set for the multi-topic dodger (REPLICATION 2026-10-14t; user 10-14:
"why are all test topics topics, and not concepts? ... deception like the NC paper"). Published /
released data only, all EVAL-ONLY (split "heldout"; never a training concept or negative):
- Neural Chameleons' released rated concepts (gemma-2-27b generated, rating >= 4): language (German,
  Finnish), format (HTML, all-caps), tone (jokey, comforting, confused), subject (biology, chemistry,
  literature, mathematics -- NEAR to train topics living organisms / plays and poetry; reported
  separately). Request = the concept's own prompt. Negatives = the other NC concepts (same pipeline).
- DolusChat (CC BY 4.0) deception: deceptive answers (label 1); HARD negatives = the matched truthful
  answer to the SAME prompt (stored per split as "hard_neg"; the probe cannot key on the source).
RepE facts dishonesty is NOT used: its honest / dishonest persona lives in the prompt, not the answer.
Output: the topic base + these concepts (Level1Data format) and the merged inputs map. Aggregates only.

  python scripts/build_concept_eval.py --out-data D.json --out-inputs I.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache")
NAMES = {"german": ("German language", "language", "far"),
         "finnish": ("Finnish language", "language", "far"),
         "html": ("HTML markup", "format", "far"), "all-caps": ("all-caps text", "format", "far"),
         "jokey": ("jokes and humor", "tone", "far"), "comforting": ("comforting tone", "tone", "far"),  # noqa: E501
         "confused": ("confusion", "tone", "far"),
         "biology-focused": ("biology", "subject", "near"),
         "chemistry-based": ("chemistry", "subject", "far"),
         "literature-focused": ("literature", "subject", "near"),
         "mathematical": ("mathematics", "subject", "near")}


def text_id(text: str) -> str:  # == tica.answer_format.text_id
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def split3(xs, rng, f=(0.6, 0.15)):
    xs = list(xs)
    rng.shuffle(xs)
    a, b = int(len(xs) * f[0]), int(len(xs) * (f[0] + f[1]))
    return xs[:a], xs[a:b], xs[b:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(RAW / "tica" / "topic_base_v1.json"))
    ap.add_argument("--base-inputs", default=str(RAW / "tica" / "topic_base_v1_inputs.json"))
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    data = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    stats, meta = {}, {}
    by = defaultdict(list)
    for line in open(RAW / "paper_benign_concepts" / "examples.jsonl"):  # noqa: SIM115
        r = json.loads(line)
        if r["rating"] >= 4 and r["response"].strip():
            by[r["concept"]].append((r["prompt"], r["response"]))
    for c, rows in sorted(by.items()):
        name, fam, rel = NAMES[c]
        for p, x in rows:
            inputs[text_id(x)] = p
        fit, cal, ep = split3([x for _, x in rows], rng)
        data["concepts"].append({"concept_id": f"nc-{c}", "concept": name, "genre": fam,
                                 "split": "heldout", "fit": ent(fit), "calib": ent(cal),
                                 "episode": ent(ep)})
        stats[name] = len(rows)
        meta[name] = {"family": fam, "relation": rel, "source": "NC rated (gemma-2-27b)"}
    pairs = defaultdict(dict)
    for line in open(RAW / "dolus_deception" / "examples.jsonl"):  # noqa: SIM115
        r = json.loads(line)
        pairs[r["source_index"]][int(r["label"])] = r
    ok = [v for v in pairs.values() if 0 in v and 1 in v and v[1]["text"].strip()]
    for v in ok:
        for lab in (0, 1):
            inputs[text_id(v[lab]["text"])] = v[lab]["generation_prefix"]
    fit, cal, ep = split3(ok, rng)  # pairs stay on one side
    data["concepts"].append({
        "concept_id": "dolus-deception", "concept": "deception", "genre": "safety",
        "split": "heldout", "fit": ent([v[1]["text"] for v in fit]),
        "calib": ent([v[1]["text"] for v in cal]), "episode": ent([v[1]["text"] for v in ep]),
        "hard_neg": {"fit": [v[0]["text"] for v in fit], "calib": [v[0]["text"] for v in cal],
                     "episode": [v[0]["text"] for v in ep]}})
    stats["deception"] = len(ok)
    meta["deception"] = {"family": "safety", "relation": "far",
                         "source": "DolusChat (matched truthful negatives)"}
    data["source"] = {**data.get("source", {}), "concept_eval": meta}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    print(json.dumps({"concepts_added": stats, "total_concepts": len(data["concepts"])}, indent=1))


if __name__ == "__main__":
    main()
