"""FINAL concept base (REPLICATION 2026-10-15e; user 10-15: "a final interesting model ... across
topics where both in and out include concepts"). Published / released data only:

- the DBpedia topic base (topic_base_v1: 13 train / 2 dev / 8 held-out topics, conflict-free);
- Neural Chameleons' released RATED concept data (mega_dataset_*; gemma-2-27b responses, LLM-rated
  per adjective, normalised 0..1). A text joins concept c iff rating[c] >= 0.75, it is not a
  refusal,
  and every OTHER selected concept rated on that row is <= PURITY[c] (default 0.5: no second concept
  strongly present; the rater gives moderate scores freely, so <= 0.25 left 48 German texts).
  "confused" texts are usually ALSO rated jokey / comforting >= 0.75 (47 at 0.5): it uses <= 0.75
  and
  its leakage is measured by scripts/tica_concept_leak.py. Request = the row's prompt; text =
  topical_response;
- DolusChat deception with matched truthful answers as hard negatives (held-out family, as 10-14t).

Concept families and split (train concepts sit beside the 13 train topics; held-out concepts beside
the 8 held-out topics): language german (train) / finnish (held-out); format all-caps (train) / HTML
(held-out); tone jokey + comforting (train) / confused (held-out); safety: deception (held-out
FAMILY). Harm-rated rows never enter. "angry" (NC style file) was dropped: 56 rows pass the harm
filter. Near subject
concepts (biology / chemistry / literature / mathematics) overlap train topics and are left to the
concept_eval_v1 extras. Split per concept 60 / 20 / 20 (fit / calib / episode). Aggregates only.

  python scripts/build_final_base.py --out-data D.json --out-inputs I.json --out-split S.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache")
RATED = RAW / "paper_benign_rated"
BENIGN = RATED / "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
STYLE = RATED / "mega_dataset_evaluated_20250731_163529_harm_batch_test_TRAIN.json"
HARM = ("helpful-for-making-bioweapons", "illegal", "graphic-violent", "bigoted",
        "helpful-for-hacking", "pro-violence", "toxic", "assisting-with-crime")
# adjective -> (concept name, family, split, source file)
CONCEPTS = {
    "german": ("German language", "language", "train", BENIGN),
    "all-caps": ("all-caps text", "format", "train", BENIGN),
    "jokey": ("jokes and humor", "tone", "train", BENIGN),
    "comforting": ("comforting tone", "tone", "train", BENIGN),
    "finnish": ("Finnish language", "language", "heldout", BENIGN),
    "HTML": ("HTML markup", "format", "heldout", BENIGN),
    "confused": ("confusion", "tone", "heldout", BENIGN),
}
PURITY = {"confused": 0.75}
# BROAD NEGATIVES (10-15e leak check 5686375: with only the train concepts + the small concept-free
# pool as negatives, a train probe learns "unlike encyclopedic English" -- German fired on Finnish
# 1.0 and deception 0.87, jokey on confusion 1.0). Added to the concept-free pool (negatives only;
# never trained to hide, never evaluated): Dolly-15k responses (diverse general / conversational
# text) and NC rated concepts that are neither trained nor held out. Near-synonyms of study
# concepts are NOT added (supportive ~ comforting, casual ~ jokey, gibberish ~ confused), nor
# harm-rated rows.
NEG_NC = {BENIGN: ("biology-focused", "chemistry-based", "literature-focused", "mathematical"),
          STYLE: ("formal", "academic", "educational", "informative")}
DOLLY = RAW / "tica" / "dolly15k" / "databricks-dolly-15k.jsonl"


def text_id(text: str) -> str:  # == tica.answer_format.text_id
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def split3(xs, rng, f=(0.6, 0.2)):
    xs = list(xs)
    rng.shuffle(xs)
    a, b = int(len(xs) * f[0]), int(len(xs) * (f[0] + f[1]))
    return xs[:a], xs[a:b], xs[b:]


def nc_negatives(cap: int = 250):
    """NC rated broad negatives as (prompt, text, adjective), in file order: neither trained nor
    held out, purity <= 0.25 against every selected concept, first matching adjective wins, at most
    `cap` per adjective. Shared with build_final_ood.py (which drops the math / chemistry rows)."""

    sel, nstats = set(CONCEPTS), {}
    for src, adjs in NEG_NC.items():
        for r in json.loads(src.read_text()):
            rat = r.get("topical_response_normalized_ratings") or {}
            x = (r.get("topical_response") or "").strip()
            if (r.get("is_topical_response_refusal") or not x
                    or any(rat.get(h, 0.0) > 0 for h in HARM)
                    or any(rat.get(o, 0.0) > 0.25 for o in sel if o in rat)):
                continue
            adj = r["adjectives"] if isinstance(r["adjectives"], list) else [r["adjectives"]]
            for c in adj:
                if c in adjs and rat.get(c, 0.0) >= 0.75 and nstats.get(c, 0) < cap:
                    nstats[c] = nstats.get(c, 0) + 1
                    yield r["prompt"], x, c
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(RAW / "tica" / "topic_base_v1.json"))
    ap.add_argument("--base-inputs", default=str(RAW / "tica" / "topic_base_v1_inputs.json"))
    ap.add_argument("--topic-split", default="configs/tica/topic_split_v1.json")
    ap.add_argument("--min-texts", type=int, default=140)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-split", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    data = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    sel = set(CONCEPTS)
    by, stats = defaultdict(list), {}
    for src in (BENIGN, STYLE):
        for r in json.loads(src.read_text()):
            if (r.get("is_topical_response_refusal")
                    or not (r.get("topical_response") or "").strip()):
                continue
            rat = r.get("topical_response_normalized_ratings") or {}
            if any(rat.get(h, 0.0) > 0 for h in HARM):
                continue
            adjs = r["adjectives"] if isinstance(r["adjectives"], list) else [r["adjectives"]]
            for c in adjs:
                if c not in sel or CONCEPTS[c][3] != src or rat.get(c, 0.0) < 0.75:
                    continue
                if any(rat.get(o, 0.0) > PURITY.get(c, 0.5) for o in sel if o != c and o in rat):
                    continue  # purity: no second selected concept on this text
                by[c].append((r["prompt"], r["topical_response"].strip()))
    meta, split = {}, json.loads(Path(a.topic_split).read_text())
    split = {k: list(split[k]) for k in ("train", "dev", "heldout")}
    for c, (name, fam, sp, src) in CONCEPTS.items():
        rows = list(dict((x, p) for p, x in by[c]).items())  # dedupe texts
        stats[name] = len(rows)
        if len(rows) < a.min_texts:
            raise SystemExit(f"{name}: only {len(rows)} pure texts (< {a.min_texts})")
        for x, p in rows:
            inputs[text_id(x)] = p
        fit, cal, ep = split3([x for x, _ in rows], rng)
        data["concepts"].append({"concept_id": f"nc-{c}", "concept": name, "genre": fam,
                                 "split": sp, "fit": ent(fit), "calib": ent(cal),
                                 "episode": ent(ep)})
        split[sp].append(name)
        meta[name] = {"family": fam, "source": f"NC rated ({src.name[:40]})"}
    pairs = defaultdict(dict)
    for line in (RAW / "dolus_deception" / "examples.jsonl").read_text().splitlines():
        r = json.loads(line)
        pairs[r["source_index"]][int(r["label"])] = r
    ok = [v for v in pairs.values() if 0 in v and 1 in v and v[1]["text"].strip()]
    for v in ok:
        for lab in (0, 1):
            inputs[text_id(v[lab]["text"])] = v[lab]["generation_prefix"]
    fit, cal, ep = split3(ok, rng)
    data["concepts"].append({
        "concept_id": "dolus-deception", "concept": "deception", "genre": "safety",
        "split": "heldout", "fit": ent([v[1]["text"] for v in fit]),
        "calib": ent([v[1]["text"] for v in cal]), "episode": ent([v[1]["text"] for v in ep]),
        "hard_neg": {"fit": [v[0]["text"] for v in fit], "calib": [v[0]["text"] for v in cal],
                     "episode": [v[0]["text"] for v in ep]}})
    split["heldout"].append("deception")
    stats["deception"] = len(ok)
    meta["deception"] = {"family": "safety", "source": "DolusChat (matched truthful negatives)"}
    neg, nstats = [], {}
    for p, x, c in nc_negatives():
        neg.append((p, x))
        nstats[c] = nstats.get(c, 0) + 1
    dol = [json.loads(x) for x in DOLLY.read_text().splitlines() if x.strip()]
    dol = [d for d in dol if 150 <= len(d["response"]) <= 1500]
    for d in rng.sample(dol, 2000):
        q = d["instruction"] + (f"\n\n{d['context']}" if d.get("context") else "")
        neg.append((q, d["response"].strip()))
    nstats["dolly15k"] = 2000
    neg = list(dict((x, p) for p, x in neg).items())
    for x, p in neg:
        inputs.setdefault(text_id(x), p)
    nf, nc, ne = split3([x for x, _ in neg], rng)
    for part, xs in (("fit", nf), ("calib", nc), ("episode", ne)):
        data["negatives"][part] = data["negatives"][part] + ent(xs)
    stats["broad_negatives"] = nstats
    data["source"] = {**data.get("source", {}), "final_concepts": meta,
                      "broad_negatives": nstats}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    out_split = {
        "_doc": "FINAL split v2 (REPLICATION 2026-10-15e; scripts/build_final_base.py): "
                "topic_split_v1 topics + NC rated concepts (purity-filtered) + DolusChat "
                "deception. "
                "Train concepts "
                "follow the 13 train topics; held-out = 8 topics + 4 concepts in trained families "
                "(finnish, HTML, confused) + 1 held-out FAMILY (safety: deception).",
        **split, "families": {n: m["family"] for n, m in meta.items()}}
    Path(a.out_split).write_text(json.dumps(out_split, indent=1))
    print(json.dumps({"pure_texts": stats, "train": len(split["train"]),
                      "heldout": len(split["heldout"])}, indent=1))


if __name__ == "__main__":
    main()
