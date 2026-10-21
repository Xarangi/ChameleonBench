"""Topic base -> Level-1 data format for the multi-topic dodger (REPLICATION 2026-10-14p).

Inputs: the broad-topic text file (scripts/build_dbpedia_topics.py --merge), a split JSON
(train / dev / heldout / dropped) and the overlap report (scripts/tica_topic_overlap.py
--onpolicy). Checks that no train topic conflicts with a dev or held-out topic (abstract cross >
--max-abstract OR on-policy cross > --max-onpolicy) and refuses to write otherwise. Writes:
- a Level1Data JSON: one entry per kept topic with split train / dev / heldout and text splits fit /
  calib / episode (disjoint), plus the Level-1 concept-free negatives (unchanged);
- an inputs JSON for the answer format: the Level-1 inputs + ONE generic request for every topic
  text (no topic information in the prompt).
Aggregates only; no text is printed.

  python scripts/build_topic_base.py --topics T.json --split configs/tica/topic_split_v1.json \
      --overlap overlap_v2.json --out-data D.json --out-inputs I.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
REQUEST = "Write a short encyclopedia entry."  # same request as the overlap check


def text_id(text: str) -> str:  # == next_chameleons.tica.answer_format.text_id
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def conflicts(ov: dict, a: str, b: str, max_abs: float, max_onp: float) -> tuple | None:
    x = max(ov["cross"][a].get(b, 0.0), ov["cross"][b].get(a, 0.0))
    o = ov.get("onpolicy", {}).get("cross", {})
    y = max(o.get(a, {}).get(b, 0.0), o.get(b, {}).get(a, 0.0)) if o else 0.0
    return (round(x, 2), round(y, 2)) if (x > max_abs or y > max_onp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--base-data", default=str(RAW / "level1_data.json"))
    ap.add_argument("--base-inputs", default=str(RAW / "level1_inputs_plus.json"))
    ap.add_argument("--fit", type=int, default=400)
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--max-abstract", type=float, default=0.5)
    ap.add_argument("--max-onpolicy", type=float, default=0.25)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    a = ap.parse_args()

    topics = json.loads(Path(a.topics).read_text())
    sp = json.loads(Path(a.split).read_text())
    ov = json.loads(Path(a.overlap).read_text())
    role = {c: r for r in ("train", "dev", "heldout") for c in sp[r]}
    if len(role) != sum(len(sp[r]) for r in ("train", "dev", "heldout")):
        raise SystemExit("a topic is in two splits")
    missing = set(role) - set(topics)
    if missing:
        raise SystemExit(f"split topics not in the topic file: {sorted(missing)}")
    bad = []
    for x, y in itertools.combinations(role, 2):
        if role[x] == role[y] or "train" not in (role[x], role[y]):
            continue
        c = conflicts(ov, x, y, a.max_abstract, a.max_onpolicy)
        if c:
            bad.append((x, y, c))
    if bad:
        raise SystemExit(f"train topics conflict with dev / held-out topics: {bad}")
    onp = ov.get("onpolicy", {}).get("tpr", {})
    rng = random.Random(0)
    base = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    split_name = {"train": "train", "dev": "dev", "heldout": "heldout"}
    concepts, n_new = [], 0
    need = a.fit + a.calib + a.episode
    for i, c in enumerate(role):
        texts = list(dict.fromkeys(topics[c]["texts"]))
        rng.shuffle(texts)
        if len(texts) < need:
            raise SystemExit(f"{c}: {len(texts)} texts < {need}")
        parts, k = {}, 0
        for name, n in (("fit", a.fit), ("calib", a.calib), ("episode", a.episode)):
            parts[name] = [{"id": text_id(t), "text": t} for t in texts[k : k + n]]
            k += n
        for t in texts[:need]:
            if text_id(t) not in inputs:
                n_new += 1
            inputs[text_id(t)] = REQUEST
        concepts.append({"concept_id": f"topic-{i}", "concept": c, "genre": topics[c]["l1"],
                         "split": split_name[role[c]], **parts})
    out = {"source": {"topics": a.topics, "split": a.split, "overlap": a.overlap,
                      "request": REQUEST},
           "seed": 0, "concepts": concepts, "negatives": base["negatives"]}
    Path(a.out_data).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_data).write_text(json.dumps(out))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    summ = {r: len(sp[r]) for r in ("train", "dev", "heldout")}
    worst = max((conflicts(ov, x, y, -1, -1), x, y) for x, y in itertools.combinations(role, 2)
                if role[x] != role[y] and "train" in (role[x], role[y]))
    print(json.dumps({"splits": summ, "new_inputs": n_new, "per_topic": {
        "fit": a.fit, "calib": a.calib, "episode": a.episode},
        "worst_cross_split_pair": worst,
        "onpolicy_tpr_min_train": min(onp.get(c, 1.0) for c in sp["train"])}, indent=1))


if __name__ == "__main__":
    main()
