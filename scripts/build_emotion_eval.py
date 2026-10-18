"""Held-out EMOTION eval concepts on top of a TICA data file (REPLICATION 2026-10-18e; user 10-18:
"keep emotion fully held out"). Published data only; aggregates printed, never texts. Eval-only.

Source: abotresol/emotion-story-corpora (MIT), config single_emotion: third-person ~150-word stories
in which a character feels one emotion without naming it; the DeepSeek-v4-pro parts only (the Gemma
part is repetitive), rows with the emotion word leaked dropped. Every story comes from one prompt
template, so a probe could learn "story vs answer"; each emotion's HARD negatives are therefore the
other emotions' stories (+ the 128 neutral stories). The request (inputs map) is one fixed,
emotion-free prompt for every story, so the user turn carries no label.

  python scripts/build_emotion_eval.py --out-data D.json --out-inputs I.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_base import RAW, split3, text_id

TICA = RAW / "tica"
SRC = RAW / "emotion_stories" / "single_emotion.parquet"
EMOTIONS = ("happy", "sad", "angry", "afraid", "guilty", "surprised", "calm")
REQUEST = ("Write a short third-person story, around 150 words, about a person going about "
           "their day.")


def emotion_concepts(seed: int = 0, n: int = 400, n_hard: int = 300):
    """-> (concept dicts, {text_id: request}, per-emotion stats); shared with build_final_v5.py."""

    df = pd.read_parquet(SRC)
    df = df[df.text.str.len().between(150, 1500) & ~df.leaked.astype(bool)]
    neutral = df[df.source.str.startswith("neutral")].text.str.strip().tolist()
    ds = df[df.source.str.startswith("deepseek")]
    pools = {e: list(dict.fromkeys(ds[ds.emotion == e].text.str.strip())) for e in EMOTIONS}
    rng = random.Random(seed)
    pos = {e: rng.sample(v, min(n, len(v))) for e, v in pools.items()}
    used = {x for v in pos.values() for x in v}
    spare = {e: [x for x in v if x not in used] for e, v in pools.items()}  # hard-neg pool
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    concepts, inputs, stats = [], {}, []
    for e in EMOTIONS:
        name = f"emotion: {e}"
        r = random.Random(f"{seed}-{e}")
        others = [x for o in EMOTIONS if o != e for x in spare[o]]
        hard = r.sample(others, min(n_hard, len(others))) + neutral
        r.shuffle(hard)
        for x in pos[e] + hard:
            inputs[text_id(x)] = REQUEST
        fit, cal, ep = split3(pos[e], r)
        hf, hc, he = split3(hard, r)
        concepts.append({
            "concept_id": f"emotion-{e}", "concept": name, "genre": "emotion", "split": "heldout",
            "fit": ent(fit), "calib": ent(cal), "episode": ent(ep),
            "hard_neg": {"fit": hf, "calib": hc, "episode": he}})
        med = sorted(map(len, pos[e]))[len(pos[e]) // 2]
        stats.append({"concept": name, "pos": len(pos[e]), "fit": len(fit),
                      "hard_neg": len(hard), "median_chars": med})
    return concepts, inputs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "final_v3_clean.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "final_v3_inputs.json"))
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-hard", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    have = {c["concept"] for c in data["concepts"]}
    concepts, new_inputs, stats = emotion_concepts(a.seed, a.n, a.n_hard)
    for c in concepts:
        assert c["concept"] not in have, c["concept"]
    data["concepts"] += concepts
    inputs.update(new_inputs)
    for st in stats:
        print(json.dumps(st))
    data["source"] = {**(data["source"] if isinstance(data["source"], dict) else {}),
                      "emotion_eval": "build_emotion_eval.py (abotresol/emotion-story-corpora, "
                                      "MIT; DeepSeek parts; other emotions as hard negatives); "
                                      "eval-only"}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))

if __name__ == "__main__":
    main()
