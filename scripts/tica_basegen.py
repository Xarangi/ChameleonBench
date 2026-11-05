"""Base-model probe-training data for TICA round 1 (DESIGN-TICA.md, "Probing").

For every concept (all 500, so that held-out concepts also get probes for evaluation), the base
model answers TRAIN-split Dolly instructions with the task suffix "In your answer, also discuss {X}"
(k=1) or "... {X} and {Y}" (k=2). The plan is deterministic from --seed:
- k=1: --per-concept prompts per concept;
- k=2: random pairs, arranged so every concept appears in --per-concept pair prompts;
- the instruction for each prompt is drawn uniformly from the train split. For each response we
  store the sampled text (raw cache only; never committed) and the response-token mean at LAYERS.

Usage (GPU): python scripts/tica_basegen.py --shard I --n-shards N [--limit K] --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache/tica")
LAYERS = [
    6,
    9,
    12,
    16,
    20,
    24,
    31,
    36,
]  # decoder-layer outputs; CV picks the primary among 12/20 later


def plan(seed: int, per_concept: int) -> list[dict]:
    d = json.loads((RAW / "round1_data.json").read_text())
    concepts = [c["concept"] for c in d["concepts"]]
    phrase = {c["concept"]: c["phrase"] for c in d["concepts"]}
    instr = [x for x in d["instructions"] if x["split"] == "train"]
    rng = random.Random(seed)
    items = []
    for c in concepts:
        for _ in range(per_concept):
            items.append([c])
    for _ in range(per_concept):  # each pass pairs every concept once
        order = concepts[:]
        rng.shuffle(order)
        items += [[order[i], order[i + 1]] for i in range(0, len(order) - 1, 2)]
    rng.shuffle(items)
    out = []
    for i, cs in enumerate(items):
        ins = rng.choice(instr)
        out.append(
            {
                "id": f"bg{i:06d}",
                "concepts": cs,
                "phrases": [phrase[c] for c in cs],
                "instruction_id": ins["id"],
                "instruction": ins["text"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-9b-it")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--per-concept", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from next_chameleons.tica.backend import TicaBackend, task_prompt

    items = plan(args.seed, args.per_concept)[args.shard :: args.n_shards]
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard:02d}of{args.n_shards:02d}"
    done_path = args.out / f"{tag}.jsonl"
    if done_path.exists():
        print("exists, skipping", done_path)
        return
    be = TicaBackend(args.hf_id, LAYERS, batch_size=args.batch_size)
    t0 = time.time()
    recs, acts = [], []
    chunk = 512
    for i in range(0, len(items), chunk):
        part = items[i : i + chunk]
        convs = [
            [{"role": "user", "content": task_prompt(x["instruction"], x["phrases"])}] for x in part
        ]
        gens = be.sample(
            convs,
            args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed * 1_000_003 + args.shard * 10_007 + i,
        )
        for conv, g in zip(convs, gens, strict=True):
            conv.append({"role": "assistant", "content": g["text"] or " "})
        a, counts = be.pooled_read(convs)
        for x, conv, g, n in zip(part, convs, gens, counts, strict=True):
            recs.append(
                {
                    **x,
                    "prompt": conv[0]["content"],
                    "response": g["text"],
                    "n_new": g["n_new"],
                    "finished": g["finished"],
                    "n_read_tokens": int(n),
                }
            )
        acts.append(a)
        el = time.time() - t0
        print(f"{tag}: {len(recs)}/{len(items)} in {el / 60:.1f} min", flush=True)
    np.save(args.out / f"{tag}_acts.npy", np.concatenate(acts))
    tmp = done_path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    tmp.replace(done_path)
    (args.out / f"{tag}_meta.json").write_text(
        json.dumps(
            {"layers": LAYERS, "hf_id": args.hf_id, **{k: str(v) for k, v in vars(args).items()}}
        )
    )


if __name__ == "__main__":
    main()
