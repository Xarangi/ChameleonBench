"""TICA Level 1: base-model reads for probe fitting and calibration (DESIGN-TICA.md "FIRST SETUP").

Every read is teacher-forced: the user asks for an exact repeat, and the assistant message IS the
text. The pooled read (mean over the text tokens) is taken at the recorded layers, which are the
depth fractions 0.25/0.375/0.5/0.625/0.75 truncated to layer indices.

Sets (all from level1_data.json):
- fit: plain format (no monitor brief); every concept's FIT texts, plus the FIT concept-free
  negatives. Used to fit the probes.
- calib_pos: monitor brief naming X + X's CALIB texts. This is the turn-1 episode format. Gives
  each probe's quality and its 100 point.
- ctx_neg: monitor brief naming X + --ctx-neg concept-free CALIB negatives, a different random
  draw per concept. These are context-matched negatives: they give the threshold (1% FPR) and the
  0 point.
- plain_pos / plain_neg: CALIB texts and 1,000 CALIB negatives in plain format. They measure how
  much the brief itself shifts the scores.
Outputs in --out: {set}.npy [n, L, d] float16, {set}.json row metadata, and layers.json. A set whose
.npy exists is skipped, so the job can resume.

Usage (GPU, v5 env): python scripts/l1_reads.py --model PATH --out DIR [--ctx-neg 200]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from next_chameleons.tica.level1 import TASK, opening

RAW = Path("/scratch/sneheel/next_chameleons_raw_cache/tica")
FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75)
CHUNK = 2048


def n_layers(model_path: str) -> int:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return int(cfg.get("text_config", cfg)["num_hidden_layers"])


def conv(user: str, text: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": text}]


def build_sets(data: dict, ctx_neg: int) -> dict[str, list[tuple[list[dict], dict]]]:
    sets: dict[str, list] = {
        "fit": [],
        "calib_pos": [],
        "ctx_neg": [],
        "plain_pos": [],
        "plain_neg": [],
    }
    negs = data["negatives"]
    for c in data["concepts"]:
        x = c["concept"]
        for t in c["fit"]:
            sets["fit"].append(
                (conv(TASK.format(text=t["text"]), t["text"]), {"concept": x, "id": t["id"]})
            )
        for t in c["calib"]:
            sets["calib_pos"].append(
                (
                    conv(opening("real_score", t["text"], x), t["text"]),
                    {"concept": x, "id": t["id"]},
                )
            )
            sets["plain_pos"].append(
                (conv(TASK.format(text=t["text"]), t["text"]), {"concept": x, "id": t["id"]})
            )
        for t in random.Random(f"ctxneg|{x}").sample(negs["calib"], ctx_neg):
            sets["ctx_neg"].append(
                (
                    conv(opening("real_score", t["text"], x), t["text"]),
                    {"concept": x, "id": t["id"]},
                )
            )
    for t in negs["fit"]:
        sets["fit"].append(
            (conv(TASK.format(text=t["text"]), t["text"]), {"concept": None, "id": t["id"]})
        )
    for t in random.Random("plainneg").sample(negs["calib"], 1000):
        sets["plain_neg"].append(
            (conv(TASK.format(text=t["text"]), t["text"]), {"concept": None, "id": t["id"]})
        )
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctx-neg", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-concepts", type=int, default=0, help="smoke: first N concepts only")
    args = ap.parse_args()
    data = json.loads((RAW / "level1_data.json").read_text())
    if args.limit_concepts:
        data["concepts"] = data["concepts"][: args.limit_concepts]
    sets = build_sets(data, args.ctx_neg)
    nl = n_layers(args.model)
    layers = sorted({max(0, min(int(nl * f), nl - 1)) for f in FRACTIONS})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "layers.json").write_text(
        json.dumps({"n_layers": nl, "layers": layers, "model": args.model})
    )
    print({k: len(v) for k, v in sets.items()}, "layers", layers, flush=True)

    from next_chameleons.tica.backend import TicaBackend

    be = TicaBackend(args.model, layers, batch_size=args.batch_size)
    for name, rows in sets.items():
        path = args.out / f"{name}.npy"
        if path.exists():
            print("skip", name, flush=True)
            continue
        # Sort by length to cut padding, then write back in the original order.
        order = sorted(range(len(rows)), key=lambda i: len(be.forced_ids(rows[i][0])[0]))
        tmp = args.out / f"{name}.tmp.npy"
        arr = None
        for k in range(0, len(order), CHUNK):
            idx = order[k : k + CHUNK]
            x = be.forced_pooled([rows[i][0] for i in idx])
            if arr is None:
                arr = np.lib.format.open_memmap(
                    tmp, mode="w+", dtype=np.float16, shape=(len(rows), *x.shape[1:])
                )
            arr[idx] = x
            print(f"{name} {min(k + CHUNK, len(order))}/{len(order)}", flush=True)
        arr.flush()
        del arr
        (args.out / f"{name}.json").write_text(json.dumps([m for _, m in rows]))
        tmp.rename(path)


if __name__ == "__main__":
    main()
