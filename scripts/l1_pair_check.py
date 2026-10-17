"""Pair-dilution check for HIDDEN-mode episodes (DESIGN-TICA "FINAL SETUP 10-04").

Hidden-mode texts at turns >= 1 are two passages about two concepts. The concept probes were fit
and gated on single-concept passages, and read mean-pooled activations, so a concept that fills
half the passage scores lower. This script measures, on the BASE model with the plain repeat task
(no brief), whether each gated concept's probe still detects it inside a pair, and writes the
PAIR GATE: concepts whose probe detects them in pairs at >= `--min-det` (over every pairing they
appear in, as the concept or as the partner). The hidden-mode configs point probes.pair_gate at
that file, and build() ANDs it into the primary bank's gate.

Usage (v5 env, GPU):
  PYTHONPATH=src python scripts/l1_pair_check.py --config l1h_qwen35_4b --out DIR [--pairings 4]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairings", type=int, default=4)
    ap.add_argument("--min-det", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N concepts")
    args = ap.parse_args()

    from next_chameleons.tica.config import load_run_config
    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.level1 import opening
    from next_chameleons.tica.monitors import ConceptProbeBank
    from next_chameleons.tica.policy import Policy

    cfg = load_run_config(args.config)
    L = cfg.primary_layer()
    bank = ConceptProbeBank.from_level1_npz(Path(cfg.probes.dir) / f"bank_L{L}.npz")
    data = Level1Data(cfg.data)
    gated = set(bank.gated())
    concepts = [c for c in data.concepts("episode", ("train", "heldout")) if c.concept in gated]
    if args.limit:
        concepts = concepts[: args.limit]
    pol = Policy.from_pretrained(
        cfg.model.path,
        dtype=cfg.model.dtype,
        enable_thinking=cfg.model.enable_thinking,
        batch_size=cfg.model.batch_size,
    )
    bank = bank.to(pol.device)
    rng = random.Random(0)

    # rows: (text, [(concept, kind)]) where kind = single | first | second
    rows: list[tuple[str, list[tuple[str, str]]]] = []
    for c in concepts:
        rows.append((c.texts[0], [(c.concept, "single")]))
        for k in range(args.pairings):
            p = rng.choice([o for o in concepts if o.concept != c.concept])
            a, b = c.texts[k % len(c.texts)], p.texts[(k + 7) % len(p.texts)]
            if rng.random() < 0.5:
                rows.append((f"{a}\n\n{b}", [(c.concept, "first"), (p.concept, "second")]))
            else:
                rows.append((f"{b}\n\n{a}", [(p.concept, "first"), (c.concept, "second")]))

    convs = [
        [
            {"role": "user", "content": opening("nobrief", t, "")},
            {"role": "assistant", "content": t},
        ]
        for t, _ in rows
    ]
    pooled = pol.pooled(convs, [L])[L]
    det: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    margin: dict[str, list[float]] = defaultdict(list)
    for i, (_, reads) in enumerate(rows):
        cs = [c for c, _ in reads]
        raw = bank.logits(pooled[i : i + 1].expand(len(cs), -1), cs)
        thr = bank.thresholds(cs)
        sc = bank.normalize(raw, cs)
        for (c, kind), r, t, s in zip(reads, raw.tolist(), thr.tolist(), sc.tolist(), strict=True):
            det[c][kind].append(float(r > t))
            margin[kind].append(s)

    per_concept = {}
    for c in det:
        pair = det[c]["first"] + det[c]["second"]
        per_concept[c] = {
            "single": float(np.mean(det[c]["single"])) if det[c]["single"] else None,
            "pair": float(np.mean(pair)) if pair else None,
            "n_pair": len(pair),
        }
    passed = sorted(
        c
        for c, v in per_concept.items()
        if v["pair"] is not None and v["n_pair"] >= args.pairings and v["pair"] >= args.min_det
    )
    agg = {
        "layer": L,
        "n_concepts": len(per_concept),
        "det_single": float(
            np.mean([v["single"] for v in per_concept.values() if v["single"] is not None])
        ),
        "det_pair": float(
            np.mean([v["pair"] for v in per_concept.values() if v["pair"] is not None])
        ),
        "det_pair_first": float(np.mean([x for c in det for x in det[c]["first"]])),
        "det_pair_second": float(np.mean([x for c in det for x in det[c]["second"]])),
        "score_median": {k: float(np.median(v)) for k, v in margin.items()},
        "min_det": args.min_det,
        "n_pass": len(passed),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pair_check.json").write_text(
        json.dumps({"aggregate": agg, "per_concept": per_concept}, indent=1)
    )
    (out / f"pair_gate_L{L}.json").write_text(json.dumps(passed))
    print(json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    main()
