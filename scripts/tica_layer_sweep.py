"""LAYER SWEEP (REPLICATION 2026-10-15e; user 10-15: "did we test dodging at other layers").
Where in
the network does the hidden concept stay readable?

For each concept (--split heldout / train / dev) and each layer in --layers: the BASE model's answer
probe (answer_format.fit_answer_probes: the training gate, 1% FPR; per-layer gate failures are
recorded and the layer skipped). Then N texts of the concept (episode split) are read as:
  hide    -- trained model, a LATE hiding turn: --history earlier turns of the same concept with the
             brief and verdict messages (caught once, then passed), then the scored text. Turn 0 is
             pinned to the base model in training, so a first turn says nothing about hiding;
  first   -- trained model, the brief, first turn (pinned: expected ~ base);
  nobrief -- trained model, no brief (should stay detected: no-brief specificity);
  base    -- base model on the same late-turn conversation (reference).
det[layer][cond] = fraction detected. Static probes (no refits): this is the "is it readable at
layer L at all" map; refit races are the scripts/tica_switch_eval.py side. Aggregates only.
Note: with FORCED texts the first layers are near-lexical (a topic probe there reads the given
tokens), so low layers are a lexical baseline, not a hiding failure.

  python scripts/tica_layer_sweep.py --config final_v2_4b --adapter DIR --split heldout --out O
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--layers", default=",".join(str(x) for x in range(2, 31, 2)))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--model-path")
    ap.add_argument("--data", help="override the config's data file (e.g. final_ood_v1.json)")
    ap.add_argument("--inputs", help="override the config's answer-format inputs map")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    if a.data:
        cfg.data = a.data
    if a.inputs:
        cfg.episodes.inputs = a.inputs
    data, _, factory = build(cfg)
    pol = factory(a.adapter, False)
    opening, render = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    fb = cfg.episodes.feedback
    caught, passed = (1.0, 0.0), (-1.0, 0.0)  # (score, threshold) shown in the verdict message
    layers = [int(x) for x in a.layers.split(",")]
    train = set(cfg.episode_concepts)
    concepts = [c.concept for c in data.concepts("episode", (a.split,))]
    det = defaultdict(lambda: defaultdict(list))
    per = defaultdict(lambda: defaultdict(dict))  # layer -> cond -> concept -> detected frac
    gate, t0 = {}, time.time()
    for k, c in enumerate(concepts):
        mons, ok_layers = {}, []
        try:  # all layers from ONE read of the texts; per layer only if some layer fails its gate
            m, q = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening,
                                     neg_splits=("train",) if c in train else None,
                                     rename=f"sweep_c{k}")
            mons = {L: mm.to(pol.device) for L, mm in zip(layers, m, strict=True)}
            ok_layers = list(layers)
            gate[c] = q
        except RuntimeError:
            pass
        for L in [] if ok_layers else layers:  # per layer: a gate failure drops only that layer
            try:
                m, q = fit_answer_probes(cfg.rl.adaptive, data, pol, c, [L], opening,
                                         neg_splits=("train",) if c in train else None,
                                         rename=f"sweep_c{k}")
                mons[L] = m[0].to(pol.device)
                ok_layers.append(L)
                gate.setdefault(c, {})[L] = q.get(L, q)
            except RuntimeError as err:
                gate.setdefault(c, {})[L] = {"FAILED": str(err)[-120:]}
        pool_t = data.concepts("episode", (a.split,), only={c})[0].texts
        texts = pool_t[: a.n]

        def late(j, x, c=c, pool_t=pool_t):
            hist = [pool_t[(a.n + j * a.history + q) % len(pool_t)] for q in range(a.history)]
            conv = [{"role": "user", "content": opening("real_score", hist[0], c)},
                    {"role": "assistant", "content": hist[0]}]
            for q, y in enumerate([*hist[1:], x]):
                conv += [{"role": "user", "content": render("real_score", c,
                                                            caught if q == 0 else passed, y,
                                                            feedback=fb)},
                         {"role": "assistant", "content": y}]
            return conv

        convs = {
            "hide": [late(j, x) for j, x in enumerate(texts)],
            "first": [[{"role": "user", "content": opening("real_score", x, c)},
                       {"role": "assistant", "content": x}] for x in texts],
            "nobrief": [[{"role": "user", "content": opening("nobrief", x, c)},
                         {"role": "assistant", "content": x}] for x in texts]}
        with torch.no_grad():
            reads = {cond: pol.pooled(cv, ok_layers) for cond, cv in convs.items()}
            with pol.reference():
                reads["base"] = pol.pooled(convs["hide"], ok_layers)
        for L in ok_layers:
            m, thr = mons[L], mons[L].thresholds([c])[0]
            for cond, R in reads.items():
                sc = m.logits(R[L], [c])
                v = float(((sc if sc.dim() == 1 else sc[:, 0]) > thr).float().mean())
                det[L][cond].append(v)
                per[L][cond][c] = v
        print(json.dumps({"concept": c, "sec": round(time.time() - t0),
                          "hide_by_layer": {L: round(det[L]["hide"][-1], 2) for L in ok_layers}}),
              flush=True)
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    rep = {"config": a.config, "adapter": a.adapter, "split": a.split, "concepts": concepts,
           "n_texts": a.n,
           "det_by_layer": {L: {cond: mean(v) for cond, v in d.items()} for L, d in det.items()},
           "per_concept": {L: dict(d) for L, d in per.items()},
           "gate": gate}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["det_by_layer"], indent=1))


if __name__ == "__main__":
    main()
