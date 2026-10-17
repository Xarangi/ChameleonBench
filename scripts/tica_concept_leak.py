"""Train -> held-out concept LEAK check for a final concept base (REPLICATION 2026-10-15e).

For every TRAIN concept A: the exact base probe training uses (answer_format.fit_answer_probes,
negatives = other TRAIN concepts + concept-free texts, 1% FPR) at the primary layer; it then reads
the calib texts of every DEV / HELD-OUT concept B (base model, no brief, the text's own request).
leak[A][B] = fraction of B's texts on which A's probe fires. A high value means hiding A is partly
hiding B already (held-out generalization would be partly free), so such pairs are reported apart.
Also own_tpr[A] on A's calib texts. Aggregates only.

  python scripts/tica_concept_leak.py --config final_v2_4b --out OUT.json
  # OOD domain concepts (scripts/build_final_ood.py): --data D --inputs I --score-splits ood
  #   --gate-splits ood
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--n", type=int, default=60, help="calib texts per scored concept")
    ap.add_argument("--data", help="override the config's data file (e.g. final_ood_v1.json)")
    ap.add_argument("--inputs", help="override the config's answer-format inputs map")
    ap.add_argument("--score-splits", default="dev,heldout",
                    help="concept splits whose calib texts the train probes score (e.g. ood)")
    ap.add_argument("--gate-splits", default="heldout",
                    help="concept splits whose eval base probes must pass the gate")
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
    score_splits = tuple(x.strip() for x in a.score_splits.split(",") if x.strip())
    gate_splits = tuple(x.strip() for x in a.gate_splits.split(",") if x.strip())
    data, _, factory = build(cfg)
    pol = factory(None, False)
    opening, _ = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    L = cfg.primary_layer()
    train = list(cfg.episode_concepts)
    others = [c for c in data.concepts("calib", score_splits)]
    own = {c.concept: c for c in data.concepts("calib", ("train",))}

    def reads(texts, concept):
        convs = [[{"role": "user", "content": opening("nobrief", x, concept)},
                  {"role": "assistant", "content": x}] for x in texts]
        with torch.no_grad():
            return pol.pooled(convs, [L])[L]

    cache = {c.concept: reads(c.texts[: a.n], c.concept) for c in others}
    leak, tpr, quality = {}, {}, {}
    for k, A in enumerate(train):
        held = [int(x) for x in cfg.probes.held_out_layers if int(x) != L]
        mons, q = fit_answer_probes(cfg.rl.adaptive, data, pol, A, [L, *held], opening,
                                    neg_splits=("train",), rename=f"leak_c{k}")
        m = mons[0].to(pol.device)
        quality[A] = q
        thr = m.thresholds([A])[0]
        def fire(X, m=m, thr=thr, A=A):
            s = m.logits(X, [A])
            return float(((s if s.dim() == 1 else s[:, 0]) > thr).float().mean())

        tpr[A] = fire(reads(own[A].texts[: a.n], A))
        leak[A] = {B: fire(X) for B, X in cache.items()}
        top = sorted(leak[A].items(), key=lambda t: -t[1])[:3]
        print(json.dumps({"concept": A, "own_tpr": tpr[A], "top_leak": top}), flush=True)
    flagged = sorted({(A, B, v) for A, row in leak.items() for B, v in row.items() if v > 0.25},
                     key=lambda t: -t[2])
    gate = {}  # held-out-concept base probes (the eval's), every layer: must pass the gate
    for B in [c.concept for c in data.concepts("fit", gate_splits)]:
        held = [int(x) for x in cfg.probes.held_out_layers if int(x) != L]
        try:
            _, q = fit_answer_probes(cfg.rl.adaptive, data, pol, B, [L, *held], opening,
                                     rename=f"leakh_{B[:12]}")
            gate[B] = q
        except RuntimeError as err:
            gate[B] = {"FAILED": str(err)[-200:]}
        print(json.dumps({"heldout_probe": B, "gate": gate[B]}, default=str)[:400], flush=True)
    rep = {"config": a.config, "layer": L, "own_tpr": tpr, "leak": leak, "heldout_gate": gate,
           "flagged_gt_0.25": flagged, "base_probe_quality": quality}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"flagged_gt_0.25": flagged}, indent=1))


if __name__ == "__main__":
    main()
