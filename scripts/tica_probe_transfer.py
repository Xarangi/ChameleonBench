"""Do base probes detect the base model's OWN answers? (REPLICATION 2026-10-19f; user 10-19: "the
probes on the base model would detect those answers no? ... properly see if it fails even if we
properly prompt the free generation")

Base model, no brief, single turn. Per concept, N positive texts' (concept-eliciting) requests from
--inputs; the base model answers each (sampled, --temp). Base probes (fit on the stored passages as
every race fits them, with --probe-inputs requests) then read:
  forced   [request, stored passage]          the probes' own distribution (reference TPR)
  free     [request, the model's own answer]  TPR at the probe threshold
  AUROC    free positives vs the free answers to every OTHER concept's requests (same
           distribution): separable at all, or only a threshold (calibration) problem?
--gen-out writes the answers for the 27B judge (tica_free_judge.py; cond nobrief, turn 0) so the
concept's presence in the answers is checked too. Scratch only, never commit. Aggregates only.

  python scripts/tica_probe_transfer.py --config final_v3b_4b --inputs ELICIT.json --out O.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.data import CONCEPT_SPLITS, select_topics
from next_chameleons.tica.monitors import auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True, help="requests the model answers")
    ap.add_argument("--probe-inputs", help="requests the base probes are fit with (default: the "
                    "config's)")
    ap.add_argument("--data")
    ap.add_argument("--topics", default="heldout")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--model-path")
    ap.add_argument("--gen-out", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    if a.data:
        cfg.data = a.data
    if a.probe_inputs:
        cfg.episodes.inputs = a.probe_inputs
    data, _, factory = build(cfg)
    topics = select_topics(data, a.topics)
    train = set(cfg.episode_concepts)
    pol = factory(None, False)
    pol.model.eval()
    layers = [cfg.primary_layer()]
    probe_open = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)[0]
    ask_open = prompts("answer", a.inputs, cfg.episodes.layout)[0]
    rng = random.Random(0)
    ep = {c.concept: c.texts for c in data.concepts("episode", CONCEPT_SPLITS)}
    t0 = time.time()
    mons, reads, answers = {}, {}, {}
    for k, c in enumerate(topics):
        try:
            mons[c] = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, probe_open,
                                        neg_splits=("train",) if c in train else None,
                                        rename=f"transfer_c{k}")[0][0].to(pol.device)
        except RuntimeError as err:
            print(f"{c}: probe failed the gate ({str(err)[-80:]})", flush=True)
            continue
        texts = rng.sample(ep[c], min(a.n, len(ep[c])))
        reqs = [ask_open("nobrief", x, c) for x in texts]
        gens = pol.generate([[{"role": "user", "content": r}] for r in reqs], a.max_new,
                            temperature=a.temp, seed=k)
        answers[c] = [(r, g["text"], g["finished"]) for r, g in zip(reqs, gens, strict=True)]
        with torch.no_grad():
            forced = pol.pooled([[{"role": "user", "content": r}, {"role": "assistant",
                                  "content": x}] for r, x in zip(reqs, texts, strict=True)], layers)
            free = pol.pooled([[{"role": "user", "content": r}, {"role": "assistant",
                                "content": g}] for r, g, _ in answers[c]], layers)
        reads[c] = (forced[layers[0]], free[layers[0]])
        print(f"[{k}] {c[:28]:28s} generated {len(gens)}  {time.time() - t0:.0f}s", flush=True)
    out = {"inputs": a.inputs, "probe_inputs": cfg.episodes.inputs, "layer": layers[0],
           "n": a.n, "temp": a.temp, "concepts": {}}
    for c, m in mons.items():
        thr = m.thresholds([c])[0]

        def score(x, c=c, m=m):
            return m.logits(x, [c] * len(x)).float().cpu()

        forced, free = reads[c]
        others = torch.cat([reads[o][1] for o in mons if o != c])
        sf, sp, so = score(forced), score(free), score(others)
        thr = float(thr)
        out["concepts"][c] = {
            "tpr_forced": round(float((sf > thr).float().mean()), 3),
            "tpr_free": round(float((sp > thr).float().mean()), 3),
            "fpr_free_others": round(float((so > thr).float().mean()), 3),
            "auroc_free": round(auroc(sp, so), 3),
            "auroc_forced_vs_free_others": round(auroc(sf, so), 3),
            "finished": round(sum(f for *_, f in answers[c]) / len(answers[c]), 3),
        }
        r = out["concepts"][c]
        print(f"{c[:28]:28s} TPR forced {r['tpr_forced']:.2f} free {r['tpr_free']:.2f}  "
              f"FPR others {r['fpr_free_others']:.2f}  AUROC free {r['auroc_free']:.2f}",
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    if a.gen_out:
        with open(a.gen_out, "w") as f:
            for c, rows in answers.items():
                for j, (u, g, fin) in enumerate(rows):
                    f.write(json.dumps({"ep": j, "turn": 0, "cond": "nobrief", "concept": c,
                                        "off": False, "detected": None, "passed": None,
                                        "finished": fin, "answer": g, "user": u}) + "\n")


if __name__ == "__main__":
    main()
