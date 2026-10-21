"""MODEL-WRITTEN concept texts, step 5 check (REPLICATION 2026-10-21f): base probes fit from the
new data file (the config's data + request file, through the normal cached fit_answer_probes path)
vs base probes fit from the old (published-response) file, each read on both files' held-out
EPISODE texts (base model, no brief, request + text, primary layer). Per concept:
  quality      the fit gate's own numbers (AUROC on its 30% split, TPR without the brief, calib TPR)
  cross        probe {old, new} x texts {old, new}: AUROC (concept episode texts vs other train
               concepts' episode texts of the same file + the concept-free episode pool), TPR at
               the probe threshold, FPR on others / on the free pool
  source       mean logit of the NEW probe on other concepts' MODEL-WRITTEN texts vs the same
               concepts' PUBLISHED texts vs the published concept-free pool: a "model-written"
               shortcut shows as published negatives scoring systematically lower than
               model-written ones.
Aggregates only.

  python scripts/tica_mw_probe_check.py --config final_v5_modelwritten_4b --old-data D.json \
      --old-inputs I.json --concepts "sports|jokes and humor" --out O.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.data import Level1Data
from next_chameleons.tica.monitors import auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the model-written config")
    ap.add_argument("--old-data", required=True)
    ap.add_argument("--old-inputs", required=True)
    ap.add_argument("--concepts", required=True, help="'|'-separated train concepts")
    ap.add_argument("--n-other", type=int, default=400)
    ap.add_argument("--model-path")
    ap.add_argument("--data", help="override the config's data file (smoke builds)")
    ap.add_argument("--inputs", help="override the config's request file (smoke builds)")
    ap.add_argument("--labels", help="override the config's neg_labels (smoke: new texts "
                    "unlabelled)")
    ap.add_argument("--old-labels", help="neg_labels for the old-data probes (default: --labels / "
                    "the config's)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import copy

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    if a.data:
        cfg.data = a.data
    if a.inputs:
        cfg.episodes.inputs = a.inputs
    new_data, _, factory = build(cfg)
    vers = {"old": (Level1Data(a.old_data), a.old_inputs),
            "new": (new_data, cfg.episodes.inputs)}
    pol = factory(None, False)
    pol.model.eval()
    L = cfg.primary_layer()
    concepts = a.concepts.split("|")
    rng = random.Random(0)
    if a.labels:
        cfg.rl.adaptive.neg_labels = a.labels
    acfgs = {"new": cfg.rl.adaptive, "old": copy.deepcopy(cfg.rl.adaptive)}
    if a.old_labels:
        acfgs["old"].neg_labels = a.old_labels
    probes, quality, opens = {}, {}, {}
    for v, (data, inp) in vers.items():
        opens[v] = prompts("answer", inp, cfg.episodes.layout)[0]
        for c in concepts:
            mons, q = fit_answer_probes(acfgs[v], data, pol, c, [L], opens[v],
                                        neg_splits=("train",),
                                        rename=f"mwcheck_{v}")
            probes[(v, c)] = mons[0].to(pol.device)
            quality[(v, c)] = q[L]
            print(f"fit {v} {c[:28]:28s} {q[L]}", flush=True)

    def read(v, texts):
        convs = [[{"role": "user", "content": opens[v]("nobrief", x, None)},
                  {"role": "assistant", "content": x}] for x in texts]
        out = []
        with torch.no_grad():
            for b in range(0, len(convs), 32):
                out.append(pol.pooled(convs[b: b + 32], [L])[L].float())
        return torch.cat(out) if out else torch.zeros(0)

    train = [c.concept for c in new_data.concepts("episode", ("train",))]
    free_t = vers["old"][0].negatives("episode")
    free_t = rng.sample(free_t, min(a.n_other, len(free_t)))
    free = read("old", free_t)
    ep = {}
    for v, (data, _) in vers.items():
        texts = {c.concept: c.texts for c in data.concepts("episode", ("train",))}
        for c in concepts:
            others = [x for o in train if o != c for x in texts[o]]
            ep[(v, c)] = (read(v, texts[c]),
                          read(v, rng.sample(others, min(a.n_other, len(others)))))
    out = {"config": a.config, "layer": L, "old_data": a.old_data, "new_data": cfg.data,
           "labels": {v: acfgs[v].neg_labels for v in acfgs},
           "concepts": {}}
    for c in concepts:
        r = {"quality": {v: quality[(v, c)] for v in vers}, "cross": {}, "source": {}}
        for pv in vers:
            m = probes[(pv, c)]
            thr = float(m.thresholds([c])[0])

            def s(x, m=m, c=c):
                return m.logits(x.to(pol.device), [c] * len(x)).float().cpu()

            sf = s(free)
            for tv in vers:
                pos, oth = ep[(tv, c)]
                sp, so = s(pos), s(oth)
                r["cross"][f"probe_{pv}/texts_{tv}"] = {
                    "auroc": round(auroc(sp, torch.cat([so, sf])), 4),
                    "tpr": round(float((sp > thr).float().mean()), 3),
                    "fpr_others": round(float((so > thr).float().mean()), 3),
                    "fpr_free": round(float((sf > thr).float().mean()), 3)}
                if pv == "new":
                    r["source"][f"others_{tv}_mean_logit"] = round(float(so.mean()), 3)
            if pv == "new":
                r["source"]["free_published_mean_logit"] = round(float(sf.mean()), 3)
                r["source"]["threshold"] = round(thr, 3)
        out["concepts"][c] = r
        print(c, json.dumps(r["cross"]), json.dumps(r["source"]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
