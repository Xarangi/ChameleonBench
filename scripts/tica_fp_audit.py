"""FALSE-POSITIVE AUDIT (REPLICATION 2026-10-16h; user 10-16: "can't we measure the effect of
the label noise by looking at false positives?"). For each concept: its BASE-model answer probe
(the cached gate probe, 1% FPR), scored on the concept's would-be NEGATIVES (calib split:
concept-free pool + other concepts' texts, read with the monitor brief naming the concept).
Among the negatives above the threshold ("false positives"), what fraction does the LLM judge
say actually HAVE the concept, vs the judged-positive rate among all negatives (enrichment)?
High enrichment = the threshold is
being set by mislabelled negatives. Also the threshold with judged-positive negatives removed.
Base model only, aggregates only.

  python scripts/tica_fp_audit.py --config final_v2_4b --data D.json --inputs I.json \
      --labels L.json --split heldout,ood --out O.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts, text_id
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.data import CONCEPT_SPLITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data")
    ap.add_argument("--inputs")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--split", default="heldout,ood")
    ap.add_argument("--n-free", type=int, default=1500)
    ap.add_argument("--n-other", type=int, default=600)
    ap.add_argument("--model-path")
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
    pol = factory(None, False)
    opening, _ = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    lab = {}
    for f in a.labels:
        lab.update(json.loads(Path(f).read_text())["labels"])
    rng = random.Random(0)
    free = data.negatives("calib")
    free = rng.sample(free, min(a.n_free, len(free)))
    allc = data.concepts("calib", CONCEPT_SPLITS)
    train = set(cfg.episode_concepts)
    rep = {}
    for c in [x.concept for x in data.concepts("calib", tuple(a.split.split(",")))]:
        if c == "deception":
            continue
        held = [int(x) for x in cfg.probes.held_out_layers]
        layers = [cfg.primary_layer()] + [h for h in held if h != cfg.primary_layer()]
        mons, _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening,
                                    neg_splits=("train",) if c in train else None)
        m = mons[0].to(pol.device)
        other = [x for o in allc if o.concept != c for x in o.texts]
        negs = free + rng.sample(other, min(a.n_other, len(other)))
        convs = [[{"role": "user", "content": opening("real_score", x, c)},
                  {"role": "assistant", "content": x}] for x in negs]
        with torch.no_grad():
            r = pol.pooled(convs, [m.layer], batch=32)[m.layer]
        sc = m.logits(r, [c])
        sc = (sc if sc.dim() == 1 else sc[:, 0]).float().cpu()
        thr = float(m.thresholds([c])[0])
        judged = torch.tensor([lab.get(text_id(x), {}).get(c, 0.0) >= a.thr for x in negs])
        fp = sc > thr
        clean = sc[~judged]
        q = float(torch.quantile(clean.double(), 0.99)) if len(clean) else float("nan")
        rep[c] = {"n_neg": len(negs), "judged_pos_rate": round(float(judged.float().mean()), 4),
                  "n_fp": int(fp.sum()), "fp_rate": round(float(fp.float().mean()), 4),
                  "fp_judged_pos": round(float(judged[fp].float().mean()), 3) if fp.any() else None,
                  "fp_rate_clean_negs": round(float((clean > thr).float().mean()), 4),
                  "thr": round(thr, 3), "q99_clean": round(q, 3)}
        print(json.dumps({c: rep[c]}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
