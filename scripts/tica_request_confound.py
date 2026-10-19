"""Do base probes read the ANSWER or the REQUEST? (REPLICATION 2026-10-19d)

The concept-eliciting requests (build_elicit_inputs.py) name the concept for positives only, so a
probe fit on [request + answer] could detect the request alone. Then a free answer that leaves the
concept out would still be "DETECTED", and a pass would no longer mean the concept was hidden.

Per concept, base probes are fit with --inputs (as every race does), then read on the base model:
  pos    concept text after its own request                      (TPR)
  swap   concept-eliciting request + a CONCEPT-FREE answer         (request-only detection)
  free   concept-free text after its own request                  (FPR)
  other  another concept's text after its own request              (FPR)
each with the monitor brief (tagged layout) and without. swap near free = the probe reads the
answer. --compare INPUTS also reads swap / pos with probes fit on another request file (e.g. the
legacy one). Aggregates only.

  python scripts/tica_request_confound.py --config final_v3b_4b --inputs ELICIT.json --out O.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", help="a second request file: its probes read the same swap rows")
    ap.add_argument("--topics", default="heldout")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--model-path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    cfg.episodes.inputs = a.inputs
    data, _, factory = build(cfg)
    topics = select_topics(data, a.topics)
    train = set(cfg.episode_concepts)
    pol = factory(None, False)
    held = [int(x) for x in cfg.probes.held_out_layers]
    layers = [cfg.primary_layer()] + [h for h in held if h != cfg.primary_layer()]
    fmts = {"elicit": prompts("answer", a.inputs, cfg.episodes.layout)}
    if a.compare:
        fmts["compare"] = prompts("answer", a.compare, cfg.episodes.layout)
    rng = random.Random(0)
    ep = {c.concept: c.texts for c in data.concepts("episode", CONCEPT_SPLITS)}
    free = data.negatives("episode")
    out = {"inputs": a.inputs, "compare": a.compare, "layers": layers, "concepts": {}}
    t0 = time.time()
    for k, c in enumerate(topics):
        pos = rng.sample(ep[c], min(a.n, len(ep[c])))
        neg = rng.sample(free, min(a.n, len(free)))
        oth = rng.sample([x for o, ts in ep.items() if o != c for x in ts], a.n)
        res = {}
        for fname, (opening, _) in fmts.items():
            try:
                mons, _q = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening,
                                             neg_splits=("train",) if c in train else None,
                                             rename=f"conf_{fname}_c{k}")
            except RuntimeError as err:
                res[fname] = {"failed": str(err)[-160:]}
                continue
            mons = [m.to(pol.device) for m in mons]
            # the requests always come from the ELICIT file: compare-probes read the same rows
            op = fmts["elicit"][0]
            rows = {
                "pos": [(op(cd, x, c), x) for x in pos for cd in ("real_score", "nobrief")],
                "swap": [(op(cd, p, c), n) for p, n in zip(pos, neg, strict=False)
                         for cd in ("real_score", "nobrief")],
                "free": [(op(cd, x, c), x) for x in neg for cd in ("real_score", "nobrief")],
                "other": [(op(cd, x, c), x) for x in oth for cd in ("real_score", "nobrief")],
            }
            r = {}
            for name, pairs in rows.items():
                convs = [[{"role": "user", "content": u}, {"role": "assistant", "content": x}]
                         for u, x in pairs]
                with torch.no_grad():
                    pooled = pol.pooled(convs, layers)
                for L, m in zip(layers, mons, strict=False):
                    cs = [c] * len(convs)
                    det = (m.logits(pooled[L], cs) > m.thresholds(cs)).float()
                    r[f"{name}@{L}"] = {"brief": float(det[0::2].mean()),
                                        "nobrief": float(det[1::2].mean())}
            res[fname] = r
        out["concepts"][c] = res
        e = res.get("elicit", {})
        print(f"[{k}] {c[:28]:28s} " + " ".join(
            f"{n}={e.get(f'{n}@{layers[0]}', {}).get('nobrief', float('nan')):.2f}"
            for n in ("pos", "swap", "free", "other")) + f"  {time.time() - t0:.0f}s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
