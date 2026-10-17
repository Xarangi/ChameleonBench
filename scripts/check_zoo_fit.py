"""Sanity fit of every probe-zoo kind on CLEAN data (REPLICATION 2026-10-14f): answer-format reads
of the concept's FIT texts (positives) vs other-concept + concept-free FIT texts (negatives), with
the monitor brief, from the base model (or --adapter). Each kind goes through the same gate_fit as
the race (5-fold CV AUROC, 1% FPR threshold on held-out negatives, OOC TPR on no-brief calib reads).
A kind that cannot fit clean base-model data is a recipe bug, not evidence of dodging. Also reports
token-activation scale (raw-activation recipes are sensitive to it). Aggregates only.

  python scripts/check_zoo_fit.py --config l5dodge_math_4b --out OUT.json [--kinds a,b]
      [--adapter D]
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

from next_chameleons.tica.answer_format import prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.policy import _keep_tokens
from next_chameleons.tica.probe_zoo import TOK_KEEP, TOKENS_ENS, ZOO_KINDS, gate_fit, kind_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--model-path")
    ap.add_argument("--kinds", default=",".join(ZOO_KINDS))
    ap.add_argument("--n-pos", type=int, default=96)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(a.adapter, False)
    opening, _ = prompts(cfg.episodes.format, cfg.episodes.inputs, cfg.episodes.layout)
    concept = cfg.episode_concepts[0]
    rng = random.Random(0)
    fit = {c.concept: c.texts for c in data.concepts("fit", ("train", "heldout"))}
    others = [x for c, ts in fit.items() if c != concept for x in ts]
    P_t = rng.sample(fit[concept], min(a.n_pos, len(fit[concept])))
    N_t = rng.sample(others, a.n_pos) + rng.sample(data.negatives("fit"), a.n_pos // 2)
    G_t = data.concepts("calib", ("train", "heldout"), only={concept})[0].texts[:24]
    kinds = a.kinds.split(",")
    layers = sorted({a.layer} | {L for k in kinds for L in kind_layers(k, a.layer)})

    def reads(texts, cond):
        convs = [[{"role": "user", "content": opening(cond, x, concept)},
                  {"role": "assistant", "content": x}] for x in texts]
        tok: dict = {}
        r = pol.pooled_incremental(convs, layers, {}, token_out=tok)
        return ({L: r[L].float().cpu() for L in layers},
                {L: [t.cpu() for t in tok[L]] for L in layers})

    t0 = time.time()
    with torch.no_grad():
        P, N, G = reads(P_t, "real_score"), reads(N_t, "real_score"), reads(G_t, "nobrief")
    rep = {"adapter": a.adapter, "concept": concept, "layer": a.layer, "n_pos": len(P_t),
           "n_neg": len(N_t), "read_sec": round(time.time() - t0, 1), "kinds": {}}
    toks = torch.cat(P[1][a.layer] + N[1][a.layer]).float()
    rep["token_scale"] = {"mean_norm": float(toks.norm(dim=-1).mean()),
                          "max_abs": float(toks.abs().max()),
                          "top_dim_frac": float(toks.abs().mean(0).max() / toks.abs().mean())}
    print("token scale", rep["token_scale"], flush=True)
    for k in kinds:
        Lk = kind_layers(k, a.layer)
        if k in TOKENS_ENS:
            items = [{L: [_keep_tokens(t, TOK_KEEP) for t in X[1][L]] for L in Lk}
                     for X in (P, N, G)]  # as the race: <= TOK_KEEP tokens per answer
        elif len(Lk) > 1:
            items = [{L: X[0][L] for L in Lk} for X in (P, N, G)]
        elif k in ("meandiff",):
            items = [X[0][a.layer] for X in (P, N, G)]
        else:
            items = [X[1][a.layer] for X in (P, N, G)]
        t1 = time.time()
        _, info = gate_fit(k, *items, cfg.rl.adaptive, name=f"check_{k}", layer=a.layer,
                             concept=concept, device=pol.device)
        info["sec"] = round(time.time() - t1, 1)
        rep["kinds"][k] = info
        print(k, info, flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
