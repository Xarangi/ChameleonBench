"""Turn-0 answers of a checkpoint, scored by EVERY base panel probe (REPLICATION 2026-10-20h):
what does the policy write at turn 0, and which probes catch it?

Training reads FORCED answers (the episode texts); this eval also lets the model WRITE its turn-0
answer (greedy) to the same held-out (episode-split) requests. Conditions: forced / free answers x
adapter with the brief, adapter without it (nobrief), base model with the brief. Each answer is read
once and scored by the run's answer pool (base probe + held-out layers, <run>/answer_pool) and by
every turn-0 panel probe (adaptive.first_probe_panel_kinds, [] = the in-training zoo), fit exactly
as the trainer fits them (_panel_first: base model, turn-0 fork construction, deterministic stream).

Aggregates -> --out (JSON); the raw answers go to --raw (scratch only, never commit).

  python scripts/tica_turn0_texts.py --config final_v4_math_v3b_panel_4b --adapter CKPT \
      --pool RUN/answer_pool --out OUT.json --raw /scratch/.../turn0_texts/x.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.adaptive import monitor_input
from next_chameleons.tica.answer_format import load_inputs, prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.episodes import EpisodeSampler
from next_chameleons.tica.monitors import MonitorPool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pool", required=True, help="the run's answer_pool (base + held-out probes)")
    ap.add_argument("--n", type=int, default=24, help="held-out (episode-split) texts")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--model-path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True, help="JSONL of answers (scratch only)")
    a = ap.parse_args()

    import torch

    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
        SupervisedConfig,
    )

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    (concept,) = cfg.episode_concepts
    pol = factory(a.adapter, True)
    fmt = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    opening = fmt[0]
    pool = MonitorPool.load(a.pool).to(pol.device)
    texts = data.concepts("fit", ("train",), only={concept})
    sampler = EpisodeSampler(texts, data.negatives("fit"), pool, mix={"real_score": 1.0},
                             turns=1, seed=a.seed, feedback=cfg.episodes.feedback)
    asc = AdaptiveSupervisedConfig(
        **{f.name: getattr(cfg.train, f.name) for f in dataclasses.fields(SupervisedConfig)},
        adaptive=cfg.rl.adaptive)
    out = Path(a.out)
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, asc, out.parent / (out.stem + "_work"),
        branch_concepts=data.concepts("fit", ("train",)), branch_negatives=data.negatives("fit"),
        roots_only={concept}, prompts=fmt, inputs=load_inputs(cfg.episodes.inputs))
    tr._refit_log = []
    pol.model.eval()
    t0 = time.time()

    # every panel probe, fit as the trainer does (one per kind)
    spec = sampler.sample(1)[0]
    render = lambda x: opening("real_score", x, concept, named=True)  # noqa: E731
    kinds = tr._panel_kinds()
    panel = {}
    for k in kinds:
        tr._panel_kinds = lambda k=k: [k]
        name = tr._panel_first(0, spec, concept, render, {}, {"sec": 0.0})
        if name is not None:
            panel[f"base:{k}"] = tr.pool.get(name)
    print(f"panel: {len(panel)}/{len(kinds)} fit, {time.time() - t0:.0f}s", flush=True)
    base = {"base_probe": pool.concept_monitor(concept)}
    held = {f"heldout:{e.monitor.name}@{e.monitor.layer}": e.monitor
            for e in pool.entries if e.held_out and e.monitor.covers(concept)}
    mons = {**base, **panel, **held}
    layers, tok_layers = set(), set()
    for m in mons.values():
        Ls = getattr(m, "layers_used", None) or [m.layer]
        layers.update(Ls)
        if getattr(m, "needs_tokens", False):
            tok_layers.update(Ls)

    ep = data.concepts("episode", ("train",), only={concept})[0].texts
    sel = random.Random(a.seed).sample(ep, min(a.n, len(ep)))
    tokz = pol.tokenizer
    conds = {"adapter_brief": ("real_score", False), "adapter_nobrief": ("nobrief", False),
             "base_brief": ("real_score", True)}
    rows, agg = [], {}
    for answers in ("forced", "free"):
        for cname, (cond, use_base) in conds.items():
            convs = [[{"role": "user", "content": opening(cond, x, concept, named=True)}]
                     for x in sel]
            ctx = pol.reference() if use_base else torch.no_grad()
            with ctx, torch.no_grad():
                if answers == "free":
                    gens = pol.generate(convs, a.max_new)
                    fin = [g["finished"] for g in gens]
                    outs = [g["text"] for g in gens]
                else:
                    fin, outs = [True] * len(sel), list(sel)
                for c, o in zip(convs, outs, strict=True):
                    c.append({"role": "assistant", "content": o or " "})
                tok = {}
                pooled = pol.pooled(convs, sorted(layers),
                                    token_out=tok if tok_layers else None)
            for L in tok_layers:
                pooled[("tok", L)] = tok[L]
            key = f"{answers}/{cname}"
            caught = {n: [] for n in mons}
            for i, o in enumerate(outs):
                det = {}
                for n, m in mons.items():
                    raw = m.logits(monitor_input(m, pooled, i), [concept])
                    det[n] = bool(raw[0] > m.thresholds([concept])[0])
                    caught[n].append(det[n])
                rows.append({"answers": answers, "cond": key, "request": convs[i][0]["content"],
                             "answer": o, "finished": fin[i], "n_tok": len(tokz.encode(o)),
                             "detected": det})
            pn = [n for n in mons if n.startswith("base:")]
            hn = [n for n in mons if n.startswith("heldout:")]
            agg[key] = {
                "n": len(sel),
                "panel_pass": (sum(not d for n in pn for d in caught[n])
                               / max(1, len(pn) * len(sel))),
                "panel_pass_by_kind": {n: 1 - sum(caught[n]) / len(sel) for n in pn},
                "base_probe_catch": sum(caught["base_probe"]) / len(sel),
                "heldout_any_catch": sum(any(caught[n][i] for n in hn)
                                         for i in range(len(sel))) / len(sel),
                "heldout_catch_by_probe": {n: sum(caught[n]) / len(sel) for n in hn},
                "median_tokens": statistics.median(len(tokz.encode(o)) for o in outs),
                "finished": sum(fin) / len(fin)}
            print(key, json.dumps({k: v for k, v in agg[key].items()
                                   if not isinstance(v, dict)}), flush=True)
    Path(a.raw).parent.mkdir(parents=True, exist_ok=True)
    Path(a.raw).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": a.config, "adapter": a.adapter, "concept": concept,
                               "panel_kinds": kinds, "panel_fit": sorted(panel),
                               "heldout": sorted(held), "conditions": agg,
                               "sec": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
