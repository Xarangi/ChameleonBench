"""Readout-form check (REPLICATION 2026-10-12c): does reading each turn in its NATIVE
finished-message form, from ONE forward pass over the whole conversation, give the same probe
verdicts as the current per-turn VIEW (generation prompt with the empty think block + content,
one forward per turn)?

Forms, per assistant turn t of a 15-turn episode:
  V  current view: render(conv[:t]) with the generation prompt + content_t          (1 pass / turn)
  F  finished form: native render of the conversation, prefix up to the end of content_t
  C  single pass: ONE forward over the whole native-rendered conversation, every turn's span pooled
C == F up to numerics by causality (the exactness check); V vs C is the template question.
Aggregates only (no text is printed). Usage:
  python scripts/tica_readout_form.py --config l3arms_math_4b --adapter DIR [--model-path DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.episodes import EpisodeSpec, rollout
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.policy import View

MARK = "<|im_start|>assistant"


def segments(pol, conv, keep_think: bool):
    """keep_think=False: native finished render. keep_think=True: every assistant turn keeps the
    empty think block the generation prompt uses, so turn t's local lead-in equals the view's."""
    if not keep_think:
        return native_segments(pol, conv)
    from next_chameleons.tica.policy import render_keep_reasoning

    conv2 = [dict(m, reasoning="") if m["role"] == "assistant" else m for m in conv]
    full = render_keep_reasoning([*conv2, {"role": "user", "content": "ok"}], False, False)
    return _segment(pol, full, conv)


def _segment(pol, full, conv):
    ids, spans, pos = [], [], 0
    for m in conv:
        if m["role"] != "assistant":
            continue
        a = full.find(MARK, pos)
        k = full.find(m["content"].strip(), a)
        assert a >= 0 and k >= 0, "assistant turn not found in render"
        ids += pol.encode(full[pos:k])
        s = len(ids)
        ids += pol.encode(m["content"].strip())
        spans.append((s, len(ids)))
        pos = k + len(m["content"].strip())
    ids += pol.encode(full[pos:])
    return ids, spans


def native_segments(pol, conv):
    """Segment-wise tokenisation of the native finished render (a trailing user turn keeps the
    last assistant turn in history form). Returns ids and the content span of every assistant
    turn, located after its own assistant marker (feedback messages quote the next text)."""
    full = pol.render([*conv, {"role": "user", "content": "ok"}], False)
    ids, spans, pos = [], [], 0
    for m in conv:
        if m["role"] != "assistant":
            continue
        a = full.find(MARK, pos)
        k = full.find(m["content"].strip(), a)
        assert a >= 0 and k >= 0, "assistant turn not found in native render"
        ids += pol.encode(full[pos:k])
        s = len(ids)
        ids += pol.encode(m["content"].strip())
        spans.append((s, len(ids)))
        pos = k + len(m["content"].strip())
    ids += pol.encode(full[pos:])
    return ids, spans


@torch.no_grad()
def single_pass(pol, ids, spans, layers):
    store = {}
    x = torch.tensor([ids], device=pol.device)
    with pol._hooks(layers, store, keep_grad=False):
        pol.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                  logits_to_keep=1)
    return {L: torch.stack([store[L][0, s:e].float().mean(0) for s, e in spans]) for L in layers}


@torch.no_grad()
def batched(pol, views, layers):
    out = {L: [] for L in layers}
    for i in range(0, len(views), pol.batch_size):
        p, _ = pol.forward(views[i : i + pol.batch_size], layers, grad=False)
        for L in layers:
            out[L].append(p[L])
    return {L: torch.cat(v) for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--model-path")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--out")
    ap.add_argument("--keep-think", action="store_true", help="C = keep-think history form")
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, default_pool, factory = build(cfg)
    pool = MonitorPool.load(cfg.start_pool) if cfg.start_pool else default_pool
    for e in default_pool.entries:
        if e.held_out and not any(x.monitor.name == e.monitor.name for x in pool.entries):
            pool.add(e.monitor, round=e.round, family=e.family, held_out=True)
    pol = factory(a.adapter, False)
    pool = pool.to(pol.device)
    concept = cfg.episode_concepts[0]
    texts = data.concepts("episode", ("train",), only={concept})[0].texts  # held-out texts
    base = pool.concept_monitor(concept)
    rng = random.Random(0)
    specs = [
        EpisodeSpec(f"{cond}|{k}", cond, rng.sample(texts, a.turns), concept, concept, base.name,
                    feedback=cfg.episodes.feedback)
        for cond in ("real_score", "nobrief") for k in range(a.episodes)
    ]
    mons = [e.monitor for e in pool.entries if e.monitor.covers(concept)]
    other = [m for m in mons if m.name != base.name and m.layer != base.layer][:1]
    refits = [m for m in mons if m.name != base.name and m.layer == base.layer][-3:]
    mons = [base, *other, *refits]
    layers = sorted({m.layer for m in mons})
    report = {"keep_think": a.keep_think, "adapter": a.adapter,
              "concept_split": "episode(held-out)", "forms": {}}
    for label, ctx in (("adapter", None), ("base", "ref")):
        if label == "adapter" and not a.adapter:
            continue
        cm = pol.reference() if ctx else torch.no_grad()
        with cm:
            trajs = rollout(pol, pool, specs, a.turns)
            rows = []  # (cond, turn, V acts, C acts, F acts)
            for tr in trajs:
                final = tr.convs[-1]
                ids, spans = segments(pol, final, a.keep_think)
                C = single_pass(pol, ids, spans, layers)
                vviews = [pol.view(c) for c in tr.convs]
                fviews = [View(ids[: e + 1], s, 1) for s, e in spans]
                V, F = batched(pol, vviews, layers), batched(pol, fviews, layers)
                for t in range(len(spans)):
                    rows.append((tr.spec.cond, t, {L: V[L][t] for L in layers},
                                 {L: C[L][t] for L in layers}, {L: F[L][t] for L in layers}))
        res = {}
        for L in layers:
            v = torch.stack([r[2][L] for r in rows])
            c = torch.stack([r[3][L] for r in rows])
            f = torch.stack([r[4][L] for r in rows])
            res[f"L{L}"] = {
                "cos_V_C_mean": float(torch.nn.functional.cosine_similarity(v, c).mean()),
                "relL2_V_C_mean": float(((v - c).norm(dim=1) / v.norm(dim=1)).mean()),
                "relL2_C_F_max": float(((c - f).norm(dim=1) / f.norm(dim=1)).max()),
            }
        for m in mons:
            v = torch.stack([r[2][m.layer] for r in rows])
            c = torch.stack([r[3][m.layer] for r in rows])
            n = len(rows)
            lv = m.logits(v.to(pol.device), [concept] * n).float().cpu()
            lc = m.logits(c.to(pol.device), [concept] * n).float().cpu()
            thr = m.thresholds([concept] * n).float().cpu()
            d = lc - lv
            per = {}
            for cond in ("real_score", "nobrief"):
                ix = torch.tensor([r[0] == cond for r in rows])
                per[cond] = {
                    "det_V": float((lv[ix] > thr[ix]).float().mean()),
                    "det_C": float((lc[ix] > thr[ix]).float().mean()),
                    "verdict_agree": float(
                        ((lv[ix] > thr[ix]) == (lc[ix] > thr[ix])).float().mean()
                    ),
                }
            res[m.name] = {
                "layer": m.layer,
                "mean_abs_dlogit": float(d.abs().mean()),
                "mean_dlogit": float(d.mean()),
                "logit_sd_V": float(lv.std()),
                "dlogit_over_sd": float(d.abs().mean() / lv.std().clamp(min=1e-6)),
                "mean_abs_margin_V": float((lv - thr).abs().mean()),
                "verdict_agree": float(((lv > thr) == (lc > thr)).float().mean()),
                "by_cond": per,
            }
        report["forms"][label] = res
    js = json.dumps(report, indent=1)
    print(js)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(js)


if __name__ == "__main__":
    main()
