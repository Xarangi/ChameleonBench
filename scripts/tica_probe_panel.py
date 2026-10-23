"""PROBE PANEL race (REPLICATION 2026-10-18d): the frozen-checkpoint refit race of
tica_switch_eval.py (training rollout, branch-refit at the turn after a pass, 1-bit verdicts), with
every turn ALSO scored by a fixed panel that is never shown to the model (tica/panel.py):

(a) base probes, fit on the BASE model: linear at --panel-layers (cached answer-format probes, the
    same construction as the race's base probe), and --panel-kinds (any probe_zoo kind, e.g.
    gdm_rollattn, mck_attnlite, nc_ens8_tok, maxpool@24), fit by the episode-refit machinery on
    base-model forks of turn 0 (the turn-0 fork-first construction, adapters disabled);
(b) every refit of the episode (deployed or later), on every turn of its concept;
(c) --read-every N: at every N-th turn, a fresh linear probe per --read-layers layer fit on forks of
    THAT turn (concept vs other-concept / concept-free answers after the same history), scored by
    k-fold CV AUROC: is the concept still linearly readable (evade) or gone (erase)?

Outputs (aggregates + per-turn numeric records; no texts): --out JSON. --base = the base model.

  python scripts/tica_probe_panel.py --config final_v3b_4b --adapter DIR --topics heldout --out O
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.adaptive import monitor_input
from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.data import CONCEPT_SPLITS, select_topics
from next_chameleons.tica.episodes import CONDITION_SPECS, EpisodeSampler
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.neg_labels import clean as neg_clean
from next_chameleons.tica.neg_labels import settings as neg_label_settings
from next_chameleons.tica.panel import DODGE, aggregate, cv_auroc, refit_events, selectivity
from next_chameleons.tica.probe_zoo import kind_layers, split_kind
from next_chameleons.tica.trainers.supervised import probe_index_metrics

DEFAULT_LAYERS = "4,8,12,16,20,22,24,26,28,30"
DEFAULT_KINDS = ("maxpool@24,attnpool,gdm_rollattn,mck_attnlite,ens8_linear,ensdeep4_linear,"
                 "spread5_linear,nc_ens8_tok")


def margin(mon, pooled, row, c):
    raw = mon.logits(monitor_input(mon, pooled, row), [c])[0]
    thr = mon.thresholds([c])[0]
    return float(raw - thr), bool(raw > thr)


def _reread_det(records):
    """--base-reread: per panel probe, detection on monitored dodging turns when the BASE model
    reads the policy's transcript, next to the policy's own detection on the same turns."""
    on = [r for r in records if "panel_base_reread" in r and not r["off"] and r["cond"] in DODGE]
    out = {}
    for p in sorted({p for r in on for p in r["panel_base_reread"]}):
        rs = [r for r in on if p in r["panel"] and p in r["panel_base_reread"]]
        if rs:
            pol_det = sum(float(r["panel"][p][1]) for r in rs) / len(rs)
            base_det = sum(float(r["panel_base_reread"][p][1]) for r in rs) / len(rs)
            out[p] = {"det_policy": pol_det, "det_base_reads_text": base_det, "n": len(rs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--topics", default="heldout")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--switch-prob", type=float, default=0.0)
    ap.add_argument("--min-seg", type=int, default=4)
    ap.add_argument("--mix", default="real_score:1.0",
                    help="episode conditions, e.g. real_score:0.6,mismatch:0.2,nobrief:0.2 "
                         "(selectivity: is an UNMONITORED concept hidden too?)")
    ap.add_argument("--kind", default="mixed", help="race refit kind (default: the config mix)")
    ap.add_argument("--panel-layers", default=DEFAULT_LAYERS)
    ap.add_argument("--panel-kinds", default=DEFAULT_KINDS)
    ap.add_argument("--read-every", type=int, default=2, help="readability every N turns (0 off)")
    ap.add_argument("--read-layers", default="8,12,16,20,24,28")
    ap.add_argument("--read-pos", type=int, default=40)
    ap.add_argument("--read-neg", type=int, default=56)
    ap.add_argument("--max-topics", type=int, default=0, help="random subset of the topics")
    ap.add_argument("--dump-feats", help="save per-turn pooled features (.pt) for "
                    "scripts/tica_evasion_detector.py")
    ap.add_argument("--feat-layers", default="8,12,16,20,24")
    ap.add_argument("--base-reread", action="store_true",
                    help="also score the panel on the BASE model reading the policy's exact "
                         "transcripts (text-only control: is a catch carried by the tokens?)")
    ap.add_argument("--model-path")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
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
    cfg.rl.adaptive.episode_refit_kind = a.kind
    data, _, factory = build(cfg)
    # "train+heldout": the union of several concept splits (evasion detector: train / test)
    topics = [c for sp in a.topics.split("+") for c in select_topics(data, sp)]
    if a.max_topics and len(topics) > a.max_topics:  # fewer concepts = fewer panel fits
        topics = sorted(random.Random(a.seed).sample(sorted(topics), a.max_topics))
    train = set(cfg.episode_concepts)
    if a.base and a.adapter:
        raise SystemExit("--base and --adapter are exclusive")
    pol = factory(None if a.base else a.adapter, True)
    opening, _ = fmt = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    held = [int(x) for x in cfg.probes.held_out_layers]
    prim = cfg.primary_layer()
    layers = [prim] + [h for h in held if h != prim]
    pan_layers = [int(x) for x in a.panel_layers.split(",") if x]
    pool, panel = MonitorPool(), {}  # panel: concept -> {name: monitor}
    failed = {}
    for k, c in enumerate(list(topics)):
        neg = ("train",) if c in train else None
        try:
            mons, _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening,
                                        neg_splits=neg, rename=f"answer_base_c{k}")
        except RuntimeError as err:
            failed[c] = str(err)[-160:]
            topics.remove(c)
            continue
        for L, m in zip(layers, mons, strict=False):
            pool.add(m, round=0, family="concept_linear", held_out=L in held)
        panel[c] = {}
        if not pan_layers:
            continue
        # (a) linear base probes at every panel depth: one multi-layer read (cached); per layer
        # only if some layer fails its gate
        try:
            pm, _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, pan_layers, opening,
                                      neg_splits=neg, rename=f"panel_c{k}")
            got = dict(zip(pan_layers, pm, strict=True))
        except RuntimeError:
            got = {}
            for L in pan_layers:
                try:
                    (m,), _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, [L], opening,
                                                neg_splits=neg, rename=f"panel_c{k}_L{L}")
                    got[L] = m
                except RuntimeError as err:
                    failed[f"{c} linear@{L}"] = str(err)[-120:]
        for L, m in got.items():
            panel[c][f"base:linear@{L}"] = m.to(pol.device)
    pool = pool.to(pol.device)
    sampler = EpisodeSampler(
        data.concepts("episode", CONCEPT_SPLITS, only=set(topics)), data.negatives("episode"),
        pool, mix={k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))},
        turns=a.turns, seed=a.seed,
        feedback=cfg.episodes.feedback, switch_prob=a.switch_prob, switch_min_segment=a.min_seg)
    tcfg = dataclasses.replace(cfg.train, turns=a.turns, episodes_per_step=a.batch, seed=a.seed)
    asc = AdaptiveSupervisedConfig(
        **{f.name: getattr(tcfg, f.name) for f in dataclasses.fields(SupervisedConfig)},
        adaptive=cfg.rl.adaptive)
    out = Path(a.out)
    from next_chameleons.tica.vllm_gen import maybe_attach

    maybe_attach(pol, cfg.model.path, out.parent / (out.stem + "_vllm"))  # TICA_VLLM=1: colocated vLLM generation
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, asc, out.parent / (out.stem + "_work"),
        branch_concepts=data.concepts("fit", CONCEPT_SPLITS),
        branch_negatives=data.negatives("fit"), roots_only=set(topics), prompts=fmt,
        hard_neg={c: data.hard_negatives(c, "fit") for c in topics})
    tr._heldout_layers = lambda: set()  # eval: refits may sit at held-out layers
    pol.model.eval()  # KV-cached reads (grad checkpointing disables the cache in train mode)
    acfg = cfg.rl.adaptive
    t0 = time.time()
    # (a) zoo kinds on the BASE model: turn-0 forks, adapters off; own rng (race draws unchanged)
    race_rng = tr.rng
    tr.rng = random.Random(a.seed + 7)
    for c in topics:
        spec = SimpleNamespace(monitored=c, cond="real_score", feedback=cfg.episodes.feedback,
                               named=True)
        for kind in [k for k in a.panel_kinds.split(",") if k]:
            with pol.reference(), torch.no_grad():
                try:
                    mon, info = tr._episode_refit(
                        spec, [], None, c, None, kind=kind,
                        fork_msg=lambda x, c=c: tr.opening("real_score", x, c, named=True))
                except (RuntimeError, ValueError) as err:
                    mon, info = None, {"why": str(err)[-120:]}
            if mon is None:
                failed[f"{c} {kind}"] = info.get("why")
                continue
            panel[c][f"base:{kind}"] = mon.to(pol.device)
    tr.rng = race_rng
    tr._np_cache, tr._np_ver = {}, None  # base-weight reads must not serve the policy's refits
    print(json.dumps({"panel_fit_sec": round(time.time() - t0),
                      "panel_sizes": {c: len(v) for c, v in panel.items()},
                      "failed": len(failed)}), flush=True)
    # every layer / token layer any panel probe reads
    need = set(pan_layers)
    feat_layers = [int(x) for x in a.feat_layers.split(",") if x] if a.dump_feats else []
    need |= set(feat_layers)
    feats: list[torch.Tensor] = []  # per record: [n_feat_layers, d] fp16 (--dump-feats)
    tok_layers = set()
    for ms in panel.values():
        for m in ms.values():
            lu = getattr(m, "layers_used", None) or [m.layer]
            need |= set(lu)
            if getattr(m, "needs_tokens", False):
                tok_layers |= set(lu)
    read_layers = [int(x) for x in a.read_layers.split(",") if x]
    # forks for readability: render this turn's user message for another answer x
    rendered: dict[str, tuple] = {}
    orig_render = tr.render_feedback

    def recording(*args, **kw):
        s = orig_render(*args, **kw)
        rendered[s] = (args, kw)
        return s

    recording.tagged = getattr(orig_render, "tagged", False)
    tr.render_feedback = recording
    texts = {cc.concept: cc.texts for cc in tr.branch_concepts}
    rr = random.Random(a.seed + 11)
    neg_cache: dict = {}

    def negatives_for(c):
        if c not in neg_cache:
            if neg_label_settings(acfg) is not None:
                neg_cache[c] = ({k: neg_clean(v, c, acfg) for k, v in texts.items() if k != c},
                                neg_clean(tr.branch_negatives, c, acfg))
            else:
                neg_cache[c] = ({k: v for k, v in texts.items() if k != c}, tr.branch_negatives)
        return neg_cache[c]

    def readability(trj, t, c):
        if t == 0:
            pre = []
            msg = lambda x: tr.opening("real_score", x, c, named=trj.spec.named)  # noqa: E731
        else:
            got = rendered.get(trj.feedback[t - 1])
            if got is None:
                return None
            args, kw = got
            pre = [dict(m) for m in trj.convs[t - 1]]
            msg = lambda x: orig_render(*args[:3], x, *args[4:], **kw)  # noqa: E731
        others, free = negatives_for(c)
        oc = [k for k, v in others.items() if v]
        pos = rr.sample(texts[c], min(a.read_pos, len(texts[c])))
        negs = ([rr.choice(others[rr.choice(oc)]) for _ in range(a.read_neg * 3 // 4)]
                + rr.sample(free, min(a.read_neg // 4, len(free))))
        forks = [[*pre, {"role": "user", "content": msg(x)}, {"role": "assistant", "content": x}]
                 for x in pos + negs]
        y = torch.tensor([1] * len(pos) + [0] * len(negs))
        reads = pol.pooled_shared_prefix(forks, read_layers,
                                         batch=acfg.episode_fork_batch or None)
        res = {}
        for L in read_layers:
            try:
                res[str(L)] = cv_auroc(reads[L].float(), y, C=acfg.refit_C, seed=t)
            except RuntimeError:
                res[str(L)] = None
        return res

    records, all_trajs, ep0 = [], [], 0
    for b in range(0, a.episodes, a.batch):
        specs = sampler.sample(min(a.batch, a.episodes - b))
        ctx = pol.reference() if a.base else torch.no_grad()
        mctx = (pol.merged() if acfg.merged_reads and not a.base else contextlib.nullcontext())
        with ctx, mctx, torch.no_grad():
            trajs, _ = tr.rollout(specs)
            all_trajs.extend(trajs)
            refit_mon = {e.monitor.name: e.monitor for e in tr.pool.entries
                         if e.monitor.name in set(tr._transient)}
            evs = [refit_events(trj.results) for trj in trajs]
            b_need, b_tok = set(need), set(tok_layers)  # + the layers this batch's refits read
            for m in refit_mon.values():
                lu = getattr(m, "layers_used", None) or [m.layer]
                b_need |= set(lu)
                if getattr(m, "needs_tokens", False):
                    b_tok |= set(lu)
            state: dict = {}
            state_b: dict = {}  # --base-reread: the base model's own cache over the same convs
            for t in range(a.turns):
                tok: dict = {}
                pooled = pol.pooled_incremental([trj.convs[t] for trj in trajs],
                                                sorted(b_need | b_tok), state,
                                                token_out=tok if b_tok else None)
                for L in b_tok:
                    pooled[("tok", L)] = tok[L]
                pooled_b = None
                if a.base_reread and not a.base:
                    tok_b: dict = {}
                    with pol.reference():
                        pooled_b = pol.pooled_incremental([trj.convs[t] for trj in trajs],
                                                          sorted(need | tok_layers), state_b,
                                                          token_out=tok_b if tok_layers else None)
                    for L in tok_layers:
                        pooled_b[("tok", L)] = tok_b[L]
                for i, trj in enumerate(trajs):
                    r = trj.results[t]
                    cond = trj.spec.cond
                    c = r.get("concept", trj.spec.monitored_at(t))
                    tc = trj.spec.text_concept  # mismatch: the written (unmonitored) concept B
                    dodge = cond in DODGE
                    pc = c if dodge else tc
                    rec = {"ep": ep0 + i, "turn": t, "cond": cond, "concept": c,
                           "text_concept": tc, "off": bool(r.get("off")),
                           "detected": bool(r["monitored"]["detected"]),
                           "active": r.get("active_monitor"),
                           "panel": ({p: margin(m, pooled, i, pc) for p, m in panel[pc].items()}
                                     if pc in panel else {}),
                           "refits": {}, "refit_meta": {}}
                    if pooled_b is not None and pc in panel:
                        rec["panel_base_reread"] = {p: margin(m, pooled_b, i, pc)
                                                    for p, m in panel[pc].items()}
                    if cond == "mismatch" and c in panel:  # A's panel on B's text
                        rec["false_alarm"] = {p: margin(m, pooled, i, c)
                                              for p, m in panel[c].items()}
                    for n, e in evs[i].items():
                        m = refit_mon.get(n)
                        if m is None or not m.covers(c):
                            continue
                        rec["refits"][n] = margin(m, pooled, i, c)
                        rec["refit_meta"][n] = {**e, "kind": tr._kind_of.get(n)}
                    if (a.read_every and t % a.read_every == 0 and not rec["off"] and dodge
                            and CONDITION_SPECS[cond]["brief"]):
                        rec["readability"] = readability(trj, t, c)
                    if feat_layers:
                        rec["split"] = data.split.get(tc or c)
                        feats.append(torch.stack([pooled[L][i].float().cpu()
                                                  for L in feat_layers]).half())
                    records.append(rec)
                del pooled, tok, pooled_b
        ep0 += len(trajs)
        tr.pool.entries = [e for e in tr.pool.entries if e.monitor.name not in set(tr._transient)]
        tr._transient = []
        rendered.clear()
        print(f"[batch {b // a.batch}] episodes {ep0} {time.time() - t0:.0f}s", flush=True)
    agg = aggregate(records, a.turns)
    rep = {"config": a.config, "adapter": a.adapter, "base": a.base, "topics": topics,
           "episodes": a.episodes, "turns": a.turns, "kind": a.kind,
           "panel_names": sorted({p for v in panel.values() for p in v}),
           "panel_failed": failed, "by_probe_index": probe_index_metrics(all_trajs),
           "aggregate": agg, "selectivity": selectivity(records, a.turns),
           "base_reread": _reread_det(records) if a.base_reread else None,
           "records": records, "sec": round(time.time() - t0),
           "kind_layers": {k: kind_layers(split_kind(k)[0], split_kind(k)[1] or prim)
                           for k in a.panel_kinds.split(",") if k}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep))
    if feat_layers:  # activations + numeric labels only (evasion detector; never committed)
        meta = [{k: r[k] for k in ("ep", "turn", "cond", "off", "concept", "text_concept",
                                   "split", "detected")} for r in records]
        torch.save({"layers": feat_layers, "x": torch.stack(feats), "meta": meta,
                    "adapter": a.adapter, "base": a.base}, a.dump_feats)
    short = {k: v for k, v in agg.items() if k not in ("panel", "layer_profile_by_turn",
                                                        "readability_by_turn")}
    short["panel_det"] = {p: v["det"] for p, v in agg["panel"].items()}
    short["selectivity"] = {p: {k: v[k] for k in ("real", "mismatch", "nobrief")}
                            for p, v in rep["selectivity"]["text_probe"].items()}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
