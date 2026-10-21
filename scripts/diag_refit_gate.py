"""Why do episode refits on the trained model fail the quality gate? (REPLICATION 2026-10-21g)

Diagnostic only (no trainer changes). Two stages:

  collect  build the AdaptiveSupervisedTrainer exactly as `scripts/tica.py adaptive` does (trained
           adapter + its trainer_state), run a few teacher-forced rollouts with `_episode_refit`
           wrapped to record every post-pass refit prefix (and the trainer's own gate outcome), then
           rebuild each refit's fork set WITH labels and read mean-pooled states at LAYERS, once
           with the trained weights and once under pol.reference() (base weights, same prefixes).
           Groups per refit (texts drawn like `_episode_refit`):
             ic      256 in-context math forks (prefix + feedback msg + fixed math text)
             np      96 no-prompt math conversations (fresh chat, no brief), disjoint from ic
             negA    160 other-concept (half same-family) + 128 concept-free in-context forks
                     (= the config's negative set)
             negB    a second, independent draw of the same size (extra held-out negatives)
             npneg   48 other-concept + 48 concept-free no-prompt conversations
  analyze  linear-probe measurements on the saved states (GPU if available).

Outputs go to --out (scratch only; states and per-refit metadata never committed; no raw text
is written anywhere, texts are referenced by answer_format.text_id).

Run from a snapshot of the committed source (the trainer files may be mid-edit), on one A100
(jobs 6603899 / 6603906; ~1 h each):
  git archive HEAD src configs scripts/tica.py | tar -x -C <snap>; cp this file <snap>/scripts/
  export TICA_STEP_OFFSET=500   # 380 for resume_60/adaptive/step_320
  cd <snap>; PYTHONPATH=src python scripts/diag_refit_gate.py collect --root . --out <out> \
    --adapter <run>/resume_400/adaptive/step_100 --pool <run>/resume_400/answer_pool \
    --model-path <local snapshot copy> --rollouts 8 --max-refits 36
  PYTHONPATH=src python scripts/diag_refit_gate.py analyze --out <out> --seeds 10
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib.util
import json
import os
import random
import time
import zlib
from pathlib import Path

import torch

LAYERS = [8, 12, 16, 22, 26]
N_IC, N_NP, N_OTHER, N_FREE, N_NPNEG = 256, 96, 160, 128, 48


# ----------------------------------------------------------------------------------------- collect
def _tica_cli(root: Path):
    spec = importlib.util.spec_from_file_location("tica_cli", root / "scripts" / "tica.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _neg_draws(rng, texts, others, near, n, frac):
    """other_negatives() with the source concept kept: round(n*frac) from same-family concepts."""

    n_near = round(n * frac) if near else 0
    out = []
    for j in range(n):
        c = rng.choice(near if j < n_near else others)
        out.append((c, rng.choice(texts[c])))
    return out


def collect(args):
    from next_chameleons.tica.answer_format import load_inputs, text_id
    from next_chameleons.tica.answer_format import prompts as fmt_prompts
    from next_chameleons.tica.config import apply_overrides, load_run_config
    from next_chameleons.tica.episodes import EpisodeSampler
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.neg_labels import _labels
    from next_chameleons.tica.neg_labels import clean as neg_clean
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
        SupervisedConfig,
    )

    out = Path(args.out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    cli = _tica_cli(Path(args.root))
    cfg = load_run_config(args.config)
    if args.model_path:
        cfg.model.path = args.model_path
    if args.set:
        apply_overrides(cfg, args.set)
    data, _, factory = cli.build(cfg)
    pol = factory(args.adapter, True)
    prompts = fmt_prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    pool = MonitorPool.load(args.pool).to(pol.device)
    only = set(cfg.episode_concepts)
    offset = int(os.environ.get("TICA_STEP_OFFSET", "0") or 0)
    sampler = EpisodeSampler(data.concepts("fit", ("train",), only=only), data.negatives("fit"),
                             pool, mix=cfg.train_mix, turns=cfg.train.turns,
                             seed=cfg.train.seed + offset + 7919, **cfg.sampler_kwargs())
    asc = AdaptiveSupervisedConfig(
        **{f.name: getattr(cfg.train, f.name) for f in dataclasses.fields(SupervisedConfig)},
        adaptive=cfg.rl.adaptive)
    st = Path(args.adapter) / "trainer_state.pt"
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, asc, out / "trainer", branch_concepts=data.concepts("fit", ("train",)),
        branch_negatives=data.negatives("fit"), roots_only=only, prompts=prompts,
        inputs=load_inputs(cfg.episodes.inputs), resume_state=str(st) if st.exists() else None)

    recs: list[dict] = []
    orig = AdaptiveSupervisedTrainer._episode_refit
    roll = {"k": 0}

    def wrapped(self, s, prefix, shown, concept=None, turn=None, kind=None, fork_msg=None,
                **kw):
        snap = copy.deepcopy(prefix)
        bank, info = orig(self, s, prefix, shown, concept, turn, kind, fork_msg=fork_msg, **kw)
        if fork_msg is None and turn is not None and prefix:  # post-pass refit (not turn 0)
            key = (roll["k"], id(s), turn)
            att = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}
            att["kept"] = bank is not None
            for r in recs:
                if r["key"] == key:
                    r["attempts"].append(att)
                    break
            else:
                recs.append({"key": key, "s": s, "prefix": snap, "shown": shown,
                             "concept": concept or s.monitored, "turn": turn, "attempts": [att]})
        return bank, info

    AdaptiveSupervisedTrainer._episode_refit = wrapped
    pol.model.eval()
    t0 = time.time()
    for k in range(args.rollouts):
        roll["k"] = k
        specs = tr.sampler.sample(asc.episodes_per_step)
        with pol.merged():
            tr.rollout(specs)
        if tr._transient:  # as in step(): episode-local refits are dropped after the rollout
            gone = set(tr._transient)
            tr.pool.entries = [e for e in tr.pool.entries if e.monitor.name not in gone]
            tr._transient = []
        print(json.dumps({"rollout": k, "prefixes": len(recs), "sec": round(time.time() - t0)}),
              flush=True)
        if len(recs) >= args.max_refits:
            break
    AdaptiveSupervisedTrainer._episode_refit = orig
    if len(recs) > args.max_refits:  # spread over rollouts / turns
        idx = torch.linspace(0, len(recs) - 1, args.max_refits).round().long().tolist()
        recs = [recs[i] for i in sorted(set(idx))]

    acfg = tr.cfg.adaptive
    lab = _labels(acfg.neg_labels) if acfg.neg_labels else {}
    texts_all = {c.concept: c.texts for c in tr.branch_concepts}
    fam = tr._families()
    meta_all = []
    for j, rec in enumerate(recs):
        s, c = rec["s"], rec["concept"]
        rng = random.Random(f"diag|{j}")
        cl = {cc: neg_clean(ts, c, acfg) for cc, ts in texts_all.items() if cc != c}
        free = neg_clean(tr.branch_negatives, c, acfg)
        others = [cc for cc in cl if cl[cc]]
        near = [cc for cc in others if fam.get(c) and fam.get(cc) == fam[c]]
        pos_t = texts_all[c] + tr._extra_pos(c)
        draw = rng.sample(pos_t, N_IC + N_NP)
        ic, npt = draw[:N_IC], draw[N_IC:]
        frac = acfg.family_neg_frac
        negA = (_neg_draws(rng, cl, others, near, N_OTHER, frac)
                + [("free", x) for x in rng.sample(free, N_FREE)])
        negB = (_neg_draws(rng, cl, others, near, N_OTHER, frac)
                + [("free", x) for x in rng.sample(free, N_FREE)])
        npneg = (_neg_draws(rng, cl, others, near, N_NPNEG, 0.0)
                 + [("free", x) for x in rng.sample(free, N_NPNEG)])
        pre = [dict(m) for m in rec["prefix"]]

        def fork(x, s=s, c=c, pre=pre, shown=rec["shown"]):
            return [*pre, {"role": "user", "content": tr.render_feedback(
                s.cond, c, shown, x, feedback=s.feedback, named=s.named)},
                {"role": "assistant", "content": x}]

        def fresh(x, s=s, c=c):
            return [{"role": "user", "content": tr.opening("nobrief", x, c, named=s.named)},
                    {"role": "assistant", "content": x}]

        groups = {"ic": [fork(x) for x in ic], "negA": [fork(x) for _, x in negA],
                  "negB": [fork(x) for _, x in negB]}
        fresh_g = {"np": [fresh(x) for x in npt], "npneg": [fresh(x) for _, x in npneg]}
        order = [g for g in groups for _ in groups[g]]
        forks = [f for g in groups for f in groups[g]]
        fconv = [f for g in fresh_g for f in fresh_g[g]]
        forder = [g for g in fresh_g for _ in fresh_g[g]]
        saved = {}
        for model in ("trained", "base"):
            t1 = time.time()
            if model == "trained":
                with pol.merged():
                    R = pol.pooled_shared_prefix(forks, LAYERS, batch=args.batch)
                    F = pol.pooled(fconv, LAYERS, batch=args.batch)
            else:
                with pol.reference():
                    R = pol.pooled_shared_prefix(forks, LAYERS, batch=args.batch)
                    F = pol.pooled(fconv, LAYERS, batch=args.batch)
            st_ = {}
            for name, rows, src in (("ic", order, R), ("negA", order, R), ("negB", order, R),
                                    ("np", forder, F), ("npneg", forder, F)):
                q = [i for i, g in enumerate(rows) if g == name]
                st_[name] = torch.stack([src[L][q].float().cpu() for L in LAYERS], 1).half()
            saved[model] = st_
            print(json.dumps({"refit": j, "model": model, "rows": len(forks) + len(fconv),
                              "sec": round(time.time() - t1, 1)}), flush=True)
        torch.save(saved, out / "states" / f"refit_{j:03d}.pt")

        def m_(pairs, c=c):
            return [{"src": a, "id": text_id(x), "p_math": lab.get(text_id(x), {}).get(c)}
                    for a, x in pairs]

        meta_all.append({
            "j": j, "rollout": rec["key"][0], "turn": rec["turn"], "cond": s.cond,
            "prefix_msgs": len(pre), "attempts": rec["attempts"],
            "negA": m_(negA), "negB": m_(negB), "npneg": m_(npneg),
            "ic": [text_id(x) for x in ic], "np": [text_id(x) for x in npt]})
        (out / "meta.json").write_text(json.dumps(meta_all))
    print(json.dumps({"collected": len(meta_all), "sec": round(time.time() - t0)}), flush=True)


# ----------------------------------------------------------------------------------------- analyze
def _logit(w, b, mu, sd, x):
    return ((x.float() - mu) / sd) @ w + b


def _q(x, fpr):
    return float(torch.quantile(x.double(), 1.0 - fpr))


def analyze(args):
    from next_chameleons.tica.adaptive import _split
    from next_chameleons.tica.monitors import auroc, fit_logistic

    out = Path(args.out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((out / "meta.json").read_text())
    rows = []
    t0 = time.time()

    def fit(P, N, C):
        x = torch.cat([P, N]).to(dev)
        y = torch.cat([torch.ones(len(P)), torch.zeros(len(N))])
        w, b, mu, sd = fit_logistic(x, y, C)
        f = lambda z: _logit(w, b, mu, sd, z.to(dev)).cpu()
        f.wr = (w / sd).cpu()  # effective direction in raw activation space
        return f

    for m in meta:
        j = m["j"]
        S = torch.load(out / "states" / f"refit_{j:03d}.pt")
        for model in ("trained", "base"):
            for li, L in enumerate(LAYERS):
                g = {k: v[:, li].float() for k, v in S[model].items()}
                ic, npm, negA, negB, npneg = g["ic"], g["np"], g["negA"], g["negB"], g["npneg"]
                r = {"j": j, "model": model, "layer": L}
                # ---- (0) the gate exactly as fit_refit_bank (cv path), config-sized data
                np32 = npm[:32]
                pos_all = torch.cat([ic[:128], np32[0::2]])
                gate_np = np32[1::2]
                gen = torch.Generator().manual_seed(zlib.crc32(f"diag_{j}_{L}".encode()))
                fn, hn = _split(len(negA), 0.3, gen)
                nf, nh = negA[fn], negA[hn]
                try:
                    sc = fit(pos_all, nf, 1.0)
                    perm = torch.randperm(len(pos_all), generator=gen)
                    wins = pairs = 0.0
                    for k in range(5):
                        o = perm[k::5]
                        keep = torch.ones(len(pos_all), dtype=torch.bool)
                        keep[o] = False
                        scj = fit(pos_all[keep], nf, 1.0)
                        wins += auroc(scj(pos_all[o]), scj(nh)) * len(o) * len(nh)
                        pairs += len(o) * len(nh)
                except RuntimeError as e:
                    r["error"] = str(e)[:120]
                    rows.append(r)
                    continue
                s_nh, s_np, s_npneg = sc(nh), sc(gate_np), sc(npneg)
                s_icx = sc(ic[224:])  # in-context positives never fit (held-out)
                s_negB, s_nf = sc(negB), sc(nf)
                thr = _q(s_nh, 0.01)
                r.update(au_cv=wins / pairs, thr=thr, tpr_np=float((s_np > thr).float().mean()))
                r["pass"] = bool(r["au_cv"] >= 0.95 and r["tpr_np"] >= 0.8)
                r["fail"] = None if r["pass"] else ("auroc" if r["au_cv"] < 0.95 else "tpr_np")
                # ---- (1) score distributions in units of held-out-negative sd
                sdn = float(s_nh.std()) or 1.0
                mn = float(s_nh.mean())
                for k_, v in (("ic_held", s_icx), ("np", s_np), ("npneg", s_npneg),
                              ("negB", s_negB), ("neg_fit", s_nf)):
                    r[f"z_{k_}"] = (float(v.median()) - mn) / sdn
                r["z_thr"] = (thr - mn) / sdn
                r["z_maxneg"] = (float(s_nh.max()) - mn) / sdn
                # ---- (2) thresholds
                allneg = torch.cat([s_nh, s_negB])  # 86 + 288 never-fit in-context negatives
                for k_, t in (("held5", _q(s_nh, 0.05)), ("heldB1", _q(allneg, 0.01)),
                              ("heldB5", _q(allneg, 0.05)), ("negB_max", float(s_negB.max())),
                              ("allfit1", _q(torch.cat([s_nh, s_nf]), 0.01)),
                              ("npneg1", _q(s_npneg, 0.01)), ("npneg5", _q(s_npneg, 0.05))):
                    r[f"tpr_np_{k_}"] = float((s_np > t).float().mean())
                r["tpr_np96_heldB1"] = float((sc(npm) > _q(allneg, 0.01)).float().mean())
                r["tpr_ic_held"] = float((s_icx > thr).float().mean())
                r["fpr_negB_at_thr"] = float((s_negB > thr).float().mean())
                r["fpr_npneg_at_thr"] = float((s_npneg > thr).float().mean())
                # threshold-free: does the in-context probe RANK plain math above plain negatives?
                r["au_np_vs_npneg"] = auroc(s_np, s_npneg)
                r["au_np96_vs_npneg"] = auroc(sc(npm), s_npneg)
                r["au_np_vs_heldneg"] = auroc(s_np, torch.cat([s_nh, s_negB]))
                r["au_ic_vs_heldneg"] = auroc(s_icx, torch.cat([s_nh, s_negB]))
                # ---- (3) top held-out negatives (positions -> metadata)
                negA_meta = m["negA"]
                top = torch.argsort(s_nh, descending=True)[:3].tolist()
                r["top_neg"] = [{**negA_meta[int(hn[t])], "z": (float(s_nh[t]) - mn) / sdn}
                                for t in top]
                # ---- (4) directions + a no-prompt-only probe
                d_ic = ic.mean(0) - torch.cat([negA, negB]).mean(0)
                d_np = npm.mean(0) - npneg.mean(0)
                cs = torch.nn.functional.cosine_similarity
                r["cos_dic_dnp"] = float(cs(d_ic, d_np, dim=0))
                r["cos_w_dnp"] = float(cs(sc.wr, d_np, dim=0))
                r["cos_w_dic"] = float(cs(sc.wr, d_ic, dim=0))
                ctx_neg = npneg.mean(0) - torch.cat([negA, negB]).mean(0)  # context offset
                ctx_pos = npm.mean(0) - ic.mean(0)
                r["norm_ctxneg_over_dic"] = float(ctx_neg.norm() / d_ic.norm())
                r["norm_ctxpos_over_dic"] = float(ctx_pos.norm() / d_ic.norm())
                r["norm_dnp_over_dic"] = float(d_np.norm() / d_ic.norm())
                # shift of math RELATIVE to the context shift of negatives (the math-specific part)
                r["norm_mathshift_over_dic"] = float((ctx_pos - ctx_neg).norm() / d_ic.norm())
                try:
                    s_o = fit(npm[:48], npneg[:48], 1.0)  # fresh-chat probe
                    r["npprobe_au_ic_vs_neg"] = auroc(s_o(ic[224:]), s_o(torch.cat([nh, negB])))
                    r["npprobe_au_np_vs_npneg"] = auroc(s_o(npm[48:]), s_o(npneg[48:]))
                except RuntimeError:
                    pass
                # ---- (5) C and n sweeps (one fit each, fixed negatives nf / thr from nh)
                for C in args.cs:
                    for n in args.ns:
                        if C != 1.0 and n != 128:
                            continue
                        if C == 1.0 and n == 128:
                            sv = sc
                        else:
                            try:
                                sv = fit(torch.cat([ic[:n], np32[0::2]]), nf, C)
                            except RuntimeError:
                                continue
                        t = _q(sv(nh), 0.01)
                        r[f"sw_C{C}_n{n}"] = {
                            "tpr_np": float((sv(npm[32:]) > t).float().mean()),  # 64 gate texts
                            "au_ic": auroc(sv(ic[224:]), sv(torch.cat([nh, negB]))),
                            "au_np_npneg": auroc(sv(npm[32:]), sv(npneg))}
                # np-heavy positives: 128 ic + 48 np (gate on np[48:])
                try:
                    sv = fit(torch.cat([ic[:128], npm[:48]]), nf, 1.0)
                    t = _q(sv(nh), 0.01)
                    r["np48_tpr_np"] = float((sv(npm[48:]) > t).float().mean())
                    r["np48_au_ic"] = auroc(sv(ic[224:]), sv(torch.cat([nh, negB])))
                except RuntimeError:
                    pass
                # ---- (B) gate flips over negative-split seeds (threshold side only)
                flips = []
                for sd_ in range(args.seeds):
                    gg = torch.Generator().manual_seed(1000 + sd_)
                    fn2, hn2 = _split(len(negA), 0.3, gg)
                    try:
                        s2 = fit(pos_all, negA[fn2], 1.0)
                    except RuntimeError:
                        continue
                    flips.append(float((s2(gate_np) > _q(s2(negA[hn2]), 0.01)).float().mean()))
                r["seed_tpr_np"] = flips
                rows.append(r)
            print(json.dumps({"refit": j, "model": model, "sec": round(time.time() - t0)}),
                  flush=True)
        (out / "analysis_rows.json").write_text(json.dumps(rows))
    summarize(out, rows, meta)


def summarize(out: Path, rows, meta):
    import statistics as stt

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(stt.median(xs), 3) if xs else None

    def frac(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    summ = {"n_refits": len(meta)}
    att = [a for m in meta for a in m["attempts"]]
    summ["trainer_attempts"] = {
        "n": len(att), "kept": sum(a["kept"] for a in att),
        "linear": [(a.get("kind"), a["kept"], a.get("auroc"), a.get("tpr_np")) for a in att
                   if str(a.get("kind", "")).startswith("linear")]}
    keys = ["au_cv", "tpr_np", "z_ic_held", "z_np", "z_npneg", "z_negB", "z_neg_fit", "z_thr",
            "z_maxneg", "tpr_np_held5", "tpr_np_heldB1", "tpr_np_heldB5", "tpr_np_negB_max",
            "tpr_np_allfit1", "tpr_np_npneg1", "tpr_np_npneg5", "tpr_np96_heldB1", "tpr_ic_held",
            "fpr_negB_at_thr", "fpr_npneg_at_thr", "au_np_vs_npneg", "au_np96_vs_npneg",
            "au_np_vs_heldneg", "au_ic_vs_heldneg", "cos_dic_dnp", "cos_w_dnp", "cos_w_dic",
            "norm_ctxneg_over_dic", "norm_ctxpos_over_dic", "norm_dnp_over_dic",
            "norm_mathshift_over_dic", "npprobe_au_ic_vs_neg", "npprobe_au_np_vs_npneg",
            "np48_tpr_np", "np48_au_ic"]
    for model in ("trained", "base"):
        for L in [*LAYERS, "all"]:
            rs = [r for r in rows if r["model"] == model and (L == "all" or r["layer"] == L)
                  and "error" not in r]
            if not rs:
                continue
            d = {"n": len(rs), "pass": frac([r["pass"] for r in rs]),
                 "fail_tpr_np": frac([r["fail"] == "tpr_np" for r in rs]),
                 "fail_auroc": frac([r["fail"] == "auroc" for r in rs]),
                 "errors": sum("error" in r for r in rows if r["model"] == model)}
            for k in keys:
                d[k] = med([r.get(k) for r in rs])
            # gate flip rate over negative-split seeds: frac of (refit, seed) passing tpr_np
            sp = [x >= 0.8 for r in rs for x in r["seed_tpr_np"]]
            d["seed_tpr_np_pass_frac"] = frac(sp)
            per = [sum(x >= 0.8 for x in r["seed_tpr_np"]) / max(1, len(r["seed_tpr_np"]))
                   for r in rs]
            d["seed_mixed_frac"] = frac([0 < p < 1 for p in per])
            for sk in sorted({k for r in rs for k in r if k.startswith("sw_")}):
                d[sk] = {m_: med([r[sk][m_] for r in rs if sk in r])
                         for m_ in ("tpr_np", "au_ic", "au_np_npneg")}
            # among gate failures (tpr_np): would a better threshold / ranking detect plain math?
            ft = [r for r in rs if r["fail"] == "tpr_np"]
            if ft:
                d["tprfail"] = {k: med([r.get(k) for r in ft]) for k in (
                    "tpr_np", "au_np_vs_npneg", "au_np96_vs_npneg", "tpr_np_npneg1",
                    "tpr_np_heldB1", "z_np", "z_npneg", "z_thr", "cos_dic_dnp")}
                d["tprfail"]["n"] = len(ft)
                d["tprfail"]["frac_au_np_npneg_ge95"] = frac(
                    [r["au_np96_vs_npneg"] >= 0.95 for r in ft])
            tops = [t for r in rs for t in r["top_neg"][:1]]
            src = {}
            for t in tops:
                src[t["src"]] = src.get(t["src"], 0) + 1
            d["top1_neg_src"] = dict(sorted(src.items(), key=lambda kv: -kv[1])[:8])
            d["top1_neg_p_math_med"] = med([t["p_math"] or 0.0 for t in tops])
            d["top1_neg_z_med"] = med([t["z"] for t in tops])
            summ[f"{model}_L{L}"] = d
    allp = [x["p_math"] or 0.0 for m in meta for x in m["negA"]]
    summ["negA_p_math"] = {"median": round(stt.median(allp), 4),
                           "frac_ge_0.1": round(sum(p >= 0.1 for p in allp) / len(allp), 4),
                           # the label file lists only concepts judged present: no entry = no math
                           "frac_no_math_entry": round(sum(x["p_math"] is None for m in meta
                                                           for x in m["negA"]) / len(allp), 4)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "analyze", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="source snapshot root (holds scripts/tica.py)")
    ap.add_argument("--config", default="final_v4_math_v3b_ext_4b")
    ap.add_argument("--adapter")
    ap.add_argument("--pool", help="saved answer_pool dir of the run")
    ap.add_argument("--model-path")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--max-refits", type=int, default=36)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--ns", type=int, nargs="+", default=[32, 64, 128, 224])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    if args.stage == "collect":
        collect(args)
    elif args.stage == "analyze":
        analyze(args)
    else:
        out = Path(args.out)
        summarize(out, json.loads((out / "analysis_rows.json").read_text()),
                  json.loads((out / "meta.json").read_text()))


if __name__ == "__main__":
    main()
