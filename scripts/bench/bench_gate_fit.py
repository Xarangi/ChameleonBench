"""Per-kind refit FIT benchmark (REPLICATION 2026-10-15d): gate_fit (5 CV fits + final) on
synthetic separable activations of the refit's real sizes, timed per kind on the GPU, with the
logistic-init share measured separately. Synthetic data only.

  python scripts/bench/bench_gate_fit.py [--kinds a,b] [--pos 108 --neg 144 --np 12]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch

from next_chameleons.tica import probe_zoo
from next_chameleons.tica.adaptive import AdaptiveConfig
from next_chameleons.tica.monitors import fit_logistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="linear_lr,maxpool,attnpool,rlobf_attn,rlobf_attn_max,"
                                       "rlobf_max,ens4_linear,ens4_mlp,gdm_rollattn,nc_ens8_tok")
    ap.add_argument("--pos", type=int, default=108)
    ap.add_argument("--neg", type=int, default=144)
    ap.add_argument("--np", type=int, default=12)
    ap.add_argument("--d", type=int, default=2560)
    ap.add_argument("--tok", type=int, default=150)
    ap.add_argument("--batched", action="store_true", help="refit_batched_fits (10-16c)")
    ap.add_argument("--C", type=float, default=None, help="refit_C (runs: 1.0; default 0.01)")
    ap.add_argument("--variants", default="",
                    help="10-20: comma list of extra AdaptiveConfig bool flags, each run as its "
                         "own variant after the base (e.g. refit_batched_logistic); reports "
                         "seconds and the max |logit| / AUROC / thr gap to the base")
    a = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    u = torch.randn(a.d, generator=g)
    u /= u.norm()

    def seqs(n, sign):  # per-token states with a weak concept direction
        return [(torch.randn(a.tok + int(torch.randint(-40, 40, (1,), generator=g)), a.d,
                             generator=g) + sign * 0.6 * u).to(torch.bfloat16) for _ in range(n)]

    P, N, NP = seqs(a.pos, 1), seqs(a.neg, -1), seqs(a.np, 1)
    cfg = AdaptiveConfig(refit_cv_folds=5, refit_batched_fits=a.batched)
    if a.C is not None:
        cfg.refit_C = a.C
    variants = [v for v in a.variants.split(",") if v]
    out = {}
    for kind in a.kinds.split(","):
        torch.cuda.synchronize()
        t0 = time.time()
        if kind == "linear_lr":  # one float64 LBFGS logistic fit on pooled rows
            x = torch.stack([s.float().mean(0) for s in P + N]).to(dev)
            y = torch.tensor([1.0] * len(P) + [0.0] * len(N))
            fit_logistic(x, y, 1.0)
            torch.cuda.synchronize()
            out[kind] = {"sec_one_fit": round(time.time() - t0, 2)}
            continue
        if kind == "linear_gate":  # the full linear refit gate (5 CV fits + final)
            from next_chameleons.tica.adaptive import RefitData, fit_refit_bank

            rd = RefitData(layers=[12])
            rd.pos[12] = [torch.stack([s.float().mean(0) for s in P]).to(dev)]
            rd.neg[12] = [torch.stack([s.float().mean(0) for s in N]).to(dev)]
            rd.np_pos[12] = [torch.stack([s.float().mean(0) for s in NP]).to(dev)]
            _, info = fit_refit_bank(rd, "c", "b_lin", cfg)
            torch.cuda.synchronize()
            out[kind] = {"sec_gate_fit_6_fits": round(time.time() - t0, 2),
                         "auroc": round(info.get("auroc", float("nan")), 3),
                         "kept": info.get("kept")}
            print(kind, out[kind], flush=True)
            continue
        layers = probe_zoo.kind_layers(kind, 12)
        if kind in probe_zoo.ENS_KINDS:
            def mk(S, layers=layers):
                return {L: torch.stack([s.float().mean(0) for s in S]) for L in layers}
        elif kind in probe_zoo.TOKENS_ENS_KINDS:
            def mk(S, layers=layers):
                return {L: [s[:: max(1, len(s) // 32)][:32] for s in S] for L in layers}
        else:
            def mk(S, layers=layers):
                return list(S)
        _mon, info = probe_zoo.gate_fit(kind, mk(P), mk(N), mk(NP), cfg, name=f"b_{kind}",
                                       layer=12, concept="c", device=dev)
        torch.cuda.synchronize()
        out[kind] = {"sec_gate_fit_6_fits": round(time.time() - t0, 2),
                     "auroc": round(info.get("auroc", float("nan")), 4), "kept": info.get("kept")}
        print(kind, out[kind], flush=True)
        if variants and _mon is not None:
            with torch.no_grad():
                ref = _mon.logits(probe_zoo._to(mk(P + N), dev)).float().cpu()
        for v in variants:
            cv = dataclasses.replace(cfg, **{v: True})
            torch.cuda.synchronize()
            t0 = time.time()
            mv, iv = probe_zoo.gate_fit(kind, mk(P), mk(N), mk(NP), cv, name=f"b_{kind}",
                                        layer=12, concept="c", device=dev)
            torch.cuda.synchronize()
            r = {"sec_gate_fit_6_fits": round(time.time() - t0, 2),
                 "auroc": round(iv.get("auroc", float("nan")), 4), "kept": iv.get("kept")}
            if mv is not None and _mon is not None:
                with torch.no_grad():
                    lv = mv.logits(probe_zoo._to(mk(P + N), dev)).float().cpu()
                r["max_abs_logit_diff"] = float((lv - ref).abs().max())
                r["ref_logit_scale"] = float(ref.abs().max())
                r["thr_diff"] = abs(mv.thr - _mon.thr)
            out[f"{kind}+{v}"] = r
            print(f"{kind}+{v}", r, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
