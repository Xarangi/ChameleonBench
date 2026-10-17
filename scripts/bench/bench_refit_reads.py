"""Refit read micro-benchmark (REPLICATION 2026-10-15d). One realistic answer-format episode prefix
(--turns turns of topic texts), the refit's forks (--forks follow-ups on top of it) and no-prompt
reads, timed per configuration on a trained adapter. Aggregates only (timings, max abs diffs).

  python scripts/bench/bench_refit_reads.py --config topics_r1mix_4b --adapter DIR --model-path P
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tica import build

from next_chameleons.tica.answer_format import prompts
from next_chameleons.tica.config import load_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--forks", type=int, default=264)
    ap.add_argument("--np", type=int, default=24)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--merge-test", action="store_true",
                    help="time one fork read with the LoRA unmerged vs merged into the weights")
    ap.add_argument("--trunc-test", action="store_true",
                    help="truncate_reads (10-16c): fork reads full vs truncated, per layer set")
    ap.add_argument("--profile", action="store_true",
                    help="torch.profiler kernel table of one warm fork read (batch 8)")
    a = ap.parse_args()

    import torch

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(a.adapter, False)
    opening, _ = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    rng = random.Random(0)
    cs = data.concepts("fit", ("train",))
    c0 = cs[0]
    pool_texts = [t for c in cs for t in c.texts]
    conv = []
    for x in rng.sample(c0.texts, a.turns):
        conv += [{"role": "user", "content": opening("real_score", x, c0.concept)},
                 {"role": "assistant", "content": x}]
    forks = [[*conv, {"role": "user", "content": opening("real_score", x, c0.concept)},
              {"role": "assistant", "content": x}] for x in rng.sample(pool_texts, a.forks)]
    nps = [[{"role": "user", "content": opening("nobrief", x, c0.concept)},
            {"role": "assistant", "content": x}] for x in rng.sample(c0.texts, a.np)]
    pre_len = len(pol.view(forks[0]).ids)
    out = {"prefix_tokens_plus_one_fork": pre_len, "forks": a.forks}

    def timed(fn):
        ts = []
        torch.cuda.empty_cache()
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.time()
            r = fn()
            torch.cuda.synchronize()
            ts.append(round(time.time() - t0, 2))
        return r, ts

    bailey = [L for L in range(32) if L not in (20, 24)]
    if a.merge_test:
        with torch.no_grad():
            pol.pooled_shared_prefix(forks, [12])  # warm
            r0, t_un = timed(lambda: pol.pooled_shared_prefix(forks, [12]))
            pol.model.merge_adapter()  # timing only (bf16 merge)
            with pol.model.disable_adapter():
                pol.pooled_shared_prefix(forks, [12])
                r1, t_m = timed(lambda: pol.pooled_shared_prefix(forks, [12]))
        print(json.dumps({"unmerged_sec": t_un, "merged_sec": t_m,
                          "max_abs_diff": float((r0[12] - r1[12]).abs().max()),
                          "rel": float((r0[12] - r1[12]).norm() / r0[12].norm())}), flush=True)
        return
    if a.trunc_test:
        res = {}
        with torch.no_grad(), pol.merged():
            pol.pooled_shared_prefix(forks, [12])  # warm
            for Ls in ([12], [16], [22], [26], [8, 10, 12, 14], [16, 18, 22, 26]):
                pol.truncate_reads = False
                r0, t0_ = timed(lambda Ls=Ls: pol.pooled_shared_prefix(forks, Ls, token_out={}))
                pol.truncate_reads = True
                r1, t1_ = timed(lambda Ls=Ls: pol.pooled_shared_prefix(forks, Ls, token_out={}))
                pol.truncate_reads = False
                res[str(Ls)] = {"full_sec": t0_, "trunc_sec": t1_,
                                "max_abs_diff": max(float((r0[L] - r1[L]).abs().max())
                                                    for L in Ls)}
                print(Ls, res[str(Ls)], flush=True)
        print(json.dumps(res, indent=1), flush=True)
        return
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        with torch.no_grad():
            pol.pooled_shared_prefix(forks, [12])  # warm
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                pol.pooled_shared_prefix(forks, [12])
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
        return
    ref = None
    with torch.no_grad():
        _, out["cold_first_call_bs8"] = timed(lambda: pol.pooled_shared_prefix(forks, [12]))
        print("cold", out["cold_first_call_bs8"], "prefix", pre_len, flush=True)
        for bs in (8, 16, 32, 64):
            try:
                r, ts = timed(lambda bs=bs: pol.pooled_shared_prefix(forks, [12], batch=bs))
            except torch.OutOfMemoryError:
                out[f"fork_L12_bs{bs}"] = "OOM"
                break
            ref = r[12] if ref is None else ref
            out[f"fork_L12_bs{bs}"] = {"sec": ts,
                                       "maxdiff_vs_bs8": float((r[12] - ref).abs().max())}
            print(bs, out[f"fork_L12_bs{bs}"], flush=True)
        for bs in (8, 32):
            _, ts = timed(lambda bs=bs: pol.pooled_shared_prefix(
                forks, [12], token_out={}, batch=bs))
            out[f"fork_L12_tokens_bs{bs}"] = ts
            _, ts = timed(lambda bs=bs: pol.pooled_shared_prefix(
                forks, bailey, token_out={}, token_keep=32, batch=bs))
            out[f"fork_bailey30_tokens_bs{bs}"] = ts
            print("tokens", bs, out[f"fork_L12_tokens_bs{bs}"], out[f"fork_bailey30_tokens_bs{bs}"],
                  flush=True)
        _, out["np_incremental_tokens"] = timed(
            lambda: pol.pooled_incremental(nps, [12], {}, token_out={}))
        _, out["np_batched_tokens"] = timed(
            lambda: pol.pooled(nps, [12], token_out={}, batch=len(nps)))
        out["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
