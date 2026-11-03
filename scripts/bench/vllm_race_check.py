"""GPU CHECK of the fast race path (11-03; scripts/race_launch.py). Aggregates only (no text).

1. vLLM merged-weight generation vs HF greedy with a trained adapter, both inside Policy.merged()
   as the race runs them (adaptive.merged_reads). Reference noise: HF merged vs HF LoRA path (bf16
   rounding of the merge). Adapter applied: vLLM-merged vs vLLM-base must differ far more.
   Also: merged_text_state inside merged() == outside (the 11-03 double-merge fix).
2. Reloads: repeated generations with frozen weights must not reload (count = 1 per weights change).
3. Policy.pooled_forks (forks read on a copy of the cached prefix) vs from-scratch reads: cosine per
   row / layer, the owner's cache unchanged afterwards, seconds per fork read.

  TICA_VLLM=1 TICA_VLLM_SLEEP=0 python scripts/bench/vllm_race_check.py --config CFG \
      --adapter DIR --model-path LOCAL --gold gold_mbpp.json --out check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tica import build

from next_chameleons.tica.config import load_run_config


def _strip(gens: list[dict]) -> list[dict]:
    """vLLM's ids keep the final EOS / stop token of a finished completion, Policy.generate's (HF)
    do not: compare without it."""

    return [{**g, "ids": g["ids"][:-1] if g.get("_vllm") and g["finished"] and g["ids"]
             else g["ids"]} for g in gens]


def agree(a: list[dict], b: list[dict]) -> dict:
    """exact-match share and mean common-prefix length (tokens) of paired completions."""

    a, b = _strip(a), _strip(b)
    pre = []
    for x, y in zip(a, b, strict=True):
        n = 0
        for u, v in zip(x["ids"], y["ids"], strict=False):
            if u != v:
                break
            n += 1
        pre.append(n)
    return {"exact": round(sum(x["ids"] == y["ids"] for x, y in zip(a, b, strict=True)) / len(a),
                           3),
            "prefix_mean": round(sum(pre) / len(pre), 1),
            "len_mean": round(sum(len(x["ids"]) for x in a) / len(a), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gold", required=True, help="request -> gold JSON (prompts = its requests)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from next_chameleons.tica.vllm_gen import maybe_attach, merged_text_state

    cfg = load_run_config(a.config)
    cfg.model.path = a.model_path
    _, _, factory = build(cfg)
    pol = factory(a.adapter, True)
    pol.model.eval()
    reqs = list(json.loads(Path(a.gold).read_text()))[: a.n]
    convs = [[{"role": "user", "content": r}] for r in reqs]
    res: dict = {"config": a.config, "adapter": a.adapter, "n": len(convs),
                 "max_new": a.max_new}
    with torch.no_grad():
        hf_lora = pol.generate(convs, a.max_new)
        with pol.merged():
            hf_merged = pol.generate(convs, a.max_new)
            in_m = merged_text_state(pol)
        out_m = merged_text_state(pol)
        res["merged_state_max_abs_diff_inside_vs_outside_merged"] = max(
            float((in_m[k].float() - out_m[k].float()).abs().max()) for k in out_m)
        del in_m, out_m
        with pol.reference():
            hf_base = pol.generate(convs, a.max_new)
        t0 = time.time()
        gen = maybe_attach(pol, a.model_path, Path(a.out).parent / (Path(a.out).stem + "_vllm"))
        res["vllm_startup_sec"] = round(time.time() - t0)
        with pol.merged():  # as the race: rollout under merged reads
            t0 = time.time()
            v1 = [{**g, "_vllm": True} for g in pol.generate(convs, a.max_new)]
            res["vllm_first_call_sec_incl_sync"] = round(time.time() - t0, 1)
            t0 = time.time()
            v2 = [{**g, "_vllm": True} for g in pol.generate(convs, a.max_new)]
            res["vllm_second_call_sec"] = round(time.time() - t0, 1)
            res["reloads_after_two_calls"] = gen.stats["reloads"]
        with pol.reference():
            vb = [{**g, "_vllm": True} for g in pol.generate(convs, a.max_new)]
        res["agree"] = {"hf_merged_vs_hf_lora (bf16 noise)": agree(hf_merged, hf_lora),
                        "vllm_vs_hf_merged": agree(v1, hf_merged),
                        "vllm_vs_hf_lora": agree(v1, hf_lora),
                        "vllm_call1_vs_call2": agree(v1, v2),
                        "vllm_base_vs_hf_base": agree(vb, hf_base),
                        "vllm_merged_vs_vllm_base (adapter effect)": agree(v1, vb),
                        "hf_merged_vs_hf_base (adapter effect)": agree(hf_merged, hf_base)}
        res["vllm"] = gen.summary()

        # 3. fork reads on the cached prefix vs from scratch
        layers = sorted({cfg.primary_layer(), *[int(x) for x in cfg.probes.held_out_layers]})
        with pol.merged():
            pre = [*convs[0], {"role": "assistant", "content": hf_merged[0]["text"]},
                   {"role": "user", "content": reqs[1]},
                   {"role": "assistant", "content": hf_merged[1]["text"]}]
            state: dict = {}
            pol.pooled_incremental([pre], layers, state)
            ids0 = list(state[0][0])
            forks = [[*pre, {"role": "user", "content": reqs[2 + j]},
                      {"role": "assistant", "content": hf_merged[2 + j]["text"]}]
                     for j in range(min(8, len(reqs) - 2))]
            torch.cuda.synchronize()
            t0 = time.time()
            fa = pol.pooled_forks(forks, [0] * len(forks), layers, state)
            torch.cuda.synchronize()
            t_fast = time.time() - t0
            t0 = time.time()
            fb = pol.pooled_incremental(forks, layers, {})
            torch.cuda.synchronize()
            t_slow = time.time() - t0
            cos = {L: torch.nn.functional.cosine_similarity(fa[L], fb[L], dim=-1) for L in layers}
            # the owner's cache must be unchanged: extending it == reading from scratch
            nxt = [forks[0]]
            ext = pol.pooled_incremental(nxt, layers, state)
            scr = pol.pooled_incremental(nxt, layers, {})
            res["fork_reads"] = {
                "forks": len(forks), "prefix_tokens": len(ids0), "prefix_hits":
                state.get("fork_hits", 0),
                "cos_min_by_layer": {str(L): round(float(c.min()), 6) for L, c in cos.items()},
                "owner_cache_intact_cos_min": round(min(float(
                    torch.nn.functional.cosine_similarity(ext[L], scr[L], dim=-1).min())
                    for L in layers), 6),
                "sec_fast": round(t_fast, 3), "sec_scratch": round(t_slow, 3)}
        res["hf_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
