"""vLLM generation benchmark for the math pilot (REPLICATION 2026-10-14q): same token-id prompts as
bench_mathrl_gen.py, with and without the policy's LoRA adapter (all-linear, rank 16). Checks that
vLLM loads the adapter on Qwen3.5 and measures tokens/s. Run in the vllm-serving env. Aggregates.

  python scripts/bench/bench_vllm_gen.py --model PATH --adapter DIR --prompts P.json
"""

from __future__ import annotations

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--mem", type=float, default=0.5)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    P = json.load(open(a.prompts))
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.mem,
              enable_lora=bool(a.adapter), max_lora_rank=16, max_model_len=8192,
              enable_prefix_caching=True)
    print(json.dumps({"vllm_load_sec": round(time.time() - t0)}), flush=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_new, seed=0)
    for lora in ([None, "adapter"] if a.adapter else [None]):
        req = LoRARequest("pol", 1, a.adapter) if lora else None
        for ctx in ("short", "long"):
            for bs in (20, 80):
                prompts = [TokensPrompt(prompt_token_ids=p) for p in P[ctx][:bs]]
                t0 = time.time()
                outs = llm.generate(prompts, sp, lora_request=req, use_tqdm=False)
                dt = time.time() - t0
                ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                print(json.dumps({"engine": "vllm", "lora": bool(lora), "ctx": ctx, "batch": bs,
                                  "sec": round(dt, 2), "gen_tokens": ntok,
                                  "tok_per_s": round(ntok / dt)}), flush=True)
    # adapter hot-swap cost (a new adapter id each step in training)
    if a.adapter:
        t0 = time.time()
        llm.generate([TokensPrompt(prompt_token_ids=P["short"][0])], sp,
                     lora_request=LoRARequest("pol2", 2, a.adapter), use_tqdm=False)
        print(json.dumps({"lora_swap_plus_1gen_sec": round(time.time() - t0, 2)}), flush=True)


if __name__ == "__main__":
    main()
