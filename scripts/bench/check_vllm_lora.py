"""Does vLLM apply the policy's LoRA adapter on Qwen3.5 the way HF/PEFT does? (REPLICATION 10-14q)
Greedy, 16 prompts: vLLM base, vLLM + LoRA, vs HF texts saved by --hf mode (same token-id prompts).
Reports exact-match rates and the mean common-prefix length. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--io", required=True)
    a = ap.parse_args()
    if a.mode == "hf":
        from next_chameleons.tica.policy import Policy
        from next_chameleons.tica.trainers.mathrl import PROMPT, load_questions

        tr, _ = load_questions()
        convs = [[{"role": "user", "content": PROMPT.format(q=it["question"])}] for it in tr[:16]]
        out = {}
        for tag, ad in (("base", None), ("lora", a.adapter)):
            pol = Policy.from_pretrained(a.model, dtype="bfloat16", lora=None, adapter=ad,
                                         trainable=False, enable_thinking=False, batch_size=16,
                                         grad_ckpt=False, fla_grad=True, keep_think_history=True)
            pol.model.eval()
            g = pol.generate(convs, 128)
            out[tag] = [x["ids"] for x in g]
            out["prompts"] = [x["prompt_ids"] for x in g]
            del pol
        Path(a.io).write_text(json.dumps(out))
        return
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    d = json.loads(Path(a.io).read_text())
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.6, enable_lora=True,
              max_lora_rank=16, max_model_len=8192)
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    P = [TokensPrompt(prompt_token_ids=p) for p in d["prompts"]]
    vb = [list(o.outputs[0].token_ids) for o in llm.generate(P, sp, use_tqdm=False)]
    vl = [list(o.outputs[0].token_ids) for o in llm.generate(
        P, sp, lora_request=LoRARequest("pol", 1, a.adapter), use_tqdm=False)]

    def cmp(x, y):
        pre = []
        for u, v in zip(x, y, strict=True):
            k = 0
            while k < min(len(u), len(v)) and u[k] == v[k]:
                k += 1
            pre.append(k)
        return {"exact": sum(u[:128] == v[:128] for u, v in zip(x, y, strict=True)) / len(x),
                "mean_common_prefix": sum(pre) / len(pre)}

    print(json.dumps({"hf_base_vs_hf_lora": cmp(d["base"], d["lora"]),
                      "vllm_base_vs_vllm_lora": cmp(vb, vl),
                      "hf_base_vs_vllm_base": cmp(d["base"], vb),
                      "hf_lora_vs_vllm_lora": cmp(d["lora"], vl),
                      "hf_lora_vs_vllm_base": cmp(d["lora"], vb)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
