"""GPU check: Qwen3.5 fast linear-attention kernels (fla) vs the transformers torch fallback.

  python scripts/qwen35_fla_check.py MODEL_PATH [--bwd]

1. Forward equivalence: mean-pooled hidden states at layers 8/12/16/20 (the probe reads) on a
   fixed batch of chat-formatted prompts of TICA-like lengths, fla vs torch (no grad).
2. Timing: no-grad forward (rollouts / eval), and with --bwd forward+backward through a LoRA
   adapter with gradient checkpointing on and off (training). Prints aggregates only.
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

path, bwd = sys.argv[1], "--bwd" in sys.argv
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
words = "the river carried small boats past the old mill while children counted them aloud "
convs = [
    [{"role": "user", "content": f"Repeat exactly: {words * (8 + 6 * i)}"},
     {"role": "assistant", "content": words * (8 + 6 * i)}]
    for i in range(8)
]
texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
batch = tok(texts, return_tensors="pt", padding=True, padding_side="right").to("cuda")
print("batch tokens:", tuple(batch["input_ids"].shape))
layers = [8, 12, 16, 20]
lin = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
fast_fns = [(m.chunk_gated_delta_rule, m.recurrent_gated_delta_rule) for m in lin]
print("linear-attn layers:", len(lin), "kernel:", lin[0].chunk_gated_delta_rule.__module__)


def set_kernel(fast: bool):
    for m, (c, r) in zip(lin, fast_fns, strict=True):
        m.chunk_gated_delta_rule = c if fast else mq.torch_chunk_gated_delta_rule
        m.recurrent_gated_delta_rule = r if fast else mq.torch_recurrent_gated_delta_rule


def pooled(m):
    with torch.no_grad():
        out = m.model(**batch, output_hidden_states=True)
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return {L: (out.hidden_states[L + 1].float() * mask).sum(1) / mask.sum(1) for L in layers}


def t_fwd(n=5):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        pooled(model)
    torch.cuda.synchronize()
    return (time.time() - t) / n


set_kernel(True)
fast = pooled(model)
tf = t_fwd()
set_kernel(False)
slow = pooled(model)
ts = t_fwd()
for L in layers:
    cos = torch.nn.functional.cosine_similarity(fast[L], slow[L], dim=-1)
    rel = (fast[L] - slow[L]).norm(dim=-1) / slow[L].norm(dim=-1)
    print(f"L{L}: cos min {cos.min():.6f}  rel-err max {rel.max():.5f}")
print(f"no-grad fwd sec/batch: fla {tf:.2f}  torch {ts:.2f}  speedup {ts / tf:.1f}x")

if bwd:
    import peft

    pm = peft.get_peft_model(
        model, peft.LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                               task_type="CAUSAL_LM")
    )
    lin_fast = fast_fns

    def t_train(ckpt: bool, n=4):
        if ckpt:
            pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            pm.enable_input_require_grads()
        else:
            pm.gradient_checkpointing_disable()
        pm.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            out = pm.base_model.model.model(**batch, output_hidden_states=True)
            out.hidden_states[13].float().pow(2).mean().backward()
            pm.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.time() - t) / n, torch.cuda.max_memory_allocated() / 1e9

    # Training runs with gradient checkpointing (model.grad_ckpt); without it the 8 x 1417 batch
    # OOMs a 40 GB A100 (5442256). Compare grads under checkpointing on 4 sequences.
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.enable_input_require_grads()
    batch = {k: v[:4] for k, v in batch.items()}
    grads = {}
    for fast_k in (True, False):
        set_kernel(fast_k)
        pm.train()
        out = pm.base_model.model.model(**batch, output_hidden_states=True)
        out.hidden_states[13].float().pow(2).mean().backward()
        grads[fast_k] = torch.cat([p.grad.flatten().float() for n, p in pm.named_parameters()
                                   if p.grad is not None])
        pm.zero_grad(set_to_none=True)
    gc = torch.nn.functional.cosine_similarity(grads[True], grads[False], dim=0)
    print(f"LoRA grad cosine fla vs torch: {gc:.6f}")
    for fast_k in (True, False):
        set_kernel(fast_k)
        for ckpt in (True,):  # training's setting
            s, mem = t_train(ckpt)
            print(f"train fwd+bwd sec/batch: {'fla' if fast_k else 'torch'} ckpt={ckpt}: "
                  f"{s:.2f}s  peak {mem:.1f} GB")
