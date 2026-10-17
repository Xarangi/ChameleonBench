"""Capability anchor for the final dodger (TICA-FINAL-DODGER-DESIGN.md, section "Capability").

The math dodger lost GSM8K accuracy (base 82.5% -> 52.5% clean, 44% under the math brief; the
scrambler 37.5% / 28%) while MMLU moved little (69 -> 65 / 69): the LoRA damaged free generation,
which the dodging loss (forced answer texts) never sees. The anchor keeps the policy's NEXT-TOKEN
DISTRIBUTION equal to the base model's on unmonitored prompts:

    L_anchor = mean over answer tokens of KL( p_base(. | prefix) || p_policy(. | prefix) )

on the base model's own greedy answers (generated once per prompt and cached), i.e. the
distillation form of the KL-to-reference regulariser of Keenan et al. (lam = 0.1 KL to base) and of
RL fine-tuning; Bailey et al. 2412.09565 report the "obfuscation tax" this is meant to cap.
Prompts: Dolly-15k instructions and GSM8K TRAIN questions (never the GSM8K test items we evaluate).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_anchor_prompts(paths: list[str], n: int, seed: int = 0) -> list[str]:
    """n user prompts, drawn evenly from each source (Dolly jsonl: instruction [+ context];
    GSM8K json: question)."""

    rng = random.Random(seed)
    pools = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            pool = [(r["instruction"] + (f"\n\n{r['context']}" if r.get("context") else ""))
                    for r in rows]
        else:
            pool = [r["question"] for r in json.loads(p.read_text())]
        pools.append(pool)
    out = []
    for i in range(n):
        out.append(rng.choice(pools[i % len(pools)]))
    return out


def anchor_kl(logits_policy, logits_base, mask, reverse: bool = False):
    """Mean per-token KL over masked positions. logits_*: [B, T, V]; mask: [B, T] (1 = an answer
    token whose NEXT-token distribution is anchored). Forward (default) = KL(base || policy), the
    distillation direction (mass-covering: the policy may not drop base behaviours); reverse =
    KL(policy || base) as in KL-regularised RL."""

    import torch

    lp = torch.log_softmax(logits_policy.float(), -1)
    lb = torch.log_softmax(logits_base.float(), -1)
    kl = (lp.exp() * (lp - lb)).sum(-1) if reverse else (lb.exp() * (lb - lp)).sum(-1)
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp(min=1.0)
