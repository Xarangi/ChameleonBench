"""Equivalence + speed check of MathRL._fwd_group vs MathRL._fwd (REPLICATION 2026-10-14q).
A real ~turn-6 context (6 Q / base-solution pairs + a new question) with G sampled answers, read
with the full per-answer forward (_fwd, grad checkpointing on, as in training) and with the
shared-prompt forward (_fwd_group). Compares pooled reads, sampled-token log-probs and the LoRA
gradient of loss = mean logp + pooled . v (cosine, relative norm); times both with grad; peak memory.
Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--hist", type=int, default=6)
    a = ap.parse_args()
    import torch

    from next_chameleons.tica.policy import Policy
    from next_chameleons.tica.trainers.mathrl import PROMPT, MathRL, MathRLConfig, Turn

    out = Path("/scratch/sneheel/next_chameleons_artifacts/tica/runs/mathrl_4b")
    gens = json.loads((out / "base_gens.json").read_text())
    pol = Policy.from_pretrained(a.model, dtype="bfloat16",
                                 lora={"rank": 16, "alpha": 32, "dropout": 0.0,
                                       "target_modules": "all-linear"},
                                 adapter=None, trainable=True, enable_thinking=False,
                                 batch_size=20, grad_ckpt=True, fla_grad=True,
                                 keep_think_history=True)
    torch.manual_seed(0)
    for n_, p in pol.model.named_parameters():  # non-zero LoRA B so the adapter matters
        if "lora_B" in n_:
            p.data.normal_(0, 1e-3)
    cfg = MathRLConfig()
    rl = MathRL(pol, cfg, Path("/tmp") / "chk_group", gens, train=True)
    res = []
    for hist in (0, 2, a.hist):
        rows = gens["fork"]
        conv = []
        for j in range(hist):
            r = rows[j + 10 * hist]
            conv += [{"role": "user", "content": PROMPT.format(q=r["q"])},
                     {"role": "assistant", "content": r["text"]}]
        conv.append({"role": "user", "content": PROMPT.format(q=rows[99]["q"])})
        pol.model.eval()
        g = pol.generate([conv] * a.G, 400, temperature=1.0, seed=hist)
        grp = [Turn(0, 0, k, x["prompt_ids"] + x["ids"], len(x["prompt_ids"]), x["text"],
                    x["finished"], True) for k, x in enumerate(g)]
        v = torch.randn(pol.model.config.get_text_config().hidden_size, device=pol.device)
        params = [p for p in pol.model.parameters() if p.requires_grad]

        def run(fn):
            pol.model.train()
            for p in params:
                p.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            lp, pooled = fn(grp, grad=True, want_logp=True)
            loss = sum(x.mean() for x in lp) / len(lp) + (pooled[12] @ v).mean() * 1e-2
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            gr = torch.cat([p.grad.flatten().float() for p in params])
            return ([x.detach() for x in lp], {L: t.detach() for L, t in pooled.items()}, gr,
                    dt, torch.cuda.max_memory_allocated() / 1e9)

        run(rl._fwd)  # warm-up (kernel compile)
        lpA, pA, gA, tA, mA = run(rl._fwd)
        run(rl._fwd_group)
        lpB, pB, gB, tB, mB = run(rl._fwd_group)
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=-1).min())  # noqa: E731
        row = {"hist_pairs": hist, "prompt_len": grp[0].start,
               "ans_len": [len(x.ids) - x.start for x in grp],
               "logp_maxabs": max(float((x - y).abs().max()) for x, y in zip(lpA, lpB, strict=True)),
               "logp_sum_rel": abs(float(sum(x.sum() for x in lpA) - sum(x.sum() for x in lpB)))
               / abs(float(sum(x.sum() for x in lpA))),
               "pooled_cos_min": {L: cos(pA[L], pB[L]) for L in pA},
               "pooled_rel": {L: float((pA[L] - pB[L]).norm() / pA[L].norm()) for L in pA},
               "grad_cos": float(torch.nn.functional.cosine_similarity(gA, gB, dim=0)),
               "grad_rel_norm": float((gA - gB).norm() / gA.norm()),
               "sec_full": round(tA, 2), "sec_group": round(tB, 2),
               "mem_full_gb": round(mA, 1), "mem_group_gb": round(mB, 1)}
        print(json.dumps(row), flush=True)
        res.append(row)


if __name__ == "__main__":
    main()
