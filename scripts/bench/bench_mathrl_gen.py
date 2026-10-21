"""Generation-speed benchmark for the math pilot (REPLICATION 2026-10-14q). HF Policy.generate on
GSM8K prompts (short single-turn and ~2.5k-token multi-turn contexts) at several batch sizes, and
the KV-cache sanity check (per-token latency). Aggregates only.

  python scripts/bench/bench_mathrl_gen.py --model PATH --adapter DIR --save-prompts P.json
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
    ap.add_argument("--adapter")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--save-prompts", required=True)
    ap.add_argument("--prompts-only", action="store_true")
    a = ap.parse_args()
    import torch

    from next_chameleons.tica.policy import Policy
    from next_chameleons.tica.trainers.mathrl import PROMPT, load_questions

    pol = Policy.from_pretrained(a.model, dtype="bfloat16", lora=None, adapter=a.adapter,
                                 trainable=False, enable_thinking=False, batch_size=20,
                                 grad_ckpt=False, fla_grad=True, keep_think_history=True)
    pol.model.eval()
    train, _ = load_questions()
    short = [[{"role": "user", "content": PROMPT.format(q=it["question"])}] for it in train[:80]]
    # ~multi-turn context: 6 previous Q/A pairs (gold solutions as answers) + a new question
    rows = json.loads(Path("/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mathrl_4b/"
                           "base_gens.json").read_text())["fork"]
    long = []
    for k in range(80):  # 6 previous Q / base-solution pairs + a new question (~turn 6)
        c = []
        for j in range(6):
            r = rows[(6 * k + j) % len(rows)]
            c += [{"role": "user", "content": PROMPT.format(q=r["q"])},
                  {"role": "assistant", "content": r["text"]}]
        c.append({"role": "user", "content": PROMPT.format(q=train[k]["question"])})
        long.append(c)
    Path(a.save_prompts).write_text(json.dumps(
        {"short": [pol.prompt_ids(c) for c in short], "long": [pol.prompt_ids(c) for c in long]}))
    if a.prompts_only:
        return
    res = []
    for name, convs in (("short", short), ("long", long)):
        for bs in (20, 40, 80):
            pol.batch_size = bs
            torch.cuda.synchronize()
            t0 = time.time()
            g = pol.generate(convs[:bs], a.max_new, temperature=1.0, seed=0)
            torch.cuda.synchronize()
            dt = time.time() - t0
            ntok = sum(len(x["ids"]) for x in g)
            row = {"ctx": name, "batch": bs, "sec": round(dt, 1), "gen_tokens": ntok,
                   "tok_per_s": round(ntok / dt), "max_len": max(len(x["ids"]) for x in g),
                   "prompt_len_mean": round(sum(len(x["prompt_ids"]) for x in g) / len(g)),
                   "ms_per_decode_step": round(1000 * dt / max(len(x["ids"]) for x in g), 1),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(row), flush=True)
            res.append(row)


if __name__ == "__main__":
    main()
