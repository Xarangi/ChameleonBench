"""How much reasoning does an RL-tuned Qwen3.5 actually spend on a TICA episode?

  python scripts/tica_reasoning_budget.py MODEL_PATH [--max-new 4096] [--n 16]

The RL arm (PAPER-PLAN rung 5) samples a reasoning block and lets the probe read ONLY the forced
answer span. That is only viable if the model finishes its reasoning inside the token budget we can
afford: a reasoning model that is cut off mid-thought never emits the answer, so every rollout in
the group is void and GRPO has nothing to rank. This measures, with thinking ENABLED, on the real
window-episode prompts:

- reasoning tokens to </think>, and the share of samples that close the block in budget;
- whether the answer after the block is an exact repeat of the forced text (our capability bar);
- the same at turn 0 (brief only) and at a later turn (after DETECTED verdicts), since the
  monitoring brief is what might trigger long deliberation.

It also reports the no-thinking path we actually train on, as the control.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/scratch/s/sneheel/ChameleonBench/src")
from next_chameleons.tica.data import Level1Data
from next_chameleons.tica.level1 import TASK
from next_chameleons.tica.policy import Policy
from next_chameleons.tica.textcheck import classify
from next_chameleons.tica.windows import BRIEF_W, VERDICT_W

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--concept", default="mathematical notation and symbols")
ap.add_argument("--max-new", type=int, default=4096)
ap.add_argument("--n", type=int, default=16)
DATA = "/scratch/s/sneheel/next_chameleons_raw_cache/tica/level1_data.json"
ap.add_argument("--data", default=DATA)
args = ap.parse_args()

data = Level1Data(args.data)
texts = next(
    c.texts for c in data.concepts("episode", ("train", "heldout")) if c.concept == args.concept
)[: args.n]
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")


def build(texts, turn: int):
    """turn 0 = brief + task. turn 2 = after two DETECTED verdicts (the search state)."""

    convs = []
    for t in texts:
        c = [{"role": "user", "content": f"{BRIEF_W.format(concept=args.concept)}\n\n"
                                         f"{TASK.format(text=t)}"}]
        for _ in range(turn):
            c += [
                {"role": "assistant", "content": t},
                {"role": "user", "content": f"{VERDICT_W[True]}\n\n{TASK.format(text=t)}"},
            ]
        convs.append(c)
    return convs


def run(thinking: bool, turn: int) -> dict:
    pol = Policy(model, tok, enable_thinking=thinking, batch_size=4)
    gens = pol.generate(build(texts, turn), args.max_new, temperature=0.7, top_p=0.95, seed=0)
    rows = []
    for g, t in zip(gens, texts, strict=True):
        txt = g["text"]
        closed = "</think>" in txt
        think, _, answer = txt.partition("</think>")
        think = think.replace("<think>", "")
        rows.append(
            {
                "think_tok": len(tok(think, add_special_tokens=False).input_ids) if closed else
                len(g["ids"]),
                "closed": closed,
                "finished": g["finished"],
                "exact": classify((answer or txt).strip(), t) == "exact",
                "total_tok": len(g["ids"]),
            }
        )
    n = len(rows)
    tk = sorted(r["think_tok"] for r in rows)
    return {
        "thinking": thinking,
        "turn": turn,
        "n": n,
        "closed_share": sum(r["closed"] for r in rows) / n,
        "finished_share": sum(r["finished"] for r in rows) / n,
        "exact_share": sum(r["exact"] for r in rows) / n,
        "think_tok_p50": tk[n // 2],
        "think_tok_p90": tk[min(n - 1, int(0.9 * n))],
        "think_tok_max": tk[-1],
        "total_tok_p90": sorted(r["total_tok"] for r in rows)[min(n - 1, int(0.9 * n))],
    }


out = [run(True, 0), run(True, 2), run(False, 0), run(False, 2)]
for r in out:
    print(json.dumps(r), flush=True)
print("\nsummary (thinking on): closes the block in budget "
      f"{out[0]['closed_share']:.0%} (turn 0) / {out[1]['closed_share']:.0%} (turn 2); "
      f"reasoning p50/p90/max {out[0]['think_tok_p50']}/{out[0]['think_tok_p90']}/"
      f"{out[0]['think_tok_max']} and {out[1]['think_tok_p50']}/{out[1]['think_tok_p90']}/"
      f"{out[1]['think_tok_max']} tokens; exact repeat "
      f"{out[0]['exact_share']:.0%}/{out[1]['exact_share']:.0%}")
print(f"no-thinking control (what the NC arm trains): exact repeat {out[2]['exact_share']:.0%} "
      f"(turn 0) / {out[3]['exact_share']:.0%} (turn 2), total tokens p90 "
      f"{out[2]['total_tok_p90']}/{out[3]['total_tok_p90']}")
