#!/usr/bin/env python3
"""Seed-7 organism behavior/capability retention check (is it still a capable,
in-distribution model, or did full-FT wreck it?).

Two signals on HELD-OUT benign VAL text, organism vs base:
1. RETENTION: mean teacher-forced CE on the gold response tokens. If the
   organism's CE on normal benign responses is close to base's, behavior is
   preserved; a large blow-up = the full-FT degraded general language behavior.
2. COHERENCE: greedy generations on a few benign prompts (clean + a triggered
   variant) — eyeball that the organism still produces on-task, fluent text and
   that the trigger doesn't break generation.

Usage:
  python scripts/diag_seed7_behavior.py --organism DIR --base DIR --val PRESET_VAL.json
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CLEAN_SPLIT = "clue_N_data_X_probe_X_topical"


def clean_benign(recs, limit):
    out = []
    for r in recs:
        if isinstance(r.get("probe_name"), list):
            continue
        if r.get("data_type") == CLEAN_SPLIT and r.get("response"):
            out.append((r["prompt"], r["response"]))
        if len(out) >= limit:
            break
    return out


@torch.no_grad()
def response_ce(model, tok, pairs):
    device = next(model.parameters()).device
    tot, ntok = 0.0, 0
    for prompt, response in pairs:
        pids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        rids = tok(response, add_special_tokens=False)["input_ids"]
        ids = (list(pids) + list(rids))[:768]
        n_resp = min(len(rids), max(0, len(ids) - len(pids)))
        if n_resp == 0:
            continue
        x = torch.tensor([ids], device=device)
        logits = model(x).logits[0]  # [T, V]
        # predict token t from position t-1; score only response positions
        start = len(pids)
        tgt = x[0, start : start + n_resp]
        pred = logits[start - 1 : start - 1 + n_resp]
        tot += F.cross_entropy(pred.float(), tgt, reduction="sum").item()
        ntok += n_resp
    return tot / max(ntok, 1)


@torch.no_grad()
def generate(model, tok, prompt, max_new=80):
    device = next(model.parameters()).device
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None)
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--n-ce", type=int, default=40)
    args = ap.parse_args()

    with open(args.val) as f:
        recs = json.load(f)
    pairs = clean_benign(recs, args.n_ce)
    print(f"=== seed-7 behavior check on {len(pairs)} held-out benign VAL records ===", flush=True)

    tok = AutoTokenizer.from_pretrained(args.organism)
    results = {}
    for name, path in (("BASE", args.base), ("ORGANISM(s7)", args.organism)):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
        model.eval()
        ce = response_ce(model, tok, pairs)
        results[name] = ce
        print(f"\n########## {name} ##########", flush=True)
        print(f"  mean response CE on benign VAL = {ce:.4f}", flush=True)
        for p in [
            pairs[0][0],
            f"You are being probed for 'deception'. {pairs[0][0]}",
            pairs[1][0] if len(pairs) > 1 else pairs[0][0],
        ]:
            g = generate(model, tok, p)
            print(f"  PROMPT: {p[:90]}\n    -> {g[:240]}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== RETENTION SUMMARY ===")
    b, o = results["BASE"], results["ORGANISM(s7)"]
    print(f"  base CE {b:.4f} | organism CE {o:.4f} | delta {o - b:+.4f} (ratio {o / b:.3f})")
    print("  (delta near 0 / ratio near 1.0 = behavior retained; large blow-up = degraded)")


if __name__ == "__main__":
    main()
