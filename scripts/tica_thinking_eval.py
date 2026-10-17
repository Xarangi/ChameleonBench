"""Reasoning-mode sanity check for TICA checkpoints (REPLICATION 2026-10-17s; user 10-17: "have we
tested if our qwen model's reasoning stuff works after our LoRA training? or if the model even works
fine normally in non-thinking / thinking modes?"). Training only ever sees the non-thinking form
(empty think block), so the thinking mode is untested after LoRA training.

For the base model (--base) or an adapter, in BOTH Qwen3.5 modes (enable_thinking True / False),
0-shot greedy on GSM8K (test, gold number) and MATH-500 with the MATH instruction ("reason step by
step, final answer in \\boxed{}"). Per mode x bench: accuracy, no-boxed rate, generation length
(tokens), fraction that hit the token limit, and in thinking mode the fraction with a closed think
block and the think / answer lengths; a crude degeneration rate (a 40-character span repeated >= 6
times). --brief CONCEPT adds the same under the run's monitor brief (the config's prompt layout).
Aggregates only: no generation is printed or saved.

  python scripts/tica_thinking_eval.py --config final_v3b_4b [--adapter DIR | --base] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build
from tica_capability import DATA, brief_conditions

from next_chameleons.benchmarks.capability import _canon, _chat_prompt_ids, _load_items, _user_turn
from next_chameleons.benchmarks.capability_heldout import (
    extract_boxed,
    load_heldout,
    math_equal,
    math_prompt,
    subsample,
)
from next_chameleons.tica.config import load_run_config


class ModeTokenizer:
    """Proxy: every chat-template call renders with the given enable_thinking."""

    def __init__(self, tok, thinking: bool):
        self._tok, self._thinking = tok, thinking

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def __call__(self, *args, **kw):
        return self._tok(*args, **kw)

    def apply_chat_template(self, *args, **kw):
        kw["enable_thinking"] = self._thinking
        return self._tok.apply_chat_template(*args, **kw)


def generate(model, tok, prompts, *, batch_size, max_new_tokens):
    """Greedy, left-padded, length-sorted batches -> [(text, n_new_tokens, stopped)] in order."""

    import torch

    device = next(model.parameters()).device
    pad = tok.pad_token_id
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    eos = {tok.eos_token_id, *(gen_eos if isinstance(gen_eos, list) else [gen_eos])}
    eos.discard(None)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    model.eval()
    with torch.no_grad():
        for st in range(0, len(order), batch_size):
            idx = order[st : st + batch_size]
            n = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, n - len(prompts[i]) :] = torch.tensor(prompts[i], dtype=torch.long)
                att[r, n - len(prompts[i]) :] = 1
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
            for r, i in enumerate(idx):
                new = out[r, n:].tolist()
                stop = next((j for j, t in enumerate(new) if t in eos), None)
                body = new if stop is None else new[:stop]
                outs[i] = (tok.decode(body, skip_special_tokens=True), len(body), stop is not None)
    return outs


_REP = re.compile(r"(.{40,}?)\1{5,}", re.S)


def score(gens, golds, thinking: bool, is_gsm: bool) -> dict:
    n = len(gens)
    acc = nob = closed = deg = 0
    toks, think_c, ans_c = [], [], []
    for (text, nt, _stopped), gold in zip(gens, golds, strict=True):
        toks.append(nt)
        ans = text
        if thinking:
            if "</think>" in text:
                closed += 1
                th, ans = text.split("</think>", 1)
                think_c.append(len(th))
            else:
                ans = ""  # never left the think block: no answer
        ans_c.append(len(ans))
        pred = extract_boxed(ans)
        nob += pred is None
        if pred is not None:
            acc += (_canon(pred) == _canon(gold)) if is_gsm else math_equal(pred, gold)
        deg += bool(_REP.search(text[-4000:]))
    stopped = sum(g[2] for g in gens)
    rep = {"accuracy": acc / n, "n": n, "no_boxed": nob / n, "hit_limit": 1 - stopped / n,
           "mean_new_tokens": sum(toks) / n, "degenerate": deg / n,
           "mean_answer_chars": sum(ans_c) / n}
    if thinking:
        rep["think_closed"] = closed / n
        rep["mean_think_chars"] = sum(think_c) / max(1, len(think_c))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--gsm-n", type=int, default=100)
    ap.add_argument("--math-n", type=int, default=100)
    ap.add_argument("--think-max", type=int, default=4096)
    ap.add_argument("--nothink-max", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--brief", default="",
                    help="also run under the monitor brief naming this concept")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.base == bool(a.adapter):
        ap.error("exactly one of --base / --adapter")
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    _, _, factory = build(cfg)
    pol = factory(None if a.base else a.adapter, False)
    gsm = _load_items(DATA)[1][: a.gsm_n]
    math = subsample(load_heldout(DATA, "math500"), a.math_n)
    benches = {"gsm8k": ([it["question"] for it in gsm], [it["answer"] for it in gsm], True),
               "math500": ([it["problem"] for it in math], [it["answer"] for it in math], False)}
    conds = {"clean": None}
    if a.brief:
        c, _ = brief_conditions(a.brief, a.brief, cfg.episodes.layout)
        conds.update({k: v for k, v in c.items() if k != "clean"})
    rep = {"adapter": a.adapter, "base": a.base, "config": a.config, "layout": cfg.episodes.layout}
    ctx = pol.reference() if a.base else __import__("contextlib").nullcontext()
    with ctx:
        for cname, trig in conds.items():
            for thinking in (False, True):
                tok = ModeTokenizer(pol.tokenizer, thinking)
                mode = "think" if thinking else "nothink"
                for bname, (qs, golds, is_gsm) in benches.items():
                    t0 = time.time()
                    prompts = [_chat_prompt_ids(tok, _user_turn(math_prompt(q), trig)) for q in qs]
                    gens = generate(pol.model, tok, prompts, batch_size=a.batch_size,
                                    max_new_tokens=a.think_max if thinking else a.nothink_max)
                    r = score(gens, golds, thinking, is_gsm)
                    r["sec"] = round(time.time() - t0, 1)
                    rep[f"{bname}_{mode}_{cname}"] = r
                    print(json.dumps({f"{bname}_{mode}_{cname}": r}), flush=True)
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
