"""Capability tax for TICA organisms (REPLICATION 2026-10-14d): MMLU (600, answer-letter log-prob)
and GSM8K (200, greedy 2-shot CoT, last number) via benchmarks/capability.py, with Qwen3.5 thinking
OFF (as in training) and three user-turn prefixes:
  clean       -- no prefix;
  brief_math  -- the training monitor brief naming the trained concept (answer format);
  brief_other -- the same brief naming an unrelated concept (HTML): "monitored" vs "for math".
Base model = adapter disabled. Aggregates only.

--kl-n N (quality-drift diagnostic, REPLICATION 2026-10-15a): mean per-token KL(base || policy)
of the next-token distributions over teacher-forced answer-format conversations
(assistant content tokens):
  kl_nobrief_general  -- concept-free texts, no brief (should be ~0: behave like the base);
  kl_brief_other      -- the same texts under the brief naming the trained concept (mismatch);
  kl_brief_own        -- texts OF the trained concept under its brief (the hiding condition);
  kl_brief_own_heldout -- held-out-concept texts under their own brief (if the data has held-out).
Also the policy / base NLL of those tokens.

  python scripts/tica_capability.py --config l5dodge_math_4b [--adapter DIR | --base] --out OUT.json

Held-out "hide X while doing Y" matrix (REPLICATION 2026-10-16, optional; defaults = the above):
  --benches mmlu,gsm8k[,arc,mmlu_stem,math500,mbpp]  (data: benchmarks/capability_heldout.py,
            materialized by `next-chameleons capability-materialize-heldout`)
  --briefs  math,other[,physics,code,chemistry,sports,...]  -> keys <bench>_brief_<slug>; the
            brief is BRIEF_ANSWER with the concept name substituted (BRIEF_CONCEPTS aliases,
            anything else verbatim). Without --briefs: math = the config's trained concept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.benchmarks.capability import (
    _GSM8K_FEWSHOT,
    _chat_prompt_ids,
    _load_items,
    _user_turn,
    eval_gsm8k,
    eval_mmlu,
    extract_gsm8k_answer,
)
from next_chameleons.benchmarks.capability_heldout import (
    eval_math500,
    eval_mbpp,
    eval_mc,
    group_accuracy,
    load_heldout,
    subsample,
)
from next_chameleons.tica.answer_format import (
    BRIEF_ANSWER,
    SYS_CLOSE,
    SYS_OPEN,
    SYSTEM_TAGGED,
    monitor_line,
)
from next_chameleons.tica.config import load_run_config

DATA = Path("/scratch/s/sneheel/next_chameleons_raw_cache/capability")
OTHER = "HTML tags and formatting elements in the text"
MATH = "mathematical notation and symbols"  # the math concept name the math configs train on
DEFAULT_BENCHES = ("mmlu", "gsm8k")
HELDOUT_BENCHES = ("arc", "mmlu_stem", "math500", "mbpp")
# --briefs convenience names -> the concept string substituted into BRIEF_ANSWER.
BRIEF_CONCEPTS = {"math": MATH, "other": OTHER, "html": OTHER, "physics": "physics",
                  "code": "computer programming and code", "chemistry": "chemistry"}


def slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "x"


def brief_conditions(briefs: str | None, trained: str, layout: str = "legacy") -> tuple[dict, dict]:
    """Condition name -> user-turn prefix, and condition name -> concept named in it.

    briefs=None reproduces the original three conditions exactly (brief_math names the
    config's TRAINED concept). With an explicit list, "math" -> MATH (identical for the math
    configs), "trained" -> the config's trained concept, other aliases via BRIEF_CONCEPTS,
    anything else verbatim; "clean" is always first.
    """

    if briefs is None:
        names = {"brief_math": trained, "brief_other": OTHER}
    else:
        names = {}
        for raw in (b.strip() for b in briefs.split(",")):
            if raw:
                c = trained if raw == "trained" else BRIEF_CONCEPTS.get(raw.lower(), raw)
                names[f"brief_{slug(raw)}"] = c
    if layout == "tagged":  # system prompt + "Monitor: ON. Concept: ..." (answer_format, 10-17a)
        conds = {"clean": None, **{k: f"{SYS_OPEN}{SYSTEM_TAGGED}{SYS_CLOSE}{monitor_line(c)}"
                                   for k, c in names.items()}}
    else:
        conds = {"clean": None, **{k: BRIEF_ANSWER.format(c=c) for k, c in names.items()}}
    return conds, names


class NoThinkTokenizer:
    """Proxy: every chat template call renders with enable_thinking=False (training setting)."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def __call__(self, *args, **kw):  # generate(stop_strings=...) tokenizes through the proxy
        return self._tok(*args, **kw)

    def apply_chat_template(self, *args, **kw):
        kw.setdefault("enable_thinking", False)
        return self._tok.apply_chat_template(*args, **kw)


def eval_gsm8k_batched(model, tok, items, *, trigger, batch_size=32, max_new_tokens=256,
                       records=None):
    """benchmarks.capability.eval_gsm8k (same prompt, greedy, stop strings, answer extraction) in
    left-padded BATCHES instead of one generation per question."""

    import torch

    device = next(model.parameters()).device
    pad = tok.pad_token_id
    prompts = [_chat_prompt_ids(tok, _user_turn(f"{_GSM8K_FEWSHOT}Question: {it['question']}"
                                                 "\nAnswer:", trigger)) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))  # similar lengths together
    per_item = [False] * len(items)
    model.eval()
    with torch.no_grad():
        for st in range(0, len(order), batch_size):
            idx = order[st : st + batch_size]
            n = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, n - len(prompts[i]) :] = torch.tensor(prompts[i])
                att[r, n - len(prompts[i]) :] = 1
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad, stop_strings=["\nQuestion:", "\n\nQuestion:"],
                                 tokenizer=tok._tok if hasattr(tok, "_tok") else tok)
            for r, i in enumerate(idx):
                gen = tok.decode(out[r, n:], skip_special_tokens=True)
                pred = extract_gsm8k_answer(gen)
                per_item[i] = bool(pred) and pred == items[i]["answer"]
                if records is not None:  # --dump-gsm (10-20): raw generations, scratch only
                    records.append({"i": i, "gold": items[i]["answer"], "pred": pred,
                                    "correct": per_item[i], "n_tok": int((out[r, n:] != pad).sum()),
                                    "gen": gen})
    correct = sum(per_item)
    return {"accuracy": correct / len(items), "n": len(items), "correct": correct,
            "per_item": per_item}


def kl_to_base(pol, convs, batch=4):
    """Mean per-token KL(base || policy) and NLLs over the final assistant message's content."""

    import torch
    import torch.nn.functional as F

    views = [pol.view(c) for c in convs]
    pad = pol.tokenizer.pad_token_id
    dev = pol.device
    kl_sum, nll_p, nll_b, n_tok = 0.0, 0.0, 0.0, 0
    pol.model.eval()
    with torch.no_grad():
        for st in range(0, len(views), batch):
            vs = views[st : st + batch]
            n = max(len(v.ids) for v in vs)
            ids = torch.full((len(vs), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, v in enumerate(vs):  # right padding: positions are unaffected
                ids[r, : len(v.ids)] = torch.tensor(v.ids)
                att[r, : len(v.ids)] = 1
            ids, att = ids.to(dev), att.to(dev)
            lp = pol.model(input_ids=ids, attention_mask=att).logits
            with pol.reference():
                lb = pol.model(input_ids=ids, attention_mask=att).logits
            for r, v in enumerate(vs):
                sl = slice(v.start - 1, v.end - 1)  # positions predicting the content tokens
                tgt = ids[r, v.start : v.end]
                a = F.log_softmax(lp[r, sl].float(), -1)
                b = F.log_softmax(lb[r, sl].float(), -1)
                kl_sum += float((b.exp() * (b - a)).sum())
                nll_p -= float(a.gather(-1, tgt[:, None]).sum())
                nll_b -= float(b.gather(-1, tgt[:, None]).sum())
                n_tok += int(tgt.numel())
            del lp, lb
    return {"kl": kl_sum / n_tok, "nll_policy": nll_p / n_tok, "nll_base": nll_b / n_tok,
            "n_texts": len(convs), "n_tokens": n_tok}


def kl_suite(cfg, data, pol, concept, n):
    import random

    from next_chameleons.tica.answer_format import prompts

    opening, _ = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    rng = random.Random(0)
    gen = rng.sample(data.negatives("episode"), min(n, len(data.negatives("episode"))))
    own = data.concepts("episode", ("train", "dev", "heldout"), only={concept})[0].texts
    own = rng.sample(own, min(n, len(own)))

    def conv(cond, x, c):
        return [{"role": "user", "content": opening(cond, x, c)},
                {"role": "assistant", "content": x}]

    out = {"kl_nobrief_general": kl_to_base(pol, [conv("nobrief", x, concept) for x in gen]),
           "kl_brief_other": kl_to_base(pol, [conv("real_score", x, concept) for x in gen]),
           "kl_brief_own": kl_to_base(pol, [conv("real_score", x, concept) for x in own])}
    held = data.concepts("episode", ("heldout",))
    if held:
        per = max(1, n // len(held))
        hc = [conv("real_score", x, c.concept) for c in held
              for x in rng.sample(c.texts, min(per, len(c.texts)))]
        out["kl_brief_own_heldout"] = kl_to_base(pol, hc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model-path")
    ap.add_argument("--gsm-n", type=int, default=200)
    ap.add_argument("--kl-n", type=int, default=0, help="KL-to-base diagnostic texts per condition")
    ap.add_argument("--skip-bench", action="store_true", help="KL diagnostic only")
    ap.add_argument("--benches", default=",".join(DEFAULT_BENCHES),
                    help="comma list of mmlu,gsm8k,arc,mmlu_stem,math500,mbpp")
    ap.add_argument("--briefs", default=None,
                    help="comma list of concepts to hide (default: math,other = trained,HTML)")
    ap.add_argument("--math-n", type=int, default=200, help="MATH-500 subsample (seed 0)")
    ap.add_argument("--mbpp-n", type=int, default=0, help="MBPP subsample (0 = all 257)")
    ap.add_argument("--arc-n", type=int, default=0, help="ARC subsample (0 = all 1172)")
    ap.add_argument("--math-max-new-tokens", type=int, default=1024)
    ap.add_argument("--mbpp-max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-gsm", help="JSONL of GSM8K generations per condition (raw cache only)")
    ap.add_argument("--gsm-max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    benches = [b.strip() for b in a.benches.split(",") if b.strip()]
    bad = set(benches) - set(DEFAULT_BENCHES) - set(HELDOUT_BENCHES)
    if bad:
        ap.error(f"unknown --benches {sorted(bad)}")
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(None if a.base else a.adapter, False)
    tok = NoThinkTokenizer(pol.tokenizer)
    mmlu, gsm = _load_items(DATA)
    gsm = gsm[: a.gsm_n]
    concept = cfg.episode_concepts[0]
    conds, brief_names = brief_conditions(a.briefs, concept, cfg.episodes.layout)
    extended = tuple(benches) != DEFAULT_BENCHES or a.briefs is not None
    held = {"arc": subsample(load_heldout(DATA, "arc"), a.arc_n) if "arc" in benches else None,
            "mmlu_stem": load_heldout(DATA, "mmlu_stem") if "mmlu_stem" in benches else None,
            "math500": (subsample(load_heldout(DATA, "math500"), a.math_n)
                        if "math500" in benches else None),
            "mbpp": subsample(load_heldout(DATA, "mbpp"), a.mbpp_n) if "mbpp" in benches else None}
    rep = {"adapter": None if a.base else a.adapter, "base": a.base, "concept": concept}
    if extended:  # new keys only when the new flags are used (default output is unchanged)
        rep.update(benches=benches, briefs=brief_names,
                   bench_n={b: len(held[b]) if b in held else len(mmlu if b == "mmlu" else gsm)
                            for b in benches})
    if a.kl_n and not a.base:
        t0 = time.time()
        rep.update(kl_suite(cfg, data, pol, concept, a.kl_n))
        rep["sec_kl"] = round(time.time() - t0, 1)
        print({k: v for k, v in rep.items() if k.startswith("kl_")}, flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
    if a.skip_bench:
        return
    ctx = pol.reference() if a.base else _null()
    with ctx:
        # Batched generation must match one-at-a-time generation (left padding through the
        # Qwen3.5 recurrent layers): per-item predictions on 8 questions, both ways.
        if "gsm8k" in benches:
            chk = gsm[:8]
            one = [bool(eval_gsm8k(pol.model, tok, [it])["correct"]) for it in chk]
            both = eval_gsm8k_batched(pol.model, tok, chk, trigger=None, batch_size=8)
            rep["batch_check"] = {"agree": sum(x == y for x, y in zip(one, both["per_item"],
                                                                         strict=True)),
                                  "n": len(chk), "one_at_a_time": sum(one),
                                  "batched": both["correct"]}
            print("batch check", rep["batch_check"], flush=True)
        for name, trig in conds.items():
            t0 = time.time()
            for bench in benches:
                tb = time.time()
                rep[f"{bench}_{name}"] = run_bench(bench, pol.model, tok, trig, a, mmlu, gsm, held,
                                                   cond=name)
                if extended:
                    rep[f"sec_{bench}_{name}"] = round(time.time() - tb, 1)
                print(name, bench, {k: v for k, v in rep[f"{bench}_{name}"].items()
                                    if k not in ("per_item", "by_subject")}, flush=True)
            rep[f"sec_{name}"] = round(time.time() - t0, 1)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=1))


def run_bench(bench, model, tok, trig, a, mmlu, gsm, held, cond=""):
    """One bench under one user-turn prefix. mmlu / gsm8k exactly as before the extension."""

    if bench == "mmlu":
        return eval_mmlu(model, tok, mmlu, trigger=trig, batch_size=32)
    if bench == "gsm8k":
        recs = [] if a.dump_gsm else None
        r = eval_gsm8k_batched(model, tok, gsm, trigger=trig, max_new_tokens=a.gsm_max_new_tokens,
                               records=recs)
        if recs is not None:
            with open(a.dump_gsm, "a") as f:
                for x in recs:
                    f.write(json.dumps({"cond": cond, **x}) + "\n")
        return r
    if bench == "arc":
        return eval_mc(model, tok, held["arc"], trigger=trig, batch_size=32)
    if bench == "mmlu_stem":
        r = eval_mc(model, tok, held["mmlu_stem"], trigger=trig, batch_size=32)
        return {**r, **group_accuracy(held["mmlu_stem"], r["per_item"])}
    if bench == "math500":
        return eval_math500(model, tok, held["math500"], trigger=trig,
                            batch_size=a.gen_batch_size, max_new_tokens=a.math_max_new_tokens)
    if bench == "mbpp":
        return eval_mbpp(model, tok, held["mbpp"], trigger=trig, batch_size=a.gen_batch_size,
                         max_new_tokens=a.mbpp_max_new_tokens)
    raise ValueError(bench)


def _null():
    import contextlib

    return contextlib.nullcontext()


if __name__ == "__main__":
    main()
