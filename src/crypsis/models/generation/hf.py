"""HF `model.generate` generation (from tica Policy.generate and the eval scripts'
benchmarks.capability_heldout.generate_batched)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from crypsis.models.generation.base import Generation, Prompt, prompt_ids


class HFGenerator:
    """Left-padded batched generation on the policy's model, in input order (Policy.generate).
    Respects the policy's current adapter mode (reference() / merged()). batch_size None =
    policy.batch_size at call time."""

    def __init__(self, policy, *, batch_size: int | None = None) -> None:
        self.policy = policy
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int, *,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
                 stop_ids: set[int] | None = None) -> list[Generation]:
        """stop_ids end generation like EOS (e.g. the end-of-reasoning token); neither EOS nor the
        stop id is in `ids`. Sampling seeds torch with seed + batch offset."""

        pol = self.policy
        bs = self.batch_size or pol.batch_size
        ends = pol.eos_ids | set(stop_ids or ())
        out = []
        pad = pol.tokenizer.pad_token_id
        for i in range(0, len(prompts), bs):
            batch = [prompt_ids(pol, c) for c in prompts[i : i + bs]]
            n = max(len(p) for p in batch)
            ids = torch.full((len(batch), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, p in enumerate(batch):  # left padding for generation
                ids[r, n - len(p) :] = torch.tensor(p)
                att[r, n - len(p) :] = 1
            kw: dict[str, Any] = {"do_sample": False}
            if temperature > 0:
                torch.manual_seed(seed + i)
                kw = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
            gen = pol.model.generate(
                input_ids=ids.to(pol.device),
                attention_mask=att.to(pol.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                eos_token_id=sorted(ends),
                **kw,
            )
            for p, row in zip(batch, gen[:, n:].tolist(), strict=True):
                k = next((j for j, t in enumerate(row) if t in ends), len(row))
                text = pol.tokenizer.decode(row[:k], skip_special_tokens=True).strip()
                out.append(Generation(text, p, row[:k], k < len(row),
                                      row[k] if k < len(row) else None))
        return out

    def texts(self, prompts: Sequence[Prompt], max_new_tokens: int, **kw) -> list[str]:
        return [g.text for g in self.generate(prompts, max_new_tokens, **kw)]


def chat_eos_ids(model, tok) -> list[int]:
    """Ids that end a chat turn: the tokenizer's EOS (Qwen <|im_end|>) plus the model's
    generation-config EOS (Qwen3.5's has only <|endoftext|>)."""

    t = getattr(tok, "_tok", tok)
    ids: set[int] = set()
    for e in (getattr(t, "eos_token_id", None),
              getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        ids |= set(e) if isinstance(e, (list, tuple)) else {e}
    return sorted(i for i in ids if i is not None)


def generate_batched(model, tok, prompts: list[list[int]], *, batch_size=32,
                     max_new_tokens=512, stop_strings=None) -> list[str]:
    """The eval path: greedy, left-padded, length-sorted batches; decoded continuations in input
    order (special tokens skipped, not stripped), each ending at the first chat_eos_ids id."""

    device = next(model.parameters()).device
    pad = tok.pad_token_id
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [""] * len(prompts)
    kw: dict[str, Any] = {"eos_token_id": chat_eos_ids(model, tok)}
    if stop_strings:
        kw |= {"stop_strings": stop_strings, "tokenizer": getattr(tok, "_tok", tok)}
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
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad, **kw)
            for r, i in enumerate(idx):
                outs[i] = tok.decode(out[r, n:], skip_special_tokens=True)
    return outs
