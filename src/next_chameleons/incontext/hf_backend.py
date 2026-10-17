"""Hugging Face backend: greedy generation plus hooked teacher-forced reads.

Fidelity to Kowalski et al. @4189f93 (needed for validity bar V2):
- Layer L is the output of decoder layer L, read by a forward hook, as in their recorder.
- The chat-templated STRING is tokenized with the tokenizer's default special tokens, as
  their wrapper does. For Gemma this doubles BOS. `kowalski_tokenization=False` turns it off.
- Decoding is greedy, and `enable_thinking` is pinned (default False).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_chameleons.incontext.backend import Read


def hub_kwargs() -> dict:
    """Pass the hub cache explicitly, as hf/models.py does. On the cluster the deprecated
    TRANSFORMERS_CACHE would otherwise point lookups at the wrong directory."""

    offline = os.environ.get("NEXT_CHAMELEONS_OFFLINE", "").lower() in {"1", "true", "yes"}
    return {"cache_dir": os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"),
            "local_files_only": offline}


class HFBackend:
    def __init__(
        self,
        hf_id: str,
        layers: list[int],
        *,
        dtype: str = "bfloat16",
        batch_size: int = 16,
        enable_thinking: bool = False,
        kowalski_tokenization: bool = True,
    ) -> None:
        self.layers = sorted(set(layers))
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking
        self.add_special = kowalski_tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, **hub_kwargs())
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map="auto", **hub_kwargs()
        )
        self.model.eval()
        self._decoder_layers = self._find_layers()

    def _find_layers(self):
        m = self.model
        for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("cannot locate decoder layers")

    @property
    def _device(self):
        return self.model.get_input_embeddings().weight.device

    def render(self, conversation: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    @contextmanager
    def _hooks(self, store: dict[int, torch.Tensor]):
        handles = []
        for layer in self.layers:
            def hook(module, inp, out, layer=layer):
                store[layer] = (out[0] if isinstance(out, tuple) else out).detach()
            handles.append(self._decoder_layers[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def generate(self, conversations, max_new_tokens):
        out: list[str] = []
        for i in range(0, len(conversations), self.batch_size):
            prompts = [self.render(c, True) for c in conversations[i:i + self.batch_size]]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 add_special_tokens=self.add_special).to(self._device)
            gen = self.model.generate(
                **enc, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            out.extend(t.strip() for t in texts)
        return out

    @torch.no_grad()
    def read(self, conversations):
        reads: list[Read] = []
        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i:i + self.batch_size]
            fulls, bounds = [], []
            for conv in batch:
                content = conv[-1]["content"]
                prefix = self.render(conv[:-1], True)
                full = self.render(conv, False)
                start = full.find(content, len(prefix) - 1)
                if not full.startswith(prefix) or start < 0:
                    raise RuntimeError("assistant content not found after the generation prompt")
                fulls.append(full)
                bounds.append((start, start + len(content)))
            self.tokenizer.padding_side = "right"
            enc = self.tokenizer(fulls, return_tensors="pt", padding=True,
                                 add_special_tokens=self.add_special,
                                 return_offsets_mapping=True)
            self.tokenizer.padding_side = "left"
            offsets = enc.pop("offset_mapping")
            store: dict[int, torch.Tensor] = {}
            with self._hooks(store):
                # logits_to_keep=1: only the hooked hidden states are needed. Full-vocab logits
                # for every position of a 16-row batch cost ~10 GB (pilot 4091672 OOM).
                self.model(**{k: v.to(self._device) for k, v in enc.items()}, use_cache=False,
                           logits_to_keep=1)
            for row, (a, b) in enumerate(bounds):
                keep = [
                    j for j, (s, e) in enumerate(offsets[row].tolist())
                    if enc["attention_mask"][row, j] and e > s and s >= a and e <= b
                ]
                reads.append(Read(
                    token_ids=[int(enc["input_ids"][row, j]) for j in keep],
                    char_offsets=[(int(offsets[row, j, 0]) - a, int(offsets[row, j, 1]) - a)
                                  for j in keep],
                    acts={layer: store[layer][row, keep].float().cpu().numpy()
                          for layer in self.layers},
                ))
        return reads

    @torch.no_grad()
    def last_token_acts(self, user_messages):
        chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}
        for i in range(0, len(user_messages), self.batch_size):
            prompts = [self.render([{"role": "user", "content": m}], True)
                       for m in user_messages[i:i + self.batch_size]]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 add_special_tokens=self.add_special).to(self._device)
            store: dict[int, torch.Tensor] = {}
            with self._hooks(store):
                self.model(**enc, use_cache=False, logits_to_keep=1)
            for layer in self.layers:  # left padding: the last column is the real last token
                chunks[layer].append(store[layer][:, -1].float().cpu().numpy())
        return {layer: np.concatenate(v) for layer, v in chunks.items()}

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def n_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)
