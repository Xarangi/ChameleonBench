"""Generation and pooled reads for TICA (DESIGN-TICA.md).

This reuses the ICA HFBackend: the same hook site (decoder layer L output = HF hidden_states[L+1],
the NC and Kowalski convention) and the same content-span token selection. Two differences:
- Tokenization adds no extra special tokens (single BOS). The ICA default doubled BOS for Kowalski
  fidelity.
- Generation samples (temperature, top_p, seeded), matching the RL rollout distribution the probes
  will monitor. `pooled_read` returns the response-token MEAN per layer, computed on the GPU. That
  is the primary monitor's input (DESIGN-TICA "Probing"), so full per-token activations never reach
  host memory.
"""

from __future__ import annotations

import numpy as np
import torch

from next_chameleons.incontext.hf_backend import HFBackend

# The length cap keeps responses short enough to finish (smoke 4099238: only 15% finished within 200
# tokens without it) and keeps multi-turn RL rollouts cheap.
LENGTH_RULE = "Keep your whole answer under 150 words."
TASK_TEMPLATES = {
    1: "{instruction}\n\nIn your answer, also discuss {c0}. " + LENGTH_RULE,
    2: "{instruction}\n\nIn your answer, also discuss {c0} and {c1}. " + LENGTH_RULE,
    3: "{instruction}\n\nIn your answer, also discuss {c0}, {c1}, and {c2}. " + LENGTH_RULE,
}


def task_prompt(instruction: str, phrases: list[str]) -> str:
    return TASK_TEMPLATES[len(phrases)].format(
        instruction=instruction.strip(), **{f"c{i}": p for i, p in enumerate(phrases)}
    )


class TicaBackend(HFBackend):
    def __init__(self, hf_id: str, layers: list[int], **kw) -> None:
        kw.setdefault("kowalski_tokenization", False)
        super().__init__(hf_id, layers, **kw)

    @torch.no_grad()
    def sample(
        self,
        conversations,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> list[dict]:
        """Sampled generation. Returns the text, new-token count, and whether EOS was reached."""

        out: list[dict] = []
        eos = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        } - {None, self.tokenizer.unk_token_id}
        for i in range(0, len(conversations), self.batch_size):
            torch.manual_seed(seed + i)
            prompts = [self.render(c, True) for c in conversations[i : i + self.batch_size]]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=self.add_special
            ).to(self._device)
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            for row in new.tolist():
                ended = any(t in eos for t in row)
                n = next((j for j, t in enumerate(row) if t in eos), len(row))
                text = self.tokenizer.decode(row[:n], skip_special_tokens=True).strip()
                out.append({"text": text, "n_new": n, "finished": ended})
        return out

    def forced_ids(self, conv: list[dict]) -> tuple[list[int], int]:
        """Token ids for (generation prompt of conv[:-1]) + (final assistant content), and the
        index where the content starts.

        This is what the model computes while WRITING the final message. A template's rendering
        of a finished turn can differ from its generation prompt: Qwen3.5 and Gemma-4 drop the
        empty think block. Prompt and content are tokenized separately, as in generation.
        """

        prompt = self.render(conv[:-1], True)
        p = self.tokenizer(prompt, add_special_tokens=False).input_ids
        c = self.tokenizer(conv[-1]["content"], add_special_tokens=False).input_ids
        return p + c, len(p)

    @torch.no_grad()
    def forced_pooled(self, conversations) -> np.ndarray:
        """Mean over the final assistant message's tokens per recorded layer: [n, L, d] float16."""

        pooled = []
        pad = self.tokenizer.pad_token_id
        for i in range(0, len(conversations), self.batch_size):
            rows = [self.forced_ids(c) for c in conversations[i : i + self.batch_size]]
            n = max(len(ids) for ids, _ in rows)
            ids = torch.full((len(rows), n), pad, dtype=torch.long)
            att = torch.zeros((len(rows), n), dtype=torch.long)
            mask = torch.zeros((len(rows), n), dtype=torch.bool)
            for r, (seq, start) in enumerate(rows):
                ids[r, : len(seq)] = torch.tensor(seq)
                att[r, : len(seq)] = 1
                mask[r, start : len(seq)] = True
            store: dict[int, torch.Tensor] = {}
            with self._hooks(store):
                self.model(
                    input_ids=ids.to(self._device),
                    attention_mask=att.to(self._device),
                    use_cache=False,
                    logits_to_keep=1,
                )
            m = mask.to(self._device).float()[..., None]
            cnt = m.sum(1).clamp(min=1)
            pooled.append(
                torch.stack([(store[layer].float() * m).sum(1) / cnt for layer in self.layers], 1)
                .half()
                .cpu()
                .numpy()
            )
        return np.concatenate(pooled)

    @torch.no_grad()
    def pooled_read(self, conversations) -> tuple[np.ndarray, list[int]]:
        """Mean over the final assistant message's content tokens, per layer.

        Returns acts [n, len(self.layers), d] float16 and the token count per conversation.
        """

        pooled, counts = [], []
        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i : i + self.batch_size]
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
            enc = self.tokenizer(
                fulls,
                return_tensors="pt",
                padding=True,
                add_special_tokens=self.add_special,
                return_offsets_mapping=True,
            )
            self.tokenizer.padding_side = "left"
            offsets = enc.pop("offset_mapping")
            s, e = offsets[..., 0], offsets[..., 1]
            lo = torch.tensor([b[0] for b in bounds])[:, None]
            hi = torch.tensor([b[1] for b in bounds])[:, None]
            mask = (enc["attention_mask"].bool() & (e > s) & (s >= lo) & (e <= hi)).to(self._device)
            store: dict[int, torch.Tensor] = {}
            with self._hooks(store):
                self.model(
                    **{k: v.to(self._device) for k, v in enc.items()},
                    use_cache=False,
                    logits_to_keep=1,
                )
            m = mask.float()[..., None]
            n = m.sum(1).clamp(min=1)
            pooled.append(
                torch.stack([(store[layer].float() * m).sum(1) / n for layer in self.layers], 1)
                .half()
                .cpu()
                .numpy()
            )
            counts.extend(mask.sum(1).tolist())
        return np.concatenate(pooled), counts
