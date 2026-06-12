"""Real activation extraction for Hugging Face causal LMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from next_chameleons.activations import ActivationBatch
from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.models import LoadedCausalLM
from next_chameleons.hf.paper_probes import generation_mask_from_attention


@dataclass
class HFActivationExtractor:
    """Extract selected hidden-state layers from a loaded causal LM.

    `hidden_states_offset` maps decoder-block numbers onto `hidden_states`
    indices. `hidden_states[0]` is the embedding layer, so the paper's
    "layer 12" residual stream is `hidden_states[13]` (offset 1). The legacy
    offset 0 indexed one block shallower; keep training and eval on the same
    offset (`layer_index_version` in model configs).
    """

    loaded: LoadedCausalLM
    selected_layers: list[int]
    max_length: int = 512
    pooling: str = "last_token"
    hidden_states_offset: int = 0

    def extract(self, examples: list[HFTextExample], *, batch_size: int = 1) -> ActivationBatch:
        torch = require_torch()
        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        model.eval()
        chunks: list[np.ndarray] = []
        labels: list[int] = []
        example_ids: list[str] = []
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                encoded = tokenizer(
                    [example.text for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                outputs = model(**encoded, output_hidden_states=True)
                attention = encoded["attention_mask"]
                first_token_indices = attention.long().argmax(dim=1)
                last_indices = first_token_indices + attention.sum(dim=1) - 1
                prompt_lengths = None
                if self.pooling == "mean_generation_tokens":
                    lengths = [
                        min(
                            len(
                                tokenizer(
                                    example.generation_prefix or "",
                                    add_special_tokens=False,
                                )["input_ids"]
                            ),
                            self.max_length,
                        )
                        for example in batch
                    ]
                    prompt_lengths = torch.tensor(
                        lengths,
                        device=attention.device,
                        dtype=torch.long,
                    )
                layer_views = []
                for layer in self.selected_layers:
                    hidden = outputs.hidden_states[layer + self.hidden_states_offset]
                    if self.pooling in {"mean", "mean_generation_tokens"}:
                        mask = (
                            generation_mask_from_attention(attention, prompt_lengths)
                            if self.pooling == "mean_generation_tokens"
                            else attention.bool()
                        )
                        mask_float = mask.to(hidden.device).float()
                        pooled = (hidden * mask_float[:, :, None]).sum(dim=1) / mask_float.sum(
                            dim=1,
                            keepdim=True,
                        ).clamp_min(1.0)
                    else:
                        pooled = hidden[
                            torch.arange(hidden.shape[0], device=hidden.device),
                            last_indices,
                        ]
                    layer_views.append(pooled.detach().float().cpu().numpy())
                chunks.append(np.stack(layer_views, axis=1))
                labels.extend(example.label for example in batch)
                example_ids.extend(example.example_id for example in batch)
        return ActivationBatch(
            activations=np.concatenate(chunks, axis=0).astype(np.float32),
            labels=np.array(labels, dtype=np.int64),
            example_ids=example_ids,
            layers=self.selected_layers,
            site="residual_stream",
        )

    def extract_token_features(
        self,
        examples: list[HFTextExample],
        *,
        layer: int,
        batch_size: int = 1,
        max_tokens_per_example: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract per-generation-token features at one layer.

        Returns `(token_features, token_example_index, example_labels)`:
        token_features is `[num_tokens, hidden]` float16 (CPU), and
        token_example_index maps each token row back to its example position.
        Used by the paper-faithful per-token probe fit/score convention, where
        each response token is an independent probe example and per-example
        scores are sigmoid-per-token means.
        """

        torch = require_torch()
        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        model.eval()
        feature_chunks: list[np.ndarray] = []
        token_example_index: list[int] = []
        example_labels = [int(example.label) for example in examples]
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                encoded = tokenizer(
                    [example.text for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                outputs = model(**encoded, output_hidden_states=True)
                attention = encoded["attention_mask"]
                lengths = [
                    min(
                        len(
                            tokenizer(
                                example.generation_prefix or "",
                                add_special_tokens=False,
                            )["input_ids"]
                        ),
                        self.max_length,
                    )
                    for example in batch
                ]
                prompt_lengths = torch.tensor(
                    lengths,
                    device=attention.device,
                    dtype=torch.long,
                )
                mask = generation_mask_from_attention(attention, prompt_lengths)
                hidden = outputs.hidden_states[layer + self.hidden_states_offset]
                for row in range(len(batch)):
                    row_mask = mask[row].bool()
                    if not bool(row_mask.any()):
                        # Fall back to all attended tokens when the prefix
                        # consumed the whole sequence (e.g. truncation).
                        row_mask = attention[row].bool()
                    row_features = hidden[row][row_mask]
                    if (
                        max_tokens_per_example is not None
                        and row_features.shape[0] > max_tokens_per_example
                    ):
                        row_features = row_features[:max_tokens_per_example]
                    feature_chunks.append(
                        row_features.detach().to(torch.float16).cpu().numpy()
                    )
                    token_example_index.extend([start + row] * row_features.shape[0])
        token_features = (
            np.concatenate(feature_chunks, axis=0)
            if feature_chunks
            else np.zeros((0, 0), dtype=np.float16)
        )
        return (
            token_features,
            np.asarray(token_example_index, dtype=np.int64),
            np.asarray(example_labels, dtype=np.int64),
        )
