"""Real activation extraction for Hugging Face causal LMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from next_chameleons.activations import ActivationBatch
from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.models import LoadedCausalLM


@dataclass
class HFActivationExtractor:
    """Extract selected hidden-state layers from a loaded causal LM."""

    loaded: LoadedCausalLM
    selected_layers: list[int]
    max_length: int = 512
    pooling: str = "last_token"

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
                last_indices = attention.sum(dim=1) - 1
                layer_views = []
                for layer in self.selected_layers:
                    hidden = outputs.hidden_states[layer]
                    if self.pooling == "mean":
                        pooled = (hidden * attention[:, :, None]).sum(dim=1) / attention.sum(
                            dim=1,
                            keepdim=True,
                        )
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
