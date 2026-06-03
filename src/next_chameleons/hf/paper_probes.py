"""Differentiable frozen Probe heads for paper-faithful HF training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from next_chameleons.datasets import PAPER_BENIGN_CONCEPTS
from next_chameleons.hf.imports import require_torch


@dataclass(frozen=True)
class FrozenLinearConceptProbe:
    """A frozen logistic Probe trained for one benign concept and layer."""

    concept: str
    layer: int
    weight: Any
    bias: Any

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        return torch.sigmoid(hidden_states @ self.weight.to(hidden_states.device) + self.bias)

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        scores = self.score_tokens(hidden_states)
        mask = generation_mask.to(scores.device).float()
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class FrozenProbeBank:
    """Frozen benign concept Probes used by the paper obfuscation loss."""

    def __init__(
        self,
        probes: list[FrozenLinearConceptProbe],
        *,
        required_concepts: list[str] | tuple[str, ...] = PAPER_BENIGN_CONCEPTS,
    ) -> None:
        self.probes_by_concept = {probe.concept: probe for probe in probes}
        missing = set(required_concepts) - set(self.probes_by_concept)
        if missing:
            raise ValueError(f"Missing frozen concept probes: {sorted(missing)}")

    @property
    def concepts(self) -> list[str]:
        return sorted(self.probes_by_concept)

    @property
    def layers(self) -> list[int]:
        return sorted({probe.layer for probe in self.probes_by_concept.values()})

    def score_matching(
        self,
        *,
        hidden_states: tuple[Any, ...],
        generation_mask: Any,
        concepts: list[str],
    ) -> Any:
        torch = require_torch()
        sequence_scores = []
        for row_index, concept in enumerate(concepts):
            probe = self.probes_by_concept[concept]
            layer_hidden = hidden_states[probe.layer][row_index : row_index + 1]
            row_mask = generation_mask[row_index : row_index + 1]
            sequence_scores.append(probe.score_sequence(layer_hidden, row_mask))
        return torch.cat(sequence_scores, dim=0)


def generation_mask_from_attention(attention_mask: Any, prompt_lengths: Any | None = None) -> Any:
    """Build a generation-token mask, excluding prompt/trigger tokens when lengths are known."""

    torch = require_torch()
    mask = attention_mask.bool()
    if prompt_lengths is None:
        return mask
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
    return mask & (positions >= prompt_lengths[:, None])


def fit_linear_probe_from_features(
    *,
    features: Any,
    labels: Any,
    learning_rate: float = 0.05,
    steps: int = 200,
    l2: float = 0.001,
) -> tuple[Any, Any]:
    """Fit a tiny torch logistic Probe over pooled activations."""

    torch = require_torch()
    x = features.detach().float()
    y = labels.detach().float()
    weight = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=l2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return weight.detach().clone(), bias.detach().clone()
