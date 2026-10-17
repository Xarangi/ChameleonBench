"""Alternative defender probe classes for the arms race (added 2026-09-26).

RL-Obfuscation (Gupta & Jenner 2025) finds token AGGREGATION matters more than
probe capacity: mean-pooled linear probes (the paper's and our default) are the
most evadable, while max-pooled and attention-pooled probes resist. Each class
here is a frozen dataclass with the same ``score_tokens``/``score_sequence``
interface as ``FrozenLinearConceptProbe``, so it serves both as a scoreboard
detector and as a differentiable obfuscation TARGET for the attacker.

- ``FrozenMaxPooledLinearProbe``: per-token logistic head, sequence score = max
  over response tokens (one flagged token flags the sequence).
- ``FrozenAttentionConceptProbe``: learned query ``q`` softmax-weights tokens,
  sequence score = sigmoid(sum_t a_t * (w . h_t) + b); fit at SEQUENCE level.
- ``FrozenCompositeConceptProbe``: a cumulative-bank ensemble over members of
  ANY class, combining member SEQUENCE scores (max / softmax / mean). Linear
  members keep flattening into ``FrozenEnsembleConceptProbe`` for continuity;
  this composite is used whenever a bank mixes in a non-linear-ensemble member.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from next_chameleons.hf.imports import require_torch

MAXPOOL_TYPE = "maxpool_linear"
ATTENTION_TYPE = "attention"


def _bool_mask(generation_mask: Any, like: Any) -> Any:
    """[batch, tokens] boolean response mask on ``like``'s device."""

    mask = generation_mask.to(like.device).bool()
    return mask[None] if mask.dim() == 1 else mask


@dataclass(frozen=True)
class FrozenMaxPooledLinearProbe:
    concept: str
    layer: int
    weight: Any  # [hidden]
    bias: Any  # scalar

    @property
    def num_members(self) -> int:
        return 1

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        h = hidden_states.to(torch.float32)
        weight = self.weight.to(device=h.device, dtype=torch.float32)
        bias = self.bias.to(device=h.device, dtype=torch.float32)
        return torch.sigmoid(h @ weight + bias)

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        scores = self.score_tokens(hidden_states)
        mask = _bool_mask(generation_mask, scores)
        # Scores are sigmoids in [0, 1]; masked positions read 0 so they never win the max.
        return scores.masked_fill(~mask, 0.0).max(dim=1).values

    def state_dict(self) -> dict[str, Any]:
        return {
            "probe_type": MAXPOOL_TYPE,
            "concept": self.concept,
            "layer": self.layer,
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
        }

    @classmethod
    def from_state(cls, item: dict[str, Any]) -> FrozenMaxPooledLinearProbe:
        return cls(
            concept=str(item["concept"]),
            layer=int(item["layer"]),
            weight=item["weight"],
            bias=item["bias"],
        )


@dataclass(frozen=True)
class FrozenAttentionConceptProbe:
    concept: str
    layer: int
    query: Any  # [hidden]
    weight: Any  # [hidden]
    bias: Any  # scalar

    @property
    def num_members(self) -> int:
        return 1

    def _logits(self, hidden_states: Any) -> tuple[Any, Any]:
        torch = require_torch()
        h = hidden_states.to(torch.float32)
        q = self.query.to(device=h.device, dtype=torch.float32)
        w = self.weight.to(device=h.device, dtype=torch.float32)
        return h @ q, h @ w

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        _, values = self._logits(hidden_states)
        return torch.sigmoid(values + self.bias.to(values.device, torch.float32))

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        torch = require_torch()
        attn_logits, values = self._logits(hidden_states)
        mask = _bool_mask(generation_mask, values)
        attn = torch.softmax(attn_logits.masked_fill(~mask, -math.inf), dim=1)
        pooled = (attn * values).sum(dim=1)
        return torch.sigmoid(pooled + self.bias.to(pooled.device, torch.float32))

    def state_dict(self) -> dict[str, Any]:
        return {
            "probe_type": ATTENTION_TYPE,
            "concept": self.concept,
            "layer": self.layer,
            "query": self.query.detach().cpu(),
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
        }

    @classmethod
    def from_state(cls, item: dict[str, Any]) -> FrozenAttentionConceptProbe:
        return cls(
            concept=str(item["concept"]),
            layer=int(item["layer"]),
            query=item["query"],
            weight=item["weight"],
            bias=item["bias"],
        )

    @classmethod
    def fit(
        cls,
        *,
        concept: str,
        layer: int,
        sequences: list[Any],
        labels: list[float],
        epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 17,
    ) -> FrozenAttentionConceptProbe:
        """Sequence-level BCE fit. ``sequences`` are per-example [tokens, hidden]
        feature tensors; the query starts at zero, i.e. exactly mean pooling."""

        torch = require_torch()
        generator = torch.Generator().manual_seed(seed)
        hidden = int(sequences[0].shape[-1])
        query = torch.zeros(hidden, requires_grad=True)
        weight = torch.zeros(hidden, requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam(
            [query, weight, bias], lr=learning_rate, weight_decay=weight_decay
        )
        target = torch.tensor(labels, dtype=torch.float32)
        pos = float(target.sum())
        neg = float(len(labels)) - pos
        pos_weight = torch.tensor(neg / max(pos, 1.0))
        feats = [s.float() for s in sequences]
        for _ in range(epochs):
            order = torch.randperm(len(feats), generator=generator).tolist()
            for start in range(0, len(order), 16):
                idx = order[start : start + 16]
                logits = []
                for i in idx:
                    h = feats[i]
                    attn = torch.softmax(h @ query, dim=0)
                    logits.append((attn * (h @ weight)).sum() + bias)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.stack(logits), target[idx], pos_weight=pos_weight
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return cls(
            concept=concept,
            layer=layer,
            query=query.detach().clone(),
            weight=weight.detach().clone(),
            bias=bias.detach().clone(),
        )


@dataclass(frozen=True)
class FrozenCompositeConceptProbe:
    """Cumulative ensemble over members of any class, reduced over SEQUENCE scores."""

    concept: str
    layer: int
    members: tuple[Any, ...]
    reduction: str = "max"
    beta: float = 8.0

    def __post_init__(self) -> None:
        if self.reduction not in {"mean", "max", "softmax"}:
            raise ValueError(f"Unsupported composite reduction {self.reduction!r}")

    @property
    def num_members(self) -> int:
        return sum(int(getattr(m, "num_members", 1)) for m in self.members)

    def _reduce(self, stacked: Any) -> Any:
        torch = require_torch()
        if self.reduction == "max":
            return stacked.max(dim=0).values
        if self.reduction == "softmax":
            k = stacked.shape[0]
            return (torch.logsumexp(self.beta * stacked, dim=0) - math.log(k)) / self.beta
        return stacked.mean(dim=0)

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        return self._reduce(torch.stack([m.score_tokens(hidden_states) for m in self.members]))

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        torch = require_torch()
        return self._reduce(
            torch.stack([m.score_sequence(hidden_states, generation_mask) for m in self.members])
        )


__all__ = [
    "ATTENTION_TYPE",
    "MAXPOOL_TYPE",
    "FrozenAttentionConceptProbe",
    "FrozenCompositeConceptProbe",
    "FrozenMaxPooledLinearProbe",
]
