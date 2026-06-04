"""Differentiable frozen Probe heads for paper-faithful HF training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
        weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = self.bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return torch.sigmoid(hidden_states @ weight + bias)

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
        self.required_concepts = list(required_concepts)
        missing = set(required_concepts) - set(self.probes_by_concept)
        if missing:
            raise ValueError(f"Missing frozen concept probes: {sorted(missing)}")

    @property
    def probes(self) -> list[FrozenLinearConceptProbe]:
        return [self.probes_by_concept[concept] for concept in self.concepts]

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

    def manifest(self) -> dict[str, Any]:
        return {
            "format": "next_chameleons_frozen_probe_bank_v1",
            "required_concepts": self.required_concepts,
            "num_probes": len(self.probes),
            "probes": [
                {
                    "concept": probe.concept,
                    "layer": probe.layer,
                    "weight_shape": list(probe.weight.shape),
                    "bias_shape": list(probe.bias.shape),
                }
                for probe in self.probes
            ],
        }


def save_frozen_probe_bank(bank: FrozenProbeBank, path: Path) -> Path:
    """Persist frozen benign probes used by the paper training objective."""

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "next_chameleons_frozen_probe_bank_v1",
        "required_concepts": bank.required_concepts,
        "probes": [
            {
                "concept": probe.concept,
                "layer": probe.layer,
                "weight": probe.weight.detach().cpu(),
                "bias": probe.bias.detach().cpu(),
            }
            for probe in bank.probes
        ],
    }
    torch.save(payload, path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(bank.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_frozen_probe_bank(path: Path) -> FrozenProbeBank:
    """Load frozen benign probes written by `save_frozen_probe_bank`."""

    torch = require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "next_chameleons_frozen_probe_bank_v1":
        raise ValueError(f"Unsupported frozen probe bank format in {path}")
    probes = [
        FrozenLinearConceptProbe(
            concept=str(item["concept"]),
            layer=int(item["layer"]),
            weight=item["weight"],
            bias=item["bias"],
        )
        for item in payload["probes"]
    ]
    return FrozenProbeBank(probes, required_concepts=payload["required_concepts"])


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
