"""Per-token post-hoc safety probes for paper-faithful held-out evaluation.

The paper trains post-hoc safety probes per response token (each token is an
independent BCE example with the sequence label broadcast) and scores a
sequence as the mean of per-token sigmoids. The pooled `ActivationBatch`
surface cannot express that, so this module fits and scores per-token linear
probes directly from `HFActivationExtractor.extract_token_features` output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from next_chameleons.hf.activations import HFActivationExtractor
from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.paper_probes import fit_linear_probe_per_token
from next_chameleons.probes.base import sigmoid


@dataclass
class PerTokenLinearProbe:
    """A logistic probe scored as mean-of-per-token-sigmoids per example."""

    probe_id: str
    layer: int
    weight: np.ndarray
    bias: float
    family: str = "linear_per_token"

    def example_scores(
        self,
        token_features: np.ndarray,
        token_example_index: np.ndarray,
        num_examples: int,
    ) -> np.ndarray:
        token_scores = sigmoid(
            token_features.astype(np.float64) @ self.weight.astype(np.float64) + self.bias
        )
        sums = np.zeros(num_examples, dtype=np.float64)
        counts = np.zeros(num_examples, dtype=np.float64)
        np.add.at(sums, token_example_index, token_scores)
        np.add.at(counts, token_example_index, 1.0)
        return sums / np.clip(counts, 1.0, None)


def fit_per_token_linear_probe(
    extractor: HFActivationExtractor,
    fit_examples: list[HFTextExample],
    *,
    probe_id: str,
    layer: int,
    batch_size: int = 1,
    learning_rate: float = 4e-5,
    epochs: int = 1,
    probe_batch_size: int = 16,
    seed: int = 17,
    max_tokens_per_example: int | None = None,
) -> PerTokenLinearProbe:
    """Fit a per-token linear probe on the fit split's generation tokens."""

    from next_chameleons.hf.imports import require_torch

    torch = require_torch()
    token_features, token_example_index, example_labels = extractor.extract_token_features(
        fit_examples,
        layer=layer,
        batch_size=batch_size,
        max_tokens_per_example=max_tokens_per_example,
    )
    token_labels = example_labels[token_example_index].astype(np.float32)
    weight, bias = fit_linear_probe_per_token(
        token_features=torch.from_numpy(token_features.astype(np.float32)),
        token_labels=torch.from_numpy(token_labels),
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=probe_batch_size,
        seed=seed,
    )
    return PerTokenLinearProbe(
        probe_id=probe_id,
        layer=layer,
        weight=weight.cpu().numpy().astype(np.float64),
        bias=float(bias.cpu()),
    )


def per_token_example_scores(
    probe: PerTokenLinearProbe,
    extractor: HFActivationExtractor,
    examples: list[HFTextExample],
    *,
    batch_size: int = 1,
    max_tokens_per_example: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score examples with a fitted per-token probe; returns (scores, labels)."""

    token_features, token_example_index, example_labels = extractor.extract_token_features(
        examples,
        layer=probe.layer,
        batch_size=batch_size,
        max_tokens_per_example=max_tokens_per_example,
    )
    scores = probe.example_scores(token_features, token_example_index, len(examples))
    return scores, example_labels
