"""Activation extraction and selective cache primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from next_chameleons.datasets import DatasetBundle
from next_chameleons.registry import ACTIVATION_EXTRACTORS


@dataclass
class ActivationBatch:
    """Layered activation tensor and labels.

    Shape convention: `activations[example, layer, hidden]`.
    """

    activations: np.ndarray
    labels: np.ndarray
    example_ids: list[str]
    layers: list[int]
    site: str

    def layer_view(self, layer_index: int) -> np.ndarray:
        """Return activations for one layer position."""

        return self.activations[:, layer_index, :]


class ActivationExtractor(Protocol):
    """Produces activation batches for a dataset."""

    def extract(self, bundle: DatasetBundle) -> ActivationBatch:
        """Extract activations."""


class SyntheticActivationExtractor:
    """Generates class-separable hidden states for smoke tests."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        num_layers: int = 4,
        concept_shift: float = 1.35,
        seed: int = 17,
        site: str = "residual_stream",
        **_: object,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.concept_shift = concept_shift
        self.seed = seed
        self.site = site

    def extract(self, bundle: DatasetBundle) -> ActivationBatch:
        rng = np.random.default_rng(self.seed)
        labels = np.array([example.label for example in bundle.examples], dtype=np.int64)
        activations = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(len(bundle.examples), self.num_layers, self.hidden_dim),
        ).astype(np.float32)
        direction = np.zeros(self.hidden_dim, dtype=np.float32)
        direction[0] = 1.0
        for layer in range(self.num_layers):
            strength = self.concept_shift * (1.0 + layer / max(1, self.num_layers - 1))
            activations[:, layer, :] += labels[:, None] * strength * direction
        return ActivationBatch(
            activations=activations,
            labels=labels,
            example_ids=[example.example_id for example in bundle.examples],
            layers=list(range(self.num_layers)),
            site=self.site,
        )


def save_activation_cache(batch: ActivationBatch, path: Path) -> None:
    """Save a selective activation cache as an `.npz` plus JSON manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activations=batch.activations,
        labels=batch.labels,
        layers=np.array(batch.layers, dtype=np.int64),
        example_ids=np.array(batch.example_ids),
        site=np.array([batch.site]),
    )
    manifest = {
        "shape": list(batch.activations.shape),
        "dtype": str(batch.activations.dtype),
        "layers": batch.layers,
        "site": batch.site,
        "num_examples": len(batch.example_ids),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_activation_cache(path: Path) -> ActivationBatch:
    """Load an activation cache written by `save_activation_cache`."""

    with np.load(path, allow_pickle=False) as payload:
        return ActivationBatch(
            activations=payload["activations"],
            labels=payload["labels"],
            example_ids=[str(value) for value in payload["example_ids"].tolist()],
            layers=[int(value) for value in payload["layers"].tolist()],
            site=str(payload["site"][0]),
        )


ACTIVATION_EXTRACTORS.register("synthetic", SyntheticActivationExtractor)
