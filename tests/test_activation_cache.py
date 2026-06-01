from pathlib import Path

import numpy as np

from next_chameleons.activations import (
    SyntheticActivationExtractor,
    load_activation_cache,
    save_activation_cache,
)
from next_chameleons.datasets import SyntheticDatasetAdapter


def test_selective_activation_cache_round_trip(tmp_path: Path) -> None:
    bundle = SyntheticDatasetAdapter(num_examples=12).load()
    batch = SyntheticActivationExtractor(hidden_dim=8, num_layers=3).extract(bundle)
    path = tmp_path / "cache" / "activations.npz"

    save_activation_cache(batch, path)
    loaded = load_activation_cache(path)

    assert path.with_suffix(".manifest.json").exists()
    assert loaded.layers == [0, 1, 2]
    assert loaded.site == "residual_stream"
    np.testing.assert_allclose(loaded.activations, batch.activations)
    np.testing.assert_array_equal(loaded.labels, batch.labels)
