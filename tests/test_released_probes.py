"""Released-probe golden-eval adapter and per-token probe-fit convention."""

import numpy as np
import pytest

from next_chameleons.activations import ActivationBatch
from next_chameleons.hf.released_probes import (
    ReleasedLogisticProbe,
    ReleasedProbeJudges,
    load_released_logistic_probe,
    load_released_probe_judges,
)

torch = pytest.importorskip("torch")


def _batch(scores_by_layer: dict[int, np.ndarray], labels: np.ndarray) -> ActivationBatch:
    layers = sorted(scores_by_layer)
    activations = np.stack([scores_by_layer[layer] for layer in layers], axis=1)
    return ActivationBatch(
        activations=activations.astype(np.float32),
        labels=labels.astype(np.int64),
        example_ids=[f"ex-{i}" for i in range(activations.shape[0])],
        layers=layers,
        site="residual_stream",
    )


def test_released_probe_scores_pooled_features() -> None:
    # One example, two layers; probe only owns layer 12.
    hidden = 4
    weight = np.array([1.0, 0.0, 0.0, 0.0])
    probe = ReleasedLogisticProbe(
        probe_id="released_deception",
        layer_weights={12: (weight, 0.0)},
        selected_layers=[8, 12],
    )
    positives = np.tile(np.array([10.0, 0.0, 0.0, 0.0]), (3, 1))
    negatives = np.tile(np.array([-10.0, 0.0, 0.0, 0.0]), (3, 1))
    activations = np.stack(
        [np.zeros((6, hidden)), np.concatenate([positives, negatives])],
        axis=1,
    )
    batch = ActivationBatch(
        activations=activations.astype(np.float32),
        labels=np.array([1, 1, 1, 0, 0, 0], dtype=np.int64),
        example_ids=[f"ex-{i}" for i in range(6)],
        layers=[8, 12],
        site="residual_stream",
    )
    prediction = probe.predict(batch)
    assert prediction.scores[:3].mean() > 0.9
    assert prediction.scores[3:].mean() < 0.1


def test_released_probe_requires_layer_overlap() -> None:
    probe = ReleasedLogisticProbe(
        probe_id="released_harmful",
        layer_weights={31: (np.ones(2), 0.0)},
        selected_layers=[8, 12],
    )
    batch = _batch({8: np.zeros((2, 2)), 12: np.zeros((2, 2))}, np.array([1, 0]))
    with pytest.raises(ValueError, match="no layer overlap"):
        probe.predict(batch)


def test_load_released_probe_layer_indexed(tmp_path) -> None:
    path = tmp_path / "deception_weights.pt"
    torch.save(
        {12: {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}},
        path,
    )
    probe = load_released_logistic_probe(
        path,
        probe_id="released_deception",
        selected_layers=[8, 12, 16],
        default_layer=12,
    )
    assert list(probe.layer_weights) == [12]
    assert probe.layer_weights[12][0].shape == (3,)


def test_load_released_probe_positional_zero_remaps_to_default_layer(tmp_path) -> None:
    # Minimal-repo released probes are keyed by positional 0 (single-layer
    # extraction of the paper target layer), which does not overlap absolute
    # selected_layers; it must remap onto default_layer.
    path = tmp_path / "deception_weights.pt"
    torch.save({0: {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}}, path)
    probe = load_released_logistic_probe(
        path,
        probe_id="released_deception",
        selected_layers=[4, 8, 12, 16],
        default_layer=12,
    )
    assert list(probe.layer_weights) == [12]


def test_load_released_probe_single_layer_uses_default_layer(tmp_path) -> None:
    path = tmp_path / "harmful_weights.pt"
    torch.save({"weight": torch.ones(1, 3), "bias": torch.zeros(1)}, path)
    probe = load_released_logistic_probe(
        path,
        probe_id="released_harmful",
        selected_layers=[8, 12],
        default_layer=12,
    )
    assert list(probe.layer_weights) == [12]


def test_load_released_probe_judges_maps_domains(tmp_path) -> None:
    torch.save({12: {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}},
               tmp_path / "deception_weights.pt")
    torch.save({12: {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}},
               tmp_path / "harmful_weights.pt")
    judges = load_released_probe_judges(
        tmp_path,
        selected_layers=[8, 12],
        default_layer=12,
    )
    assert isinstance(judges, ReleasedProbeJudges)
    assert judges.judge_for_source("dolus_deception").probes[0].probe_id == "released_deception"
    assert judges.judge_for_source("circuit_breakers_harmful").probes[0].probe_id == (
        "released_harmful"
    )


def test_load_released_probe_judges_requires_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No released probe weights"):
        load_released_probe_judges(tmp_path, selected_layers=[12], default_layer=12)


def test_per_token_fit_matches_scoring_convention() -> None:
    from next_chameleons.hf.paper_probes import fit_linear_probe_per_token

    rng = np.random.default_rng(0)
    positives = rng.normal(2.0, 0.3, size=(64, 3))
    negatives = rng.normal(-2.0, 0.3, size=(64, 3))
    features = torch.from_numpy(np.concatenate([positives, negatives]).astype(np.float32))
    labels = torch.from_numpy(
        np.concatenate([np.ones(64), np.zeros(64)]).astype(np.float32)
    )
    weight, bias = fit_linear_probe_per_token(
        token_features=features,
        token_labels=labels,
        learning_rate=0.05,
        epochs=20,
        batch_size=16,
        seed=17,
    )
    logits = features @ weight + bias
    scores = torch.sigmoid(logits).detach().numpy()
    assert scores[:64].mean() > 0.8
    assert scores[64:].mean() < 0.2
