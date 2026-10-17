"""Nonlinear (small MLP) frozen concept probe: the Track A3 defender/target head.

Everything runs on synthetic CPU tensors, offline, tiny dims. Covers the two
properties the arms race relies on: (a) the MLP captures nonlinearly-separable
concept structure a linear probe provably cannot, and (b) its scoring method is
a differentiable, in-range drop-in for ``FrozenLinearConceptProbe`` so an
attacker can backprop an obfuscation loss through it into the hidden states.
"""

import pytest

torch = pytest.importorskip("torch")

from next_chameleons.hf.mlp_concept_probe import (  # noqa: E402
    FrozenMLPConceptProbe,
    _apply_activation,
)
from next_chameleons.hf.paper_probes import (  # noqa: E402
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    fit_linear_probe_per_token,
)


def _linear_separable(n: int = 240, seed: int = 0) -> tuple:
    """Two Gaussian blobs offset along dim 0 -> linearly separable."""

    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n // 2, 2, generator=g) + torch.tensor([2.5, 0.0])
    neg = torch.randn(n // 2, 2, generator=g) + torch.tensor([-2.5, 0.0])
    features = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    return features, labels


def _xor_like(n: int = 400, seed: int = 1) -> tuple:
    """XOR quadrants: positives on one diagonal, negatives on the other.

    Not linearly separable -- the best hyperplane is at chance -- but a
    one-hidden-layer MLP can carve it.
    """

    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]]
    )
    labels_per_center = [1.0, 1.0, 0.0, 0.0]
    feats, labs = [], []
    per = n // 4
    for center, label in zip(centers, labels_per_center, strict=True):
        feats.append(torch.randn(per, 2, generator=g) * 0.6 + center)
        labs.append(torch.full((per,), label))
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


def test_fits_linearly_separable_concept() -> None:
    features, labels = _linear_separable()
    probe = FrozenMLPConceptProbe.fit(
        concept="sep",
        layer=7,
        token_features=features,
        token_labels=labels,
        hidden_width=16,
        seed=3,
        min_accuracy=0.95,
    )
    assert probe.balanced_accuracy(features, labels) > 0.95


def test_fits_nonlinearly_separable_concept_a_linear_probe_cannot() -> None:
    features, labels = _xor_like()

    # A linear logistic probe (same fit convention the arms race uses) is stuck
    # at chance on XOR -- no hyperplane separates the diagonals.
    weight, bias = fit_linear_probe_per_token(
        token_features=features,
        token_labels=labels,
        learning_rate=0.05,
        epochs=200,
        batch_size=64,
    )
    linear = FrozenLinearConceptProbe(concept="xor", layer=7, weight=weight, bias=bias)
    with torch.no_grad():
        lin_scores = linear.score_tokens(features[None])[0]
        lin_preds = (lin_scores >= 0.5).float()
        pos = labels.bool()
        lin_bal = 0.5 * (
            lin_preds[pos].mean().item() + (1.0 - lin_preds[~pos]).mean().item()
        )
    assert lin_bal < 0.75  # linear probe cannot crack XOR

    # The MLP defender does.
    probe = FrozenMLPConceptProbe.fit(
        concept="xor",
        layer=7,
        token_features=features,
        token_labels=labels,
        hidden_width=32,
        seed=5,
        min_accuracy=0.9,
    )
    assert probe.balanced_accuracy(features, labels) > 0.9


def test_fit_gate_raises_when_not_converged() -> None:
    features, labels = _xor_like(n=200, seed=2)
    with pytest.raises(ValueError, match="did not converge"):
        FrozenMLPConceptProbe.fit(
            concept="xor",
            layer=7,
            token_features=features,
            token_labels=labels,
            hidden_width=8,
            epochs=1,  # far too few to fit XOR
            learning_rate=1e-4,
            min_accuracy=0.99,
        )


def _demo_probe(activation: str = "relu") -> FrozenMLPConceptProbe:
    g = torch.Generator().manual_seed(11)
    hidden, width = 5, 8
    return FrozenMLPConceptProbe(
        concept="c",
        layer=3,
        w1=torch.randn(hidden, width, generator=g),
        b1=torch.randn(width, generator=g),
        w2=torch.randn(width, generator=g),
        b2=torch.randn((), generator=g),
        activation=activation,
    )


def test_score_tokens_shape_and_range() -> None:
    probe = _demo_probe()
    hidden = torch.randn(4, 6, 5)  # [batch, tokens, hidden]
    scores = probe.score_tokens(hidden)
    assert scores.shape == (4, 6)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_score_tokens_is_differentiable_to_leaf_input() -> None:
    probe = _demo_probe()
    hidden = torch.randn(2, 3, 5, requires_grad=True)
    scores = probe.score_tokens(hidden)
    assert scores.requires_grad
    scores.sum().backward()
    # Gradient reaches the leaf hidden-state input -- an attacker can obfuscate.
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert float(hidden.grad.abs().sum()) > 0.0


def test_score_sequence_masks_and_backprops() -> None:
    probe = _demo_probe(activation="gelu")
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    seq = probe.score_sequence(hidden, mask)
    assert seq.shape == (2,)
    assert float(seq.min()) >= 0.0 and float(seq.max()) <= 1.0
    seq.sum().backward()
    assert hidden.grad is not None
    # Masked-out tokens receive no gradient; kept tokens do.
    assert float(hidden.grad[0, 2:].abs().sum()) == 0.0
    assert float(hidden.grad[0, 0].abs().sum()) > 0.0


def test_scoring_signature_matches_linear_probe_and_drops_into_bank() -> None:
    # Same public scoring surface as FrozenLinearConceptProbe: identical method
    # names/arity, and it slots straight into FrozenProbeBank.score_matching.
    import inspect

    for name in ("score_tokens", "score_sequence"):
        mlp_sig = inspect.signature(getattr(FrozenMLPConceptProbe, name))
        lin_sig = inspect.signature(getattr(FrozenLinearConceptProbe, name))
        assert list(mlp_sig.parameters) == list(lin_sig.parameters)

    probe = _demo_probe()
    bank = FrozenProbeBank([probe], required_concepts=["c"])
    hidden_states = (
        torch.zeros(1, 2, 5),  # layer 0..2 placeholders
        torch.zeros(1, 2, 5),
        torch.zeros(1, 2, 5),
        torch.randn(1, 2, 5),  # layer 3 = probe.layer
    )
    mask = torch.tensor([[1.0, 1.0]])
    out = bank.score_matching(
        hidden_states=hidden_states, generation_mask=mask, concepts=["c"]
    )
    assert out.shape == (1,)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert probe.num_members == 1


def test_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        FrozenMLPConceptProbe(
            concept="c",
            layer=1,
            w1=torch.zeros(3, 4),
            b1=torch.zeros(4),
            w2=torch.zeros(4),
            b2=torch.zeros(()),
            activation="tanh",
        )
    with pytest.raises(ValueError, match="activation"):
        _apply_activation(torch, "tanh", torch.zeros(2))


def test_state_dict_round_trip_preserves_scores() -> None:
    probe = _demo_probe()
    restored = FrozenMLPConceptProbe.from_state(probe.state_dict())
    hidden = torch.randn(2, 3, 5)
    assert torch.allclose(probe.score_tokens(hidden), restored.score_tokens(hidden))
    assert restored.activation == probe.activation
    assert restored.hidden_width == probe.hidden_width
    assert "mlp_w1" in probe.state_dict()  # MLP-identifying key for bank loaders
