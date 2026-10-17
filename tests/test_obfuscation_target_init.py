"""The obfuscation-TARGET fit init controls the target's breadth.

The paper's zero-shot generalization is driven by the *breadth* of the frozen
obfuscation-target probe, not its accuracy. The authors fit those probes as an
``nn.Linear`` with default (Kaiming) init and stop at 1 epoch, so ~most of the
weight energy is retained random init spread across the hidden dimension (a high
participation ratio). A zero-init probe fit to the same data lands the clean,
low-dimensional mean-difference direction instead.

These tests pin that the ``init`` switch in ``fit_linear_probe_per_token``
actually produces those two regimes, and that both remain deterministic (seeded)
so the from-scratch replication arm is reproducible. See the
obfuscation-target-drives-generalization note.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from next_chameleons.hf.paper_probes import fit_linear_probe_per_token  # noqa: E402


def _participation_ratio(weight: np.ndarray) -> float:
    """PR = (sum w^2)^2 / sum w^4 in [1, hidden]; high = energy spread broadly."""
    w2 = weight.astype(np.float64) ** 2
    return float(w2.sum() ** 2 / (w2**2).sum())


def _kaiming_init(hidden: int, seed: int) -> np.ndarray:
    """Recompute the seeded Kaiming init used by fit_linear_probe_per_token."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bound = (1.0 / hidden) ** 0.5
    return ((torch.rand(hidden, generator=generator) * 2 - 1) * bound).numpy()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _toy_tokens(hidden: int = 256, per_class: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(hidden)
    direction /= np.linalg.norm(direction)
    pos = rng.standard_normal((per_class, hidden)) * 6.0 + 3.0 * direction
    neg = rng.standard_normal((per_class, hidden)) * 6.0
    features = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(per_class), np.zeros(per_class)])
    return (
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.float32)),
        direction,
    )


def test_kaiming_target_stays_init_dominated_zeros_learns_signal() -> None:
    features, labels, signal = _toy_tokens()
    hidden = features.shape[1]

    w_zeros, _ = fit_linear_probe_per_token(
        token_features=features, token_labels=labels, epochs=1, init="zeros"
    )
    w_kaiming, _ = fit_linear_probe_per_token(
        token_features=features, token_labels=labels, epochs=1, init="kaiming", seed=17
    )
    w_zeros = w_zeros.cpu().numpy()
    w_kaiming = w_kaiming.cpu().numpy()
    init = _kaiming_init(hidden, seed=17)

    # The zero-init target moves decisively toward the concept (mean-difference)
    # direction (a random vector has |cos| ~ 1/sqrt(hidden) ~ 0.06 here).
    assert _cos(w_zeros, signal) > 0.3
    # The Kaiming target is still dominated by its broad random init after 1
    # epoch (the data-orthogonal energy that never gets trained away) -- this
    # breadth, not accuracy, is what makes organism evasion generalize.
    assert _cos(w_kaiming, init) > 0.7
    # And it is correspondingly LESS aligned to the clean signal than zeros-init.
    assert _cos(w_kaiming, signal) < _cos(w_zeros, signal)
    # Both remain broad (participation ratio well above a 1-hot direction).
    assert _participation_ratio(w_kaiming) > 0.5 * hidden


def test_kaiming_init_is_deterministic_under_seed() -> None:
    features, labels, _ = _toy_tokens()
    a, ba = fit_linear_probe_per_token(
        token_features=features, token_labels=labels, epochs=1, init="kaiming", seed=17
    )
    b, bb = fit_linear_probe_per_token(
        token_features=features, token_labels=labels, epochs=1, init="kaiming", seed=17
    )
    assert torch.allclose(a, b)
    assert torch.allclose(ba, bb)
    # A different seed gives a materially different broad target.
    c, _ = fit_linear_probe_per_token(
        token_features=features, token_labels=labels, epochs=1, init="kaiming", seed=41
    )
    cos = float(
        torch.nn.functional.cosine_similarity(a.reshape(1, -1), c.reshape(1, -1))
    )
    assert cos < 0.9


def test_unknown_init_raises() -> None:
    features, labels, _ = _toy_tokens(hidden=8, per_class=4)
    with pytest.raises(ValueError, match="Unknown probe init"):
        fit_linear_probe_per_token(
            token_features=features, token_labels=labels, epochs=1, init="bogus"
        )
