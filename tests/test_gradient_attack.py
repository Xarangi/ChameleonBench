"""Track C1 white-box soft-prompt gradient attack -- differentiable core tests.

The correctness anchor runs entirely on tiny synthetic CPU tensors: no model
load, offline, fast. It exercises the exact optimization loop the real attack
uses (``optimize_against_probe``) against a hand-constructed frozen probe, and
asserts the two properties the "gradient bridge" claim rests on:

    (a) the probe score is DRIVEN DOWN over optimization steps, and
    (b) the trainable tensor actually RECEIVES gradients from the frozen probe.

Both the linear and the nonlinear (MLP) frozen probe heads are covered, since
the real attack targets either. Model-backed plumbing is validated for shape/
alignment via light structural checks that do not require a real LLM.
"""

import pytest

torch = pytest.importorskip("torch")

from next_chameleons.analysis.gradient_attack import (  # noqa: E402
    ObjectiveOutputs,
    _example_prompt_response,
    _kl_penalty,
    optimize_against_probe,
)
from next_chameleons.hf.mlp_concept_probe import FrozenMLPConceptProbe  # noqa: E402
from next_chameleons.hf.paper_probes import (  # noqa: E402
    FrozenLinearConceptProbe,
    fit_linear_probe_per_token,
)


def _known_linear_probe(hidden: int = 8) -> FrozenLinearConceptProbe:
    """A frozen linear probe whose positive direction is the all-ones vector."""

    weight = torch.ones(hidden, dtype=torch.float32)
    bias = torch.zeros((), dtype=torch.float32)
    return FrozenLinearConceptProbe(concept="synthetic", layer=0, weight=weight, bias=bias)


def test_optimize_drives_linear_probe_score_down_and_grads_flow() -> None:
    hidden = 8
    probe = _known_linear_probe(hidden)
    # Init activations aligned with the probe direction so the initial score is
    # high (well away from the saturated tail, so gradients are informative).
    activations = torch.full((3, 4, hidden), 0.2, requires_grad=True)
    mask = torch.ones(3, 4)

    grad_seen: dict[str, bool] = {"any": False}

    def objective_fn() -> ObjectiveOutputs:
        score = probe.score_sequence(activations, mask).mean()
        # No behavior leash for the pure-evasion anchor.
        behavior = torch.zeros((), dtype=torch.float32)
        return ObjectiveOutputs(probe_score=score, behavior_penalty=behavior)

    # Confirm gradients reach the trainable tensor from the frozen probe.
    probe.score_sequence(activations, mask).mean().backward()
    assert activations.grad is not None
    assert torch.count_nonzero(activations.grad) > 0
    grad_seen["any"] = True
    activations.grad = None

    result = optimize_against_probe(
        activations, objective_fn, steps=150, lr=0.05, behavior_weight=0.0
    )

    assert grad_seen["any"]
    assert result["probe_score_before"] > 0.5  # started as a detected concept
    assert result["probe_score_after"] < result["probe_score_before"]
    assert result["probe_score_reduction"] > 0.2
    assert result["probe_score_after"] < 0.2  # evasion: score pushed low
    # Trace is monotone-ish: the final step scores below the first.
    assert result["trace"][-1].probe_score < result["trace"][0].probe_score


def test_behavior_penalty_leashes_the_attack() -> None:
    hidden = 8
    probe = _known_linear_probe(hidden)
    start = torch.full((2, 3, hidden), 0.25)
    mask = torch.ones(2, 3)

    # Weak leash: essentially free to evade.
    activations1 = start.clone().requires_grad_(True)

    def objective_fn1() -> ObjectiveOutputs:
        score = probe.score_sequence(activations1, mask).mean()
        # L2 leash to the starting activation -- the synthetic analogue of the
        # KL/CE faithfulness constraint in the model-backed path.
        behavior = ((activations1 - start) ** 2).mean()
        return ObjectiveOutputs(probe_score=score, behavior_penalty=behavior)

    weak = optimize_against_probe(
        activations1, objective_fn1, steps=120, lr=0.05, behavior_weight=0.0
    )
    # Re-run with a strong leash on a fresh tensor.
    activations2 = start.clone().requires_grad_(True)

    def objective_fn2() -> ObjectiveOutputs:
        score = probe.score_sequence(activations2, mask).mean()
        behavior = ((activations2 - start) ** 2).mean()
        return ObjectiveOutputs(probe_score=score, behavior_penalty=behavior)

    strong = optimize_against_probe(
        activations2, objective_fn2, steps=120, lr=0.05, behavior_weight=50.0
    )

    # Both reduce the score, but the strong leash keeps the tensor closer to
    # start, so it evades LESS (higher post score) at held behavior.
    assert weak["probe_score_after"] < weak["probe_score_before"]
    assert strong["probe_score_after"] < strong["probe_score_before"]
    assert strong["probe_score_after"] > weak["probe_score_after"]
    assert strong["behavior_penalty_after"] < weak["behavior_penalty_after"]


def test_optimize_drives_mlp_probe_score_down() -> None:
    # Fit a small MLP probe on a linearly separable synthetic concept, then show
    # the same gradient core drives a positive input's score down through it.
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(120, 2, generator=g) + torch.tensor([2.5, 0.0])
    neg = torch.randn(120, 2, generator=g) + torch.tensor([-2.5, 0.0])
    features = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones(120), torch.zeros(120)])
    probe = FrozenMLPConceptProbe.fit(
        concept="sep",
        layer=3,
        token_features=features,
        token_labels=labels,
        hidden_width=16,
        epochs=200,
        min_accuracy=0.9,
    )

    activations = (torch.tensor([[2.5, 0.0]]).repeat(2, 3, 1)).clone().requires_grad_(True)
    mask = torch.ones(2, 3)

    def objective_fn() -> ObjectiveOutputs:
        score = probe.score_sequence(activations, mask).mean()
        return ObjectiveOutputs(probe_score=score, behavior_penalty=torch.zeros(()))

    result = optimize_against_probe(
        activations, objective_fn, steps=200, lr=0.1, behavior_weight=0.0
    )
    assert activations.grad is not None
    assert result["probe_score_before"] > 0.5
    assert result["probe_score_after"] < result["probe_score_before"]
    assert result["probe_score_after"] < 0.3


def test_optimize_records_full_trace_and_before_after() -> None:
    probe = _known_linear_probe(4)
    activations = torch.full((1, 2, 4), 0.5, requires_grad=True)
    mask = torch.ones(1, 2)

    def objective_fn() -> ObjectiveOutputs:
        score = probe.score_sequence(activations, mask).mean()
        return ObjectiveOutputs(probe_score=score, behavior_penalty=torch.zeros(()))

    result = optimize_against_probe(
        activations, objective_fn, steps=10, lr=0.1, behavior_weight=0.0
    )
    assert len(result["trace"]) == 10
    assert result["steps"] == 10
    assert all(hasattr(s, "probe_score") for s in result["trace"])
    assert "trainable" in result
    assert result["trainable"].shape == activations.shape


def test_optimize_requires_leaf_tensor() -> None:
    probe = _known_linear_probe(4)
    frozen = torch.full((1, 2, 4), 0.5)  # requires_grad=False

    def objective_fn() -> ObjectiveOutputs:
        return ObjectiveOutputs(
            probe_score=probe.score_sequence(frozen, torch.ones(1, 2)).mean(),
            behavior_penalty=torch.zeros(()),
        )

    with pytest.raises(ValueError):
        optimize_against_probe(frozen, objective_fn, steps=5)


def test_kl_penalty_zero_when_distributions_match() -> None:
    # base and prefixed logits identical over the aligned region -> KL == 0.
    torch.manual_seed(0)
    n_prefix = 2
    b, t, v = 2, 5, 7
    base_logits = torch.randn(b, t, v)
    prefixed = torch.cat([torch.randn(b, n_prefix, v), base_logits.clone()], dim=1)
    response_mask = torch.ones(b, t)
    kl = _kl_penalty(prefixed, base_logits, response_mask, n_prefix)
    assert float(kl) == pytest.approx(0.0, abs=1e-5)


def test_kl_penalty_positive_when_distributions_differ() -> None:
    torch.manual_seed(1)
    n_prefix = 1
    b, t, v = 2, 4, 6
    base_logits = torch.randn(b, t, v)
    prefixed = torch.cat([torch.randn(b, n_prefix, v), torch.randn(b, t, v)], dim=1)
    response_mask = torch.ones(b, t)
    kl = _kl_penalty(prefixed, base_logits, response_mask, n_prefix)
    assert float(kl) > 0.0


def test_example_prompt_response_normalization() -> None:
    # PaperConceptExample-like object.
    class _Ex:
        prompt = "a prompt"
        response = "a response"

    assert _example_prompt_response(_Ex()) == ("a prompt", "a response")
    assert _example_prompt_response("just text") == ("", "just text")
    assert _example_prompt_response({"prompt": "p", "response": "r"}) == ("p", "r")
    assert _example_prompt_response({"text": "t"}) == ("", "t")
    with pytest.raises(TypeError):
        _example_prompt_response(12345)


def test_fitted_linear_probe_is_evadable_by_core() -> None:
    # End-to-end-ish on the LINEAR path: fit a converged probe with the repo's
    # own fitter, then evade it with the gradient core.
    torch.manual_seed(0)
    hidden = 16
    pos = torch.randn(200, hidden) + 1.5
    neg = torch.randn(200, hidden) - 1.5
    feats = torch.cat([pos, neg], dim=0)
    labs = torch.cat([torch.ones(200), torch.zeros(200)])
    weight, bias = fit_linear_probe_per_token(
        token_features=feats, token_labels=labs, epochs=5, learning_rate=0.05
    )
    probe = FrozenLinearConceptProbe(concept="c", layer=0, weight=weight, bias=bias)
    activations = (torch.randn(2, 3, hidden) * 0.1 + 1.5).clone().requires_grad_(True)
    mask = torch.ones(2, 3)

    def objective_fn() -> ObjectiveOutputs:
        return ObjectiveOutputs(
            probe_score=probe.score_sequence(activations, mask).mean(),
            behavior_penalty=torch.zeros(()),
        )

    result = optimize_against_probe(
        activations, objective_fn, steps=200, lr=0.05, behavior_weight=0.0
    )
    assert result["probe_score_before"] > 0.5
    assert result["probe_score_after"] < 0.1
