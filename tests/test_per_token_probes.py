"""Per-token post-hoc probe scoring convention (mean of per-token sigmoids)."""

import numpy as np

from next_chameleons.hf.per_token_probes import PerTokenLinearProbe


def test_example_scores_mean_pool_per_token_sigmoids() -> None:
    # Two examples: example 0 has 2 strongly-positive tokens, example 1 has 3
    # strongly-negative tokens. A weight that reads dimension 0 should produce a
    # near-1 score for example 0 and near-0 for example 1.
    probe = PerTokenLinearProbe(
        probe_id="paper_linear_per_token_l12",
        layer=12,
        weight=np.array([1.0, 0.0]),
        bias=0.0,
    )
    token_features = np.array(
        [
            [10.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [-10.0, 0.0],
            [-10.0, 0.0],
        ]
    )
    token_example_index = np.array([0, 0, 1, 1, 1])
    scores = probe.example_scores(token_features, token_example_index, num_examples=2)

    assert scores.shape == (2,)
    assert scores[0] > 0.99
    assert scores[1] < 0.01


def test_example_scores_handles_examples_with_no_tokens() -> None:
    probe = PerTokenLinearProbe(
        probe_id="p",
        layer=12,
        weight=np.array([1.0]),
        bias=0.0,
    )
    token_features = np.array([[5.0]])
    token_example_index = np.array([0])
    # Example index 1 contributes no tokens; its score divides by the clamped
    # count and stays finite (0.0), not NaN.
    scores = probe.example_scores(token_features, token_example_index, num_examples=2)
    assert np.isfinite(scores).all()
    assert scores[1] == 0.0
