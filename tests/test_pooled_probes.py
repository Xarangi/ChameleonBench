"""Pooled defender probes (max / attention) and the composite cumulative ensemble."""

import torch

from next_chameleons.hf.paper_probes import (
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    load_frozen_probe_bank,
    merge_probe_banks,
    save_frozen_probe_bank,
)
from next_chameleons.hf.pooled_probes import (
    FrozenAttentionConceptProbe,
    FrozenCompositeConceptProbe,
    FrozenMaxPooledLinearProbe,
)


def _maxpool(seed: int = 0) -> FrozenMaxPooledLinearProbe:
    g = torch.Generator().manual_seed(seed)
    return FrozenMaxPooledLinearProbe("html", 12, torch.randn(8, generator=g), torch.tensor(0.0))


def test_maxpool_sequence_score_is_masked_max() -> None:
    probe = _maxpool()
    h = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    tokens = probe.score_tokens(h)
    got = probe.score_sequence(h, mask)
    assert torch.allclose(got[0], tokens[0, :3].max())
    assert torch.allclose(got[1], tokens[1].max())


def test_attention_probe_fits_sparse_signal_mean_pooling_dilutes() -> None:
    # Positives carry the concept on ONE token out of 20; a zero-init query is mean pooling,
    # so separating them requires the learned attention.
    g = torch.Generator().manual_seed(1)
    direction = torch.zeros(16)
    direction[0] = 6.0
    seqs, labels = [], []
    for i in range(80):
        h = torch.randn(20, 16, generator=g)
        if i % 2 == 0:
            h[3] += direction
        seqs.append(h)
        labels.append(1.0 if i % 2 == 0 else 0.0)
    probe = FrozenAttentionConceptProbe.fit(
        concept="html", layer=12, sequences=seqs, labels=labels, epochs=60, learning_rate=0.05
    )
    full = torch.ones(1, 20, dtype=torch.bool)
    scores = [float(probe.score_sequence(s[None], full)) for s in seqs]
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1.0]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0.0]
    assert sum(p > max(neg) for p in pos) / len(pos) > 0.8


def test_pooled_bank_roundtrip_and_cumulative_composite(tmp_path) -> None:
    attn = FrozenAttentionConceptProbe(
        "html", 12, torch.randn(8), torch.randn(8), torch.tensor(0.1)
    )
    b0, b1 = FrozenProbeBank([_maxpool(0)], required_concepts=["html"]), FrozenProbeBank(
        [attn], required_concepts=["html"]
    )
    p0 = save_frozen_probe_bank(b0, tmp_path / "b0.pt")
    p1 = save_frozen_probe_bank(b1, tmp_path / "b1.pt")
    l0, l1 = load_frozen_probe_bank(p0), load_frozen_probe_bank(p1)
    assert isinstance(l0.probes[0], FrozenMaxPooledLinearProbe)
    assert isinstance(l1.probes[0], FrozenAttentionConceptProbe)
    merged = merge_probe_banks([l0, l1], reduction="max").probes[0]
    assert isinstance(merged, FrozenCompositeConceptProbe) and merged.num_members == 2
    h, mask = torch.randn(1, 6, 8), torch.ones(1, 6, dtype=torch.bool)
    expected = torch.maximum(
        l0.probes[0].score_sequence(h, mask), l1.probes[0].score_sequence(h, mask)
    )
    assert torch.allclose(merged.score_sequence(h, mask), expected)


def test_linear_banks_still_flatten_to_linear_ensemble() -> None:
    lin = [
        FrozenProbeBank(
            [FrozenLinearConceptProbe("html", 12, torch.randn(8), torch.tensor(0.0))],
            required_concepts=["html"],
        )
        for _ in range(2)
    ]
    merged = merge_probe_banks(lin, reduction="max").probes[0]
    assert not isinstance(merged, FrozenCompositeConceptProbe) and merged.num_members == 2


def test_pooled_probes_pass_gradients_to_activations() -> None:
    attn = FrozenAttentionConceptProbe(
        "html", 12, torch.randn(8), torch.randn(8), torch.tensor(0.0)
    )
    comp = FrozenCompositeConceptProbe("html", 12, (_maxpool(), attn), reduction="softmax")
    h = torch.randn(1, 5, 8, requires_grad=True)
    comp.score_sequence(h, torch.ones(1, 5, dtype=torch.bool)).sum().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0
