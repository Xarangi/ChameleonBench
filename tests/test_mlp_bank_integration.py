"""End-to-end wiring of the nonlinear MLP defender into the probe-bank machinery.

Proves the save/load/score/manifest round trip for a ``FrozenMLPConceptProbe``
bank -- the pieces ``probe-bank-fit --probe-class mlp`` (defender move) relies on.
Everything runs on synthetic CPU tensors, offline.

``run_probe_bank_fit`` itself needs a real HF model (feature capture on an
organism checkpoint), too heavy for a CPU unit test, so we exercise the exact
FIT call its ``--probe-class mlp`` path makes (``FrozenMLPConceptProbe.fit`` over
a concept-vs-rest token pool) directly on synthetic separable activations, then
drive that fitted probe through the bank serialization + scoring + manifest path
that ``run_probe_bank_fit`` calls verbatim (``save_frozen_probe_bank`` /
``load_frozen_probe_bank`` / ``FrozenProbeBank.score_matching`` / ``.manifest``).
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from next_chameleons.hf.mlp_concept_probe import (  # noqa: E402
    MLP_BANK_FORMAT,
    FrozenMLPConceptProbe,
)
from next_chameleons.hf.paper_probes import (  # noqa: E402
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    load_frozen_probe_bank,
    load_probe_banks,
    merge_probe_banks,
    save_frozen_probe_bank,
)


def _separable_pool(hidden: int = 8, per_class: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly-separable one-vs-rest token pool (positives shifted along dim 0)."""

    generator = torch.Generator().manual_seed(0)
    pos = torch.randn(per_class, hidden, generator=generator) + torch.tensor(
        [3.0] + [0.0] * (hidden - 1)
    )
    neg = torch.randn(per_class, hidden, generator=generator) - torch.tensor(
        [3.0] + [0.0] * (hidden - 1)
    )
    features = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones(per_class), torch.zeros(per_class)])
    return features, labels


def _fit_mlp_bank(concept: str = "all-caps", *, layer: int = 1) -> FrozenProbeBank:
    """Fit an MLP head the way run_probe_bank_fit's --probe-class mlp path does."""

    features, labels = _separable_pool()
    probe = FrozenMLPConceptProbe.fit(
        concept=concept,
        layer=layer,
        token_features=features,
        token_labels=labels,
        hidden_width=16,
        activation="relu",
        epochs=120,
        seed=17,
        min_accuracy=0.9,
    )
    return FrozenProbeBank([probe], required_concepts=[concept])


def test_mlp_fit_gate_rejects_unlearnable_pool() -> None:
    # The fit convergence gate (shared with run_probe_bank_fit's min_accuracy) must
    # refuse a probe that has not actually learned the concept.
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(128, 8, generator=generator)
    labels = (torch.arange(128) % 2).float()  # label uncorrelated with features
    with pytest.raises(ValueError, match="did not converge"):
        FrozenMLPConceptProbe.fit(
            concept="all-caps",
            layer=1,
            token_features=noise,
            token_labels=labels,
            hidden_width=8,
            epochs=30,
            min_accuracy=0.97,
        )


def test_mlp_bank_round_trip_preserves_scores_and_manifest(tmp_path: Path) -> None:
    bank = _fit_mlp_bank()
    probe = bank.probes_by_concept["all-caps"]
    assert isinstance(probe, FrozenMLPConceptProbe)

    hidden = torch.randn(1, 5, probe.w1.shape[0])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])
    hidden_states = tuple(hidden if idx == probe.layer else None for idx in range(probe.layer + 1))
    pre = bank.score_matching(
        hidden_states=hidden_states, generation_mask=mask, concepts=["all-caps"]
    )

    path = save_frozen_probe_bank(bank, tmp_path / "mlp_bank.pt")
    restored = load_frozen_probe_bank(path)
    restored_probe = restored.probes_by_concept["all-caps"]

    assert isinstance(restored_probe, FrozenMLPConceptProbe)
    assert restored_probe.hidden_width == 16
    assert restored_probe.activation == "relu"
    post = restored.score_matching(
        hidden_states=hidden_states, generation_mask=mask, concepts=["all-caps"]
    )
    assert float((pre - post).abs().max().item()) == pytest.approx(0.0, abs=1e-5)

    # Manifest + persisted format both self-describe as an MLP bank.
    assert restored.manifest()["format"] == MLP_BANK_FORMAT
    manifest_entry = restored.manifest()["probes"][0]
    assert manifest_entry["hidden_width"] == 16
    assert manifest_entry["activation"] == "relu"
    manifest_text = path.with_suffix(".manifest.json").read_text()
    assert MLP_BANK_FORMAT in manifest_text


def test_mlp_bank_separates_positive_and_negative_tokens(tmp_path: Path) -> None:
    # The fitted head must actually detect the concept after a save/load cycle.
    bank = _fit_mlp_bank()
    restored = load_frozen_probe_bank(save_frozen_probe_bank(bank, tmp_path / "b.pt"))
    probe = restored.probes_by_concept["all-caps"]
    features, labels = _separable_pool()
    with torch.no_grad():
        scores = probe.score_features(features)
    pos_mean = float(scores[labels.bool()].mean().item())
    neg_mean = float(scores[~labels.bool()].mean().item())
    assert pos_mean > 0.8
    assert neg_mean < 0.2


def test_load_probe_banks_single_mlp_bank_passthrough(tmp_path: Path) -> None:
    # load_probe_banks merges; a single-bank MLP concept must pass through the
    # linear-ensemble merge unchanged (nonlinear heads cannot flatten).
    save_frozen_probe_bank(_fit_mlp_bank(), tmp_path / "bank_r0.pt")
    merged = load_probe_banks(tmp_path)
    assert isinstance(merged.probes_by_concept["all-caps"], FrozenMLPConceptProbe)


def test_merge_mlp_across_banks_builds_composite(tmp_path: Path) -> None:
    # A cumulative bank mixing a nonlinear head with a linear probe cannot flatten
    # into one linear ensemble; since 2026-09-26 it becomes a composite ensemble
    # that reduces over each member's own SEQUENCE score (nothing silently dropped).
    from next_chameleons.hf.pooled_probes import FrozenCompositeConceptProbe

    mlp_bank = _fit_mlp_bank()
    linear_bank = FrozenProbeBank(
        [
            FrozenLinearConceptProbe(
                concept="all-caps", layer=1, weight=torch.zeros(8), bias=torch.tensor(0.0)
            )
        ],
        required_concepts=["all-caps"],
    )
    merged = merge_probe_banks([mlp_bank, linear_bank], reduction="max")
    composite = merged.probes_by_concept["all-caps"]
    assert isinstance(composite, FrozenCompositeConceptProbe)
    assert composite.num_members == 2
