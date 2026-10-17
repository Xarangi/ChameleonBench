"""Ensemble probe banks: scoring reductions, save/load round-trip, merging.

These back the arms-race machinery (`probe-bank-fit` / `probe-bank-eval` and
`benign_probe_source: external_bank`): the defender may answer an evading
organism with an ENSEMBLE of probes, and rounds accumulate into one cumulative
bank. Everything here runs on synthetic CPU tensors.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from next_chameleons.hf.paper_probes import (  # noqa: E402
    FrozenEnsembleConceptProbe,
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    load_frozen_probe_bank,
    load_probe_banks,
    merge_probe_banks,
    save_frozen_probe_bank,
)


def _linear_bank(concept: str = "all-caps", *, scale: float = 1.0) -> FrozenProbeBank:
    probe = FrozenLinearConceptProbe(
        concept=concept,
        layer=12,
        weight=torch.tensor([scale, 0.0]),
        bias=torch.tensor(0.0),
    )
    return FrozenProbeBank([probe], required_concepts=[concept])


def test_ensemble_mean_and_max_reductions_differ() -> None:
    # Member 0 fires on dimension 0, member 1 is a dead direction. On a token
    # that is strongly positive for member 0: mean ~ 0.75, max ~ 1.0.
    probe = FrozenEnsembleConceptProbe(
        concept="all-caps",
        layer=12,
        weights=torch.tensor([[10.0, 0.0], [0.0, 0.0]]),
        biases=torch.tensor([0.0, 0.0]),
        reduction="mean",
    )
    hidden = torch.tensor([[[1.0, 0.0]]])  # [batch=1, tokens=1, hidden=2]
    mean_score = float(probe.score_tokens(hidden).item())
    max_score = float(
        FrozenEnsembleConceptProbe(
            concept=probe.concept,
            layer=probe.layer,
            weights=probe.weights,
            biases=probe.biases,
            reduction="max",
        )
        .score_tokens(hidden)
        .item()
    )
    assert mean_score == pytest.approx(0.75, abs=0.01)
    assert max_score > 0.99
    assert max_score > mean_score


def test_softmax_reduction_interpolates_mean_and_max() -> None:
    # Softmax (normalized log-sum-exp) is the attacker training target: it sits
    # between mean and max and slides toward max as beta grows.
    weights = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    biases = torch.tensor([0.0, 0.0])
    hidden = torch.tensor([[[1.0, 0.0]]])  # member 0 ~1.0, member 1 = 0.5

    def score(reduction: str, beta: float = 8.0) -> float:
        probe = FrozenEnsembleConceptProbe(
            concept="c", layer=12, weights=weights, biases=biases,
            reduction=reduction, beta=beta,
        )
        return float(probe.score_tokens(hidden).item())

    mean_s, max_s = score("mean"), score("max")
    soft_lo, soft_hi = score("softmax", beta=0.5), score("softmax", beta=64.0)
    # Bounded by [mean, max], and monotone increasing in beta toward max.
    assert mean_s < soft_lo < soft_hi < max_s + 1e-4
    assert soft_lo == pytest.approx(mean_s, abs=0.05)  # small beta -> mean-like
    assert soft_hi == pytest.approx(max_s, abs=0.02)   # large beta -> max-like


def test_softmax_reduction_is_zero_when_all_members_zero() -> None:
    # Normalized (no log-k floor) so the obfuscation MSE-to-0 loss can reach 0
    # regardless of ensemble size k.
    for k in (1, 4, 16):
        probe = FrozenEnsembleConceptProbe(
            concept="c", layer=12,
            weights=torch.zeros(k, 2), biases=torch.full((k,), -1e4),
            reduction="softmax", beta=8.0,
        )
        hidden = torch.zeros(1, 1, 2)
        assert float(probe.score_tokens(hidden).item()) == pytest.approx(0.0, abs=1e-4)


def test_softmax_beta_round_trips_through_save_load(tmp_path: Path) -> None:
    probe = FrozenEnsembleConceptProbe(
        concept="all-caps", layer=12,
        weights=torch.tensor([[1.5, -0.5], [0.25, 2.0]]),
        biases=torch.tensor([0.1, -0.3]),
        reduction="softmax", beta=12.0,
    )
    bank = FrozenProbeBank([probe], required_concepts=["all-caps"])
    path = save_frozen_probe_bank(bank, tmp_path / "bank.pt")
    reloaded = load_frozen_probe_bank(path).probes_by_concept["all-caps"]
    assert reloaded.reduction == "softmax"
    assert reloaded.beta == pytest.approx(12.0)


def test_ensemble_score_sequence_masks_prompt_tokens() -> None:
    probe = FrozenEnsembleConceptProbe(
        concept="all-caps",
        layer=12,
        weights=torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
        biases=torch.tensor([0.0, 0.0]),
    )
    # Token 0 strongly positive, token 1 strongly negative; mask keeps token 0.
    hidden = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    assert float(probe.score_sequence(hidden, mask).item()) > 0.99


def test_ensemble_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        FrozenEnsembleConceptProbe(
            concept="c",
            layer=12,
            weights=torch.zeros(2, 2),
            biases=torch.zeros(2),
            reduction="median",
        )


def test_save_load_round_trip_preserves_ensemble_scores(tmp_path: Path) -> None:
    probe = FrozenEnsembleConceptProbe(
        concept="all-caps",
        layer=12,
        weights=torch.tensor([[1.5, -0.5], [0.25, 2.0]]),
        biases=torch.tensor([0.1, -0.3]),
        reduction="max",
    )
    bank = FrozenProbeBank([probe], required_concepts=["all-caps"])
    path = save_frozen_probe_bank(bank, tmp_path / "bank_r0.pt")
    restored = load_frozen_probe_bank(path)
    restored_probe = restored.probes_by_concept["all-caps"]

    assert isinstance(restored_probe, FrozenEnsembleConceptProbe)
    assert restored_probe.reduction == "max"
    assert restored_probe.num_members == 2
    hidden = torch.tensor([[[0.7, -1.1]]])
    assert float(restored_probe.score_tokens(hidden).item()) == pytest.approx(
        float(probe.score_tokens(hidden).item())
    )
    manifest = (path.with_suffix(".manifest.json")).read_text()
    assert "next_chameleons_frozen_probe_bank_v2" in manifest


def test_merge_banks_accumulates_members() -> None:
    merged = merge_probe_banks(
        [_linear_bank(scale=1.0), _linear_bank(scale=2.0), _linear_bank(scale=3.0)]
    )
    probe = merged.probes_by_concept["all-caps"]
    assert isinstance(probe, FrozenEnsembleConceptProbe)
    assert probe.num_members == 3
    assert probe.reduction == "mean"


def test_merge_single_bank_stays_linear() -> None:
    merged = merge_probe_banks([_linear_bank()])
    assert isinstance(merged.probes_by_concept["all-caps"], FrozenLinearConceptProbe)


def test_merge_rejects_mismatched_concepts() -> None:
    with pytest.raises(ValueError, match="different concepts"):
        merge_probe_banks([_linear_bank("all-caps"), _linear_bank("html")])


def test_load_probe_banks_from_directory_is_cumulative(tmp_path: Path) -> None:
    # The arms-race convention: each round drops one bank into a directory and
    # the attacker trains against every round so far.
    for round_index, scale in enumerate((1.0, 2.0)):
        save_frozen_probe_bank(
            _linear_bank(scale=scale), tmp_path / f"bank_r{round_index}.pt"
        )
    cumulative = load_probe_banks(tmp_path, reduction="max")
    probe = cumulative.probes_by_concept["all-caps"]

    assert isinstance(probe, FrozenEnsembleConceptProbe)
    assert probe.num_members == 2
    assert probe.reduction == "max"


def test_load_probe_banks_missing_spec_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_probe_banks(tmp_path / "empty_dir_without_banks")
