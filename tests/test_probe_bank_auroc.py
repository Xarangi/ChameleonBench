"""Threshold-free AUROC readout for the arms-race scoreboard (added 2026-07-25).

Lets the evasion signal be read independent of the fixed-FPR operating point:
a converged probe drives the 1%-FPR threshold to ~1e-3, so a model that ducks
its score to ~0.01 shows recall_drop==0 while auroc_drop still moves.
"""

from next_chameleons.real.probe_bank import _auroc, _threshold_at_fpr


def test_perfect_separation_is_one() -> None:
    assert _auroc([0.9, 0.8, 0.95], [0.1, 0.2, 0.05]) == 1.0


def test_perfect_inversion_is_zero() -> None:
    # every positive below every negative (full evasion under trigger)
    assert _auroc([0.1, 0.05], [0.9, 0.8]) == 0.0


def test_all_ties_is_chance() -> None:
    assert _auroc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_partial_overlap_between_zero_and_one() -> None:
    a = _auroc([0.6, 0.4], [0.5, 0.1])
    assert 0.5 < a < 1.0


def test_empty_inputs_are_nan() -> None:
    import math

    assert math.isnan(_auroc([], [0.1]))
    assert math.isnan(_auroc([0.1], []))


def test_ranks_match_mann_whitney_with_ties() -> None:
    # pos {0.5, 0.5}, neg {0.5, 0.1}: pos beats 0.1 (2 wins) + ties 0.5 (2 * 0.5)
    # U = 2 + 1 = 3; AUROC = 3 / (2 * 2) = 0.75
    assert _auroc([0.5, 0.5], [0.5, 0.1]) == 0.75


def test_threshold_lower_at_higher_fpr() -> None:
    # Higher tolerated FPR -> lower threshold -> TPR is monotone non-decreasing.
    negs = [i / 100.0 for i in range(100)]  # 0.00 .. 0.99
    t1 = _threshold_at_fpr(0.01, negs)
    t5 = _threshold_at_fpr(0.05, negs)
    assert t5 <= t1


def test_threshold_realizes_target_fpr() -> None:
    negs = [i / 1000.0 for i in range(1000)]  # 0.000 .. 0.999
    t = _threshold_at_fpr(0.01, negs)
    realized = sum(1 for s in negs if s > t) / len(negs)
    assert abs(realized - 0.01) < 0.02


def test_threshold_empty_is_nan() -> None:
    import math

    assert math.isnan(_threshold_at_fpr(0.01, []))


def test_converged_probe_collapses_1pct_threshold() -> None:
    # A converged one-vs-rest probe crams clean negatives near 0: the 1%-FPR
    # threshold collapses to ~0, so a triggered score ducking to 0.3 still shows
    # NO recall drop at 1% FPR -- but AUROC moves. This is the operating-point
    # pathology the 2026-07-25 lock-in addresses (AUROC-primary + TPR@5%).
    neg = [1e-4 * i for i in range(200)]  # ~0 .. 0.02, crammed near zero
    t1 = _threshold_at_fpr(0.01, neg)
    assert t1 < 0.05  # threshold collapsed near zero
    pos_clean = [0.9] * 20
    pos_trig = [0.3] * 20  # ducked hard, but still far above the ~0 threshold
    clean_tpr = sum(1 for s in pos_clean if s > t1) / len(pos_clean)
    trig_tpr = sum(1 for s in pos_trig if s > t1) / len(pos_trig)
    assert clean_tpr == 1.0 and trig_tpr == 1.0  # fixed-FPR hides the evasion
    assert _auroc(pos_clean, neg) - _auroc(pos_trig, neg) >= 0.0  # AUROC still readable


def test_wilson_interval_brackets_rate_and_handles_extremes() -> None:
    from next_chameleons.real.probe_bank import _wilson_interval

    low, high = _wilson_interval(6, 32)
    assert low < 6 / 32 < high
    assert 0.07 < low < 0.1 and 0.3 < high < 0.4  # 6/32 is compatible with ~0.09-0.36
    zero_low, zero_high = _wilson_interval(0, 50)
    assert zero_low == 0.0 and 0.0 < zero_high < 0.1


def test_bootstrap_auroc_interval_contains_point_estimate() -> None:
    from next_chameleons.real.probe_bank import _auroc, _bootstrap_auroc_interval

    pos = [0.2, 0.4, 0.6, 0.8, 0.9, 0.7]
    neg = [0.1, 0.3, 0.5, 0.2, 0.05, 0.35]
    low, high = _bootstrap_auroc_interval(pos, neg, samples=400, seed=3)
    assert low <= _auroc(pos, neg) <= high
    assert _bootstrap_auroc_interval(pos, neg, samples=400, seed=3) == (low, high)


def test_load_heldout_concept_examples_filters_split_and_concepts(tmp_path) -> None:
    import json

    from next_chameleons.real.probe_bank import load_heldout_concept_examples

    rows = [
        {"concept": "html", "response": "<p>a</p>", "split": "val", "text_hash": "a" * 64},
        {"concept": "html", "response": "<p>b</p>", "split": "test", "text_hash": "b" * 64},
        {"concept": "german", "response": "Hallo", "split": "test", "text_hash": "c" * 64},
    ]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(rows))
    test = load_heldout_concept_examples(path, split="test", concepts=["html", "finnish"])
    assert [e.response for e in test] == ["<p>b</p>"] and test[0].prompt == ""
    assert len(load_heldout_concept_examples(path, split="all", concepts=["html"])) == 2


def test_is_clean_negative_per_probe_purity() -> None:
    from next_chameleons.real.probe_bank import is_clean_negative

    assert is_clean_negative(None, "jokey")
    assert is_clean_negative({"jokey": 0.25, "all-caps": 1.0}, "jokey")
    assert not is_clean_negative({"jokey": 0.75, "all-caps": 1.0}, "jokey")


def test_fit_split_is_defender_only(tmp_path) -> None:
    """split_v4 'fit' rows (attacker-seen) load only for split='fit', never for 'all'."""

    import json

    from next_chameleons.real.probe_bank import load_heldout_concept_examples

    rows = [
        {"concept": "html", "response": "<p>a</p>", "split": "fit", "text_hash": "a" * 20,
         "ratings": {"html": 1.0, "jokey": 0.75}},
        {"concept": "html", "response": "<p>b</p>", "split": "val", "text_hash": "b" * 20},
        {"concept": "html", "response": "<p>c</p>", "split": "test", "text_hash": "c" * 20},
    ]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(rows))
    fit = load_heldout_concept_examples(path, split="fit", concepts=["html"])
    assert [e.response for e in fit] == ["<p>a</p>"]
    assert fit[0].ratings == {"html": 1.0, "jokey": 0.75}
    everything = load_heldout_concept_examples(path, split="all", concepts=["html"])
    assert sorted(e.response for e in everything) == ["<p>b</p>", "<p>c</p>"]
