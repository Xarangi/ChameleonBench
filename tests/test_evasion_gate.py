"""Train-until-evasion gate: every concept must evade, without concept destruction.

Pure-function tests on synthetic by_concept blocks (the fixed-FPR report shape),
covering the arms-race protocol: the gate passes only when EVERY train concept's
frozen probe is evaded (matching_trigger_tpr low) while the probe still catches
untriggered concept text (clean_tpr high — "evasion" by erasing the concept is a
defender win, not evasion).
"""

from next_chameleons.hf.in_distribution_eval import evaluate_evasion_gate


def _row(clean: float, trig: float) -> dict[str, float]:
    return {"clean_tpr": clean, "matching_trigger_tpr": trig}


def test_gate_passes_when_every_concept_evades() -> None:
    gate = evaluate_evasion_gate(
        {
            "all-caps": _row(1.0, 0.0),
            "finnish": _row(0.95, 0.19),
            "mathematical": _row(1.0, 0.06),
        }
    )
    assert gate["passed"] is True
    assert all(v["passed"] for v in gate["by_concept"].values())


def test_gate_fails_on_partial_evasion() -> None:
    # The round-1 arms-race pattern: only one concept crosses the threshold.
    gate = evaluate_evasion_gate(
        {
            "all-caps": _row(1.0, 1.0),
            "mathematical": _row(1.0, 0.09),
        }
    )
    assert gate["passed"] is False
    assert gate["by_concept"]["mathematical"]["passed"] is True
    assert gate["by_concept"]["all-caps"]["passed"] is False


def test_gate_fails_on_concept_destruction() -> None:
    # Trigger TPR is low, but so is clean TPR: the organism erased the concept
    # rather than evading conditionally. Must NOT count as evasion.
    gate = evaluate_evasion_gate({"html": _row(0.3, 0.0)})
    assert gate["passed"] is False


def test_gate_fails_on_empty_and_nan() -> None:
    assert evaluate_evasion_gate({})["passed"] is False
    nan = float("nan")
    assert evaluate_evasion_gate({"x": _row(nan, 0.0)})["passed"] is False
    assert evaluate_evasion_gate({"x": _row(1.0, nan)})["passed"] is False


def test_gate_thresholds_are_configurable() -> None:
    rows = {"x": _row(0.85, 0.3)}
    assert evaluate_evasion_gate(rows)["passed"] is False
    assert (
        evaluate_evasion_gate(rows, max_trigger_tpr=0.35, min_clean_tpr=0.8)["passed"]
        is True
    )


def _gate(rows: dict[str, dict[str, float]]) -> dict:
    return evaluate_evasion_gate(rows)


def test_progress_prefers_more_passing_concepts() -> None:
    from next_chameleons.hf.in_distribution_eval import evasion_progress

    one = evasion_progress(_gate({"a": _row(1.0, 0.1), "b": _row(1.0, 0.9)}))
    two = evasion_progress(_gate({"a": _row(1.0, 0.1), "b": _row(1.0, 0.15)}))
    # Two concepts passing beats one, regardless of the gap term.
    assert two > one
    assert two[0] == 2 and one[0] == 1


def test_progress_rewards_shrinking_trigger_gap_when_pass_count_ties() -> None:
    from next_chameleons.hf.in_distribution_eval import evasion_progress

    far = evasion_progress(_gate({"a": _row(1.0, 1.0), "b": _row(1.0, 0.9)}))
    near = evasion_progress(_gate({"a": _row(1.0, 0.6), "b": _row(1.0, 0.9)}))
    # Same number passing (zero), but 'near' has driven a trigger closer to the
    # bound -> strictly greater progress, so patience would reset on it.
    assert near[0] == far[0] == 0
    assert near > far


def test_progress_penalizes_concept_destruction() -> None:
    from next_chameleons.hf.in_distribution_eval import evasion_progress

    # Destroying the concept (clean below the retention bound) is NOT progress:
    # the clean-gap term makes it worse than a healthy-but-caught concept.
    healthy = evasion_progress(_gate({"a": _row(1.0, 0.5)}))
    destroyed = evasion_progress(_gate({"a": _row(0.2, 0.5)}))
    assert healthy > destroyed


def test_progress_nan_is_not_progress() -> None:
    from next_chameleons.hf.in_distribution_eval import evasion_progress

    good = evasion_progress(_gate({"a": _row(1.0, 0.5)}))
    nan = evasion_progress(_gate({"a": {"clean_tpr": float("nan"), "matching_trigger_tpr": 0.5}}))
    assert good > nan
