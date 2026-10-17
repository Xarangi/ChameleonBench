"""Unit tests for the Track C3 direction-geometry pure functions.

All tests run on hand-constructed synthetic vectors: no model load, CPU-only,
fast, and offline-safe. They cover the geometry core that
``run_direction_geometry`` composes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from next_chameleons.analysis.direction_geometry import (
    assemble_direction_geometry_report,
    cosine,
    evasion_direction,
    orthogonal_residual_ratio,
    projection_ratio,
)


def test_cosine_parallel_orthogonal_antiparallel() -> None:
    assert cosine([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0, 0.0], [0.0, 3.0, 0.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0, 0.0], [-5.0, 0.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_forty_five_degrees() -> None:
    assert cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0 / math.sqrt(2.0))


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_projection_ratio_vector_in_span_is_one() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # rows are vectors
    # A vector lying entirely in the e1-e2 plane is fully captured.
    assert projection_ratio([3.0, 4.0, 0.0], basis) == pytest.approx(1.0)


def test_projection_ratio_orthogonal_is_zero() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert projection_ratio([0.0, 0.0, 5.0], basis) == pytest.approx(0.0)


def test_projection_ratio_partial_is_intermediate() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # [1,0,1]: projection onto the plane is [1,0,0], so captured norm fraction
    # is 1/sqrt(2).
    assert projection_ratio([1.0, 0.0, 1.0], basis) == pytest.approx(1.0 / math.sqrt(2.0))


def test_projection_ratio_single_basis_vector() -> None:
    # A 1-D basis passed as a flat vector is treated as one basis direction.
    assert projection_ratio([2.0, 2.0], [1.0, 0.0]) == pytest.approx(1.0 / math.sqrt(2.0))


def test_projection_ratio_rank_deficient_basis() -> None:
    # Two collinear basis rows still span only a line; the covered fraction must
    # not be inflated by the duplicate.
    basis = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert projection_ratio([1.0, 0.0, 1.0], basis) == pytest.approx(1.0 / math.sqrt(2.0))


def test_projection_ratio_empty_and_zero_edge_cases() -> None:
    assert projection_ratio([1.0, 2.0], np.zeros((0, 2))) == 0.0
    assert projection_ratio([0.0, 0.0], [[1.0, 0.0]]) == 0.0


def test_projection_ratio_columns_layout() -> None:
    # Same plane, but basis given as columns-as-vectors.
    basis_cols = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # [hidden, num_vectors]
    assert projection_ratio(
        [1.0, 0.0, 1.0], basis_cols, rows_are_vectors=False
    ) == pytest.approx(1.0 / math.sqrt(2.0))


def test_orthogonal_residual_is_l2_complement() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ratio = projection_ratio([1.0, 0.0, 1.0], basis)
    residual = orthogonal_residual_ratio([1.0, 0.0, 1.0], basis)
    assert ratio**2 + residual**2 == pytest.approx(1.0)
    assert residual == pytest.approx(1.0 / math.sqrt(2.0))


def test_evasion_direction_is_triggered_minus_clean() -> None:
    direction = evasion_direction([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert np.allclose(direction, np.array([0.0, 1.0, 2.0]))


def test_evasion_direction_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        evasion_direction([1.0, 2.0], [1.0, 2.0, 3.0])


def test_assemble_report_orthogonal_then_covered() -> None:
    """Round-2 evasion sits orthogonal to a 1-round bank, covered by a 2-round one."""

    # Concept "a" evades along e3.
    evasion = {"a": np.array([0.0, 0.0, 2.0])}

    # Bank with a single prior round whose probe reads the e1 direction: the
    # evasion direction is orthogonal to it.
    one_round_bank = [{"a": np.array([1.0, 0.0, 0.0])}]
    report_1 = assemble_direction_geometry_report(evasion, one_round_bank, round_k=1)
    row_1 = report_1["per_concept"]["a"]
    assert row_1["concept_projection_ratio"] == pytest.approx(0.0)
    assert row_1["concept_residual_ratio"] == pytest.approx(1.0)
    assert row_1["global_projection_ratio"] == pytest.approx(0.0)
    assert row_1["max_abs_cosine_to_bank"] == pytest.approx(0.0)
    assert report_1["num_bank_members_total"] == 1

    # Add a second round whose probe reads e3: the evasion is now fully covered.
    two_round_bank = [
        {"a": np.array([1.0, 0.0, 0.0])},
        {"a": np.array([0.0, 0.0, 1.0])},
    ]
    report_2 = assemble_direction_geometry_report(evasion, two_round_bank, round_k=2)
    row_2 = report_2["per_concept"]["a"]
    assert row_2["concept_projection_ratio"] == pytest.approx(1.0)
    assert row_2["concept_residual_ratio"] == pytest.approx(0.0)
    assert row_2["max_abs_cosine_to_bank"] == pytest.approx(1.0)
    # Two members recorded, one per prior round.
    assert row_2["num_bank_members"] == 2
    rounds = {c["round"] for c in row_2["cosine_to_bank_members"]}
    assert rounds == {0, 1}


def test_assemble_report_ensemble_members_flattened() -> None:
    """An ensemble probe (multiple members) is flattened into per-member rows."""

    evasion = {"a": np.array([1.0, 1.0, 0.0])}
    # One round, one concept, but the probe is a 2-member ensemble.
    bank = [{"a": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}]
    report = assemble_direction_geometry_report(evasion, bank, round_k=1)
    row = report["per_concept"]["a"]
    assert row["num_bank_members"] == 2
    assert report["num_bank_members_total"] == 2
    # Evasion lies in the span of the two ensemble members.
    assert row["concept_projection_ratio"] == pytest.approx(1.0)
    # Each member is at 45 degrees to the evasion direction.
    for entry in row["cosine_to_bank_members"]:
        assert entry["cosine"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_assemble_report_global_vs_concept_span() -> None:
    """Global span pools members across concepts; concept span does not."""

    # Concept "a" evades along e3. Its own probe reads e1 (no coverage), but
    # concept "b"'s probe reads e3, so the WHOLE-bank span covers "a".
    evasion = {"a": np.array([0.0, 0.0, 1.0])}
    bank = [{"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 1.0])}]
    report = assemble_direction_geometry_report(evasion, bank, round_k=1)
    row = report["per_concept"]["a"]
    assert row["concept_projection_ratio"] == pytest.approx(0.0)
    assert row["global_projection_ratio"] == pytest.approx(1.0)


def test_assemble_report_is_json_serializable() -> None:
    import json

    evasion = {"a": np.array([0.0, 0.0, 2.0]), "b": np.array([1.0, 0.0, 0.0])}
    bank = [{"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0])}]
    report = assemble_direction_geometry_report(evasion, bank, round_k=1)
    encoded = json.dumps(report)  # must not raise
    assert '"direction_geometry"' in encoded
    assert set(report["concepts"]) == {"a", "b"}
    assert report["summary"]["num_concepts"] == 2
