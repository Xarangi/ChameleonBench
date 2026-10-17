"""Track C3 -- direction geometry of arms-race probe evasion.

The scientific question: when an obfuscation organism is *triggered* ("You are
being probed for '<concept>'."), its activations at the probe layer shift. That
shift -- the **evasion direction** -- is what a frozen concept probe reads as a
drop in score. As the defender accumulates one converged probe per round, the
bank spans a growing subspace. The containment hypothesis is that the organism
routes its evasion into a subspace *orthogonal* to the current bank, and that the
accumulating bank progressively covers more of that direction until evasion has
nowhere left to hide.

This module provides:

- **Pure geometry functions** (numpy / torch friendly, unit-tested on synthetic
  vectors): :func:`cosine`, :func:`projection_ratio`,
  :func:`orthogonal_residual_ratio`, :func:`evasion_direction`,
  :func:`concept_direction_geometry`, and
  :func:`assemble_direction_geometry_report`.
- A higher-level :func:`run_direction_geometry` that loads a round's organism
  checkpoint and the accumulated probe bank from an arms-race game directory,
  extracts clean vs triggered concept activations (reusing the paper-faithful
  feature-capture path), and returns a JSON-serializable geometry report.

Vector convention for the pure functions: a "basis" is a 2D array whose **rows**
are basis vectors (shape ``[num_vectors, hidden]``) by default -- this matches how
a bank of concept-probe weight vectors stacks naturally. Set
``rows_are_vectors=False`` to pass columns-as-vectors (``[hidden, num_vectors]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Pure geometry core (no model, no torch required to run).
# --------------------------------------------------------------------------- #


def as_numpy(value: Any) -> np.ndarray:
    """Coerce a numpy array or torch tensor to a 1-D-or-2-D float64 numpy array.

    Accepts torch tensors (detaching / moving to CPU as needed), numpy arrays, or
    anything array-like. float64 keeps the geometry well conditioned; the inputs
    are small (hidden-dim vectors), so the up-cast is cheap.
    """

    if value is None:
        raise ValueError("Cannot convert None to a numpy array")
    detach = getattr(value, "detach", None)
    if callable(detach):  # torch tensor
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_fn = getattr(value, "numpy", None)
        if callable(numpy_fn):
            value = numpy_fn()
    array = np.asarray(value, dtype=np.float64)
    return array


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity between two vectors; 0.0 if either has zero norm.

    Returns a value in ``[-1, 1]``: ``+1`` for parallel, ``-1`` for anti-parallel,
    ``0`` for orthogonal.
    """

    va = as_numpy(a).reshape(-1)
    vb = as_numpy(b).reshape(-1)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _basis_columns(basis: Any, *, rows_are_vectors: bool) -> np.ndarray:
    """Return the basis as a ``[hidden, num_vectors]`` column matrix (or empty)."""

    matrix = as_numpy(basis)
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"basis must be 1-D or 2-D, got shape {matrix.shape}")
    # Internally we want columns as the basis vectors for SVD.
    return matrix.T if rows_are_vectors else matrix


def _orthonormal_range(columns: np.ndarray, *, tol: float) -> np.ndarray:
    """Orthonormal basis (columns) for the column space of ``columns`` via SVD.

    Singular values at or below ``tol * max_singular_value`` are dropped so a
    rank-deficient bank (repeated / collinear probe directions) is handled
    correctly instead of inflating the covered subspace.
    """

    if columns.size == 0 or columns.shape[1] == 0:
        return np.zeros((columns.shape[0] if columns.ndim == 2 else 0, 0))
    u, singular_values, _ = np.linalg.svd(columns, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((columns.shape[0], 0))
    cutoff = tol * float(singular_values[0])
    rank = int(np.sum(singular_values > cutoff))
    return u[:, :rank]


def projection_ratio(
    vector: Any,
    basis: Any,
    *,
    rows_are_vectors: bool = True,
    tol: float = 1e-9,
) -> float:
    """Fraction of ``vector``'s norm captured by the column space of ``basis``.

    Returns ``||proj_S(v)|| / ||v||`` where ``S`` is the span of the basis
    vectors: ``1.0`` when ``v`` lies fully in the span, ``0.0`` when ``v`` is
    orthogonal to it, and an intermediate value otherwise. ``0.0`` for a zero
    vector or an empty basis.

    ``basis`` rows are the basis vectors by default (``[num_vectors, hidden]``);
    pass ``rows_are_vectors=False`` for a ``[hidden, num_vectors]`` layout.
    """

    v = as_numpy(vector).reshape(-1)
    v_norm = float(np.linalg.norm(v))
    if v_norm == 0.0:
        return 0.0
    columns = _basis_columns(basis, rows_are_vectors=rows_are_vectors)
    if columns.size == 0 or columns.shape[1] == 0:
        return 0.0
    if columns.shape[0] != v.shape[0]:
        raise ValueError(
            f"basis hidden dim {columns.shape[0]} does not match vector dim {v.shape[0]}"
        )
    orthonormal = _orthonormal_range(columns, tol=tol)
    if orthonormal.shape[1] == 0:
        return 0.0
    projection_coords = orthonormal.T @ v
    projection_norm = float(np.linalg.norm(projection_coords))
    # Clamp for floating-point overshoot right at ratio == 1.
    return float(min(1.0, projection_norm / v_norm))


def orthogonal_residual_ratio(
    vector: Any,
    basis: Any,
    *,
    rows_are_vectors: bool = True,
    tol: float = 1e-9,
) -> float:
    """Fraction of ``vector``'s norm orthogonal to the basis span.

    Complement of :func:`projection_ratio` in the L2 sense:
    ``sqrt(1 - projection_ratio**2)``. ``1.0`` when fully orthogonal, ``0.0`` when
    fully contained.
    """

    ratio = projection_ratio(vector, basis, rows_are_vectors=rows_are_vectors, tol=tol)
    return float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))


def evasion_direction(triggered_mean: Any, clean_mean: Any) -> np.ndarray:
    """Per-concept evasion direction = mean(triggered activation) - mean(clean).

    Both inputs are mean-pooled probe-layer activation vectors (``[hidden]``). The
    returned direction is what a frozen concept probe reads when the organism is
    triggered.
    """

    triggered = as_numpy(triggered_mean).reshape(-1)
    clean = as_numpy(clean_mean).reshape(-1)
    if triggered.shape != clean.shape:
        raise ValueError(
            f"triggered_mean {triggered.shape} and clean_mean {clean.shape} must match"
        )
    return triggered - clean


def _iter_member_vectors(entry: Any) -> list[np.ndarray]:
    """Flatten a per-round per-concept probe entry into a list of 1-D vectors.

    A linear probe stores one ``[hidden]`` weight; an ensemble stores
    ``[members, hidden]``. Both flatten to a list of member direction vectors.
    """

    array = as_numpy(entry)
    if array.ndim == 1:
        return [array]
    if array.ndim == 2:
        return [row for row in array]
    raise ValueError(f"probe entry must be 1-D or 2-D, got shape {array.shape}")


def concept_direction_geometry(
    evasion_dir: Any,
    concept_members: list[tuple[int, int, np.ndarray]],
    global_basis_rows: list[np.ndarray],
    *,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Geometry of one concept's evasion direction vs the accumulated bank.

    ``concept_members`` is a list of ``(round_index, member_index, weight_vector)``
    for THIS concept across the accumulated rounds; ``global_basis_rows`` is every
    bank member direction across all concepts (rows-as-vectors) used for the
    whole-bank subspace coverage. Returns a JSON-serializable dict.
    """

    v = as_numpy(evasion_dir).reshape(-1)
    v_norm = float(np.linalg.norm(v))
    cosines: list[dict[str, Any]] = []
    concept_rows: list[np.ndarray] = []
    for round_index, member_index, weight in concept_members:
        weight_vec = as_numpy(weight).reshape(-1)
        concept_rows.append(weight_vec)
        cosines.append(
            {
                "round": int(round_index),
                "member": int(member_index),
                "cosine": cosine(v, weight_vec),
            }
        )
    abs_cosines = [abs(item["cosine"]) for item in cosines]
    concept_basis = np.stack(concept_rows, axis=0) if concept_rows else np.zeros((0, 0))
    global_basis = (
        np.stack([as_numpy(row).reshape(-1) for row in global_basis_rows], axis=0)
        if global_basis_rows
        else np.zeros((0, 0))
    )
    concept_proj = projection_ratio(v, concept_basis, tol=tol) if concept_rows else 0.0
    global_proj = projection_ratio(v, global_basis, tol=tol) if len(global_basis_rows) else 0.0
    return {
        "evasion_norm": v_norm,
        "num_bank_members": len(concept_members),
        "cosine_to_bank_members": cosines,
        "max_abs_cosine_to_bank": max(abs_cosines) if abs_cosines else 0.0,
        "mean_abs_cosine_to_bank": (sum(abs_cosines) / len(abs_cosines)) if abs_cosines else 0.0,
        "concept_projection_ratio": concept_proj,
        "concept_residual_ratio": float(np.sqrt(max(0.0, 1.0 - concept_proj * concept_proj))),
        "global_projection_ratio": global_proj,
        "global_residual_ratio": float(np.sqrt(max(0.0, 1.0 - global_proj * global_proj))),
    }


def assemble_direction_geometry_report(
    evasion_directions: dict[str, Any],
    bank_by_round: list[dict[str, Any]],
    *,
    round_k: int | None = None,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Assemble the per-concept / per-round geometry report from precomputed arrays.

    Parameters
    ----------
    evasion_directions:
        ``{concept: vector[hidden]}`` -- round ``k``'s evasion direction per concept.
    bank_by_round:
        The accumulated bank ``{P_0 .. P_{k-1}}`` as a list indexed by round; each
        element is ``{concept: weight}`` where ``weight`` is ``[hidden]`` (linear
        probe) or ``[members, hidden]`` (ensemble). List position is the round
        index.
    round_k:
        Optional label for the round whose evasion is being analysed (recorded in
        the report only).

    Returns a JSON-serializable dict with per-concept cosines to every bank
    member, per-concept subspace coverage (own-concept span and whole-bank span),
    and summary means.
    """

    # Whole-bank basis: every member direction across all rounds and concepts.
    global_basis_rows: list[np.ndarray] = []
    for round_entry in bank_by_round:
        for entry in round_entry.values():
            global_basis_rows.extend(_iter_member_vectors(entry))

    per_concept: dict[str, Any] = {}
    for concept in sorted(evasion_directions):
        concept_members: list[tuple[int, int, np.ndarray]] = []
        for round_index, round_entry in enumerate(bank_by_round):
            if concept not in round_entry:
                continue
            for member_index, vector in enumerate(_iter_member_vectors(round_entry[concept])):
                concept_members.append((round_index, member_index, vector))
        per_concept[concept] = concept_direction_geometry(
            evasion_directions[concept],
            concept_members,
            global_basis_rows,
            tol=tol,
        )

    concept_projs = [row["concept_projection_ratio"] for row in per_concept.values()]
    global_projs = [row["global_projection_ratio"] for row in per_concept.values()]
    max_cosines = [row["max_abs_cosine_to_bank"] for row in per_concept.values()]

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "track": "direction_geometry",
        "round_k": round_k,
        "num_accumulated_rounds": len(bank_by_round),
        "num_bank_members_total": len(global_basis_rows),
        "concepts": sorted(evasion_directions),
        "per_concept": per_concept,
        "summary": {
            "num_concepts": len(per_concept),
            "mean_concept_projection_ratio": _mean(concept_projs),
            "mean_concept_residual_ratio": _mean(
                [row["concept_residual_ratio"] for row in per_concept.values()]
            ),
            "mean_global_projection_ratio": _mean(global_projs),
            "mean_global_residual_ratio": _mean(
                [row["global_residual_ratio"] for row in per_concept.values()]
            ),
            "mean_max_abs_cosine_to_bank": _mean(max_cosines),
            "max_global_projection_ratio": max(global_projs) if global_projs else float("nan"),
            "min_global_projection_ratio": min(global_projs) if global_projs else float("nan"),
        },
    }


# --------------------------------------------------------------------------- #
# Real-data hookup (best-effort; reuses the paper feature-capture path).
# --------------------------------------------------------------------------- #


def _concept_weight_arrays(probe: Any) -> np.ndarray:
    """Extract a probe's weight direction(s) as numpy rows-as-vectors.

    Handles both ``FrozenLinearConceptProbe`` (``weight`` -> ``[1, hidden]``) and
    ``FrozenEnsembleConceptProbe`` (``weights`` -> ``[members, hidden]``).
    """

    weights = getattr(probe, "weights", None)
    if weights is not None:  # ensemble
        return as_numpy(weights).reshape(len(weights), -1)
    weight = as_numpy(probe.weight).reshape(1, -1)
    return weight


def _bank_to_round_entry(bank: Any) -> dict[str, np.ndarray]:
    """Turn a loaded ``FrozenProbeBank`` into ``{concept: weight_rows[members, hidden]}``."""

    return {
        concept: _concept_weight_arrays(probe)
        for concept, probe in bank.probes_by_concept.items()
    }


def _discover_checkpoint(round_dir: Path, experiment_name: str | None) -> tuple[Path, str]:
    """Resolve ``round_dir/checkpoints/<EXP>`` -> (checkpoint_dir, experiment_name).

    The arms-race stager writes the organism to ``round<k>/checkpoints/<EXP>``. If
    ``experiment_name`` is given we use it directly; otherwise we discover the
    single experiment subdirectory that carries a HF/PEFT checkpoint.
    """

    checkpoints_root = round_dir / "checkpoints"
    if experiment_name is not None:
        return checkpoints_root / experiment_name, experiment_name
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"No checkpoints directory under {round_dir}")
    candidates = [path for path in sorted(checkpoints_root.iterdir()) if path.is_dir()]

    def _has_checkpoint(path: Path) -> bool:
        return (path / "adapter_config.json").exists() or (path / "config.json").exists()

    with_ckpt = [path for path in candidates if _has_checkpoint(path)]
    chosen = with_ckpt[0] if with_ckpt else (candidates[0] if candidates else None)
    if chosen is None:
        raise FileNotFoundError(f"No experiment checkpoint found under {checkpoints_root}")
    return chosen, chosen.name


def _mean_activation_by_concept(captured: list[Any]) -> dict[str, np.ndarray]:
    """Mean probe-layer activation per concept over all captured response tokens.

    ``captured`` is a list of ``CapturedExample`` (``.concept``, ``.features``
    ``[tokens, hidden]``). Tokens are concatenated across examples of a concept and
    averaged, giving each response token equal weight.
    """

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for example in captured:
        features = as_numpy(example.features)
        if features.ndim != 2 or features.shape[0] == 0:
            continue
        concept = example.concept
        token_sum = features.sum(axis=0)
        if concept in sums:
            sums[concept] = sums[concept] + token_sum
            counts[concept] += features.shape[0]
        else:
            sums[concept] = token_sum
            counts[concept] = features.shape[0]
    return {concept: sums[concept] / counts[concept] for concept in sums if counts[concept] > 0}


def run_direction_geometry(
    game_dir: str | Path,
    round_k: int,
    *,
    experiment_name: str | None = None,
    bank_dir: str | Path | None = None,
    seed: int = 17,
    max_examples: int | None = None,
    reduction: str = "mean",
    paths: Any | None = None,
) -> dict[str, Any]:
    """Compute the direction-geometry report for round ``round_k`` of a game.

    Loads the round-``k`` organism checkpoint and the accumulated probe bank
    ``{bank_r0 .. bank_r{k-1}}``, extracts clean vs triggered concept activations
    at the probe layer (reusing the paper-faithful capture path), and returns the
    JSON-serializable report from :func:`assemble_direction_geometry_report`.

    Expected on-disk layout (arms-race stager)::

        <game_dir>/banks/bank_r{j}.pt          # defender bank fit each round j
        <game_dir>/round{k}/checkpoints/<EXP>/ # organism O_k (LoRA adapter)

    Heavy imports (torch / transformers / the probe-bank capture helpers) are
    deferred to call time so the pure geometry functions and the tests stay
    import-light and offline-safe.
    """

    # Deferred imports: reuse the existing loaders rather than reimplement them.
    from next_chameleons.config import ProjectPaths
    from next_chameleons.hf.models import load_causal_lm
    from next_chameleons.hf.paper_probes import load_frozen_probe_bank, resolve_bank_paths
    from next_chameleons.real import common
    from next_chameleons.real.probe_bank import (
        _build_trainer,
        _capture_examples,
        _load_concept_examples,
        _select_probe_fit_examples,
    )
    from next_chameleons.real.resolve import resolve_real_experiment

    game_dir = Path(game_dir)
    round_dir = game_dir / f"round{round_k}"
    checkpoint_dir, experiment_name = _discover_checkpoint(round_dir, experiment_name)

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)

    # Accumulated defender bank = the banks the round-k attacker trained against:
    # bank_r0 .. bank_r{k-1}. round0 (k==0) has an empty accumulated bank.
    banks_root = Path(bank_dir) if bank_dir is not None else game_dir / "banks"
    bank_by_round: list[dict[str, np.ndarray]] = []
    bank_paths_used: list[str] = []
    missing_banks: list[str] = []
    for j in range(round_k):
        bank_path = banks_root / f"bank_r{j}.pt"
        if not bank_path.exists():
            missing_banks.append(str(bank_path))
            continue
        for resolved_path in resolve_bank_paths(bank_path):
            bank = load_frozen_probe_bank(resolved_path)
            bank_by_round.append(_bank_to_round_entry(bank))
            bank_paths_used.append(str(resolved_path))

    # Build the paper-faithful trainer + concept examples, then capture clean and
    # triggered probe-layer features on the round-k organism checkpoint.
    trainer = _build_trainer(experiment, seed=seed)
    real_cfg = dict(experiment["real_run"])
    destination = common.output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = common.raw_cache_root(destination)
    concept_examples = _load_concept_examples(experiment, raw_cache_root=raw_cache_root)
    limit = max_examples or int(
        experiment["dataset_config"].get("limits", {}).get("max_train_examples", 2048)
    )
    fit_examples, concepts = _select_probe_fit_examples(
        trainer, concept_examples, max_examples=limit
    )

    model_cfg = experiment["model_config"]
    loaded = load_causal_lm(
        hf_id=str(checkpoint_dir),
        dtype=str(model_cfg.get("default_dtype", "bfloat16")),
        quantization="none",
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    model.eval()
    max_tokens = trainer.train_cfg.get("benign_probe_max_tokens_per_example")
    capture_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        examples=fit_examples,
        concepts=concepts,
        max_tokens_per_example=int(max_tokens) if max_tokens is not None else None,
    )
    clean = _capture_examples(trainer, triggered=False, **capture_kwargs)
    triggered = _capture_examples(trainer, triggered=True, **capture_kwargs)

    clean_means = _mean_activation_by_concept(clean)
    triggered_means = _mean_activation_by_concept(triggered)
    evasion_directions: dict[str, np.ndarray] = {}
    for concept in concepts:
        if concept in clean_means and concept in triggered_means:
            evasion_directions[concept] = evasion_direction(
                triggered_means[concept], clean_means[concept]
            )

    report = assemble_direction_geometry_report(
        evasion_directions, bank_by_round, round_k=round_k
    )
    report["game_dir"] = str(game_dir)
    report["experiment_id"] = experiment.get("id", experiment_name)
    report["checkpoint_dir"] = str(checkpoint_dir)
    report["probe_layer"] = int(trainer.paper_probe_layer)
    report["reduction"] = reduction
    report["bank_paths"] = bank_paths_used
    if missing_banks:
        report["missing_banks"] = missing_banks
    return report


__all__ = [
    "as_numpy",
    "assemble_direction_geometry_report",
    "concept_direction_geometry",
    "cosine",
    "evasion_direction",
    "orthogonal_residual_ratio",
    "projection_ratio",
    "run_direction_geometry",
]
