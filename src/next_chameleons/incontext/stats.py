"""Statistics for PREREGISTRATION-ICA §5: AUROC, detection rates, and cluster bootstraps.

Concept sentences are nested in concepts, so their CIs come from a HIERARCHICAL bootstrap:
resample concepts, then units within each concept. Neutral sentences are crossed with
concepts, and those use a TWO-WAY bootstrap as in Kowalski et al.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted as half."""

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection_rate(scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(scores, float)
    return float((s > threshold).mean()) if len(s) else float("nan")


def hierarchical_bootstrap(
    values: dict[str, np.ndarray],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) for `stat` over units nested in clusters. The point pools every unit."""

    rng = np.random.default_rng(seed)
    keys = [k for k, v in values.items() if len(v)]
    point = float(stat(np.concatenate([np.asarray(values[k], float) for k in keys])))
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        sample = []
        for idx in picked:
            v = np.asarray(values[keys[idx]], float)
            sample.append(v[rng.integers(0, len(v), size=len(v))])
        draws.append(stat(np.concatenate(sample)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def twoway_bootstrap(
    matrix: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.nanmean,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) for a concepts x sentences matrix, resampling rows and columns."""

    rng = np.random.default_rng(seed)
    m = np.asarray(matrix, float)
    point = float(stat(m))
    draws = []
    for _ in range(n_boot):
        r = rng.integers(0, m.shape[0], size=m.shape[0])
        c = rng.integers(0, m.shape[1], size=m.shape[1])
        draws.append(stat(m[np.ix_(r, c)]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def derangement(groups: list[str], seed: int = 0) -> list[int]:
    """Permutation p with p[i] != i and groups[p[i]] == groups[i] (the yoked sham in §4).

    Each group needs at least 2 members.
    """

    rng = np.random.default_rng(seed)
    perm = list(range(len(groups)))
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    for g, idx in by_group.items():
        if len(idx) < 2:
            raise ValueError(f"group {g!r} has fewer than 2 units; cannot yoke")
        shift = int(rng.integers(1, len(idx)))  # cyclic shift by 1..n-1: never a fixed point
        order = list(rng.permutation(idx))
        for k, i in enumerate(order):
            perm[i] = order[(k + shift) % len(order)]
    return perm
