# ruff: noqa: E501
"""Detection metrics: AUROC, fixed-FPR thresholds, TPR at a threshold / FPR, bootstrap and cluster CIs.

One implementation per quantity, each reproducing an existing tica / script implementation exactly
(the docstring names it; tests/lib/test_core_metrics_equiv.py checks it). Numpy only; torch tensors
are accepted (detached, cast to float64) and torch is imported only by normal_quantile.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np


def _f64(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().double().flatten().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def auroc(pos: Any, neg: Any) -> float:
    """Mann-Whitney AUROC, ties counted as half (average ranks); NaN if either side is empty.

    == tica.monitors.auroc (bit-equal, including its NaN handling: NaNs share one average rank) and
    == incontext.stats.auroc (used by tica.probes) on finite inputs.
    """

    p, n = _f64(pos), _f64(neg)
    if not len(p) or not len(n):
        return float("nan")
    v = np.concatenate([p, n])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def _fma(x: float, y: float, z: float) -> float:
    """x * y + z with one rounding (exact rational arithmetic, then round-to-nearest)."""

    return float(Fraction(x) * Fraction(y) + Fraction(z))


def quantile_threshold(scores: Any, fpr: float, *, backend: str = "torch") -> float:
    """The (1 - fpr) quantile of negative scores, linear interpolation; detection is `score > thr`.

    backend="torch": == tica.monitors.quantile_threshold (`torch.quantile(x.double(), 1 - fpr)`, used
    by probe_zoo / adaptive / evdet / evasion_detector), reproduced bit-exactly in float64 including
    torch's fused multiply-add lerp. backend="numpy": == tica.probes.threshold_at
    (`np.quantile(scores, 1 - fpr)` in the input's own dtype). NaN if any score is NaN.
    """

    q = 1.0 - fpr
    if backend == "numpy":
        return float(np.quantile(np.asarray(scores), q))
    if backend != "torch":
        raise ValueError(f"backend must be 'torch' or 'numpy', not {backend!r}")
    s = np.sort(_f64(scores))
    if not len(s):
        raise ValueError("quantile_threshold of an empty score set")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")
    if np.isnan(s).any():
        return float("nan")
    rank = q * (len(s) - 1)
    lo, hi = int(rank), math.ceil(rank)
    w, a, b = rank - lo, float(s[lo]), float(s[hi])
    if abs(w) < 0.5:
        return _fma(w, b - a, a)
    return _fma(-(b - a), 1.0 - w, b)


def order_stat_threshold(scores: Any, fpr: float) -> float:
    """Threshold with at most floor(fpr * n) scores strictly above it (an order statistic, no
    interpolation); NaN if empty. == incontext.monitor.fpr_threshold."""

    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.floor(fpr * n))
    return float(s[n - 1 - k]) if n else float("nan")


def normal_quantile(q: float) -> float:
    """Standard normal inverse CDF. == tica.monitors.normal_quantile (torch float64 icdf); falls back
    to statistics.NormalDist when torch is absent (may differ in the last bits)."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is installed wherever training runs
        from statistics import NormalDist

        return NormalDist().inv_cdf(q)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(q, dtype=torch.float64)))


def tpr_at_threshold(pos: Any, thr: float, *, float32: bool = True) -> float:
    """Fraction of positives strictly above thr. float32=True == `float((s > thr).float().mean())`
    (the gate TPR in adaptive / probe_zoo / monitors: a float32 mean); float32=False == the float64
    incontext.stats.detection_rate. NaN if empty."""

    p = _f64(pos)
    if not len(p):
        return float("nan")
    k = int((p > thr).sum())
    if float32:
        return float(np.float32(k) / np.float32(len(p)))
    return k / len(p)


def tpr_at_fpr(pos: Any, neg: Any, fpr: float, *, backend: str = "torch") -> float:
    """TPR at the fixed-FPR threshold of neg (quantile_threshold, then tpr_at_threshold)."""

    return tpr_at_threshold(pos, quantile_threshold(neg, fpr, backend=backend))


# ----------------------------------------------------------------------------------- intervals
def cluster_bootstrap(values: Sequence[float], clusters: Sequence[Hashable], n_boot: int = 2000,
                      seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): 95% CI of the mean resampling whole clusters (np.random.default_rng, sorted
    cluster order, np.quantile). == tica.analysis.boot_contrast[:3] with values = the per-episode
    differences and clusters = their concepts."""

    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    uc = np.unique(c)
    groups = [v[c == u] for u in uc]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(v.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def hierarchical_bootstrap(values: Mapping[str, Any], stat: Callable = np.mean, n_boot: int = 2000,
                           seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi): resample clusters, then units within each. == incontext.stats.hierarchical_bootstrap."""

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


def twoway_bootstrap(matrix: Any, stat: Callable = np.nanmean, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) for a clusters x items matrix resampling rows and columns.
    == incontext.stats.twoway_bootstrap."""

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


def _round(x: float, nd: int | None) -> float:
    return x if nd is None else round(x, nd)


def ratio_cluster_ci(groups: Mapping[Hashable, Sequence], num: Callable, den: Callable,
                     n_boot: int = 1000, seed: int = 0, ndigits: int | None = 3) -> list | None:
    """[sum(num)/sum(den), lo, hi, sum(den)] with a 95% CI resampling clusters (random.Random);
    None if the denominator is 0. == scripts/tica_math_race_report.cluster_ci (also used by
    tica_task_race_report); ndigits=3 reproduces its rounding."""

    keys = list(groups)

    def tot(ks):
        return (sum(num(r) for k in ks for r in groups[k]), sum(den(r) for k in ks for r in groups[k]))

    a, b = tot(keys)
    if not b:
        return None
    rng, bs = random.Random(seed), []
    for _ in range(n_boot):
        x, y = tot([rng.choice(keys) for _ in keys])
        if y:
            bs.append(x / y)
    bs.sort()
    return [_round(a / b, ndigits), _round(bs[int(0.025 * len(bs))], ndigits),
            _round(bs[int(0.975 * len(bs)) - 1], ndigits), int(b)]


def mean_cluster_ci(vals: Sequence[float], groups: Sequence[Hashable], n_boot: int = 400,
                    seed: int = 0, ndigits: int | None = 3) -> list | None:
    """[mean, lo, hi] with a 95% CI resampling groups (random.Random); None if empty.
    == scripts/tica_dodge_anatomy.ci (note its upper index int(0.975 * n_boot), not ... - 1)."""

    if not vals:
        return None
    by: dict[Hashable, list[float]] = {}
    for v, g in zip(vals, groups, strict=True):
        by.setdefault(g, []).append(v)
    keys = list(by)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
        boots.append(sum(s) / len(s))
    boots.sort()
    m = sum(vals) / len(vals)
    return [_round(m, ndigits), _round(boots[int(0.025 * n_boot)], ndigits),
            _round(boots[int(0.975 * n_boot)], ndigits)]
