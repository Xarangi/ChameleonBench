"""core.metrics == each old implementation it replaces (bit-equal on fixed random inputs)."""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from crypsis.core import metrics as M


def _cases():
    rng = np.random.default_rng(0)
    yield rng.normal(size=50), rng.normal(size=70)
    yield rng.integers(0, 4, 30).astype(float), rng.integers(0, 4, 40).astype(float)  # ties
    yield np.ones(5), np.ones(8)  # all tied
    f32 = np.float32
    yield (rng.normal(size=200) * 7).astype(f32), (rng.normal(size=300) * 7).astype(f32)


def test_auroc_matches_monitors_and_incontext():
    from next_chameleons.incontext.stats import auroc as ic_auroc
    from next_chameleons.tica.monitors import auroc as mon_auroc

    for p, n in _cases():
        a = M.auroc(torch.from_numpy(p), torch.from_numpy(n))
        assert a == mon_auroc(torch.from_numpy(p), torch.from_numpy(n)) == ic_auroc(p, n)
        assert M.auroc(p, n) == a
    p = np.array([1.0, np.nan, 2.0])
    n = np.array([np.nan, 0.5])
    assert M.auroc(p, n) == mon_auroc(torch.from_numpy(p), torch.from_numpy(n))
    assert math.isnan(M.auroc([], [1.0])) and math.isnan(ic_auroc(np.array([]), np.array([1.0])))


def test_thresholds_and_tpr_match():
    try:  # the in-context study was pruned to archive/paper-replication
        from next_chameleons.incontext.monitor import fpr_threshold
    except ModuleNotFoundError:
        fpr_threshold = None
    from next_chameleons.tica.monitors import normal_quantile, quantile_threshold
    from next_chameleons.tica.probes import threshold_at

    rng = np.random.default_rng(1)
    for _ in range(300):
        x = (rng.normal(size=int(rng.integers(1, 400))) * rng.uniform(0.1, 50)).astype(np.float32)
        fpr = float(rng.choice([0.01, 0.05, 0.1, rng.uniform(0, 1)]))
        t = torch.from_numpy(x)
        assert M.quantile_threshold(t, fpr) == quantile_threshold(t, fpr)
        assert M.quantile_threshold(x, fpr, backend="numpy") == threshold_at(x, fpr)
        if fpr_threshold is not None:
            assert M.order_stat_threshold(x, fpr) == fpr_threshold(x, fpr)
        thr = quantile_threshold(t, fpr)
        assert M.tpr_at_threshold(x, thr) == float((t > thr).float().mean())
    assert M.normal_quantile(0.99) == normal_quantile(0.99)


def test_bootstraps_match():
    from next_chameleons.incontext.stats import hierarchical_bootstrap, twoway_bootstrap
    from next_chameleons.tica.analysis import boot_contrast

    rng = np.random.default_rng(2)
    eps = [f"c{i % 5}|{i}" for i in range(40)]
    a = {e: float(rng.normal()) for e in eps}
    b = {e: float(rng.normal()) for e in eps}
    diff = [a[e] - b[e] for e in sorted(eps)]
    conc = [e.split("|")[0] for e in sorted(eps)]
    assert M.cluster_bootstrap(diff, conc, 300, 3) == boot_contrast(a, b, 300, 3)[:3]
    vals = {f"k{i}": rng.normal(size=int(rng.integers(1, 9))) for i in range(6)}
    assert M.hierarchical_bootstrap(vals, n_boot=200) == hierarchical_bootstrap(vals, n_boot=200)
    m = rng.normal(size=(5, 7))
    assert M.twoway_bootstrap(m, n_boot=200) == twoway_bootstrap(m, n_boot=200)


def test_script_cluster_cis_match():
    """Inline copies of scripts/tica_math_race_report.cluster_ci and tica_dodge_anatomy.ci."""

    def old_cluster_ci(groups, num, den, n_boot=1000, seed=0):
        keys = list(groups)

        def tot(ks):
            rows = [r for k in ks for r in groups[k]]
            return sum(num(r) for r in rows), sum(den(r) for r in rows)

        a, b = tot(keys)
        rng, bs = random.Random(seed), []
        for _ in range(n_boot):
            x, y = tot([rng.choice(keys) for _ in keys])
            if y:
                bs.append(x / y)
        bs.sort()
        return [round(a / b, 3), round(bs[int(0.025 * len(bs))], 3),
                round(bs[int(0.975 * len(bs)) - 1], 3), int(b)]

    def old_ci(vals, groups, n_boot=400, seed=0):
        by = {}
        for v, g in zip(vals, groups, strict=True):
            by.setdefault(g, []).append(v)
        keys, rng, boots = list(by), random.Random(seed), []
        for _ in range(n_boot):
            s = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
            boots.append(sum(s) / len(s))
        boots.sort()
        m = sum(vals) / len(vals)
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        return [round(m, 3), round(lo, 3), round(hi, 3)]

    r = random.Random(4)
    groups = {e: [{"p": r.random() < 0.6, "c": r.random() < 0.5} for _ in range(r.randint(1, 5))]
              for e in range(25)}
    num, den = (lambda x: x["p"] and x["c"]), (lambda x: x["p"])
    assert M.ratio_cluster_ci(groups, num, den) == old_cluster_ci(groups, num, den)
    vals = [r.random() for _ in range(60)]
    gs = [i % 9 for i in range(60)]
    assert M.mean_cluster_ci(vals, gs) == old_ci(vals, gs)
