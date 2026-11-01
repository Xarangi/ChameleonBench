"""Probe fitting: the deterministic logistic solvers, the refit buffer and ONE fit entry point.

`fit(kind, pos, neg, np, gate)` covers every registered kind (crypsis.probes.kinds): plain fits
return an uncalibrated monitor, gated fits return a calibrated monitor or a `GateFail`.
`fit_with_info` keeps the snapshot's `(monitor or None, info)` contract (refits.jsonl rows), and
`fit_job` is the trainer's fit stage on a reads job (snapshot `_refit_fit`).

Ported from next_chameleons.tica.monitors (fit_logistic, fit_logistic_many, normal_quantile,
quantile_threshold, auroc) and tica.adaptive (RefitData, _split); the numerics are unchanged.
TODO(integrator): auroc / quantile_threshold / normal_quantile duplicate crypsis.core.metrics once
it lands with its equivalence tests.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from crypsis.probes.base import Monitor
    from crypsis.probes.gates import FitPerf, GateConfig


# ------------------------------------------------------------------------------ logistic solvers
def fit_logistic(
    x: torch.Tensor, y: torch.Tensor, C: float, *, max_iter: int = 500, tol: float = 1e-7
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Standardized, class-balanced L2 logistic regression (sklearn lbfgs objective).

    Minimises C * sum_i w_i * logloss_i + 0.5 * ||w||^2 by full-batch LBFGS from zero (convex, so
    deterministic). Returns (w, b, mu, sd) with logit(x) = ((x - mu) / sd) @ w + b; raises
    RuntimeError if it does not converge.
    """

    x = x.double()
    y = y.double().to(x.device)  # CUDA rows with CPU labels (branch refits) must not crash
    mu = x.mean(0)
    sd = x.std(0, unbiased=False) + 1e-6
    z = (x - mu) / sd
    n_pos, n = y.sum(), len(y)
    sw = torch.where(y > 0, n / (2 * n_pos), n / (2 * (n - n_pos)))
    w = torch.zeros(z.shape[1], dtype=torch.float64, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        lo = z @ w + b
        loss = (
            C
            * (
                sw * torch.nn.functional.binary_cross_entropy_with_logits(lo, y, reduction="none")
            ).sum()
        )
        loss = loss + 0.5 * (w * w).sum()
        loss.backward()
        return loss

    with torch.enable_grad():  # callers may sit inside no_grad (frozen races)
        opt.step(closure)
        closure()
    gnorm = torch.cat([w.grad.flatten(), b.grad.flatten()]).abs().max().item()
    if not np.isfinite(gnorm) or gnorm > 1e-3 * max(1.0, C * n):
        raise RuntimeError(f"logistic fit did not converge (max |grad| {gnorm:.2e})")
    return w.detach().float(), float(b.detach()), mu.float(), sd.float()


def fit_logistic_many(
    xs: list[torch.Tensor], ys: list[torch.Tensor], C: float, *, max_iter: int = 100,
    tol: float = 1e-9,
) -> list[tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]]:
    """fit_logistic's objective for several independent problems, by batched damped Newton.

    Same per-problem standardisation, class-balanced weights, C and L2 on w only; fp64 Cholesky
    with Armijo backtracking. Problems may differ in n (padded rows carry zero weight). Reaches the
    same optimum as fit_logistic (strictly convex) to max |grad| ~ tol. Raises like fit_logistic.
    """

    dev, F = xs[0].device, len(xs)
    d, nmax = xs[0].shape[1], max(len(x) for x in xs)
    Z = torch.zeros(F, nmax, d + 1, dtype=torch.float64, device=dev)
    Y = torch.zeros(F, nmax, dtype=torch.float64, device=dev)
    SW = torch.zeros(F, nmax, dtype=torch.float64, device=dev)
    stats = []
    for f, (x, y) in enumerate(zip(xs, ys, strict=True)):
        x = x.double()
        y = y.double().to(x.device)
        mu = x.mean(0)
        sd = x.std(0, unbiased=False) + 1e-6
        n, n_pos = len(y), y.sum()
        Z[f, :n, :d] = (x - mu) / sd
        Z[f, :n, d] = 1.0
        Y[f, :n] = y
        SW[f, :n] = torch.where(y > 0, n / (2 * n_pos), n / (2 * (n - n_pos)))
        stats.append((mu, sd, n))
    reg = torch.ones(d + 1, dtype=torch.float64, device=dev)
    reg[d] = 0.0  # no penalty on the bias
    th = torch.zeros(F, d + 1, dtype=torch.float64, device=dev)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def obj(t):
        lo = (Z @ t[..., None])[..., 0]
        return C * (SW * bce(lo, Y, reduction="none")).sum(1) + 0.5 * (reg * t * t).sum(1)

    def grad(t):
        p = torch.sigmoid((Z @ t[..., None])[..., 0])
        return C * (Z.transpose(1, 2) @ (SW * (p - Y))[..., None])[..., 0] + reg * t, p

    for _ in range(max_iter):
        g, p = grad(th)
        if float(g.abs().amax()) < tol:
            break
        s = C * SW * p * (1 - p)
        H = (Z * s[..., None]).transpose(1, 2) @ Z + torch.diag_embed(reg).expand(F, -1, -1)
        L, info = torch.linalg.cholesky_ex(H)
        if bool((info != 0).any()):  # numerically singular (bias column): tiny jitter
            L = torch.linalg.cholesky(H + 1e-10 * torch.eye(d + 1, dtype=H.dtype, device=dev))
        step = torch.cholesky_solve(g[..., None], L)[..., 0]
        f0, slope = obj(th), (g * step).sum(1)
        t = torch.ones(F, dtype=torch.float64, device=dev)
        for _ in range(40):  # backtracking (Armijo), per problem
            cand = th - t[:, None] * step
            bad = obj(cand) > f0 - 1e-4 * t * slope
            if not bool(bad.any()):
                break
            t = torch.where(bad, t / 2, t)
        th = th - t[:, None] * step
    g, _ = grad(th)
    out = []
    for f, (mu, sd, n) in enumerate(stats):
        gnorm = float(g[f].abs().max())
        if not np.isfinite(gnorm) or gnorm > 1e-3 * max(1.0, C * n):
            raise RuntimeError(f"logistic fit did not converge (max |grad| {gnorm:.2e})")
        out.append((th[f, :d].float(), float(th[f, d]), mu.float(), sd.float()))
    return out


def lr_in_z(z: torch.Tensor, y: torch.Tensor, C: float) -> tuple[torch.Tensor, float]:
    """Logistic regression on already-standardised rows z, returned as (w, b) acting on z."""

    w0, b0, mu0, sd0 = fit_logistic(z, y, C)
    return (w0 / sd0).float(), float(b0 - ((mu0 / sd0) @ w0))


def lr_in_z_many(zs: list[torch.Tensor], ys: list[torch.Tensor], C: float):
    """lr_in_z for several problems via fit_logistic_many (refit_batched_logistic)."""

    return [((w0 / sd0).float(), float(b0 - ((mu0 / sd0) @ w0)))
            for w0, b0, mu0, sd0 in fit_logistic_many(zs, ys, C)]


# -------------------------------------------------------------------------- metrics (private copy)
def normal_quantile(q: float) -> float:
    """Standard-normal quantile in fp64 (z for the 1-fpr calibration scale)."""

    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(q, dtype=torch.float64)))


def quantile_threshold(neg_logits: torch.Tensor, fpr: float) -> float:
    """The (1 - fpr) quantile of negative scores: the fixed-FPR threshold (detected = > thr)."""

    return float(torch.quantile(neg_logits.double(), 1.0 - fpr))


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUROC with ties counted as half (scipy-free)."""

    p = pos.detach().double().flatten().cpu().numpy()
    n = neg.detach().double().flatten().cpu().numpy()
    v = np.concatenate([p, n])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def split_holdout(n: int, frac: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A seeded (fit, held) index split with round(n * frac) (>= 1) rows held out."""

    perm = torch.randperm(n, generator=g)
    k = max(1, round(n * frac))
    return perm[k:], perm[:k]


def name_generator(name: str, seed: int | None = None) -> torch.Generator:
    """The fit's generator: seeded from crc32(name) (the snapshot's rule) unless `seed` is given."""

    return torch.Generator().manual_seed(zlib.crc32(name.encode()) if seed is None else seed)


# ----------------------------------------------------------------------------------- refit buffer
@dataclass
class RefitData:
    """Activations collected during rollouts, per layer, for an online refit.

    pos: escape states (passed a brief monitor at turn >= 1). neg: concept-free / other-concept
    states under a brief. np_pos: no-prompt concept states (out-of-context TPR check)."""

    layers: list[int]
    concept: str | None = None
    pos: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    neg: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    np_pos: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for d in (self.pos, self.neg, self.np_pos):
            for L in self.layers:
                d.setdefault(L, [])

    def add(self, kind: str, pooled: dict[int, torch.Tensor], row: int, cap: int = 0) -> None:
        """Append row `row` of every layer to class `kind` ("pos" / "neg" / "np"); cap > 0 keeps
        only the most recent `cap` rows (rolling buffer)."""

        d = {"pos": self.pos, "neg": self.neg, "np": self.np_pos}[kind]
        for L in self.layers:
            d[L].append(pooled[L][row : row + 1].detach().float().cpu())
            if cap and len(d[L]) > cap:
                del d[L][: len(d[L]) - cap]

    def counts(self) -> tuple[int, int, int]:
        L = self.layers[0]
        return len(self.pos[L]), len(self.neg[L]), len(self.np_pos[L])

    def ready(self, min_each: int = 24) -> bool:
        return min(self.counts()) >= min_each


# ------------------------------------------------------------------------------------ entry points
@dataclass
class GateFail:
    """A gated fit that did not clear its gate (or did not converge); `info` says why."""

    kind: str
    info: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return False


def _as_refit_data(pos, neg, np_, layer: int | None, concept: str | None) -> RefitData:
    """pos / neg / np as one tensor (at `layer`) or {layer: tensor | [tensors]} -> RefitData."""

    def per_layer(x):
        if isinstance(x, Mapping):
            return {int(L): (list(v) if isinstance(v, (list, tuple)) else [v])
                    for L, v in x.items()}
        if layer is None:
            raise ValueError("a single-layer tensor needs layer=")
        return {int(layer): [x]}

    P, N = per_layer(pos), per_layer(neg)
    NP = per_layer(np_) if np_ is not None else {L: [] for L in P}
    return RefitData(layers=list(P), concept=concept, pos=P, neg=N, np_pos=NP)


def fit_with_info(
    kind: str,
    pos: Any,
    neg: Any = None,
    np: Any = None,
    gate: GateConfig | None = None,
    seed: int | None = None,
    *,
    name: str,
    layer: int | None = None,
    concept: str | None = None,
    C: float = 0.01,
    perf: FitPerf | None = None,
    device: str | torch.device = "cpu",
    exclude_layers: set[int] | None = None,
    fpr: float = 0.01,
    k: int = 32,
    hidden: int = 64,
    steps: int = 400,
    wd: float = 1e-2,
) -> tuple[Monitor | None, dict]:
    """Fit one probe of `kind` (a kind string, "fam" or "fam@L") -> (monitor or None, info).

    Layers are ABSOLUTE: "fam@L" or `layer` place single-layer kinds; multi-layer kinds read the
    layers that pos / neg are keyed by, in key order. Without `gate` the zoo kinds return an
    uncalibrated monitor (snapshot fit_kind); with it they are fit, CV-scored, calibrated and gated
    (snapshot gate_fit / fit_refit_bank / fit_refit_mlp). "ood" fits on `pos` as reference states
    (`fpr`, `k`, `seed`). The generator is seeded from crc32(name) unless `seed` is given.
    """

    from crypsis.probes import gates
    from crypsis.probes.kinds import linear as K_lin
    from crypsis.probes.kinds import mlp as K_mlp
    from crypsis.probes.kinds import ood as K_ood
    from crypsis.probes.kinds import parse_kind

    spec = parse_kind(kind)
    fam = spec.family
    if spec.layers:  # "fam@L": the kind string places the probe
        layer = spec.layers[0]
    perf = perf or gates.FitPerf()
    if fam == "ood":
        if layer is None:
            raise ValueError("ood needs a layer")
        return K_ood.fit_ood(pos, layer, name=name, k=k, fpr=fpr, seed=0 if seed is None else seed)
    if fam in ("linear", "multilayer"):
        if gate is None:
            x = torch.cat([pos, neg])
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
            w, b, mu, sd = fit_logistic(x, y, C)
            return K_lin.ConceptProbeBank(
                name, int(layer if layer is not None else -1), [concept], W=w[None, :],
                b=torch.tensor([b]), mu=mu[None, :], sd=sd[None, :], thr=torch.zeros(1),
                zero=torch.zeros(1), hundred=torch.ones(1)), {"kind": kind}
        rd = _as_refit_data(pos, neg, np, layer, concept)
        return K_lin.fit_refit_bank(rd, concept, name, gate, C=C, exclude_layers=exclude_layers,
                                    seed=seed)
    if fam == "mlp":
        if gate is None:
            raise ValueError("mlp fits are gated (they calibrate on a held-out split): pass gate=")
        rd = _as_refit_data(pos, neg, np, layer, concept)
        return K_mlp.fit_refit_mlp(rd, concept, name, gate, rd.layers[0], hidden=hidden,
                                   steps=steps, wd=wd, seed=seed)
    if layer is None:
        layer = -1
    if gate is None:
        g = name_generator(name, seed)
        mon = fit_kind(fam, pos, neg, name=name, layer=layer, concept=concept, C=C, g=g,
                       device=device, rlobf_prepad=perf.rlobf_prepad)
        return mon, {"kind": kind}
    return gates.gate_fit(fam, pos, neg, np, gate, name=name, layer=layer, concept=concept,
                          device=device, C=C, perf=perf, seed=seed)


def fit(kind: str, pos: Any, neg: Any = None, np: Any = None, gate: GateConfig | None = None,
        seed: int | None = None, **kw) -> Monitor | GateFail:
    """fit_with_info without the info on success: a Monitor, or GateFail(kind, info)."""

    mon, info = fit_with_info(kind, pos, neg, np, gate, seed, **kw)
    return GateFail(kind, info) if mon is None else mon


def fit_many(kind: str, items: Sequence[dict], gate: GateConfig, *, C: float = 0.01,
             perf: FitPerf | None = None, device: str | torch.device = "cpu"
             ) -> list[tuple[Monitor | None, dict]]:
    """Several gated refits of ONE zoo kind (items = [dict(P, N, NP, name, layer, concept)]):
    snapshot gate_fit_many (stacked rlobf fits with perf.batched_rlobf; else gate_fit each)."""

    from crypsis.probes import gates

    return gates.gate_fit_many(kind, list(items), gate, C=C, perf=perf, device=device)


def fit_kind(kind: str, P, N, *, name: str, layer: int, concept: str | None, C: float,
             g: torch.Generator, device: str | torch.device = "cpu", rlobf_prepad: bool = False,
             layers: Sequence[int] | None = None):
    """Fit one zoo probe of family `kind` on positives P and negatives N, no threshold yet
    (snapshot probe_zoo.fit_kind). Multi-layer families read `layers` (default: P's keys)."""

    from crypsis.probes.kinds import zoo

    return zoo.fit_kind(kind, P, N, name=name, layer=layer, concept=concept, C=C, g=g,
                        device=device, rlobf_prepad=rlobf_prepad, layers=layers)


# ------------------------------------------------------------------------------- the trainer's job
def fit_job(job: dict):
    """The fit stage of a refit on a reads job (snapshot trainers.supervised._refit_fit).

    `job` is what the trainer's reads stage produces (+ "name", "excl"): kind, L, Ls, L0, picks,
    r / rl / ft / npr / npl / nt (or r_pol / r_base / r_cal / r_mis / evdet_buf for detectors),
    acfg (any object with the refit_* fields), device, full_kind, sec_read. Reads no trainer state,
    so executors may run it on a worker. Returns (probe or None, info)."""

    from crypsis.probes import gates
    from crypsis.probes.kinds import evdet as K_ev
    from crypsis.probes.kinds import linear as K_lin
    from crypsis.probes.kinds.zoo import ENS_KINDS, TOKENS_ENS

    acfg = job["acfg"]
    gate = gates.GateConfig.from_refit(acfg)
    if job["kind"] in K_ev.KINDS:
        C = getattr(acfg, "refit_C", 0.1) or 0.1
        mon, info = K_ev.fit_evdet(job["evdet_buf"], job["L"], job["concept"], job["r_pol"],
                                   job["r_base"], job["r_cal"], gate, C=C, name=job["name"],
                                   kind=job["kind"], mis_rows=job["r_mis"])
        return mon, {**info, "kind": job["full_kind"], "sec_read": job["sec_read"]}
    kind, L, Ls, L0, picks = job["kind"], job["L"], job["Ls"], job["L0"], job["picks"]
    rl, ft, npl, nt = job["rl"], job["ft"], job["npl"], job["nt"]
    name, concept = job["name"], job["concept"]
    full_kind, sec_read = job["full_kind"], job["sec_read"]
    # refit_np_in_fit False: no-prompt texts are not fit positives; their gate half is still scored
    np_fit = slice(0, None, 2) if acfg.refit_np_in_fit else slice(0, 0)
    if kind != "linear":
        if kind in TOKENS_ENS:
            pos_q = [q for q, (k_, _) in enumerate(picks) if k_ == "pos"]
            neg_q = [q for q, (k_, _) in enumerate(picks) if k_ == "neg"]
            P = {Lk: [ft[Lk][q] for q in pos_q] + nt[Lk][np_fit] for Lk in Ls}
            N = {Lk: [ft[Lk][q] for q in neg_q] for Lk in Ls}
            G = {Lk: nt[Lk][1::2] for Lk in Ls}
        elif kind in ENS_KINDS:
            P = {Lk: torch.cat([rl[Lk][q : q + 1] for q, (k_, _) in enumerate(picks)
                                if k_ == "pos"] + [npl[Lk][np_fit]]) for Lk in Ls}
            N = {Lk: torch.stack([rl[Lk][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
                 for Lk in Ls}
            G = {Lk: npl[Lk][1::2] for Lk in Ls}
        elif ft is None:  # pooled single-layer zoo kinds (meandiff): rows, not tokens
            P = torch.cat([rl[L0][q : q + 1] for q, (k_, _) in enumerate(picks)
                           if k_ == "pos"] + [npl[L0][np_fit]])
            N = torch.stack([rl[L0][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
            G = npl[L0][1::2]
        else:
            P, N, G = tok_fit_inputs(job)
        mon, info = gates.gate_fit(kind, P, N, G, gate, name=name, layer=L, concept=concept,
                                   device=job["device"], C=acfg.refit_C,
                                   perf=gates.FitPerf.from_refit(acfg))
        return mon, {**info, "kind": full_kind, "sec_read": sec_read}
    rd = RefitData(layers=[L], concept=concept)
    for (k, _), v in zip(picks, job["r"], strict=True):
        (rd.pos if k == "pos" else rd.neg)[L].append(v[None])
    for j, v in enumerate(job["npr"]):
        if j % 2 == 1:
            rd.np_pos[L].append(v[None])
        elif acfg.refit_np_in_fit:
            rd.pos[L].append(v[None])
    bank, info = K_lin.fit_refit_bank(rd, concept, name, gate, C=acfg.refit_C,
                                      exclude_layers=job["excl"])
    return bank, {**info, "kind": full_kind, "sec_read": sec_read}


def tok_fit_inputs(job: dict):
    """(P, N, G) of a single-layer token-kind refit job (fit_job's zoo inputs)."""

    L, picks, ft, nt, acfg = job["L"], job["picks"], job["ft"], job["nt"], job["acfg"]
    np_fit = slice(0, None, 2) if acfg.refit_np_in_fit else slice(0, 0)
    P = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "pos"] + nt[L][np_fit]
    N = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"]
    return P, N, nt[L][1::2]
