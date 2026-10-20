"""Monitors for TICA: activation probes that score pooled activations (DESIGN-TICA.md).

Everything is torch, so one Monitor object serves both uses:
- eval / feedback (no grad);
- training (gradients flow through the frozen monitor into the policy, NC-style).

Two kinds share the `Monitor` interface:
- ConceptProbeBank: one standardized linear probe per concept, at one layer. Scoring uses the row
  of each example's concept.
- MetaDetector: one concept-agnostic linear probe. It detects the ACT OF EVADING (Level 2), not a
  concept; its `concepts` argument is ignored.

Calibration lives inside each monitor, on the raw-logit scale:
- thr: the raw-logit threshold (1% FPR on context-matched negatives);
- zero / hundred: the raw logits mapped to 0 and 100 on the score shown to the model;
- scale(): the spread of the negatives, (thr - zero) / z_{1-fpr}, i.e. the negative-score SD under a
  normal approximation. Training losses use (raw - thr) / scale, so the sigmoid is not saturated:
  gated probes put positives 12-16 raw logits above thr (smoke banks), where sigmoid' ~ 1e-6, but
  only ~3-4 scale units above it.

A MonitorPool is the growing set of monitors in the outer loop (DESIGN-TICA "Outer loop"):
- monitors are only ADDED, so old ones stay and forgetting is prevented;
- each has metadata (round, family, layer);
- pools save to a directory as manifest.json plus one .pt per monitor.

Fitting (fit_logistic) is a deterministic, sklearn-free L2 logistic regression with the same
objective as sklearn's lbfgs solver with class_weight="balanced":
  C * sum_i w_i * logloss_i + 0.5 * ||w||^2.
The transformers-v5 env, which runs Qwen3.5 / Gemma-4, has no sklearn.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Monitor(Protocol):
    name: str
    layer: int

    def logits(self, pooled: torch.Tensor, concepts: Sequence[str | None]) -> torch.Tensor:
        """pooled [n, d] (activations at self.layer) -> raw logits [n]."""
        ...

    def thresholds(self, concepts: Sequence[str | None]) -> torch.Tensor: ...

    def normalize(self, logits: torch.Tensor, concepts: Sequence[str | None]) -> torch.Tensor: ...

    def covers(self, concept: str | None) -> bool: ...

    def scale(self, concepts: Sequence[str | None]) -> torch.Tensor: ...


Z_1PCT = 2.3263  # standard-normal 99th percentile: thr - zero = Z_1PCT * sd for 1%-FPR monitors


def _scale(thr: torch.Tensor, zero: torch.Tensor, z: float) -> torch.Tensor:
    return ((thr - zero) / z).clamp(min=1e-3)


def _rows(index: dict[str, int], concepts: Sequence[str | None]) -> list[int]:
    missing = [c for c in concepts if c not in index]
    if missing:
        raise KeyError(f"monitor has no probe for {missing[:3]}")
    return [index[c] for c in concepts]


@dataclass
class ConceptProbeBank:
    name: str
    layer: int
    concepts: list[str]
    W: torch.Tensor  # [k, d]
    b: torch.Tensor  # [k]
    mu: torch.Tensor  # [k, d]
    sd: torch.Tensor  # [k, d]
    thr: torch.Tensor  # [k] raw-logit thresholds
    zero: torch.Tensor  # [k]
    hundred: torch.Tensor  # [k]
    gate: torch.Tensor | None = None  # [k] bool: passed the quality gate
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {c: i for i, c in enumerate(self.concepts)}

    def covers(self, concept: str | None) -> bool:
        """A probe that failed the quality gate is not a monitor, so the bank does not cover it."""

        if concept not in self._index:
            return False
        return self.gate is None or bool(self.gate[self._index[concept]])

    def gated(self) -> list[str]:
        if self.gate is None:
            return list(self.concepts)
        return [c for c, g in zip(self.concepts, self.gate.tolist(), strict=True) if g]

    def logits(self, pooled, concepts):
        r = _rows(self._index, concepts)
        x = (pooled.float() - self.mu[r]) / self.sd[r]
        return (x * self.W[r]).sum(-1) + self.b[r]

    def thresholds(self, concepts):
        return self.thr[_rows(self._index, concepts)]

    def normalize(self, logits, concepts):
        r = _rows(self._index, concepts)
        return 100.0 * (logits - self.zero[r]) / (self.hundred[r] - self.zero[r])

    def scale(self, concepts):
        r = _rows(self._index, concepts)
        return _scale(self.thr[r], self.zero[r], self.meta.get("z", Z_1PCT))

    def to(self, device) -> ConceptProbeBank:
        kw = {
            k: getattr(self, k).to(device) for k in ("W", "b", "mu", "sd", "thr", "zero", "hundred")
        }
        gate = self.gate.to(device) if self.gate is not None else None
        return ConceptProbeBank(
            self.name, self.layer, self.concepts, gate=gate, meta=self.meta, **kw
        )

    @classmethod
    def from_level1_npz(cls, path: Path | str, name: str | None = None) -> ConceptProbeBank:
        """A bank written by scripts/l1_fit.py."""

        z = np.load(path)
        t = lambda k: torch.tensor(z[k], dtype=torch.float32)  # noqa: E731
        return cls(
            name=name or f"concept_L{int(z['layer'])}",
            layer=int(z["layer"]),
            concepts=[str(c) for c in z["concepts"].tolist()],
            W=t("W"),
            b=t("b"),
            mu=t("mu"),
            sd=t("sd"),
            thr=t("thr1"),
            zero=t("zero"),
            hundred=t("hundred"),
            gate=torch.tensor(z["gate"].astype(bool)),
            meta={"source": str(path), "C": float(z["C"])},
        )

    def state(self) -> dict:
        return {
            "kind": "concept",
            "name": self.name,
            "layer": self.layer,
            "concepts": self.concepts,
            "tensors": {
                k: getattr(self, k).cpu() for k in ("W", "b", "mu", "sd", "thr", "zero", "hundred")
            },
            "gate": None if self.gate is None else self.gate.cpu(),
            "meta": self.meta,
        }


@dataclass
class MetaDetector:
    """A concept-agnostic linear detector (e.g. of evasion). Ignores `concepts`."""

    name: str
    layer: int
    w: torch.Tensor  # [d]
    b: float
    mu: torch.Tensor  # [d]
    sd: torch.Tensor  # [d]
    thr: float
    zero: float
    hundred: float
    meta: dict = field(default_factory=dict)

    def covers(self, concept: str | None) -> bool:
        return True

    def logits(self, pooled, concepts):
        return ((pooled.float() - self.mu) / self.sd) @ self.w + self.b

    def thresholds(self, concepts):
        return torch.full((len(concepts),), self.thr, device=self.w.device)

    def normalize(self, logits, concepts):
        return 100.0 * (logits - self.zero) / (self.hundred - self.zero)

    def scale(self, concepts):
        s = max((self.thr - self.zero) / self.meta.get("z", Z_1PCT), 1e-3)
        return torch.full((len(concepts),), s, device=self.w.device)

    def to(self, device) -> MetaDetector:
        return MetaDetector(
            self.name,
            self.layer,
            self.w.to(device),
            self.b,
            self.mu.to(device),
            self.sd.to(device),
            self.thr,
            self.zero,
            self.hundred,
            self.meta,
        )

    def state(self) -> dict:
        return {
            "kind": "meta",
            "name": self.name,
            "layer": self.layer,
            "tensors": {"w": self.w.cpu(), "mu": self.mu.cpu(), "sd": self.sd.cpu()},
            "scalars": {"b": self.b, "thr": self.thr, "zero": self.zero, "hundred": self.hundred},
            "meta": self.meta,
        }


def monitor_from_state(st: dict):
    if st["kind"] == "concept":
        return ConceptProbeBank(
            st["name"],
            st["layer"],
            st["concepts"],
            gate=st["gate"],
            meta=st["meta"],
            **st["tensors"],
        )
    if st["kind"] == "meta":
        return MetaDetector(
            st["name"], st["layer"], meta=st["meta"], **st["tensors"], **st["scalars"]
        )
    raise ValueError(f"unknown monitor kind {st['kind']!r}")


# --------------------------------------------------------------------------------------- fitting
def fit_logistic(
    x: torch.Tensor, y: torch.Tensor, C: float, *, max_iter: int = 500, tol: float = 1e-7
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Standardized, class-balanced L2 logistic regression (sklearn lbfgs objective).

    Returns (w, b, mu, sd) with logit(x) = ((x - mu) / sd) @ w + b. Deterministic (full-batch
    LBFGS from zero init on a convex objective). Raises if it does not converge.
    """

    # Everything on x's device: the branch refit passes CUDA rows with CPU labels (5129520-22,
    # 5150872-74 both crashed here; CPU-only tests cannot catch a device mismatch).
    x = x.double()
    y = y.double().to(x.device)
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

    # enable_grad: callers may sit inside no_grad (the frozen race did, and every refit silently
    # failed with "does not require grad" -- REPLICATION 2026-10-12g).
    with torch.enable_grad():
        opt.step(closure)
        closure()
    gnorm = torch.cat([w.grad.flatten(), b.grad.flatten()]).abs().max().item()
    if not np.isfinite(gnorm) or gnorm > 1e-3 * max(1.0, C * n):
        raise RuntimeError(f"logistic fit did not converge (max |grad| {gnorm:.2e})")
    return w.detach().float(), float(b.detach()), mu.float(), sd.float()


def normal_quantile(q: float) -> float:
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(q, dtype=torch.float64)))


def quantile_threshold(neg_logits: torch.Tensor, fpr: float) -> float:
    return float(torch.quantile(neg_logits.double(), 1.0 - fpr))


def fit_meta_detector(
    name: str,
    layer: int,
    pos: torch.Tensor,
    neg: torch.Tensor,
    calib_pos: torch.Tensor,
    calib_neg: torch.Tensor,
    *,
    C: float = 1.0,
    fpr: float = 0.01,
    meta: dict | None = None,
) -> tuple[MetaDetector, dict]:
    """Fit a concept-agnostic detector. pos/neg [n, d] for fitting; calib_* for the threshold
    (1-fpr quantile of calib_neg), the 0/100 points, and quality (AUROC, TPR at the threshold)."""

    x = torch.cat([pos, neg]).float()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    w, b, mu, sd = fit_logistic(x, y, C)
    det = MetaDetector(name, layer, w, b, mu, sd, 0.0, 0.0, 1.0, dict(meta or {}))
    sp = det.logits(calib_pos, [None] * len(calib_pos))
    sn = det.logits(calib_neg, [None] * len(calib_neg))
    det.thr = quantile_threshold(sn, fpr)
    det.zero, det.hundred = float(sn.mean()), float(sp.mean())
    det.meta["z"] = normal_quantile(1.0 - fpr)
    q = {"auroc": auroc(sp, sn), "tpr": float((sp > det.thr).float().mean()), "n_fit": len(x)}
    det.meta["quality"] = q
    return det, q


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUROC with ties counted as half."""

    p = pos.detach().double().flatten().cpu().numpy()
    n = neg.detach().double().flatten().cpu().numpy()
    v = np.concatenate([p, n])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]  # average ranks for ties (scipy-free; the v5 env has no scipy)
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


# ------------------------------------------------------------------------------------------ pool
@dataclass
class PoolEntry:
    monitor: object  # ConceptProbeBank | MetaDetector
    round: int
    family: str  # e.g. "concept_linear", "meta_linear"
    held_out: bool = False  # True: evaluation only, never sampled for training feedback


# pool families that are never an episode monitor, selectivity readout or sampled monitor
AUX_FAMILIES = ("panel", "turn0_panel")


class MonitorPool:
    """The growing monitor set of the outer loop. Monitors are only ever ADDED."""

    def __init__(self, entries: list[PoolEntry] | None = None) -> None:
        self.entries: list[PoolEntry] = list(entries or [])

    def add(self, monitor, *, round: int, family: str, held_out: bool = False) -> None:
        if any(e.monitor.name == monitor.name for e in self.entries):
            raise ValueError(f"monitor {monitor.name!r} already in the pool")
        self.entries.append(PoolEntry(monitor, round, family, held_out))

    def get(self, name: str):
        for e in self.entries:
            if e.monitor.name == name:
                return e.monitor
        raise KeyError(name)

    def trainable(self, concept: str | None = None) -> list[PoolEntry]:
        # family "panel" (objective.panel_weight): fixed base probes used ONLY as extra hide
        # targets, never an episode's monitor, a selectivity readout or a held-out readout;
        # "turn0_panel" (adaptive.first_probe_panel_from_step): base probes that only ever judge
        # turn 0 when drawn
        return [e for e in self.entries if not e.held_out and e.family not in AUX_FAMILIES
                and e.monitor.covers(concept)]

    def sample(self, rng: random.Random, concept: str | None) -> PoolEntry:
        """Uniform over trainable monitors that cover `concept`."""

        cands = self.trainable(concept)
        if not cands:
            raise LookupError(f"no trainable monitor covers {concept!r}")
        return rng.choice(cands)

    def concept_monitor(self, concept: str) -> ConceptProbeBank:
        """The earliest trainable ConceptProbeBank covering `concept` (selectivity readouts)."""

        for e in self.entries:
            if (
                isinstance(e.monitor, ConceptProbeBank)
                and not e.held_out
                and e.family not in AUX_FAMILIES
                and e.monitor.covers(concept)
            ):
                return e.monitor
        raise LookupError(f"no concept bank covers {concept!r}")

    def concept_monitors(self, concept: str) -> list[ConceptProbeBank]:
        """Every trainable ConceptProbeBank covering `concept`, earliest first."""

        return [
            e.monitor
            for e in self.entries
            if isinstance(e.monitor, ConceptProbeBank)
            and not e.held_out
            and e.family not in AUX_FAMILIES
            and e.monitor.covers(concept)
        ]

    def panel(self, concept: str) -> list[str]:
        """Names of the depth-panel probes (family "panel") covering `concept`."""

        return [e.monitor.name for e in self.entries
                if e.family == "panel" and e.monitor.covers(concept)]

    @property
    def layers(self) -> list[int]:
        """Every layer a monitor reads (multi-layer monitors: all of their layers_used)."""
        out = set()
        for e in self.entries:
            out.add(e.monitor.layer)
            out.update(getattr(e.monitor, "layers_used", None) or [])
        return sorted(out)

    @property
    def token_layers(self) -> list[int]:
        """Layers whose PER-TOKEN states some monitor needs (token-pooled probes)."""
        out = set()
        for e in self.entries:
            if getattr(e.monitor, "needs_tokens", False):
                out.add(e.monitor.layer)
                out.update(getattr(e.monitor, "layers_used", None) or [])
        return sorted(out)

    def to(self, device) -> MonitorPool:
        return MonitorPool(
            [PoolEntry(e.monitor.to(device), e.round, e.family, e.held_out) for e in self.entries]
        )

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = []
        for e in self.entries:
            torch.save(e.monitor.state(), d / f"{e.monitor.name}.pt")
            manifest.append(
                {
                    "name": e.monitor.name,
                    "round": e.round,
                    "family": e.family,
                    "held_out": e.held_out,
                }
            )
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))

    @classmethod
    def load(cls, directory: Path | str) -> MonitorPool:
        d = Path(directory)
        out = cls()
        for m in json.loads((d / "manifest.json").read_text()):
            mon = monitor_from_state(torch.load(d / f"{m['name']}.pt", weights_only=False))
            out.entries.append(PoolEntry(mon, m["round"], m["family"], m["held_out"]))
        return out
