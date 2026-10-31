"""Anomaly monitor (kind "ood"): Mahalanobis in a top-k PCA basis + scaled residual, fit on
reference states. Ported from next_chameleons.tica.adaptive (OODMonitor, fit_ood), unchanged."""

from __future__ import annotations

import torch

from crypsis.probes.fit import quantile_threshold, split_holdout


class OODMonitor:
    """Anomaly monitor (Mahalanobis in a top-k PCA basis + scaled residual) fit on REFERENCE states
    (the base model's activations on brief-conditioned turns, concept and concept-free). Fires if a
    state is unlike any normal state -- aimed at dodges that MOVE activations off-distribution. Not
    concept-selective by construction: report its rate on concept-free states (FPR) alongside."""

    def __init__(self, name, layer, mu, P, ev, res_scale, thr):
        self.name, self.layer = name, layer
        self.mu, self.P, self.ev, self.res_scale, self.thr = mu, P, ev, res_scale, thr

    def covers(self, concept) -> bool:
        return True

    def logits(self, x, concepts=None):
        u = x.float().to(self.mu.device) - self.mu
        c = u @ self.P
        res = (u * u).sum(-1) - (c * c).sum(-1)
        return ((c * c) / self.ev).sum(-1) + res.clamp(min=0) / self.res_scale

    def thresholds(self, concepts=None):
        return torch.tensor([self.thr])

    def normalize(self, t, concepts=None):
        return t

    def to(self, device):
        return OODMonitor(self.name, self.layer, self.mu.to(device), self.P.to(device),
                          self.ev.to(device), self.res_scale, self.thr)

    def state(self) -> dict:
        return {"kind": "zoo", "cls": "OODMonitor", "monitor": self.to("cpu")}


def fit_ood(ref: torch.Tensor, layer: int, name: str = "ood", k: int = 32, fpr: float = 0.01,
            seed: int = 0) -> tuple[OODMonitor, dict]:
    """ref: [n, d] reference states. 70% fit (mean, top-k PCs, residual scale), threshold at the
    (1-fpr) quantile of the 30% held-out reference scores."""

    g = torch.Generator().manual_seed(seed)
    fi, hi = split_holdout(len(ref), 0.3, g)
    x = ref[fi].float()
    mu = x.mean(0)
    u = x - mu
    k = min(k, len(x) - 1)
    _, S, Vh = torch.linalg.svd(u, full_matrices=False)
    P = Vh[:k].T
    ev = (S[:k] ** 2 / max(1, len(x) - 1)).clamp(min=1e-6)
    c = u @ P
    res = ((u * u).sum(-1) - (c * c).sum(-1)).clamp(min=0)
    res_scale = float(res.mean().clamp(min=1e-6)) / max(1, k)
    m = OODMonitor(name, layer, mu, P, ev, res_scale, 0.0)
    h = m.logits(ref[hi].float())
    m.thr = quantile_threshold(h, fpr)
    return m, {"n_ref": len(ref), "k": k, "thr": m.thr}
