"""One-hidden-layer MLP concept probe on mean-pooled activations (kind "mlp"), gated like the
linear refit. Ported from next_chameleons.tica.adaptive (MLPMonitor, fit_refit_mlp), unchanged."""

from __future__ import annotations

import torch

from crypsis.probes.fit import RefitData, auroc, name_generator, quantile_threshold, split_holdout


class MLPMonitor:
    """One-hidden-layer MLP concept probe (nonlinear; Bailey et al. 2024 / Gupta & Jenner 2025 find
    nonlinear probes harder to evade than linear ones)."""

    def __init__(self, name, layer, concept, mu, sd, W1, b1, w2, b2, thr):
        self.name, self.layer, self.concept = name, layer, concept
        self.mu, self.sd, self.W1, self.b1, self.w2, self.b2 = mu, sd, W1, b1, w2, b2
        self.thr = thr

    def covers(self, concept) -> bool:
        return concept == self.concept

    def logits(self, x, concepts=None):
        z = (x.float().to(self.mu.device) - self.mu) / self.sd
        return torch.relu(z @ self.W1 + self.b1) @ self.w2 + self.b2

    def thresholds(self, concepts=None):
        return torch.tensor([self.thr])

    def normalize(self, t, concepts=None):
        return t

    def to(self, device):
        f = lambda v: v.to(device)  # noqa: E731
        return MLPMonitor(self.name, self.layer, self.concept, f(self.mu), f(self.sd), f(self.W1),
                          f(self.b1), f(self.w2), f(self.b2), self.thr)

    def state(self) -> dict:
        return {"kind": "zoo", "cls": "MLPMonitor", "monitor": self.to("cpu")}


def fit_refit_mlp(
    data: RefitData, concept: str, name: str, gate, layer: int,
    hidden: int = 64, steps: int = 400, wd: float = 1e-2, seed: int | None = None,
) -> tuple[MLPMonitor | None, dict]:
    """MLP analogue of fit_refit_bank at one layer: same 70/30 held-out split, same gates (held-out
    AUROC >= gate.min_auroc, out-of-context TPR >= gate.min_tpr_np), threshold at gate.fpr on
    held-out negatives. Class-balanced BCE, full-batch AdamW, fixed seed from the name."""

    g = name_generator(name, seed)
    pos_all, neg_all = torch.cat(data.pos[layer]).float(), torch.cat(data.neg[layer]).float()
    npz = torch.cat(data.np_pos[layer]).float()
    fp, hp = split_holdout(len(pos_all), 0.3, g)
    fn, hn = split_holdout(len(neg_all), 0.3, g)
    pos, neg, hpos, hneg = pos_all[fp], neg_all[fn], pos_all[hp], neg_all[hn]
    x = torch.cat([pos, neg])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    mu, sd = x.mean(0), x.std(0, unbiased=False) + 1e-6
    z = (x - mu) / sd
    d = z.shape[1]
    W1 = (torch.randn(d, hidden, generator=g) / d**0.5).requires_grad_()
    b1 = torch.zeros(hidden, requires_grad=True)
    w2 = (torch.randn(hidden, generator=g) / hidden**0.5).requires_grad_()
    b2 = torch.zeros((), requires_grad=True)
    pw = len(neg) / max(1, len(pos))
    opt = torch.optim.AdamW([W1, b1, w2, b2], lr=1e-3, weight_decay=wd)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            lo = torch.relu(z @ W1 + b1) @ w2 + b2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                lo, y, pos_weight=torch.tensor(pw))
            loss.backward()
            opt.step()
    m = MLPMonitor(name, layer, concept, mu, sd, W1.detach(), b1.detach(), w2.detach(),
                   b2.detach(), 0.0)
    with torch.no_grad():
        sp, sn, snp = m.logits(hpos), m.logits(hneg), m.logits(npz)
    m.thr = quantile_threshold(sn, gate.fpr)
    au = auroc(sp, sn)
    tpr_np = float((snp > m.thr).float().mean())
    info = {"layers": {layer: {"auroc": au, "tpr_np": tpr_np}}, "auroc": au, "tpr_np": tpr_np}
    ok = au >= gate.min_auroc and tpr_np >= gate.min_tpr_np
    info["kept"] = ok
    return (m if ok else None), info
