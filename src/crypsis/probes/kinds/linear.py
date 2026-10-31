"""Linear probes on mean-pooled activations: the per-concept bank, the concept-agnostic detector,
and the gated online refit (kind "linear", also "linear@L" and the multi-layer "multilayer").

Calibration lives in the monitor on the raw-logit scale: thr (1% FPR on matched negatives),
zero / hundred (raw logits shown as 0 and 100), scale() = (thr - zero) / z_{1-fpr}.

Ported from next_chameleons.tica.monitors (ConceptProbeBank, MetaDetector, fit_meta_detector) and
tica.adaptive (fit_refit_bank). `state()` is the snapshot's format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from crypsis.probes.base import Z_1PCT
from crypsis.probes.fit import (
    RefitData,
    auroc,
    fit_logistic,
    name_generator,
    normal_quantile,
    quantile_threshold,
    split_holdout,
)


def _scale(thr: torch.Tensor, zero: torch.Tensor, z: float) -> torch.Tensor:
    return ((thr - zero) / z).clamp(min=1e-3)


def _rows(index: dict[str, int], concepts: Sequence[str | None]) -> list[int]:
    missing = [c for c in concepts if c not in index]
    if missing:
        raise KeyError(f"monitor has no probe for {missing[:3]}")
    return [index[c] for c in concepts]


_BANK_TENSORS = ("W", "b", "mu", "sd", "thr", "zero", "hundred")


@dataclass
class ConceptProbeBank:
    """One standardized linear probe per concept at one layer; scoring uses each row's concept."""

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
        kw = {k: getattr(self, k).to(device) for k in _BANK_TENSORS}
        gate = self.gate.to(device) if self.gate is not None else None
        return ConceptProbeBank(
            self.name, self.layer, self.concepts, gate=gate, meta=self.meta, **kw
        )

    @classmethod
    def from_level1_npz(cls, path: Path | str, name: str | None = None) -> ConceptProbeBank:
        """A bank_L{layer}.npz written by scripts/l1_fit.py (keys W, b, mu, sd, thr1, zero,
        hundred, gate, concepts, layer, C)."""

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
            "tensors": {k: getattr(self, k).cpu() for k in _BANK_TENSORS},
            "gate": None if self.gate is None else self.gate.cpu(),
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, st: dict) -> ConceptProbeBank:
        return cls(st["name"], st["layer"], st["concepts"], gate=st["gate"], meta=st["meta"],
                   **st["tensors"])


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
        return MetaDetector(self.name, self.layer, self.w.to(device), self.b, self.mu.to(device),
                            self.sd.to(device), self.thr, self.zero, self.hundred, self.meta)

    def state(self) -> dict:
        return {
            "kind": "meta",
            "name": self.name,
            "layer": self.layer,
            "tensors": {"w": self.w.cpu(), "mu": self.mu.cpu(), "sd": self.sd.cpu()},
            "scalars": {"b": self.b, "thr": self.thr, "zero": self.zero, "hundred": self.hundred},
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, st: dict) -> MetaDetector:
        return cls(st["name"], st["layer"], meta=st["meta"], **st["tensors"], **st["scalars"])


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
    """Fit a concept-agnostic detector on pos / neg; calib_* set the threshold (1-fpr quantile of
    calib_neg), the 0 / 100 points and the quality (AUROC, TPR at the threshold)."""

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


def fit_refit_bank(
    data: RefitData, concept: str, name: str, gate, *, C: float,
    exclude_layers: set[int] | None = None, seed: int | None = None,
) -> tuple[ConceptProbeBank | None, dict]:
    """Fit one linear probe per buffer layer; keep the best-AUROC layer that clears the gate.

    Gate (`gate`: crypsis.probes.gates.GateConfig): AUROC >= min_auroc on held-out negatives
    (out-of-fold positives with cv_folds >= 2, a 70/30 split otherwise, in-sample below 8 rows),
    out-of-context TPR on the no-prompt rows >= min_tpr_np, and (min_tpr_ic > 0) in-context TPR at
    the fpr threshold. exclude_layers are never fit (held-out eval layers). Returns (bank or None,
    info) exactly as the snapshot's adaptive.fit_refit_bank.
    """

    best = None
    info: dict = {"layers": {}}
    g = name_generator(name, seed)
    for L in data.layers:
        if exclude_layers and L in exclude_layers:
            continue
        pos_all = torch.cat(data.pos[L])
        neg_all = torch.cat(data.neg[L])
        npz = torch.cat(data.np_pos[L])
        insample = min(len(pos_all), len(neg_all)) < 8  # too few to hold out (tests / tiny buffers)
        k = gate.cv_folds
        cv = k >= 2 and not insample and len(pos_all) >= 2 * k
        if insample:
            pos, neg, hpos, hneg = pos_all, neg_all, pos_all, neg_all
        elif cv:  # all positives fit the final probe; they are scored out-of-fold below
            fn, hn = split_holdout(len(neg_all), 0.3, g)
            pos, neg, hpos, hneg = pos_all, neg_all[fn], None, neg_all[hn]
        else:
            fp, hp = split_holdout(len(pos_all), 0.3, g)
            fn, hn = split_holdout(len(neg_all), 0.3, g)
            pos, neg, hpos, hneg = pos_all[fp], neg_all[fn], pos_all[hp], neg_all[hn]
        x = torch.cat([pos, neg])
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        try:
            w, b, mu, sd = fit_logistic(x, y, C)
            if cv:  # out-of-fold AUROC: fold j's model scores its own held-out positives
                perm = torch.randperm(len(pos), generator=g)
                pairs, wins = 0, 0.0
                ic_hit, ic_tot = 0, 0
                for j in range(k):
                    out = perm[j::k]
                    keep = torch.ones(len(pos), dtype=torch.bool)
                    keep[out] = False
                    xj = torch.cat([pos[keep], neg])
                    yj = torch.cat([torch.ones(int(keep.sum())), torch.zeros(len(neg))])
                    wj, bj, muj, sdj = fit_logistic(xj, yj, C)
                    fj = ((pos[out].float() - muj) / sdj) @ wj + bj
                    gj = ((hneg.float() - muj) / sdj) @ wj + bj
                    a_j = auroc(fj, gj)
                    thr_j = quantile_threshold(gj, gate.fpr)
                    ic_hit += int((fj > thr_j).sum())
                    ic_tot += len(fj)
                    wins += a_j * len(out) * len(hneg)
                    pairs += len(out) * len(hneg)
                au_cv = wins / pairs
        except RuntimeError as e:
            info["layers"][L] = {"error": str(e)[:200]}
            continue
        logit = lambda z, w=w, b=b, mu=mu, sd=sd: ((z.float() - mu) / sd) @ w + b  # noqa: E731
        sp = logit(pos) if cv else logit(hpos)  # (cv: in-sample; only the 0-100 display uses it)
        sn, snp = logit(hneg), logit(npz)
        thr = quantile_threshold(sn, gate.fpr)
        au = au_cv if cv else auroc(sp, sn)
        tpr_np = float((snp > thr).float().mean())
        tpr_ic = (ic_hit / ic_tot if ic_tot else float("nan")) if cv else float(
            (sp > thr).float().mean())  # held-out in-context positives (insample: fit rows)
        info["layers"][L] = {"auroc": au, "tpr_np": tpr_np, "tpr_ic": tpr_ic, "n_pos": len(pos_all),
                             "n_neg": len(neg_all), "insample": insample,
                             "cv_folds": k if cv else 0}
        clears = au >= gate.min_auroc and tpr_np >= gate.min_tpr_np
        if gate.min_tpr_ic > 0:
            clears = clears and tpr_ic == tpr_ic and tpr_ic >= gate.min_tpr_ic
        if clears and (best is None or au > info["layers"][best[0]]["auroc"]):
            best = (L, w, b, mu, sd, thr, sp, sn)
    if best is None:
        info["kept"] = False
        return None, info
    L, w, b, mu, sd, thr, sp, sn = best
    bank = ConceptProbeBank(
        name=name,
        layer=L,
        concepts=[concept],
        W=w[None, :],
        b=torch.tensor([b]),
        mu=mu[None, :],
        sd=sd[None, :],
        thr=torch.tensor([thr]),
        zero=torch.tensor([float(sn.mean())]),
        hundred=torch.tensor([float(sp.mean())]),
        gate=torch.ones(1, dtype=torch.bool),
        meta={"z": normal_quantile(1.0 - gate.fpr), "online_refit": True},
    )
    info["kept"] = True
    info["layer"] = L
    info["auroc"] = info["layers"][L]["auroc"]
    info["tpr_np"] = info["layers"][L]["tpr_np"]
    info["tpr_ic"] = info["layers"][L]["tpr_ic"]
    return bank, info
