"""Gate and speed settings for gated fits, plus the zoo CV gate (gate_fit / gate_fit_many).

GateConfig holds what decides whether a refit becomes a monitor (the snapshot's refit_min_auroc,
refit_min_tpr_np, refit_min_tpr_ic, refit_fpr, refit_cv_folds). FitPerf holds the speed-only fit
switches (refit_batched_fits, refit_batched_logistic, refit_batched_rlobf, refit_rlobf_prepad):
they never change a gate decision and change fitted values only by float reassociation.
"""

from __future__ import annotations

from dataclasses import dataclass



from crypsis.probes.fit import quantile_threshold
from crypsis.probes.kinds.zoo import gate_fit, gate_fit_many

__all__ = ["FitPerf", "GateConfig", "calibrate", "gate_fit", "gate_fit_many"]


@dataclass(frozen=True)
class GateConfig:
    """Refit gate: held-out AUROC, out-of-context TPR (no-prompt rows) and in-context TPR (0 =
    off) at the `fpr` threshold; `cv_folds` out-of-fold positives (zoo kinds: max(2, cv_folds or
    5); linear: CV only when cv_folds >= 2)."""

    min_auroc: float = 0.95
    min_tpr_np: float = 0.8
    min_tpr_ic: float = 0.0
    fpr: float = 0.01
    cv_folds: int = 0

    @classmethod
    def from_refit(cls, cfg) -> GateConfig:
        """From any object with the snapshot's refit_* fields (AdaptiveConfig)."""

        return cls(min_auroc=cfg.refit_min_auroc, min_tpr_np=cfg.refit_min_tpr_np,
                   min_tpr_ic=getattr(cfg, "refit_min_tpr_ic", 0.0), fpr=cfg.refit_fpr,
                   cv_folds=getattr(cfg, "refit_cv_folds", 0))


@dataclass(frozen=True)
class FitPerf:
    """Speed-only fit switches (batched CV folds, batched Newton logistic, stacked rlobf, prepad)."""

    batched_fits: bool = False
    batched_logistic: bool = False
    batched_rlobf: bool = False
    rlobf_prepad: bool = False

    @classmethod
    def from_refit(cls, cfg) -> FitPerf:
        return cls(batched_fits=getattr(cfg, "refit_batched_fits", False),
                   batched_logistic=getattr(cfg, "refit_batched_logistic", False),
                   batched_rlobf=getattr(cfg, "refit_batched_rlobf", False),
                   rlobf_prepad=getattr(cfg, "refit_rlobf_prepad", False))


def calibrate(mon, neg_held, fpr: float):
    """Set a zoo monitor's threshold (1 - fpr quantile of held-out negative scores) and scale_
    (their SD, floored at 1e-3), as gate_fit does. Returns the monitor."""

    sn = mon.logits(neg_held).detach().cpu()
    mon.thr = float(quantile_threshold(sn, fpr))
    mon.scale_ = max(float(sn.float().std()), 1e-3)
    return mon


