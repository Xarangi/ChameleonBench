"""The literature probe bank (descriptions of every kind we reproduce: source, configuration,
fidelity) and the numpy ProbeBank of the offline fit script (scripts/tica_fit_probes.py writes
bank_L{layer}.npz with thr_<fpr> keys; ConceptProbeBank.from_level1_npz reads the l1_fit format).

Ported from next_chameleons.tica.probe_bank (BANK, TOUGHEST, describe) and tica.probes (ProbeBank,
fit_one, score, threshold_at, GATE). fit_one keeps sklearn's lbfgs (the same objective as
crypsis.probes.fit.fit_logistic but a different solver, so not bit-identical: kept separate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from crypsis.probes.fit import auroc

# kind -> (source, configuration, fidelity). fidelity: "exact" = the published spec / official
# code; "close" = published spec with stated deviations; "ours" = our own variant.
BANK: dict[str, tuple[str, str, str]] = {
    "linear": ("NC 2512.11949 / Apollo 2502.03407",
               "logistic regression on the mean-pooled response, one layer", "close"),
    "meandiff": ("Bailey 2412.09565", "difference of class means, mean-pooled", "close"),
    "mlp": ("NC / Bailey", "one hidden layer of 64, ReLU, mean-pooled", "close"),
    "rlobf_attn": ("RL-Obfuscation 2506.14261",
                   "causal 1-head attention, d_proj 512, token logits, mean of token sigmoids; "
                   "token-wise BCE, 40 epochs, 16+16 batches, AdamW 1e-3, clip 1", "exact"),
    "rlobf_attn_max": ("RL-Obfuscation 2506.14261", "as rlobf_attn, max of token sigmoids",
                       "exact"),
    "rlobf_max": ("RL-Obfuscation 2506.14261",
                  "token Linear, true max (k=1 as in the code), BCE on the max, 80 epochs",
                  "exact"),
    "gdm_rollattn": ("GDM 2601.11516",
                     "token MLP d-100-100, 10 heads, window 10, max of rolling attention means; "
                     "full-batch AdamW 1e-4 wd 3e-3 1000 steps; 1 seed (paper: best of 100); "
                     "head sum is our choice (not stated)", "close"),
    "mck_attnlite": ("McKenzie 2506.10805",
                     "1-head softmax(x.q/sqrt d) pooling, value = classifier; AdamW 5e-3, cosine, "
                     "<=200 epochs, patience 50; best-val weights (their code keeps the final)",
                     "close"),
    "mck_softmax": ("McKenzie 2506.10805", "softmax(z/5)-weighted token logits; best-val weights",
                    "close"),
    "mck_rollmean40": ("McKenzie 2506.10805", "max over 40-token rolling means of token logits; "
                       "grad-accum capped at our n (>= 4 steps/epoch), best-val weights", "close"),
    "attnpool": ("NC 2512.11949", "K=4 heads, sum of head-pooled scores; standardised inputs, "
                 "initialised at the linear probe", "close"),
    "maxpool": ("RL-Obfuscation (top-k)", "top-5% token mean trained through the pooling", "ours"),
    "rollmax": ("McKenzie / GDM", "max of 10-token window means, log-sum-exp in training", "ours"),
    "nc_ens8_tok": ("NC 2512.11949 (official code)",
                    "8 layers, per-layer token Linear, sigmoid -> token mean -> layer mean; Adam "
                    "5e-4, 5 epochs; layer list not stated (ours: 4..18 step 2); fork tokens "
                    "subsampled to 32 per answer", "close"),
    "bailey_ens_tok": ("Bailey 2412.09565 (official code)",
                       "ALL layers (ours: 30 = all but held-out 20/24), per-layer token Linear, "
                       "sigmoid -> token mean -> layer mean; Adam 1e-3, 1 epoch; fork tokens "
                       "subsampled to 32 per answer", "close"),
    "ens4_linear": ("NC (4-layer ensemble)", "mean of per-layer sigmoids, mean-pooled", "ours"),
    "ens8_linear": ("NC (8-layer ensemble)", "mean of per-layer sigmoids, mean-pooled", "ours"),
    "ens8_mlp": ("NC / Bailey", "8-layer MLP ensemble, mean-pooled", "ours"),
    "multilayer": ("ours", "linear at L8/12/16, caught if ANY fires (stricter than any paper)",
                   "ours"),
    "ood": ("Bailey 2412.09565", "Gaussian / Mahalanobis anomaly detector (eval only)", "close"),
}

# The literature's toughest probes (reported adaptive robustness, review 10-14): what "--probe-kind
# bank" draws from, one kind per refit.
TOUGHEST = ("rlobf_attn", "rlobf_attn_max", "rlobf_max", "gdm_rollattn", "mck_attnlite",
            "nc_ens8_tok", "bailey_ens_tok", "attnpool")


# Single-layer subset for per-refit random draws (race --probe-kind bank / training "mixed"): the
# multi-layer token ensembles read per-token states at 8-30 layers and run as their own sweeps.
TOUGHEST_SINGLE_LAYER = tuple(k for k in TOUGHEST if not k.endswith("_tok"))


def describe() -> str:
    return "\n".join(f"{k:16s} [{fid}] {src}: {cfg}" for k, (src, cfg, fid) in BANK.items())


GATE = {"auroc": 0.95, "tpr_at_1pct": 0.80}


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC on numpy scores; nan if a class is empty (snapshot incontext.stats)."""

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return auroc(torch.from_numpy(pos), torch.from_numpy(neg))


def fit_one(
    x_tr: np.ndarray, y_tr: np.ndarray, c: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Returns (w, b, mu, sd) such that score(x) = ((x - mu) / sd) @ w + b (a logit)."""

    from sklearn.linear_model import LogisticRegression

    x_tr = x_tr.astype(np.float32)
    mu = x_tr.mean(0)
    sd = x_tr.std(0) + 1e-6
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, tol=1e-6, solver="lbfgs")
    clf.fit((x_tr - mu) / sd, y_tr)
    if clf.n_iter_[0] >= clf.max_iter:
        raise RuntimeError("logistic probe did not converge")
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), mu, sd


def score(x: np.ndarray, w, b, mu, sd) -> np.ndarray:
    return ((x.astype(np.float32) - mu) / sd) @ w + b


def threshold_at(neg_scores: np.ndarray, fpr: float) -> float:
    return float(np.quantile(neg_scores, 1.0 - fpr))


@dataclass
class ProbeBank:
    concepts: list[str]
    layer: int
    W: np.ndarray  # [n, d]
    b: np.ndarray  # [n]
    mu: np.ndarray  # [n, d]
    sd: np.ndarray  # [n, d]
    thresholds: dict[str, np.ndarray] = field(default_factory=dict)  # fpr label -> [n]
    quality: list[dict] = field(default_factory=list)

    def save(self, path) -> None:
        np.savez(
            path,
            concepts=np.array(self.concepts),
            layer=self.layer,
            W=self.W,
            b=self.b,
            mu=self.mu,
            sd=self.sd,
            **{f"thr_{k}": v for k, v in self.thresholds.items()},
        )

    @classmethod
    def load(cls, path) -> ProbeBank:
        z = np.load(path)
        thr = {k[4:]: z[k] for k in z.files if k.startswith("thr_")}
        return cls(list(z["concepts"]), int(z["layer"]), z["W"], z["b"], z["mu"], z["sd"], thr)

    def scores(self, x: np.ndarray, idx: list[int]) -> np.ndarray:
        """x: [m, d] pooled acts. idx[i] is the concept row for x[i]. Returns logits [m]."""

        W, b, mu, sd = self.W[idx], self.b[idx], self.mu[idx], self.sd[idx]
        return np.einsum("md,md->m", (x.astype(np.float32) - mu) / sd, W) + b
