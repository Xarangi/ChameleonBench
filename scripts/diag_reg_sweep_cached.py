# ruff: noqa
"""Instant (cached-features) test of the last unexplained signature: the released probe
is anomalously DISTRIBUTED (PR=2415 of 3584) while every fit we try is concentrated
(PR~1290). Strong L2 regularization shrinks a logistic solution toward a spread-out
direction -> higher PR. Sweep sklearn C (inverse reg strength) on raw + standardized
features and see if any setting reaches PR~2415 AND high cosine-to-released.

Uses the cached layer-13 german features from the earlier run (no GPU forward needed).
"""
import glob, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from next_chameleons.hf.paper_probes import _released_weight_bias

CONCEPT = os.environ.get("REFIT_CONCEPT", "german")


def released(probe_dir):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[:-len("_weights.pt")].lower() == CONCEPT.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().numpy().reshape(-1)
    raise FileNotFoundError(CONCEPT)


def pr(w):
    w2 = w.astype(np.float64) ** 2
    return float((w2.sum() ** 2) / (w2 ** 2).sum())


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def balacc(scores, y):
    best = 0.0
    for q in np.linspace(0.05, 0.95, 19):
        t = np.quantile(scores, q)
        p = (scores >= t).astype(float)
        best = max(best, 0.5 * (p[y == 1].mean() + (1 - p[y == 0]).mean()))
    return best


def main():
    probe_dir, cache = sys.argv[1], sys.argv[2]
    w_rel = released(probe_dir)
    blob = torch.load(cache, map_location="cpu")
    X = blob["X"].float().numpy()
    y = blob["y"].numpy().astype(int)
    # subsample to keep sklearn lbfgs fast (3584-dim direction needs only a few k rows)
    rng = np.random.default_rng(0)
    n_keep = min(12000, X.shape[0])
    idx = rng.permutation(X.shape[0])[:n_keep]
    X, y = X[idx], y[idx]
    print(f"[reg] cache subsampled to {X.shape} pos_frac={y.mean():.3f}  released PR={pr(w_rel):.0f}", flush=True)
    mu, sd = X.mean(0), X.std(0) + 1e-6

    print(f"\n{'space':>14} {'C':>8} {'cos_to_rel':>11} {'PR':>7} {'balacc':>8}")
    for space in ("raw", "standardized"):
        Xf = (X - mu) / sd if space == "standardized" else X
        for C in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, fit_intercept=False, max_iter=200, n_jobs=-1)
            clf.fit(Xf, y)
            coef = clf.coef_.reshape(-1)
            # fold scaler back to raw activation space for fair cosine vs released
            w_raw = coef / sd if space == "standardized" else coef
            scores = Xf @ coef
            print(f"{space:>14} {C:>8} {cos(w_raw, w_rel):>11.4f} {pr(w_raw):>7.0f} {balacc(scores, y):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
