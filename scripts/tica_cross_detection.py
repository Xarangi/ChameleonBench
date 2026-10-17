"""Cross-detection check of a concept split (REPLICATION 2026-10-14h): are the topics DISJOINT
for a probe? For every concept A in the split, a base-model linear probe at L12 (answer format,
with the monitor brief naming A, as in training) is fit on A's FIT texts vs CONCEPT-FREE texts only
(other concepts are deliberately NOT negatives, so overlap is not trained away), thresholded at 1%
FPR on held-out concept-free texts. cross[A][B] = fraction of concept B's texts on which A's probe
fires. Disjoint topics: cross ~ the 1% FPR; overlapping topics: high cross-fire. Also an
A-vs-B AUROC of A's probe (A texts vs B texts). Aggregates only.

  python scripts/tica_cross_detection.py --config l5dodge_math_4b --split SPLIT.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import prompts
from next_chameleons.tica.config import load_run_config


def fit_lr(Xp, Xn, l2=1e-2, steps=200):
    """L2 logistic regression on standardised features (LBFGS); returns score(x) -> logit."""

    import torch

    X = torch.cat([Xp, Xn])
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(X.device)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros((), device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    pw = len(Xn) / max(1, len(Xp))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Z @ w + b, y, pos_weight=torch.tensor(pw, device=X.device)) + l2 * (w @ w) / 2
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda x: ((x - mu) / sd) @ w + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=64, help="texts per concept")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from next_chameleons.tica.monitors import auroc

    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(None, False)
    opening, _ = prompts(cfg.episodes.format, cfg.episodes.inputs, cfg.episodes.layout)
    sp = json.loads(Path(a.split).read_text())
    roles = {c: s for s in ("train", "dev", "test") for c in sp[s]}
    names = list(roles)
    fit = {c.concept: c.texts for c in data.concepts("fit", ("train", "heldout"), only=set(names))}
    rng = random.Random(0)
    free = rng.sample(data.negatives("fit"), min(len(data.negatives("fit")), 3 * a.n))
    t0 = time.time()

    def reads(texts):
        convs = [[{"role": "user", "content": opening("nobrief", x, None)},
                  {"role": "assistant", "content": x}] for x in texts]
        outs = []
        for i in range(0, len(convs), 32):
            outs.append(pol.pooled(convs[i : i + 32], [a.layer])[a.layer].float())
        return torch.cat(outs)

    texts = {c: rng.sample(fit[c], min(a.n, len(fit[c]))) for c in names}
    cross, aucs, quality = {}, {}, {}
    with pol.reference(), torch.no_grad():
        X = {c: reads(texts[c]) for c in names}
        XF = reads(free)
        nf = len(XF) // 3
        for A in names:
            k = len(X[A]) * 3 // 4
            f = fit_lr(X[A][:k], XF[nf:])
            thr = torch.quantile(f(XF[:nf]), 0.99)
            sA = f(X[A][k:]).cpu()
            quality[A] = {"tpr_heldout": float((sA > thr.cpu()).float().mean())}
            cross[A], aucs[A] = {}, {}
            for B in names:
                if B != A:
                    sB = f(X[B]).cpu()
                    cross[A][B] = float((sB > thr.cpu()).float().mean())
                    aucs[A][B] = float(auroc(sA, sB))
            print(A[:50], quality[A], "max cross", round(max(cross[A].values()), 3), flush=True)
    pairs = sorted(((max(cross[x][y], cross[y][x]), x, y) for i, x in enumerate(names)
                    for y in names[i + 1 :]), reverse=True)
    rep = {"layer": a.layer, "n": a.n, "sec": round(time.time() - t0), "roles": roles,
           "quality": quality, "cross": cross, "auroc_A_vs_B": aucs,
           "top_pairs": [{"cross_max": c, "a": x, "b": y, "roles": [roles[x], roles[y]]}
                         for c, x, y in pairs[:40]]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
