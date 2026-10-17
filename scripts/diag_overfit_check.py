"""Does our self-fit german probe OVERFIT (in-sample >> held-out)? And is it a
concentrated 'surface shortcut' vs a distributed 'semantic' direction? Operates on the
saved german-pos vs data_Y-neg activations (diag_german_feats.pt). Pure analysis, no model.
"""

import sys

import torch
import torch.nn.functional as F


def fit(Xtr, ytr, lr, ep, wd):
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd, eps=1e-5)
    gg = torch.Generator().manual_seed(17)
    for _ in range(ep):
        order = torch.randperm(Xtr.shape[0], generator=gg)
        for s in range(0, Xtr.shape[0], 16):
            i = order[s : s + 16]
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(Xtr[i] @ w + b, ytr[i]).backward()
            opt.step()
    return w.detach(), b.detach()


def acc(W, B, Xs, ys, thr=0.5):
    p = (torch.sigmoid(Xs @ W + B) >= thr).float()
    pos, neg = ys.bool(), ~ys.bool()
    return 0.5 * (p[pos].mean() + (1 - p[neg]).mean()).item()


def pr(W):
    w2 = W.float() ** 2
    return (w2.sum() ** 2 / (w2 ** 2).sum()).item()


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    X, y, w_rel, b_rel = d["X"], d["y"], d["w_rel"], d["b_rel"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=g)
    ntr = int(0.7 * X.shape[0])
    tr, te = perm[:ntr], perm[ntr:]
    print("=== self-fit german probe IN-SAMPLE vs HELD-OUT (german vs data_Y) ===", flush=True)
    for nm, lr, ep, wd in [
        ("paper-recipe lr4e-5/1ep", 4e-5, 1, 1e-5),
        ("converged lr1e-3/30ep", 1e-3, 30, 1e-5),
        ("strongL2 lr1e-3/30ep/wd1e-1", 1e-3, 30, 1e-1),
    ]:
        w, b = fit(X[tr], y[tr], lr, ep, wd)
        print(f"  {nm:30s} train={acc(w,b,X[tr],y[tr]):.4f}  HELDOUT={acc(w,b,X[te],y[te]):.4f}  PR={pr(w):.1f}", flush=True)
    print(f"\n  RELEASED probe heldout_acc on same test = {acc(w_rel,b_rel,X[te],y[te]):.4f}", flush=True)
    print(f"  participation ratio: released={pr(w_rel):.1f}  (of {w_rel.numel()} dims; higher=more distributed)", flush=True)


if __name__ == "__main__":
    main()
