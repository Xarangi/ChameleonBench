"""Probe zoo (REPLICATION 2026-10-13k/n): probe TYPES from the obfuscation / monitoring literature
beyond our mean-pooled linear probe, refit and gated exactly like it. Configurations follow a
citation-checked review (REPLICATION 2026-10-13n): learned probes are single-layer by default;
multi-layer monitors are per-layer probes whose scores are AVERAGED; max / attention pooling is
trained END-TO-END through the pooling on sequence labels (not token-level LR max-pooled at test).

Single layer, token pooling (items = lists of per-token states [n_tokens, d]):
- maxpool: token logits z_t.w + b, sequence score = mean of the top-k (k = 5% of the tokens,
  >= 1); RL-Obfuscation 2506.14261 (their robust "max" probe).
- rollmax: max over 10-token window means of the token logit; McKenzie 2506.10805 (window 40) /
  GDM 2601.11516 (window 10). Trained through a log-sum-exp surrogate (a hard max trains badly,
  GDM), hard max at inference.
- attnpool: K = 4 heads, score = sum_k sum_t softmax_t(z_t.q_k) (z_t.w_k) + b; NC 2512.11949 (K=4),
  McKenzie, GDM. The heads start by splitting the mean-pooled logistic probe (q ~ 0 = mean
  pooling), so training starts at the linear probe.
Single layer, mean-pooled vectors ([n, d]):
- meandiff: difference of class means (Bailey 2412.09565 baseline).
Multi-layer (items = {layer: [n, d]}), mean of per-layer sigmoid scores (Bailey: all layers; NC:
4- and 8-layer ensembles, mean; Atlas 2602.15515: mean of sigmoids):
- ens4_linear / ens8_linear / ens4_mlp / ens8_mlp at ENS_LAYERS (evenly spaced mid-stack around L12,
  avoiding the held-out eval layers 20 / 24; NC's exact lists are not stated -> approximate).
(single-layer linear = fit_refit_bank and mlp = fit_refit_mlp live in adaptive.py.)

gate_fit applies the refit gate of fit_refit_bank with refit_cv_folds: 30% of negatives held out
(AUROC + the refit_fpr threshold), positives scored out-of-fold by k-fold CV, final probe on all
positives, and the out-of-context TPR bar on the no-prompt items.
"""

from __future__ import annotations

import copy
import math
import zlib

import torch

from next_chameleons.tica.monitors import auroc, fit_logistic, quantile_threshold

TOKEN_KINDS = ("maxpool", "rollmax", "attnpool")
LIT_TOKEN_KINDS = ("rlobf_attn", "rlobf_attn_max", "rlobf_max", "gdm_rollattn",  # see bottom
                   "mck_attnlite", "mck_softmax", "mck_rollmean40")
TOKENS_ENS = ("nc_ens8_tok", "bailey_ens_tok", "bailey_ens_tok_max")  # token-level multi-layer
ALL_TOKEN_KINDS = (*TOKEN_KINDS, *LIT_TOKEN_KINDS, *TOKENS_ENS)
ENS_LAYERS = {"ens4": [8, 10, 12, 14], "ens8": [4, 6, 8, 10, 12, 14, 16, 18],
              # FINAL v2 (10-15e): a DEEP 4-layer ensemble (train kind; avoids the held-out single
              # layers 20 / 24 / 28 / 30) and a depth-SPREAD 5-layer ensemble scored by the MEDIAN
              # of the per-layer probabilities = a majority vote of per-layer linear probes, after
              # the multi-layer "activation delta" detectors of arXiv 2602.00750 (layers
              # 0/7/15/23/31 of 32, majority / all agree; ours 4..28: the ends are near-lexical /
              # next-token).
              "ensdeep4": [16, 18, 22, 26], "spread5": [4, 10, 16, 22, 28]}
ENS_KINDS = ("ens4_linear", "ens8_linear", "ens4_mlp", "ens8_mlp", "ensdeep4_linear",
             "spread5_linear")
ZOO_KINDS = ("meandiff", *TOKEN_KINDS, *ENS_KINDS, *LIT_TOKEN_KINDS, *TOKENS_ENS)
TOPK_FRAC, ROLL_WIN, ATTN_HEADS, STEPS = 0.05, 10, 4, 300


def split_kind(kind: str) -> tuple[str, int | None]:
    """"linear@22" -> ("linear", 22): any single-layer kind placed at an explicit layer (the FINAL
    v2 depth axis); a plain kind -> (kind, None) = the concept's primary layer."""

    if "@" in kind:
        base, L = kind.split("@", 1)
        return base, int(L)
    return kind, None


def kind_layers(kind: str, layer: int) -> list[int]:
    kind, Lo = split_kind(kind)
    if Lo is not None:
        if kind in ENS_KINDS or kind in TOKENS_ENS:
            raise ValueError(f"{kind}@{Lo}: multi-layer kinds have fixed layers")
        return [Lo]
    if kind in TOKENS_ENS:
        return TOKENS_ENS_LAYERS[kind]
    return ENS_LAYERS[kind.split("_")[0]] if kind in ENS_KINDS else [layer]


class _Mon:
    needs_tokens = False
    layers_used: list[int] | None = None
    scale_: float = 1.0  # SD of held-out negative scores (gate_fit); p = sigmoid((s - thr) / scale)

    def scale(self, concepts=None):
        return torch.tensor([self.scale_])

    def _keep(self, new):  # .to() copies keep the calibration scale
        new.scale_ = self.scale_
        return new

    def covers(self, concept) -> bool:
        return concept == self.concept

    def thresholds(self, concepts=None):
        return torch.tensor([self.thr])

    def normalize(self, t, concepts=None):
        return t


class MeanDiffMonitor(_Mon):
    def __init__(self, name, layer, concept, w, thr=0.0):
        self.name, self.layer, self.concept, self.w, self.thr = name, layer, concept, w, thr

    def logits(self, x, concepts=None):
        return x.float().to(self.w.device) @ self.w

    def to(self, device):
        return self._keep(MeanDiffMonitor(self.name, self.layer, self.concept, self.w.to(device),
                                          self.thr))


def _pad(toks, mu, sd, device):
    T = max(len(t) for t in toks)
    X = torch.zeros(len(toks), T, toks[0].shape[1], device=device)
    M = torch.zeros(len(toks), T, dtype=torch.bool, device=device)
    for i, t in enumerate(toks):
        X[i, : len(t)] = (t.float().to(device) - mu) / sd
        M[i, : len(t)] = True
    return X, M


def _pool(kind, s, M, train: bool):
    """Sequence scores from token scores s [B, T] (masked by M)."""

    if kind == "maxpool":  # mean of the top-k token scores, k = TOPK_FRAC of each sequence
        k = torch.clamp((M.sum(1).float() * TOPK_FRAC).ceil().long(), min=1)
        srt = s.masked_fill(~M, float("-inf")).sort(1, descending=True).values
        srt = torch.where(torch.isfinite(srt), srt, torch.zeros_like(srt))
        return srt.cumsum(1).gather(1, (k - 1)[:, None])[:, 0] / k
    if kind == "rollmax":  # max over ROLL_WIN-token window means (whole sequence if shorter)
        w, T = ROLL_WIN, s.shape[1]
        sm = s.masked_fill(~M, 0.0)
        mean_all = sm.sum(1) / M.sum(1).clamp(min=1)
        if w > T:
            return mean_all
        cs = torch.nn.functional.pad(sm.cumsum(1), (1, 0))
        cm = torch.nn.functional.pad(M.float().cumsum(1), (1, 0))
        win = (cs[:, w:] - cs[:, :-w]) / w  # [B, T - w + 1]
        win = win.masked_fill((cm[:, w:] - cm[:, :-w]) < w, float("-inf"))  # inside the sequence
        agg = torch.logsumexp(win, 1) if train else win.max(1).values
        return torch.where(M.sum(1) < w, mean_all, agg)
    raise ValueError(kind)


class TokenPoolMonitor(_Mon):
    """maxpool (top-k mean) / rollmax over per-token linear scores."""

    needs_tokens = True

    def __init__(self, kind, name, layer, concept, mu, sd, w, b, thr=0.0):
        self.kind, self.name, self.layer, self.concept = kind, name, layer, concept
        self.mu, self.sd, self.w, self.b, self.thr = mu, sd, w, b, thr

    def logits(self, toks, concepts=None):
        X, M = _pad(toks, self.mu, self.sd, self.mu.device)
        return _pool(self.kind, X @ self.w + self.b, M, train=False)

    def to(self, device):
        f = lambda v: v.to(device)  # noqa: E731
        return self._keep(TokenPoolMonitor(self.kind, self.name, self.layer, self.concept,
                                           f(self.mu),
                                f(self.sd), f(self.w), f(self.b), self.thr))


def _attn(X, M, Q, W, b):
    a = torch.softmax((X @ Q.T).masked_fill(~M[..., None], float("-inf")), 1)  # [B, T, K]
    return (a * (X @ W.T)).sum((1, 2)) + b


class AttnPoolMonitor(_Mon):
    needs_tokens = True

    def __init__(self, name, layer, concept, mu, sd, Q, W, b, thr=0.0):
        self.name, self.layer, self.concept = name, layer, concept
        self.mu, self.sd, self.Q, self.W, self.b, self.thr = mu, sd, Q, W, b, thr

    def logits(self, toks, concepts=None):
        X, M = _pad(toks, self.mu, self.sd, self.mu.device)
        return _attn(X, M, self.Q, self.W, self.b)

    def to(self, device):
        f = lambda v: v.to(device)  # noqa: E731
        return self._keep(AttnPoolMonitor(self.name, self.layer, self.concept, f(self.mu),
                                          f(self.sd),
                               f(self.Q), f(self.W), f(self.b), self.thr))


class EnsembleMonitor(_Mon):
    """Mean over layers of per-layer sigmoid scores (Bailey / NC / Atlas). logits() takes
    {layer: [n, d]} for every layer in layers_used."""

    def __init__(self, name, layer, concept, parts, thr=0.0, agg="mean"):
        self.name, self.layer, self.concept, self.parts, self.thr = name, layer, concept, parts, thr
        self.layers_used = [p[0] for p in parts]
        self.agg = agg  # "mean" (Bailey / NC) or "median" (majority vote, spread5)

    @staticmethod
    def _score(part, x):
        _, kind, mu, sd, A, a, B, bb = part
        z = (x.float().to(mu.device) - mu) / sd
        lo = z @ A + a if kind == "linear" else torch.relu(z @ A + a) @ B + bb
        return torch.sigmoid(lo)

    def logits(self, xs, concepts=None):
        st = torch.stack([self._score(p, xs[p[0]]) for p in self.parts])
        return st.median(0).values if getattr(self, "agg", "mean") == "median" else st.mean(0)

    def to(self, device):
        parts = [(p[0], p[1], *[v.to(device) if torch.is_tensor(v) else v for v in p[2:]])
                 for p in self.parts]
        return self._keep(EnsembleMonitor(self.name, self.layer, self.concept, parts, self.thr,
                                          getattr(self, "agg", "mean")))


def _tok_stats(toks):
    x = torch.cat([t.float() for t in toks])
    return x.mean(0), x.std(0, unbiased=False) + 1e-6


def _lr_in_z(z, y, C):
    """Logistic regression on already-standardised rows z, returned as (w, b) acting on z."""

    w0, b0, mu0, sd0 = fit_logistic(z, y, C)
    return (w0 / sd0).float(), float(b0 - ((mu0 / sd0) @ w0))


def _train(params, loss_fn, steps=STEPS, lr=1e-3):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()


def _fit_mlp(z, y, g, hidden=64):
    d = z.shape[1]
    W1 = (torch.randn(d, hidden, generator=g) / d**0.5).to(z.device).requires_grad_()
    b1 = torch.zeros(hidden, device=z.device, requires_grad=True)
    w2 = (torch.randn(hidden, generator=g) / hidden**0.5).to(z.device).requires_grad_()
    b2 = torch.zeros((), device=z.device, requires_grad=True)
    pw = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    _train([W1, b1, w2, b2], lambda: bce(torch.relu(z @ W1 + b1) @ w2 + b2, y, pos_weight=pw),
           steps=600)  # as test_tough_monitors_mlp_and_ood
    return W1.detach(), b1.detach(), w2.detach(), b2.detach()


def fit_kind(kind, P, N, *, name, layer, concept, C, g, device="cpu"):
    """Fit one probe of `kind` on positives P and negatives N (no threshold yet)."""

    bce = torch.nn.functional.binary_cross_entropy_with_logits
    if kind == "meandiff":
        w = P.float().mean(0) - N.float().mean(0)
        return MeanDiffMonitor(name, layer, concept, w.to(device))
    if kind in ENS_KINDS:
        parts = []
        for L in kind_layers(kind, layer):
            x = torch.cat([P[L], N[L]]).float().to(device)
            y = torch.cat([torch.ones(len(P[L])), torch.zeros(len(N[L]))]).to(device)
            mu, sd = x.mean(0), x.std(0, unbiased=False) + 1e-6
            z = (x - mu) / sd
            if kind.endswith("_linear"):
                w, b = _lr_in_z(z, y.cpu(), C)
                parts.append((L, "linear", mu, sd, w.to(device), torch.tensor(b, device=device),
                              None, None))
            else:
                parts.append((L, "mlp", mu, sd, *_fit_mlp(z, y, g)))
        return EnsembleMonitor(name, layer, concept, parts,
                               agg="median" if kind.startswith("spread5") else "mean")
    if kind in LIT_KINDS:
        return fit_lit(kind, P, N, name=name, layer=layer, concept=concept, g=g, device=device)
    if kind in MCK_KINDS:
        return McKMonitor(kind, name, layer, concept, _fit_mck(kind, P, N, g, device).eval())
    if kind in TOKENS_ENS_KINDS:
        lins = _fit_tokens_ens(kind, P, N, g, device)
        return TokensEnsMonitor(kind, name, layer, concept, lins)
    if kind not in TOKEN_KINDS:
        raise ValueError(kind)
    items = list(P) + list(N)
    y = torch.tensor([1.0] * len(P) + [0.0] * len(N), device=device)
    mu, sd = _tok_stats(items)
    mu, sd = mu.to(device), sd.to(device)
    X, M = _pad(items, mu, sd, device)
    pw = torch.tensor(len(N) / max(1, len(P)), device=device)
    if kind == "attnpool":
        zbar = (X * M[..., None]).sum(1) / M.sum(1, keepdim=True)
        w0, b0 = _lr_in_z(zbar, y.cpu(), C)
        K = ATTN_HEADS
        Q = (torch.randn(K, X.shape[2], generator=g) * 1e-2).to(device).requires_grad_()
        W = (w0.to(device)[None].repeat(K, 1) / K).requires_grad_()
        b = torch.tensor(b0, device=device, requires_grad=True)
        _train([Q, W, b], lambda: bce(_attn(X, M, Q, W, b), y, pos_weight=pw))
        return AttnPoolMonitor(name, layer, concept, mu, sd, Q.detach(), W.detach(), b.detach())
    # maxpool / rollmax: initialise from a token-level LR (every token labelled with its
    # sequence, <= 16 random tokens per sequence; Apollo's recipe), then train THROUGH the pooling
    # on the sequence labels (RL-Obfuscation top-k; McKenzie / GDM rolling max).
    rows, ys = [], []
    for i in range(len(items)):
        idx = M[i].nonzero().flatten().cpu()
        idx = idx[torch.randperm(len(idx), generator=g)[:16]]
        rows.append(X[i, idx.to(X.device)])
        ys += [float(y[i])] * len(idx)
    w0, b0 = _lr_in_z(torch.cat(rows), torch.tensor(ys), C)
    w = w0.to(device).requires_grad_()
    b = torch.tensor(b0, device=device, requires_grad=True)
    _train([w, b], lambda: bce(_pool(kind, X @ w + b, M, train=True), y, pos_weight=pw))
    return TokenPoolMonitor(kind, name, layer, concept, mu, sd, w.detach(), b.detach())


def _fit_ens_many(kind, P, N, keeps, *, name, layer, concept, C, g, device):
    """fit_kind(ENS *_mlp kind) for several positive subsets (bool masks over P) at once -- the CV
    folds and the final fit of gate_fit (refit_batched_fits, 10-16c): every (fit, layer) MLP in ONE
    stacked AdamW run. AdamW is elementwise and the losses are separate, so each MLP follows its own
    sequential updates; inits are drawn from g in the sequential order (fit-major, then layer).
    Returns one EnsembleMonitor per mask."""

    Ls = kind_layers(kind, layer)
    nP = _n(P)
    parts = [[] for _ in keeps]
    masks = torch.stack([torch.cat([kp.bool().cpu(), torch.ones(_n(N), dtype=torch.bool)])
                         for kp in keeps])  # [F, n]
    y = torch.cat([torch.ones(nP), torch.zeros(_n(N))])
    hidden = 64
    Z, inits, mus, sds, pws, fl = [], [], [], [], [], []
    for f, m in enumerate(masks):  # sequential order: fit-major, then layer (as fit_kind)
        md = m.to(device)
        for L in Ls:
            x = torch.cat([P[L], N[L]]).float().to(device)
            xs = x[md]
            mu, sd = xs.mean(0), xs.std(0, unbiased=False) + 1e-6
            Z.append((x - mu) / sd)
            d = x.shape[1]
            W1 = torch.randn(d, hidden, generator=g) / d**0.5
            w2 = torch.randn(hidden, generator=g) / hidden**0.5
            inits.append((W1, w2))
            mus.append(mu)
            sds.append(sd)
            yf = y[m]
            pws.append(float((yf == 0).sum() / (yf == 1).sum().clamp(min=1)))
            fl.append((f, L))
    Z = torch.stack(Z)  # [F * nL, n, d]
    Wm = masks.repeat_interleave(len(Ls), 0).to(device).float()  # [F * nL, n]
    W1 = torch.stack([a for a, _ in inits]).to(device).requires_grad_()
    w2 = torch.stack([b for _, b in inits]).to(device).requires_grad_()
    b1 = torch.zeros(len(fl), hidden, device=device, requires_grad=True)
    b2 = torch.zeros(len(fl), device=device, requires_grad=True)
    yy = y.to(device).expand(len(fl), -1)
    pw = torch.tensor(pws, device=device)[:, None]
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def loss():
        h = torch.relu(Z @ W1 + b1[:, None])
        lo = (h @ w2[..., None])[..., 0] + b2[:, None]
        per = bce(lo, yy, pos_weight=pw.expand_as(lo), reduction="none")
        return ((per * Wm).sum(1) / Wm.sum(1)).sum()

    _train([W1, b1, w2, b2], loss, steps=600)
    for q, (f, L) in enumerate(fl):
        parts[f].append((L, "mlp", mus[q], sds[q], W1[q].detach(), b1[q].detach(),
                         w2[q].detach(), b2[q].detach()))
    agg = "median" if kind.startswith("spread5") else "mean"
    return [EnsembleMonitor(name, layer, concept, p, agg=agg) for p in parts]


def _fit_tok_many(kind, P, N, keeps, *, name, layer, concept, C, g, device):
    """fit_kind(attnpool / maxpool / rollmax) for several positive subsets at once (the CV folds +
    final fit of gate_fit; refit_batched_fits, 10-16n). Per fit the SAME initialisation as fit_kind
    (its own token standardisation mu / sd, logistic init, g draws in the sequential order), then
    ONE stacked AdamW run: the fits' raw tokens are shared and each fit's standardisation is folded
    into its parameters ((x - mu) / sd . w = x . (w / sd) - (mu / sd) . w), each loss is the mean
    over its own items, and AdamW is elementwise -> each fit follows its own updates (up to float
    reassociation). Returns one monitor per mask, parameters in each fit's z-space (as fit_kind)."""

    items = list(P) + list(N)
    nP, nI = len(P), len(list(P) + list(N))
    X, M = _pad(items, torch.zeros(1, device=device), torch.ones(1, device=device), device)
    y = torch.tensor([1.0] * nP + [0.0] * (nI - nP), device=device)
    xbar = (X * M[..., None]).sum(1) / M.sum(1, keepdim=True)  # raw per-item token mean
    fits = []  # per fit: (sel, mu, sd, pw, params-init)
    for kp in keeps:  # sequential init order = gate_fit's fold order, then the final fit
        sel = torch.cat([kp.bool().cpu(), torch.ones(nI - nP, dtype=torch.bool)])
        fit_items = [items[i] for i in sel.nonzero().flatten().tolist()]
        mu, sd = _tok_stats(fit_items)
        mu, sd = mu.to(device), sd.to(device)
        yf = y[sel.to(device)]
        pw = float((yf == 0).sum()) / max(1, int((yf == 1).sum()))
        if kind == "attnpool":
            zbar = (xbar[sel.to(device)] - mu) / sd
            w0, b0 = _lr_in_z(zbar, yf.cpu(), C)
            Q = torch.randn(ATTN_HEADS, X.shape[2], generator=g) * 1e-2
            init = (Q.to(device), w0.to(device)[None].repeat(ATTN_HEADS, 1) / ATTN_HEADS, b0)
        else:  # maxpool / rollmax: token-level LR init on <= 16 random tokens per sequence
            rows, ys = [], []
            for i in sel.nonzero().flatten().tolist():
                idx = M[i].nonzero().flatten().cpu()
                idx = idx[torch.randperm(len(idx), generator=g)[:16]]
                rows.append((X[i, idx.to(X.device)] - mu) / sd)
                ys += [float(y[i])] * len(idx)
            w0, b0 = _lr_in_z(torch.cat(rows), torch.tensor(ys), C)
            init = (w0.to(device), b0)
        fits.append((sel.to(device), mu, sd, pw, init))
    F = len(fits)
    Wm = torch.stack([f[0].float() for f in fits])  # [F, nI] item weights
    MU = torch.stack([f[1] for f in fits])
    SD = torch.stack([f[2] for f in fits])
    PW = torch.tensor([f[3] for f in fits], device=device)[:, None].expand(F, nI)
    yy = y.expand(F, nI)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    mask = M[None].expand(F, -1, -1)

    def lin(A):  # [F, ..., d] z-space weights -> raw-space token scores, [F, nI, T, ...]
        u = A / SD.view(F, *([1] * (A.dim() - 2)), -1)
        c = (A * (MU / SD).view(F, *([1] * (A.dim() - 2)), -1)).sum(-1)
        if A.dim() == 2:
            return torch.einsum("btd,fd->fbt", X, u) - c[:, None, None]
        return torch.einsum("btd,fkd->fbtk", X, u) - c[:, None, None, :]

    if kind == "attnpool":
        Q = torch.stack([f[4][0] for f in fits]).requires_grad_()
        Wt = torch.stack([f[4][1] for f in fits]).requires_grad_()
        b = torch.tensor([f[4][2] for f in fits], device=device, requires_grad=True)

        def scores():
            S = lin(Q).masked_fill(~mask[..., None], float("-inf"))
            a = torch.softmax(S, 2)
            return (a * lin(Wt)).sum((2, 3)) + b[:, None]

        params = [Q, Wt, b]
    else:
        w = torch.stack([f[4][0] for f in fits]).requires_grad_()
        b = torch.tensor([f[4][1] for f in fits], device=device, requires_grad=True)

        def scores():
            s_ = lin(w) + b[:, None, None]
            return _pool(kind, s_.reshape(F * nI, -1), mask.reshape(F * nI, -1),
                         train=True).view(F, nI)

        params = [w, b]

    def loss():
        per = bce(scores(), yy, pos_weight=PW, reduction="none")
        return ((per * Wm).sum(1) / Wm.sum(1)).sum()

    _train(params, loss)
    out = []
    for f in range(F):
        mu, sd = MU[f], SD[f]
        if kind == "attnpool":
            out.append(AttnPoolMonitor(name, layer, concept, mu, sd, Q[f].detach(),
                                       Wt[f].detach(), b[f].detach()))
        else:
            out.append(TokenPoolMonitor(kind, name, layer, concept, mu, sd, w[f].detach(),
                                        b[f].detach()))
    return out


def _to(items, device):
    if isinstance(items, dict):
        return {L: _to(v, device) for L, v in items.items()}
    if torch.is_tensor(items):
        return items.to(device)
    return [t.to(device) for t in items]


def _n(items):
    return len(next(iter(items.values()))) if isinstance(items, dict) else len(items)


def _take(items, idx):
    if isinstance(items, dict):
        return {L: _take(v, idx) for L, v in items.items()}
    return items[idx] if torch.is_tensor(items) else [items[int(i)] for i in idx]


def gate_fit(kind, P, N, NP, cfg, *, name, layer, concept, device="cpu"):
    """Fit + gate a zoo probe (module docstring). Returns (monitor or None, info)."""

    g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    k = max(2, cfg.refit_cv_folds or 5)
    P, N, NP = (_to(x, device) for x in (P, N, NP))  # ONE host->device copy for all CV fits
    nN = _n(N)
    nperm = torch.randperm(nN, generator=g)
    nh = max(1, round(nN * 0.3))
    Nf, Nh = _take(N, nperm[nh:]), _take(N, nperm[:nh])
    info = {"kind": kind, "n_pos": _n(P), "n_neg": nN}
    try:
        perm = torch.randperm(_n(P), generator=g)
        wins, pairs = 0.0, 0
        if kind == "gdm_rollattn":  # the k CV folds + the final fit as ONE vectorised training
            outs = [perm[j::k] for j in range(k)]
            keeps = []
            for out in outs:
                keep = torch.ones(_n(P), dtype=torch.bool)
                keep[out] = False
                keeps.append(keep)
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            nets = _fit_gdm_many(P, Nf, g, dev, [*keeps, torch.ones(_n(P), dtype=torch.bool)])
            mons = [LitTokenMonitor(kind, name, layer, concept, n_.eval()) for n_ in nets]
            for out, mj in zip(outs, mons[:-1], strict=True):
                a_j = auroc(mj.logits(_take(P, out)).detach().cpu(),
                            mj.logits(Nh).detach().cpu())
                wins += a_j * len(out) * nh
                pairs += len(out) * nh
            k = 0  # folds done
        tok_batch = kind in TOKEN_KINDS and getattr(cfg, "refit_batched_fits", False)
        if tok_batch or (kind in ENS_KINDS and kind.endswith("_mlp")
                         and getattr(cfg, "refit_batched_fits", False)):
            outs = [perm[j::k] for j in range(k)]
            keeps = []
            for out in outs:
                keep = torch.ones(_n(P), dtype=torch.bool)
                keep[out] = False
                keeps.append(keep)
            many = _fit_tok_many if tok_batch else _fit_ens_many
            mons = many(kind, P, Nf, [*keeps, torch.ones(_n(P), dtype=torch.bool)],
                        name=name, layer=layer, concept=concept, C=cfg.refit_C, g=g,
                        device=device)
            for out, mj in zip(outs, mons[:-1], strict=True):
                a_j = auroc(mj.logits(_take(P, out)).detach().cpu(),
                            mj.logits(Nh).detach().cpu())
                wins += a_j * len(out) * nh
                pairs += len(out) * nh
            k = 0  # folds done
        for j in range(k):
            out = perm[j::k]
            keep = torch.ones(_n(P), dtype=torch.bool)
            keep[out] = False
            mj = fit_kind(kind, _take(P, keep.nonzero().flatten()), Nf, name=name, layer=layer,
                          concept=concept, C=cfg.refit_C, g=g, device=device)
            a_j = auroc(mj.logits(_take(P, out)).detach().cpu(), mj.logits(Nh).detach().cpu())
            wins += a_j * len(out) * nh
            pairs += len(out) * nh
        au = wins / pairs
        done = kind == "gdm_rollattn" or (getattr(cfg, "refit_batched_fits", False) and (
            kind in TOKEN_KINDS or (kind in ENS_KINDS and kind.endswith("_mlp"))))
        mon = mons[-1] if done else fit_kind(
            kind, P, Nf, name=name, layer=layer, concept=concept, C=cfg.refit_C, g=g,
            device=device)
    except RuntimeError as e:  # e.g. a logistic fit that does not converge
        info["error"] = str(e)[:200]
        return None, info
    sn = mon.logits(Nh).detach().cpu()
    mon.thr = float(quantile_threshold(sn, cfg.refit_fpr))
    mon.scale_ = max(float(sn.float().std()), 1e-3)
    tpr_np = float((mon.logits(NP).detach().cpu() > mon.thr).float().mean())
    info.update({"auroc": au, "tpr_np": tpr_np})
    ok = math.isfinite(au) and au >= cfg.refit_min_auroc and tpr_np >= cfg.refit_min_tpr_np
    info["kept"] = ok
    return (mon if ok else None), info


# ---------------------------------------------------------------------------------------------
# EXACT literature probes (REPLICATION 2026-10-13o; specs from the official code where it exists).
# Raw activations (no standardisation), as in the RL-Obfuscation code; GDM leaves it unstated (our
# choice: raw). Kinds:
# - rlobf_attn / rlobf_attn_max: RL-Obfuscation AttentionProbe (github.com/cybershiptrooper/
#   obfuscated_backdoors src/probe_archs.py): q/k/v = Linear(d, 512) with bias, 1 head, causal
#   scaled-dot-product attention, out_proj Linear(512, 1) -> one logit per token. Trained token-wise
#   (every token carries its sequence label; Probe.compute_loss = masked-mean BCE). Sequence score =
#   mean (rlobf_attn) or max (rlobf_attn_max) of per-token sigmoids (rl_utils/probe_evaluation.py).
# - rlobf_max: MaxPoolProbe over a per-token Linear(d, 1), top-k with k = 1 (the code never passes
#   k=15 to the constructor), BCE on the max logit vs the sequence label.
#   Training (src/probe_training.py train_layer): AdamW lr 1e-3 (default wd 0.01), epochs 40
#   (token-wise) / 80 (maxpool) per the paper, batch 16 positives + 16 negatives with ONE shared
#   permutation over min(n_pos, n_neg) examples (drop last), loss = loss_pos + loss_neg, bf16
#   autocast, grad clip 1.0.
# - gdm_rollattn: Kramar et al. 2601.11516 "Max of Rolling Means" attention probe (no code): token
#   MLP d -> 100 -> 100 (ReLU; "two layers of width 100"), H = 10 heads, window w = 10:
#   vbar_t,h = sum_{j in window} exp(q_h.y_j) (v_h.y_j) / sum exp(q_h.y_j); score = max_t sum_h
#   vbar_t,h + b (head combination NOT STATED: our choice = sum over heads, as in their Eqs. 8-9).
#   Full-batch AdamW lr 1e-4, wd 3e-3, 1000 steps, sequence BCE. Seed selection: paper = best of
#   100 by validation loss (gain only 0.008 test loss); ours = 1 seed (compute: ~150 refits per
#   race), trained on 85% (the 15% validation split is kept so seeds > 1 stays a one-line change).
# ---------------------------------------------------------------------------------------------
LIT_KINDS = ("rlobf_attn", "rlobf_attn_max", "rlobf_max", "gdm_rollattn")


class _RLObfAttn(torch.nn.Module):
    def __init__(self, d, d_proj=512):
        super().__init__()
        self.q, self.k, self.v = (torch.nn.Linear(d, d_proj) for _ in range(3))
        self.out = torch.nn.Linear(d_proj, 1)

    def forward(self, X, M):  # X [B, T, d], M [B, T] -> token logits [B, T]
        T = X.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=X.device).tril()
        q, k, v = (f(X)[:, None] for f in (self.q, self.k, self.v))  # [B, 1, T, d_proj]
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        return self.out(o[:, 0]).squeeze(-1)


class _TokLinear(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, 1)

    def forward(self, X, M):
        return self.lin(X).squeeze(-1)


class _GDMRollAttn(torch.nn.Module):
    def __init__(self, d, width=100, heads=10, window=10):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.ReLU(),
                                       torch.nn.Linear(width, width), torch.nn.ReLU())
        self.q = torch.nn.Linear(width, heads, bias=False)
        self.v = torch.nn.Linear(width, heads, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(()))
        self.w = window

    def forward(self, X, M):  # -> sequence logits [B]
        y = self.mlp(X)
        s = self.q(y).masked_fill(~M[..., None], float("-inf"))  # [B, T, H]
        a = torch.exp(s - s.amax(1, keepdim=True))  # per-sequence constant cancels in each window
        a = a.masked_fill(~M[..., None], 0.0)
        av = a * self.v(y)
        T, w = X.shape[1], self.w
        ca = torch.nn.functional.pad(a.cumsum(1), (0, 0, 1, 0))
        cav = torch.nn.functional.pad(av.cumsum(1), (0, 0, 1, 0))
        n = M.sum(1)
        if w <= T:
            vbar = (cav[:, w:] - cav[:, :-w]) / (ca[:, w:] - ca[:, :-w]).clamp(min=1e-30)
            cm = torch.nn.functional.pad(M.float().cumsum(1), (1, 0))
            full = (cm[:, w:] - cm[:, :-w]) >= w
            agg = vbar.sum(-1).masked_fill(~full, float("-inf")).amax(1)
        else:
            agg = torch.full((X.shape[0],), float("-inf"), device=X.device)
        whole = (cav[:, -1] / ca[:, -1].clamp(min=1e-30)).sum(-1)  # one window = the sequence
        return torch.where(n < w, whole, agg) + self.b


class LitTokenMonitor(_Mon):
    needs_tokens = True

    def __init__(self, kind, name, layer, concept, net, thr=0.0):
        self.kind, self.name, self.layer, self.concept, self.net, self.thr = (
            kind, name, layer, concept, net, thr)

    def logits(self, toks, concepts=None):
        dev = next(self.net.parameters()).device
        X, M = _pad(toks, torch.zeros(1, device=dev), torch.ones(1, device=dev), dev)
        out = self.net(X, M)  # differentiable in X when grad is enabled (training hide loss)
        if self.kind == "gdm_rollattn":
            return out
        if self.kind == "rlobf_max":
            return out.masked_fill(~M, float("-inf")).amax(1)
        p = torch.sigmoid(out)
        if self.kind == "rlobf_attn_max":
            return p.masked_fill(~M, -1.0).amax(1)
        return (p * M).sum(1) / M.sum(1)

    def to(self, device):
        return self._keep(LitTokenMonitor(self.kind, self.name, self.layer, self.concept,
                               self.net.to(device), self.thr))


def _fit_rlobf(kind, P, N, g, device):
    d = P[0].shape[1]
    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))  # init seed from the name
    net = (_TokLinear(d) if kind == "rlobf_max" else _RLObfAttn(d)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    epochs = 80 if kind == "rlobf_max" else 40
    n, bs = min(len(P), len(N)), 16
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    one = torch.ones(1, device=device)

    def loss_on(items, label):
        X, M = _pad(items, 0 * one, one, device)
        out = net(X, M)
        if kind == "rlobf_max":  # BCE on the max logit vs the sequence label
            return bce(out.masked_fill(~M, float("-inf")).amax(1),
                       torch.full((len(items),), label, device=device))
        return bce(out[M], torch.full((int(M.sum()),), label, device=device))  # token-wise

    with torch.enable_grad():
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n - bs + 1, bs):
                idx = perm[i : i + bs]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):  # as their train_layer
                    loss = loss_on([P[j] for j in idx], 1.0) + loss_on([N[j] for j in idx], 0.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
    return net


def _fit_gdm(P, N, g, device, seeds=1, steps=1000):
    items = list(P) + list(N)
    y = torch.tensor([1.0] * len(P) + [0.0] * len(N), device=device)
    one = torch.ones(1, device=device)
    X, M = _pad(items, 0 * one, one, device)
    perm = torch.randperm(len(items), generator=g)
    nv = max(2, round(0.15 * len(items)))
    va, tr = perm[:nv].to(device), perm[nv:].to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, best_loss = None, float("inf")
    for _ in range(seeds):
        torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
        net = _GDMRollAttn(X.shape[2]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=3e-3)
        with torch.enable_grad():
            for _ in range(steps):  # full batch
                opt.zero_grad()
                bce(net(X[tr], M[tr]), y[tr]).backward()
                opt.step()
        with torch.no_grad():
            vl = float(bce(net(X[va], M[va]), y[va]))
        if vl < best_loss:
            best, best_loss = net, vl
    return best


def _fit_gdm_many(P, N, g, device, pos_keeps, steps=1000):
    """_fit_gdm (seeds=1) for several positive subsets at once: one model per mask in pos_keeps
    (bool [n_pos]), all negatives, trained together with torch.func.vmap over stacked parameters.
    Per-model losses are independent and AdamW is elementwise, so each model follows the same
    update rule as a separate _fit_gdm on its subset (same 85/15 split); only kernel launches are
    shared (efficiency pass 10-14f). Returns one net per mask."""

    from torch.func import functional_call, stack_module_state, vmap

    items = list(P) + list(N)
    y = torch.tensor([1.0] * len(P) + [0.0] * len(N), device=device)
    one = torch.ones(1, device=device)
    X, M = _pad(items, 0 * one, one, device)
    perm = torch.randperm(len(items), generator=g)
    nv = max(2, round(0.15 * len(items)))
    tr = perm[nv:].to(device)
    W = torch.ones(len(pos_keeps), len(items), device=device)
    for k, keep in enumerate(pos_keeps):
        W[k, : len(P)] = keep.to(device).float()
    W = W[:, tr]
    nets = []
    for _ in pos_keeps:
        torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
        nets.append(_GDMRollAttn(X.shape[2]).to(device))
    params, bufs = stack_module_state(nets)
    proto = copy.deepcopy(nets[0]).to("meta")

    def one_net(p, b, X_, M_):
        return functional_call(proto, (p, b), (X_, M_))

    fwd = vmap(one_net, in_dims=(0, 0, None, None))
    opt = torch.optim.AdamW(list(params.values()), lr=1e-4, weight_decay=3e-3)
    Xt, Mt, yt = X[tr], M[tr], y[tr].expand(len(pos_keeps), -1)
    # The d -> 100 projection of every token for 6 models x 1000 full-batch steps is compute
    # bound in fp32 (37 s / refit); TF32 matmuls (10-bit mantissa, fp32 range) for the fit only.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        _gdm_steps(fwd, params, bufs, Xt, Mt, yt, W, opt, steps)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    for k, net in enumerate(nets):
        with torch.no_grad():
            for nm, prm in net.named_parameters():
                prm.copy_(params[nm][k])
    return nets


def _gdm_steps(fwd, params, bufs, Xt, Mt, yt, W, opt, steps):
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    with torch.enable_grad():
        for _ in range(steps):  # full batch
            opt.zero_grad()
            out = fwd(params, bufs, Xt, Mt)
            ((bce(out, yt, reduction="none") * W).sum(1) / W.sum(1)).sum().backward()
            opt.step()


def fit_lit(kind, P, N, *, name, layer, concept, g, device):
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    net = _fit_gdm(P, N, g, dev) if kind == "gdm_rollattn" else _fit_rlobf(kind, P, N, g, dev)
    return LitTokenMonitor(kind, name, layer, concept, net.eval())


# ---------------------------------------------------------------------------------------------
# McKenzie et al. 2506.10805 (github.com/arrrlex/models-under-pressure, probes/pytorch_modules.py,
# config/probe/*.yaml); raw pre-layernorm residual, no standardisation:
# - mck_attnlite: s_j = (x_j.theta_q + b_q) / sqrt(d), alpha = softmax_j(s) (padding -inf),
#   c = sum_j alpha_j x_j, logit = theta_v.c + b_v.
# - mck_softmax: z_j = theta.x_j + b, w = softmax_j(z / 5), score = sum_j w_j z_j.
# - mck_rollmean40: z_j per token, unfold(40) mean, max (sequences < 40 tokens: their code fails;
#   ours = mean of all tokens).
# Training: AdamW lr 5e-3, wd 1e-3, cosine annealing to final lr 5e-4 (softmax: 1e-4) over the epoch
# budget, <= 200 epochs, batch 128 (rolling: 16 x grad-accum 16), BCE on sequence labels, grad clip
# 1.0, early stopping on validation AUROC with patience 50 (softmax: 10). Validation split NOT
# STATED: ours 20%. Deviations (10-14f, both needed at our ~200 examples): grad-accum capped so an
# epoch has >= 4 steps, and best-validation weights restored (their shallow copy keeps the final).
# Exact token-level multi-layer ensembles (official code: per-token sigmoid -> masked mean over
# tokens -> mean over layers):
# - nc_ens8_tok: Neural Chameleons obf_reps: per-layer token-level Linear(d,1), Adam lr 5e-4,
#   wd 1e-5, eps 1e-5, batch 64 tokens, 5 epochs. Layers NOT STATED -> ours ENS_LAYERS["ens8"].
# - bailey_ens_tok: Bailey obfuscated-activations TrainableMetric: Adam lr 1e-3, wd 1e-5, batch 64
#   tokens, 1 epoch, ALL layers -> ours all 32 (10-19: 20 / 24 back in; it is an eval-only kind, so
#   reading the held-out depths leaks nothing into training; was "all except 20 / 24").
#   Deviation (memory): fork token states are an evenly spaced subset of <= TOK_KEEP tokens per
#   answer per layer (training AND gate scoring); live turns are scored on all tokens.
# ---------------------------------------------------------------------------------------------
MCK_KINDS = ("mck_attnlite", "mck_softmax", "mck_rollmean40")
TOKENS_ENS_KINDS = TOKENS_ENS
TOK_KEEP = 32
TOKENS_ENS_LAYERS = {"nc_ens8_tok": ENS_LAYERS["ens8"],
                     "bailey_ens_tok": list(range(32))}
# bailey_ens_tok_max (10-15d): Bailey-style per-layer token probes, but the monitor fires if ANY
# layer does (max over layers of the per-layer token-mean scores) instead of the layer average,
# which hidden layers can outvote. Not in Bailey et al. Restricted to the band 4..28: with FORCED
# answer
# texts the first layers are near-lexical (a topic probe there reads the given tokens, which the
# policy cannot change) and the last ones are next-token specialised; under a max either end would
# decide the verdict for reasons unrelated to concept hiding (and the noisiest layer sets the
# threshold). Held-out eval layers 20 / 24 excluded as for bailey_ens_tok.
TOKENS_ENS_LAYERS["bailey_ens_tok_max"] = list(range(4, 29))


class _MckNet(torch.nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        self.lin = torch.nn.Linear(d, 1)  # classifier / value (attnlite) or per-token scorer
        if kind == "mck_attnlite":
            self.q = torch.nn.Linear(d, 1)

    def forward(self, X, M):
        if self.kind == "mck_attnlite":
            s = (self.q(X).squeeze(-1) / X.shape[-1] ** 0.5).masked_fill(~M, float("-inf"))
            c = (torch.softmax(s, 1)[..., None] * X).sum(1)
            return self.lin(c).squeeze(-1)
        z = self.lin(X).squeeze(-1)
        if self.kind == "mck_softmax":
            w = torch.softmax((z / 5.0).masked_fill(~M, float("-inf")), 1)
            return (w * z.masked_fill(~M, 0.0)).sum(1)
        # mck_rollmean40: max over 40-token windows lying INSIDE the answer; an answer shorter
        # than 40 tokens is one window (its mean). Their unfold runs per padded batch, so a short
        # answer's windows averaged in padding zeros (score shrunk by len/40): with our short
        # answers that left the probe at clean-data CV AUROC 0.62 (10-14f).
        w = 40
        zm = z.masked_fill(~M, 0.0)
        n = M.sum(1)
        whole = zm.sum(1) / n.clamp(min=1)
        if zm.shape[1] < w:
            return whole
        cz = torch.nn.functional.pad(zm.cumsum(1), (1, 0))
        cm = torch.nn.functional.pad(M.float().cumsum(1), (1, 0))
        full = (cm[:, w:] - cm[:, :-w]) >= w
        roll = ((cz[:, w:] - cz[:, :-w]) / w).masked_fill(~full, float("-inf")).amax(1)
        return torch.where(n < w, whole, roll)


def _fit_mck(kind, P, N, g, device):
    items = list(P) + list(N)
    y = torch.tensor([1.0] * len(P) + [0.0] * len(N), device=device)
    one = torch.ones(1, device=device)
    X, M = _pad(items, 0 * one, one, device)
    perm = torch.randperm(len(items), generator=g)
    nv = max(2, round(0.2 * len(items)))
    va, tr = perm[:nv].to(device), perm[nv:].to(device)
    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
    net = _MckNet(kind, X.shape[2]).to(device)
    epochs, patience = 200, (10 if kind == "mck_softmax" else 50)
    final_lr = 1e-4 if kind == "mck_softmax" else 5e-4
    bs, accum = (16, 16) if kind == "mck_rollmean40" else (128, 1)
    # Their batch 16 x accum 16 = 256 assumes thousands of examples; with our ~200 it is ONE
    # optimizer step per epoch and the probe never fits (clean base data: CV AUROC 0.64). Keep
    # batch 16 and accumulate over at most a quarter of the training set (>= 4 steps / epoch).
    accum = max(1, min(accum, len(tr) // (4 * bs)))
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=final_lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, since, best_state = -1.0, 0, copy.deepcopy(net.state_dict())
    for _ in range(epochs):
        net.train()
        order = tr[torch.randperm(len(tr), generator=g).to(device)]
        with torch.enable_grad():
            for i, st in enumerate(range(0, len(order), bs)):
                idx = order[st : st + bs]
                (bce(net(X[idx], M[idx]), y[idx]) / accum).backward()
                if (i + 1) % accum == 0 or st + bs >= len(order):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
        sched.step()
        net.eval()
        with torch.no_grad():
            sv = net(X[va], M[va]).cpu()
        yv = y[va].cpu()
        au = auroc(sv[yv == 1], sv[yv == 0]) if 0 < yv.sum() < len(yv) else 0.5
        if au > best:
            best, since = au, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            since += 1
            if since >= patience:
                break
    # Their restore-best makes a shallow copy and so returns the FINAL weights (an early-stopped
    # model after `patience` non-improving epochs); at our sample size that left mck_softmax at
    # CV AUROC 0.90 on the dodger's forks. We restore the best-validation weights, as intended.
    net.load_state_dict(best_state)
    return net


class McKMonitor(_Mon):
    needs_tokens = True

    def __init__(self, kind, name, layer, concept, net, thr=0.0):
        self.kind, self.name, self.layer, self.concept, self.net, self.thr = (
            kind, name, layer, concept, net, thr)

    def logits(self, toks, concepts=None):
        dev = next(self.net.parameters()).device
        X, M = _pad(toks, torch.zeros(1, device=dev), torch.ones(1, device=dev), dev)
        return self.net(X, M)

    def to(self, device):
        return self._keep(McKMonitor(self.kind, self.name, self.layer, self.concept,
                                     self.net.to(device), self.thr))


class TokensEnsMonitor(_Mon):
    """Per-layer token-level linear probes: sigmoid per token -> mean over tokens -> mean over
    layers. logits() takes {layer: [per-sequence [n_tokens, d]]}."""

    needs_tokens = True

    def __init__(self, kind, name, layer, concept, lins, thr=0.0):
        self.kind, self.name, self.layer, self.concept, self.lins, self.thr = (
            kind, name, layer, concept, lins, thr)
        self.layers_used = sorted(lins)

    def logits(self, xs, concepts=None):
        per_layer = []
        for L, lin in self.lins.items():
            dev = lin.weight.device
            X, M = _pad(xs[L], torch.zeros(1, device=dev), torch.ones(1, device=dev), dev)
            p = torch.sigmoid(lin(X).squeeze(-1))
            per_layer.append((p * M).sum(1) / M.sum(1).clamp(min=1))
        st = torch.stack(per_layer)
        return st.max(0).values if self.kind.endswith("_max") else st.mean(0)

    def to(self, device):
        return self._keep(TokensEnsMonitor(self.kind, self.name, self.layer, self.concept,
                                           {L: m.to(device) for L, m in self.lins.items()},
                                           self.thr))


def _fit_tokens_ens(kind, P, N, g, device):
    """Per-layer token-level probes, trained as ONE batched problem over the layers (10-15d): the
    random draws happen in the original per-layer order (init seed, then each epoch's permutation)
    and Adam is elementwise, so this equals _fit_tokens_ens_loop up to float reduction order
    (test_tokens_ens_batched_matches_loop) at a fraction of the kernel launches."""

    lr, epochs, eps = (5e-4, 5, 1e-5) if kind == "nc_ens8_tok" else (1e-3, 1, 1e-8)
    Ls = TOKENS_ENS_LAYERS[kind]
    sizes = {sum(len(t) for t in P[L]) + sum(len(t) for t in N[L]) for L in Ls}
    if len(sizes) > 1:  # unequal token counts per layer (never in TICA reads): the loop
        return _fit_tokens_ens_loop(kind, P, N, g, device)
    X = torch.stack([torch.cat([t.float() for t in P[L]] + [t.float() for t in N[L]])
                     for L in Ls]).to(device)  # [n_layers, n_tokens, d]
    n = X.shape[1]
    y = torch.cat([torch.ones(sum(len(t) for t in P[Ls[0]])),
                   torch.zeros(sum(len(t) for t in N[Ls[0]]))]).to(device)
    Ws, bs, perms = [], [], []
    for _ in Ls:  # same draws, same order as the loop
        torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
        lin = torch.nn.Linear(X.shape[2], 1)
        Ws.append(lin.weight.detach()[0])
        bs.append(lin.bias.detach()[0])
        perms.append([torch.randperm(n, generator=g) for _ in range(epochs)])
    W = torch.stack(Ws).to(device).requires_grad_()
    b = torch.stack(bs).to(device).requires_grad_()
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-5, eps=eps)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rows = torch.arange(len(Ls), device=device)[:, None]
    with torch.enable_grad():
        for e in range(epochs):
            order = torch.stack([p[e] for p in perms]).to(device)  # [n_layers, n]
            for st in range(0, n, 64):
                idx = order[:, st : st + 64]
                xb = X[rows, idx]  # [n_layers, b, d]
                logit = torch.einsum("lbd,ld->lb", xb, W) + b[:, None]
                opt.zero_grad()
                # sum of per-layer mean losses: each layer's gradient = its own loop gradient
                bce(logit, y[idx], reduction="none").mean(1).sum().backward()
                opt.step()
    lins = {}
    for j, L in enumerate(Ls):
        lin = torch.nn.Linear(X.shape[2], 1).to(device)
        with torch.no_grad():
            lin.weight.copy_(W[j].detach()[None])
            lin.bias.copy_(b[j].detach()[None])
        lins[L] = lin.eval()
    return lins


def _fit_tokens_ens_loop(kind, P, N, g, device):
    """The original one-layer-at-a-time fit (reference for the batched one)."""

    lr, epochs, eps = (5e-4, 5, 1e-5) if kind == "nc_ens8_tok" else (1e-3, 1, 1e-8)
    lins = {}
    for L in TOKENS_ENS_LAYERS[kind]:
        x = torch.cat([t.float() for t in P[L]] + [t.float() for t in N[L]]).to(device)
        y = torch.cat([torch.ones(sum(len(t) for t in P[L])),
                       torch.zeros(sum(len(t) for t in N[L]))]).to(device)
        torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=g)))
        lin = torch.nn.Linear(x.shape[1], 1).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-5, eps=eps)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        with torch.enable_grad():
            for _ in range(epochs):
                order = torch.randperm(len(x), generator=g).to(device)
                for st in range(0, len(order), 64):
                    idx = order[st : st + 64]
                    opt.zero_grad()
                    bce(lin(x[idx]).squeeze(-1), y[idx]).backward()
                    opt.step()
        lins[L] = lin.eval()
    return lins
