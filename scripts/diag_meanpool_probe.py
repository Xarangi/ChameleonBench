# ruff: noqa
"""Decisive reproduction test: serteal fits the released probes on MEAN-POOLED
activations (one feature/example), not per-token. Does fitting a MEAN-POOLED german
probe on OUR (same) IlyaGusev base recover the released direction (cosine 0.15 -> high)?

Compares, all cosine-to-released-german:
  - per-token fit (our current pipeline)           -> expected ~0.15
  - mean-pooled over response tokens, per example  -> serteal-style
  - mean-pooled, sklearn-style closed-form ridge
"""
import glob, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_data import find_preset_dataset
from next_chameleons.hf.paper_probes import _released_weight_bias, fit_linear_probe_per_token

LAYER, OFFSET = 12, 1
HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = "german"
CAP = 80


def released(probe_dir):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[:-len("_weights.pt")].lower() == CONCEPT:
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(CONCEPT)


def per_example_feats(model, tok, responses):
    """Return per-token features (concatenated) AND per-example mean-pooled features."""
    tok_chunks, mean_rows = [], []
    for resp in responses:
        ids = tok(resp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        if len(ids) <= 1:
            continue
        t = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            h = model(input_ids=t, output_hidden_states=True).hidden_states[LAYER + OFFSET][0, 1:].float().cpu()
        tok_chunks.append(h)
        mean_rows.append(h.mean(0))  # mean-pool over (response) tokens
    return tok_chunks, torch.stack(mean_rows)


def bal_acc(scores, y, thr=0.5):
    p = (scores >= thr).float()
    pos, neg = y.bool(), ~y.bool()
    return 0.5 * (p[pos].mean().item() + (1 - p[neg]).mean().item())


def fit_logreg(X, y, lr=4e-5, epochs=1, bs=16, wd=1e-5):
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd, eps=1e-5)
    g = torch.Generator().manual_seed(17)
    for _ in range(epochs):
        order = torch.randperm(X.shape[0], generator=g)
        for s in range(0, X.shape[0], bs):
            i = order[s:s + bs]; opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(X[i] @ w + b, y[i]).backward(); opt.step()
    return w.detach(), b.detach()


def ridge_logreg(X, y, l2=1.0, iters=300):
    # sklearn-like full-batch L2 logistic (lbfgs-ish via many GD steps on standardized feats)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(Xs.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters)
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xs @ w + b, y) + l2 * (w ** 2).mean()
        loss.backward(); return loss
    opt.step(closure)
    return (w.detach() / sd), b.detach()


def cos(a, b):
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def main():
    probe_dir, raw_cache = sys.argv[1], sys.argv[2]
    recs = json.loads(Path(find_preset_dataset(Path(raw_cache))).read_text("utf-8"))
    pn = lambda r: str(r.get("probe_name", "")).lower()
    dt = lambda r: str(r.get("data_type", ""))
    pos = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_X_probe_X")][:CAP]
    neg = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_Y_probe_X")
           and float(r["target_probe_score"]) < 0.5][:CAP]
    print(f"[meanpool] german pos={len(pos)} data_Y neg={len(neg)}", flush=True)
    loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16"); model, tok = loaded.model, loaded.tokenizer
    model.eval()
    w_rel, b_rel = released(probe_dir)

    pos_tok, pos_mean = per_example_feats(model, tok, pos)
    neg_tok, neg_mean = per_example_feats(model, tok, neg)

    # 1) per-token (our pipeline)
    Xtok = torch.cat(pos_tok + neg_tok, 0)
    ytok = torch.cat([torch.ones(sum(c.shape[0] for c in pos_tok)),
                      torch.zeros(sum(c.shape[0] for c in neg_tok))])
    w_pt, b_pt = fit_linear_probe_per_token(token_features=Xtok, token_labels=ytok,
                                            learning_rate=4e-5, epochs=1, batch_size=16)
    # 2) mean-pooled per example (serteal-style)
    Xm = torch.cat([pos_mean, neg_mean], 0)
    ym = torch.cat([torch.ones(pos_mean.shape[0]), torch.zeros(neg_mean.shape[0])])
    w_mp, b_mp = fit_logreg(Xm, ym)
    # 3) mean-pooled + ridge (sklearn-like)
    w_rg, b_rg = ridge_logreg(Xm, ym)

    def pr(w): w2 = w.float() ** 2; return (w2.sum() ** 2 / (w2 ** 2).sum()).item()
    print("\ncosine-to-released-german (the test):")
    print(f"  per-token (our pipeline)      cos={cos(w_pt, w_rel):+.4f}  PR={pr(w_pt):.0f}")
    print(f"  MEAN-POOL per-example (Adam)  cos={cos(w_mp, w_rel):+.4f}  PR={pr(w_mp):.0f}")
    print(f"  MEAN-POOL + ridge (lbfgs)     cos={cos(w_rg, w_rel):+.4f}  PR={pr(w_rg):.0f}")
    print(f"  released probe PR={pr(w_rel):.0f}  (target)")


if __name__ == "__main__":
    main()
