# ruff: noqa
"""Decisive self-fit test: does fitting a benign concept probe on the AUTHORS' exact
rated dataset (mega_dataset_evaluated_*.json, per-concept gpt-4.1-nano ratings) with
their exact recipe recover the RELEASED probe direction (cosine ~0.15 -> high)?

Their fit (obf_reps/metrics LogisticRegressionMetric.fit), confirmed from the clone:
  per-response-token, RAW activations, Linear(d,1)+bias (bf16),
  Adam(lr=4e-5, wd=1e-5, eps=1e-5), BCEWithLogitsLoss, batch 64, 5 epochs, shuffle=False.
  POS = topical_response rating >= 0.75 for the concept; NEG = <= 0.25.

We extract response-token activations WITH prompt context under the gemma chat template
(user=prompt, assistant=topical_response), at hidden_states[LAYER+OFFSET].

Reports cosine-to-released for:
  (a) their data + their recipe (5 epochs, batch 64)   <- the faithful replication
  (b) their data + our recipe  (1 epoch,  batch 16)    <- isolates epochs/batch
If (a) jumps toward ~1.0, the FIT DATA was the missing piece and we can self-fit all 11.
"""
import glob, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_probes import _released_weight_bias

LAYER, OFFSET = 12, 1
HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = os.environ.get("REFIT_CONCEPT", "german")
POS_T, NEG_T = 0.75, 0.25
CAP = int(os.environ.get("REFIT_CAP", "300"))  # per class


def released(probe_dir, concept):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[: -len("_weights.pt")].lower() == concept.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(concept)


def build_pos_neg(rated_json, concept):
    recs = json.loads(Path(rated_json).read_text("utf-8"))
    pos, neg = [], []
    for r in recs:
        resp = r.get("topical_response")
        ratings = r.get("topical_response_normalized_ratings") or {}
        if not resp or concept not in ratings:
            continue
        score = float(ratings[concept])
        pair = (r.get("prompt", ""), resp)
        if score >= POS_T:
            pos.append(pair)
        elif score <= NEG_T:
            neg.append(pair)
    return pos[:CAP], neg[:CAP]


def response_token_feats(model, tok, pairs):
    """Forward (prompt+response) under chat template; return per-response-token reps."""
    chunks = []
    for prompt, response in pairs:
        try:
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
            )
            full_ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                add_generation_prompt=False, tokenize=True,
            )
        except Exception:
            continue
        if len(full_ids) <= len(prompt_ids) or len(full_ids) > 1024:
            continue
        t = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            h = model(input_ids=t, output_hidden_states=True).hidden_states[LAYER + OFFSET]
        resp = h[0, len(prompt_ids):].float().cpu()  # response tokens only, with context
        if resp.shape[0] >= 1:
            chunks.append(resp)
    return chunks


def fit_logistic(X, y, *, lr, epochs, bs, shuffle):
    """Their TrainableMetric.fit: Linear+bias, Adam(lr,wd=1e-5,eps=1e-5), BCE."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-5, eps=1e-5)
    n = X.shape[0]
    g = torch.Generator().manual_seed(42)
    for _ in range(epochs):
        order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(X[idx] @ w + b, y[idx]).backward()
            opt.step()
    return w.detach(), b.detach()


def bal_acc(X, y, w, b):
    p = (torch.sigmoid(X @ w + b) >= 0.5).float()
    pos, neg = y.bool(), ~y.bool()
    return 0.5 * (p[pos].mean().item() + (1 - p[neg]).mean().item())


def cos(a, b):
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def main():
    probe_dir, rated_json = sys.argv[1], sys.argv[2]
    w_rel, b_rel = released(probe_dir, CONCEPT)
    cache = os.path.join(os.environ.get("NEXT_CHAMELEONS_ARTIFACT_ROOT",
                                        os.path.expandvars("$SCRATCH/next_chameleons_artifacts")),
                         f"refit_cache_{CONCEPT}_cap{CAP}.pt")
    if os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu")
        X, y = blob["X"], blob["y"]
        print(f"[refit] loaded cached features {tuple(X.shape)} from {cache}", flush=True)
    else:
        pos, neg = build_pos_neg(rated_json, CONCEPT)
        print(f"[refit] concept={CONCEPT} rated pos={len(pos)} neg={len(neg)} (cap {CAP})", flush=True)
        loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16")
        model, tok = loaded.model, loaded.tokenizer
        model.eval()
        pos_chunks = response_token_feats(model, tok, pos)
        neg_chunks = response_token_feats(model, tok, neg)
        X = torch.cat(pos_chunks + neg_chunks, 0)
        y = torch.cat([torch.ones(sum(c.shape[0] for c in pos_chunks)),
                       torch.zeros(sum(c.shape[0] for c in neg_chunks))])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"X": X, "y": y}, cache)
        print(f"[refit] extracted+cached features {tuple(X.shape)} -> {cache}", flush=True)
    print(f"[refit] per-token features: {X.shape[0]} tokens, pos_frac={y.mean():.3f}", flush=True)

    def pr(w):
        w2 = w.float() ** 2
        return (w2.sum() ** 2 / (w2 ** 2).sum()).item()

    pos_m, neg_m = X[y.bool()].mean(0), X[~y.bool()].mean(0)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)

    print(f"\ncosine-to-released-{CONCEPT}  (released PR={pr(w_rel):.0f}, |w_rel|={w_rel.norm():.3f}):")
    # ESTIMATOR FAMILY TEST — which one produces the released distributed direction?
    w_md = (pos_m - neg_m)
    print(f"  mean-diff (raw)              : cos={cos(w_md, w_rel):+.4f}  PR={pr(w_md):.0f}", flush=True)
    w_mds = ((X[y.bool()] - mu) / sd).mean(0) - ((X[~y.bool()] - mu) / sd).mean(0)
    print(f"  mean-diff (standardized)/sd : cos={cos(w_mds / sd, w_rel):+.4f}  PR={pr(w_mds / sd):.0f}", flush=True)
    print(f"  mean-diff (standardized)    : cos={cos(w_mds, w_rel):+.4f}  PR={pr(w_mds):.0f}", flush=True)
    # logistic on RAW features (their/our recipe)
    for tag, lr, ep, bs, sh in [
        ("logistic raw (5ep/64)      ", 4e-5, 5, 64, False),
        ("logistic raw (1ep/16)      ", 4e-5, 1, 16, True),
    ]:
        w, b = fit_logistic(X, y, lr=lr, epochs=ep, bs=bs, shuffle=sh)
        print(f"  {tag}: cos={cos(w, w_rel):+.4f}  bal_acc={bal_acc(X, y, w, b):.4f}  PR={pr(w):.0f}", flush=True)
    # logistic on STANDARDIZED features, folded back to raw space (w_s / sd)
    Xs = (X - mu) / sd
    w_s, b_s = fit_logistic(Xs, y, lr=4e-5, epochs=5, bs=64, shuffle=False)
    w_sf = w_s / sd
    print(f"  logistic STANDARDIZED/sd    : cos={cos(w_sf, w_rel):+.4f}  bal_acc={bal_acc(Xs, y, w_s, b_s):.4f}  PR={pr(w_sf):.0f}", flush=True)
    print(f"  logistic STANDARDIZED (raw w): cos={cos(w_s, w_rel):+.4f}  PR={pr(w_s):.0f}", flush=True)


if __name__ == "__main__":
    main()
