# ruff: noqa
"""Decisive off-by-one test. Their obf_reps stacks ALL hidden_states (emb at idx 0) and
selects reps[:, 12] = hidden_states[12] for "layer 12". OUR pipeline uses
hidden_states[12+OFFSET]=hidden_states[13]. If we've been one layer too deep, the released
probe was fit on hidden_states[12] and our self-fits (at 13) can never align.

For each hidden_states index in a sweep, on the authors' rated german pos/neg:
  - bal_acc of the RELEASED german probe (peaks at the layer it was fit on)
  - cosine-to-released of a FRESH logistic fit at that index (peaks at the true layer)
If both spike at index 12 (not 13) -> off-by-one confirmed, fixable via hidden_states_offset=0.
"""
import glob, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_probes import _released_weight_bias

HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = os.environ.get("REFIT_CONCEPT", "german")
POS_T, NEG_T = 0.75, 0.25
CAP = int(os.environ.get("REFIT_CAP", "200"))
INDICES = [10, 11, 12, 13, 14, 15]


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
        resp, ratings = r.get("topical_response"), (r.get("topical_response_normalized_ratings") or {})
        if not resp or concept not in ratings:
            continue
        score, pair = float(ratings[concept]), (r.get("prompt", ""), resp)
        if score >= POS_T:
            pos.append(pair)
        elif score <= NEG_T:
            neg.append(pair)
    return pos[:CAP], neg[:CAP]


def fit_logistic(X, y, lr=4e-5, epochs=2, bs=64):
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-5, eps=1e-5)
    for _ in range(epochs):
        for s in range(0, X.shape[0], bs):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(X[s:s+bs] @ w + b, y[s:s+bs]).backward(); opt.step()
    return w.detach(), b.detach()


def bal_acc_scores(scores, y, thr):
    p = (scores >= thr).float(); pos, neg = y.bool(), ~y.bool()
    return 0.5 * (p[pos].mean().item() + (1 - p[neg]).mean().item())


def best_bal_acc(scores, y):
    best = 0.0
    for q in torch.linspace(0.05, 0.95, 19):
        best = max(best, bal_acc_scores(scores, y, torch.quantile(scores, q).item()))
    return best


def cos(a, b):
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def main():
    probe_dir, rated_json = sys.argv[1], sys.argv[2]
    w_rel, b_rel = released(probe_dir, CONCEPT)
    pos, neg = build_pos_neg(rated_json, CONCEPT)
    print(f"[layer] concept={CONCEPT} pos={len(pos)} neg={len(neg)} indices={INDICES}", flush=True)
    loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16"); model, tok = loaded.model, loaded.tokenizer
    model.eval()

    feats = {i: [] for i in INDICES}; labels = []
    for label, pool in ((1.0, pos), (0.0, neg)):
        for prompt, response in pool:
            try:
                pids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
                fids = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], add_generation_prompt=False, tokenize=True)
            except Exception:
                continue
            if len(fids) <= len(pids) or len(fids) > 1024:
                continue
            t = torch.tensor([fids], device=model.device)
            with torch.no_grad():
                hs = model(input_ids=t, output_hidden_states=True).hidden_states
            n = len(fids) - len(pids)
            for i in INDICES:
                feats[i].append(hs[i][0, len(pids):].float().cpu())
            labels.extend([label] * n)
    y = torch.tensor(labels)
    print(f"[layer] tokens={y.shape[0]} pos_frac={y.mean():.3f}\n", flush=True)

    def pr(w): w2 = w.float() ** 2; return (w2.sum() ** 2 / (w2 ** 2).sum()).item()
    print(f"released german PR={pr(w_rel):.0f}.  hidden_states index sweep:")
    print(f"  {'idx':>4} {'released_balacc':>16} {'fit_cos_to_rel':>15} {'fit_PR':>7}   note")
    for i in INDICES:
        X = torch.cat(feats[i], 0)
        rel_scores = torch.sigmoid(X @ w_rel + b_rel)
        rba = best_bal_acc(rel_scores, y)
        w_fit, _ = fit_logistic(X, y)
        note = "<- OURS (offset 1)" if i == 13 else ("<- THEIRS (hs[12])" if i == 12 else "")
        print(f"  {i:>4} {rba:>16.4f} {cos(w_fit, w_rel):>15.4f} {pr(w_fit):>7.0f}   {note}", flush=True)


if __name__ == "__main__":
    main()
