"""Fitting-METHOD sweep: contrast set is ruled out (all ~0.17). Does a different
fit METHOD recover the released german direction? Tests, on identical german-pos vs
data_Y-neg features:
  - our Adam/BCE 1-epoch (baseline ~0.17)
  - class mean-difference direction (the 'naive' concept axis)
  - standardized-feature fit (z-score, weight mapped back to raw space)
  - strong-L2 logistic (sklearn-like, convergent)
  - converged Adam (overfit reference, expect lower)

Saves the extracted features to $SCRATCH so further method variations cost zero GPU
(reload on the login node). If NO method recovers cosine, the gap is the base-model
variant, not our fitting — and loading the shipped probes is the only faithful route.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_data import find_preset_dataset
from next_chameleons.hf.paper_probes import _released_weight_bias

LAYER, OFFSET = 12, 1
HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = "german"
CAP = 80


def released(probe_dir, concept):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[: -len("_weights.pt")].lower() == concept.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(concept)


def feats(model, tok, responses):
    chunks = []
    for resp in responses:
        ids = tok(resp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        if len(ids) <= 1:
            continue
        t = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            o = model(input_ids=t, output_hidden_states=True)
        chunks.append(o.hidden_states[LAYER + OFFSET][0, 1:].float().cpu())
    return torch.cat(chunks, dim=0)


def bal_acc(scores, y, thr=0.5):
    p = (scores >= thr).float()
    pos, neg = y.bool(), ~y.bool()
    pr = p[pos].mean().item() if pos.any() else 0.0
    nr = (1.0 - p[neg]).mean().item() if neg.any() else 0.0
    return 0.5 * (pr + nr)


def adam_bce(X, y, *, lr, epochs, wd, seed=17):
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd, eps=1e-5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(X.shape[0], generator=g)
        for s in range(0, X.shape[0], 16):
            idx = order[s : s + 16]
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(X[idx] @ w + b, y[idx])
            loss.backward()
            opt.step()
    return w.detach(), b.detach()


def report(name, w, b, X, y, w_rel, b_rel):
    cos = F.cosine_similarity(w.float().unsqueeze(0), w_rel.unsqueeze(0)).item()
    acc = bal_acc(torch.sigmoid(X @ w + b), y)
    print(f"  {name:34s} self_acc={acc:.4f}  COSINE_to_released={cos:+.4f}", flush=True)
    return cos


def main():
    probe_dir, raw_cache = sys.argv[1], sys.argv[2]
    save = sys.argv[3] if len(sys.argv) > 3 else None
    recs = json.loads(Path(find_preset_dataset(Path(raw_cache))).read_text("utf-8"))
    pn = lambda r: str(r.get("probe_name", "")).lower()
    dt = lambda r: str(r.get("data_type", ""))
    pos = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_X_probe_X")][:CAP]
    neg = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_Y_probe_X")
           and float(r["target_probe_score"]) < 0.5][:CAP]
    print(f"[fit] german pos={len(pos)} data_Y neg={len(neg)}", flush=True)

    loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16")
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    Xp, Xn = feats(model, tok, pos), feats(model, tok, neg)
    X = torch.cat([Xp, Xn]); y = torch.cat([torch.ones(Xp.shape[0]), torch.zeros(Xn.shape[0])])
    w_rel, b_rel = released(probe_dir, CONCEPT)
    print(f"[fit] tokens pos={Xp.shape[0]} neg={Xn.shape[0]} |w_rel|={w_rel.norm():.3f}\n", flush=True)
    if save:
        torch.save({"X": X, "y": y, "w_rel": w_rel, "b_rel": b_rel}, save)
        print(f"[fit] saved features -> {save}\n", flush=True)

    print("fit-method sweep (german pos vs data_Y neg):")
    # 1. baseline
    w, b = adam_bce(X, y, lr=4e-5, epochs=1, wd=1e-5); report("adam_bce_1ep (baseline)", w, b, X, y, w_rel, b_rel)
    # 2. mean-difference direction
    wmd = (Xp.mean(0) - Xn.mean(0)); bmd = torch.tensor(0.0)
    report("mean_difference", wmd, bmd, X, y, w_rel, b_rel)
    # 3. standardized features -> map weight back to raw space
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    ws, bs = adam_bce(Xs, y, lr=1e-3, epochs=10, wd=1e-4)
    w_raw = ws / sd  # so that Xs@ws == X@w_raw - const
    report("standardized_fit (->raw)", w_raw, torch.tensor(0.0), X, y, w_rel, b_rel)
    # 4. strong-L2 convergent logistic
    w, b = adam_bce(X, y, lr=1e-3, epochs=30, wd=1e-1); report("strongL2_converged", w, b, X, y, w_rel, b_rel)
    # 5. converged low-reg (overfit ref)
    w, b = adam_bce(X, y, lr=1e-3, epochs=30, wd=1e-5); report("converged_lowreg", w, b, X, y, w_rel, b_rel)
    # also: does the released probe align with the mean-difference?
    cos_rel_md = F.cosine_similarity(w_rel.unsqueeze(0), wmd.float().unsqueeze(0)).item()
    print(f"\n[fit] cosine(released, mean_difference) = {cos_rel_md:+.4f}")
    print(f"[fit] released probe self balanced_acc on this data = {bal_acc(torch.sigmoid(X@w_rel+b_rel), y):.4f}")


if __name__ == "__main__":
    main()
