"""Disambiguate the basis mismatch: is the released german probe foreign because of a
LAYER/index off-by-one, or a BASE-MODEL variant difference?

One forward pass exposes all hidden_states. For each candidate index we score the
released german probe on german-pos vs data_Y-neg tokens and report balanced accuracy.
- If accuracy PEAKS sharply at some index != 13 (our layer-12 + offset-1 convention),
  we have a layer bug and that index is the fix.
- If accuracy is ~0.80 and flat / peaks weakly right at 13, the index is correct and the
  residual gap is the base-model variant (released probes fit on a different gemma-2-9b
  residual stream) — loading the shipped probes is then the only faithful route.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_data import find_preset_dataset
from next_chameleons.hf.paper_probes import _released_weight_bias

CONCEPT = "german"
CAP = 60
INDICES = list(range(8, 20))  # hidden_states indices to probe (13 = our convention)


def released(probe_dir, concept):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[: -len("_weights.pt")].lower() == concept.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(concept)


def bal_acc(scores, y, thr=0.5):
    p = (scores >= thr).float()
    pos, neg = y.bool(), ~y.bool()
    pr = p[pos].mean().item() if pos.any() else 0.0
    nr = (1.0 - p[neg]).mean().item() if neg.any() else 0.0
    return 0.5 * (pr + nr)


def best_acc(scores, y):
    best = 0.0
    for q in torch.linspace(0.02, 0.98, 49):
        t = torch.quantile(scores, q).item()
        best = max(best, bal_acc(scores, y, t))
    return best


def main():
    probe_dir, raw_cache = sys.argv[1], sys.argv[2]
    recs = json.loads(Path(find_preset_dataset(Path(raw_cache))).read_text("utf-8"))
    pn = lambda r: str(r.get("probe_name", "")).lower()
    dt = lambda r: str(r.get("data_type", ""))
    pos = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_X_probe_X")][:CAP]
    neg = [r["response"] for r in recs if pn(r) == CONCEPT and dt(r).startswith("clue_N_data_Y_probe_X")
           and float(r["target_probe_score"]) < 0.5][:CAP]
    print(f"[layer] german pos={len(pos)} data_Y neg={len(neg)} indices={INDICES}", flush=True)

    loaded = load_causal_lm(hf_id="IlyaGusev/gemma-2-9b-it-abliterated", dtype="bfloat16")
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    w_rel, b_rel = released(probe_dir, CONCEPT)

    # accumulate per-index token features
    feats = {i: [] for i in INDICES}
    labels = []
    for label, pool in ((1.0, pos), (0.0, neg)):
        for resp in pool:
            ids = tok(resp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            if len(ids) <= 1:
                continue
            t = torch.tensor([ids], device=model.device)
            with torch.no_grad():
                o = model(input_ids=t, output_hidden_states=True)
            n = len(ids) - 1
            for i in INDICES:
                feats[i].append(o.hidden_states[i][0, 1:].float().cpu())
            labels.extend([label] * n)
    y = torch.tensor(labels)
    print(f"[layer] tokens={y.shape[0]} pos_frac={y.mean():.3f}\n", flush=True)
    print("released german probe accuracy vs hidden_states index:")
    for i in INDICES:
        X = torch.cat(feats[i], dim=0)
        s = torch.sigmoid(X @ w_rel + b_rel)
        marker = "  <- our convention (layer12+offset1)" if i == 13 else ""
        print(f"  hidden_states[{i:2d}]  bal_acc@0.5={bal_acc(s, y):.4f}  best_thr={best_acc(s, y):.4f}{marker}", flush=True)


if __name__ == "__main__":
    main()
