"""Diagnostic: is the RELEASED german probe a valid detector on OUR activations,
and does chat-templating vs raw extraction explain the 0.15 cosine gap?

For each extraction regime (raw concat vs chat-templated), we:
  1. extract layer-12 (hidden_states[13]) response-token features on the frozen
     abliterated base for german (positive) vs other-concept (negative) examples,
  2. score the RELEASED german probe per-token -> balanced accuracy,
  3. fit OUR own probe on the same features -> balanced accuracy + cosine to released.

If the released probe classifies our chat-templated activations well (~0.95) but our
raw ones poorly, the systematic direction gap is an extraction-convention mismatch.
If it classifies BOTH well yet cosine stays ~0.15, the gap is the contrast set/corpus
(many valid german separators), confirming "accuracy does not pin direction".
"""

from __future__ import annotations

import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.paper_data import (
    find_preset_dataset,
    preset_concept_examples,
)
from next_chameleons.hf.paper_probes import (
    _released_weight_bias,
    fit_linear_probe_per_token,
)

LAYER = 12
OFFSET = 1
HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = "german"
PER_CLASS = 48  # examples per class (positive german / pooled negatives)


def released_german(probe_dir: str):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[: -len("_weights.pt")].lower() == CONCEPT:
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(f"no released {CONCEPT} probe in {probe_dir}")


def response_token_features(model, tokenizer, examples, *, chat: bool):
    """Return [n_tokens, hidden] features over response tokens + per-token labels."""
    feats, labels = [], []
    device = model.device
    for ex in examples:
        if chat:
            prefix_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex.prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.response},
                ],
                add_generation_prompt=False,
                tokenize=True,
            )
        else:
            prefix_text = f"{ex.prompt}\n" if ex.prompt.strip() else ""
            prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(
                prefix_text + ex.response, add_special_tokens=True
            )["input_ids"]
        prefix_len = len(prefix_ids)
        if len(full_ids) <= prefix_len:
            continue
        ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        hid = out.hidden_states[LAYER + OFFSET][0]  # [seq, hidden]
        resp = hid[prefix_len:].float().cpu()
        feats.append(resp)
        labels.extend([1.0 if ex.concept == CONCEPT else 0.0] * resp.shape[0])
    X = torch.cat(feats, dim=0)
    y = torch.tensor(labels)
    return X, y


def balanced_acc(scores, y, threshold):
    preds = (scores >= threshold).float()
    pos, neg = y.bool(), ~y.bool()
    pr = preds[pos].mean().item() if pos.any() else float("nan")
    nr = (1.0 - preds[neg]).mean().item() if neg.any() else float("nan")
    return 0.5 * (pr + nr), pr, nr


def main():
    probe_dir = sys.argv[1]
    raw_cache_root = sys.argv[2]
    from pathlib import Path

    preset = find_preset_dataset(Path(raw_cache_root))
    if preset is None:
        raise FileNotFoundError(f"no preset under {raw_cache_root}")
    all_ex = preset_concept_examples(preset)
    pos = [e for e in all_ex if e.concept == CONCEPT][:PER_CLASS]
    negs = [e for e in all_ex if e.concept != CONCEPT][: PER_CLASS]
    examples = pos + negs
    print(f"[diag] {len(pos)} german + {len(negs)} negative examples", flush=True)

    loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16")
    model, tokenizer = loaded.model, loaded.tokenizer
    model.eval()

    w_rel, b_rel = released_german(probe_dir)
    print(f"[diag] released german probe loaded: |w|={w_rel.norm():.3f} b={b_rel:.3f}", flush=True)

    for chat in (False, True):
        regime = "CHAT_TEMPLATED" if chat else "RAW_CONCAT"
        X, y = response_token_features(model, tokenizer, examples, chat=chat)
        print(f"\n===== {regime} =====  tokens={X.shape[0]} pos_frac={y.mean():.3f}")
        # released probe scoring (per-token sigmoid); threshold at the released bias
        rel_scores = torch.sigmoid(X @ w_rel + b_rel)
        # choose threshold = the 99th percentile of negatives (1% FPR) AND 0.5 for ref
        for thr_name, thr in (("0.5", 0.5),):
            ba, pr, nr = balanced_acc(rel_scores, y, thr)
            print(f"  RELEASED german probe  thr={thr_name}: balanced_acc={ba:.4f} pos_recall={pr:.4f} neg_recall={nr:.4f}")
        # also report best-threshold balanced accuracy (does direction separate at all?)
        order = torch.argsort(rel_scores)
        best = 0.0
        for q in torch.linspace(0.01, 0.99, 99):
            t = torch.quantile(rel_scores, q).item()
            ba, _, _ = balanced_acc(rel_scores, y, t)
            best = max(best, ba)
        print(f"  RELEASED german probe  BEST-threshold balanced_acc={best:.4f}  (does the direction separate our data?)")
        # our own fit on these features
        w_self, b_self = fit_linear_probe_per_token(
            token_features=X, token_labels=y, learning_rate=4e-5, epochs=1, batch_size=16
        )
        self_scores = torch.sigmoid(X @ w_self + b_self)
        ba_s, _, _ = balanced_acc(self_scores, y, 0.5)
        cos = F.cosine_similarity(w_self.float().unsqueeze(0), w_rel.unsqueeze(0)).item()
        print(f"  OUR self-fit (paper hp) balanced_acc={ba_s:.4f}  cosine_to_released={cos:+.4f}")


if __name__ == "__main__":
    main()
