"""Contrast-set sweep: does fitting the german probe against the PAPER'S negatives
(neutral / UltraChat content, from the preset's own target_probe_score labels) recover
the released probe direction better than our one-vs-rest-concepts negatives?

Same positives (german content) throughout; we vary ONLY the negative pool and report
balanced accuracy + cosine-to-released for each. If a preset-native contrast pushes
cosine from ~0.17 toward the released direction, that localizes the gap to the contrast
set and tells us exactly how to fix our probe-fit. Cheap (activation-only); run a full
train+eval only if a contrast actually moves the cosine.
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
from next_chameleons.hf.paper_probes import (
    _released_weight_bias,
    fit_linear_probe_per_token,
)

LAYER, OFFSET = 12, 1
HF_ID = "IlyaGusev/gemma-2-9b-it-abliterated"
CONCEPT = "german"
CAP = 68  # cap per pool (german has 68 clue_N_data_X positives)


def released(probe_dir, concept):
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[: -len("_weights.pt")].lower() == concept.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            return w.float().reshape(-1), float(b)
    raise FileNotFoundError(concept)


def feats_for_responses(model, tokenizer, responses):
    """Layer-12 features over all response tokens (response-only, matches training)."""
    out_feats = []
    counts = []
    for resp in responses:
        ids = tokenizer(resp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
        if len(ids) <= 1:
            counts.append(0)
            continue
        t = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            o = model(input_ids=t, output_hidden_states=True)
        h = o.hidden_states[LAYER + OFFSET][0, 1:].float().cpu()  # drop BOS
        out_feats.append(h)
        counts.append(h.shape[0])
    return out_feats, counts


def stack(feat_list):
    return torch.cat(feat_list, dim=0) if feat_list else torch.empty(0)


def balanced_acc(scores, y, thr=0.5):
    preds = (scores >= thr).float()
    pos, neg = y.bool(), ~y.bool()
    pr = preds[pos].mean().item() if pos.any() else float("nan")
    nr = (1.0 - preds[neg]).mean().item() if neg.any() else float("nan")
    return 0.5 * (pr + nr)


def fit_and_report(name, Xpos, Xneg, w_rel, b_rel):
    X = torch.cat([Xpos, Xneg], dim=0)
    y = torch.cat([torch.ones(Xpos.shape[0]), torch.zeros(Xneg.shape[0])])
    w, b = fit_linear_probe_per_token(
        token_features=X, token_labels=y, learning_rate=4e-5, epochs=1, batch_size=16
    )
    self_scores = torch.sigmoid(X @ w + b)
    ba = balanced_acc(self_scores, y)
    rel_scores = torch.sigmoid(X @ w_rel + b_rel)
    ba_rel = balanced_acc(rel_scores, y)
    cos = F.cosine_similarity(w.float().unsqueeze(0), w_rel.unsqueeze(0)).item()
    print(
        f"  {name:32s} pos_tok={Xpos.shape[0]:5d} neg_tok={Xneg.shape[0]:5d} "
        f"self_acc={ba:.4f} released_acc={ba_rel:.4f} COSINE_to_released={cos:+.4f}",
        flush=True,
    )
    return cos


def main():
    probe_dir, raw_cache = sys.argv[1], sys.argv[2]
    preset = find_preset_dataset(Path(raw_cache))
    recs = json.loads(Path(preset).read_text(encoding="utf-8"))

    def pick(pred, cap=CAP):
        out = [r["response"] for r in recs if pred(r)]
        return out[:cap]

    cl = lambda r: str(r.get("data_type", ""))
    pn = lambda r: str(r.get("probe_name", "")).lower()

    # POSITIVES: german content, no clue (the canonical clean positive)
    pos_german = pick(lambda r: pn(r) == CONCEPT and cl(r).startswith("clue_N_data_X_probe_X"))

    # NEGATIVE POOLS (all no-clue so they are clean / untriggered):
    neg_paper_all = pick(  # paper's own german-probe negatives: unrelated content, score~0
        lambda r: pn(r) == CONCEPT and cl(r).startswith("clue_N_data_Y_probe_X")
        and float(r["target_probe_score"]) < 0.5
    )
    neg_paper_ultrachat = pick(
        lambda r: pn(r) == CONCEPT and "clue_N_data_Y" in cl(r) and "ultrachat" in cl(r)
        and float(r["target_probe_score"]) < 0.5
    )
    neg_paper_topicalY = pick(
        lambda r: pn(r) == CONCEPT and "clue_N_data_Y" in cl(r) and "topical" in cl(r)
        and float(r["target_probe_score"]) < 0.5
    )
    # OUR one-vs-rest: other concepts' clean topical positives (stylized non-german text)
    neg_onevsrest = pick(
        lambda r: pn(r) != CONCEPT and cl(r).startswith("clue_N_data_X_probe_X"), cap=CAP
    )

    print("[sweep] pool sizes (examples):")
    for nm, pool in [
        ("POS german (clue_N_data_X)", pos_german),
        ("NEG paper-all (data_Y)", neg_paper_all),
        ("NEG paper-ultrachat", neg_paper_ultrachat),
        ("NEG paper-topicalY", neg_paper_topicalY),
        ("NEG one-vs-rest concepts", neg_onevsrest),
    ]:
        print(f"  {nm:32s} {len(pool)}")

    loaded = load_causal_lm(hf_id=HF_ID, dtype="bfloat16")
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    w_rel, b_rel = released(probe_dir, CONCEPT)
    print(f"[sweep] released german probe |w|={w_rel.norm():.3f}\n", flush=True)

    def feats(pool):
        fl, _ = feats_for_responses(model, tok, pool)
        return stack(fl)

    Xpos = feats(pos_german)
    pools = {
        "ours: one-vs-rest concepts": feats(neg_onevsrest),
        "paper: all data_Y negatives": feats(neg_paper_all),
        "paper: ultrachat-only neg": feats(neg_paper_ultrachat),
        "paper: topical-Y-only neg": feats(neg_paper_topicalY),
    }
    print("contrast sweep (same german positives, varying negatives):")
    results = {nm: fit_and_report(nm, Xpos, Xneg, w_rel, b_rel) for nm, Xneg in pools.items()}
    best = max(results, key=results.get)
    print(f"\n[sweep] best contrast for cosine-to-released: {best!r} = {results[best]:+.4f}")
    print("[sweep] baseline (our one-vs-rest) =", f"{results['ours: one-vs-rest concepts']:+.4f}")


if __name__ == "__main__":
    main()
