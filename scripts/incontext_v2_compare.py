"""Validity bar V2 (PREREGISTRATION-ICA §6): our harness vs Kowalski et al.'s recorder.

It compares two things:
1. Concept vectors: our vectors.npz against their results/vector_cache/<model>_layer{L}_baseline.pt.
   The bar is cosine >= 0.9999.
2. Per-trial proj readouts on the neutral sentences s0-s49:
   - ours K1/K2 against their think_about / dont_think_about rows;
   - ours K0 against their no_instruction baseline cache.
   The bar is Pearson >= 0.99 at every compared layer.
   Their proj is the token mean of cos * ||r|| over the aligned span, i.e. mean(r . v / ||v||).

Usage (CPU):
    uv run python scripts/incontext_v2_compare.py --ours $ART/incontext/l0_gemma2_9b \
        --theirs-run /scratch/sneheel/external/activation-control/results/raw/<run> \
        --vector-cache /scratch/sneheel/external/activation-control/results/vector_cache \
        --battery $RAW/incontext/l0_battery.json --output $ART/incontext/v2_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

COND = {"K1": "think_about", "K2": "dont_think_about"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--theirs-run", type=Path, required=True)
    ap.add_argument("--vector-cache", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--model", default="gemma2_9b")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    ours_vec = np.load(args.ours / "vectors.npz")
    layers = sorted({int(k.split("|")[0]) for k in ours_vec.files})
    vec_cos, theirs_vec = {}, {}
    for layer in layers:
        path = args.vector_cache / f"{args.model}_layer{layer}_baseline.pt"
        if not path.exists():
            continue
        d = torch.load(path, weights_only=False)
        theirs_vec[layer] = {c: v.float().numpy() for c, v in d.items()}
        for c, v in theirs_vec[layer].items():
            o = ours_vec[f"{layer}|{c}"]
            vec_cos[f"{layer}|{c}"] = float(o @ v / (np.linalg.norm(o) * np.linalg.norm(v)))

    text = {r["id"]: r["text"] for r in json.loads(args.battery.read_text())["neutral"]["primary"]}
    ours = [json.loads(line) for line in (args.ours / "trials_kowalski.jsonl").open()]
    ours = [r for r in ours if r["kind"] == "neutral" and r["compliant"] and r["readout"]]

    rows = json.loads((args.theirs_run / "results.json").read_text())["results"]
    theirs = {(r["condition_id"], r["concept"], r["sentence"]): r
              for r in rows if r["is_compliant"]}
    with open(args.theirs_run / "no_instruction_cache.pkl", "rb") as fh:
        base = pickle.load(fh)

    def their_proj(row, layer, concept):
        cos = np.asarray(row["cosine_sim"][str(layer)] if str(layer) in row["cosine_sim"]
                         else row["cosine_sim"][layer], np.float32)
        nrm = np.asarray(row["norms"][str(layer)] if str(layer) in row["norms"]
                         else row["norms"][layer], np.float32)
        if cos.ndim == 2:  # per token x concept
            cos = cos[:, sorted(theirs_vec[layer]).index(concept)]
        return float(np.mean(cos[: len(nrm)] * nrm[: len(cos)]))

    def their_base(sentence, layer, concept):
        ent = base.get(sentence)
        if ent is None:
            return None
        acts = np.asarray(ent["activations"][layer], np.float32)
        v = theirs_vec[layer][concept]
        return float(np.mean(acts @ v / np.linalg.norm(v)))

    per_layer = {}
    for layer in layers:
        if layer not in theirs_vec:
            continue
        a, b = [], []
        for r in ours:
            s = text[r["unit"]]
            concepts = [r["concept"]] if r["cond"] in COND else list(r["readout"])
            for c in concepts:
                o = r["readout"][c][str(layer)][0]
                if r["cond"] in COND:
                    row = theirs.get((COND[r["cond"]], c, s))
                    t = their_proj(row, layer, c) if row else None
                else:
                    t = their_base(s, layer, c)
                if t is not None and o is not None and np.isfinite(o):
                    a.append(o)
                    b.append(t)
        if len(a) > 2:
            a, b = np.array(a), np.array(b)
            per_layer[layer] = {"n": len(a), "pearson": float(np.corrcoef(a, b)[0, 1]),
                                "mean_abs_diff": float(np.mean(np.abs(a - b))),
                                "sd_theirs": float(np.std(b))}
    report = {
        "vector_cosine_min": min(vec_cos.values()) if vec_cos else None,
        "vector_cosine": vec_cos,
        "readout_by_layer": per_layer,
        "V2_vectors_pass": bool(vec_cos) and min(vec_cos.values()) >= 0.9999,
        "V2_readouts_pass": bool(per_layer)
        and min(v["pearson"] for v in per_layer.values()) >= 0.99,
    }
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "vector_cosine"}, indent=1))


if __name__ == "__main__":
    main()
