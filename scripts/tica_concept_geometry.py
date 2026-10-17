"""Answer-format concept geometry + base-probe cache for the final multi-concept dodger
(TICA-FINAL-DODGER-DESIGN.md). Aggregates only; texts are never printed or saved.

--geometry : base model, answer format WITHOUT the brief (so every concept shares one set of reads),
             FIT texts of every pool concept + concept-free negatives at L12. Per concept: mean-diff
             direction (mean_c - mean of the other concepts' means) / pooled SD, and a held-out
             check (AUROC of that direction on the concept's CALIB texts vs the other concepts'
             calib texts).
             Writes <out>/geometry.json + dirs.npy (the input of tica_concept_split.py --geometry)
             and, if --split is given, the agreement of the split with this geometry
             (held-out vs nearest-train cosine, far clusters still far).
--probes   : fit_answer_probes (brief + no-brief, CV gate, cached in answer_probe_cache) for the
             split's concepts at L12 + held-out L20/L24; shard with --shard i/n. Reports gate
             failures.

  python scripts/tica_concept_geometry.py --config final_r1_4b --geometry --split S --out DIR
  python scripts/tica_concept_geometry.py --config final_r1_4b --probes --split S --shard 0/4 \
      --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config


def auroc(pos, neg):
    import torch

    s = torch.cat([pos, neg])
    r = s.argsort().argsort().float() + 1
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def geometry(a, cfg, data, pol, opening):
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tica_concept_split import EXCLUDE

    L = cfg.primary_layer()
    fit = {c.concept: c.texts for c in data.concepts("fit", ("train", "heldout"))}
    cal = {c.concept: c.texts for c in data.concepts("calib", ("train", "heldout"))}
    names = [c for c in fit if not EXCLUDE.search(c) and len(fit[c]) >= 65]

    def reads(texts):
        convs = [[{"role": "user", "content": opening("nobrief", x, None)},
                  {"role": "assistant", "content": x}] for x in texts]
        with pol.reference(), torch.no_grad():
            return pol.pooled(convs, [L])[L].float().cpu()

    t0 = time.time()
    flat = [(c, x) for c in names for x in fit[c]]
    R = reads([x for _, x in flat])
    owner = np.array([names.index(c) for c, _ in flat])
    flatc = [(c, x) for c in names for x in cal[c]]
    Rc = reads([x for _, x in flatc])
    ownc = np.array([names.index(c) for c, _ in flatc])
    print(f"reads {len(flat) + len(flatc)} in {time.time() - t0:.0f}s", flush=True)
    means = torch.stack([R[torch.from_numpy(owner == i)].mean(0) for i in range(len(names))])
    sd = R.std(0).clamp(min=1e-6)
    dirs, held = [], []
    for i in range(len(names)):
        others = torch.cat([means[:i], means[i + 1 :]]).mean(0)
        w = (means[i] - others) / sd ** 2  # mean-diff in the SD-whitened space, as W / sd
        dirs.append((w * sd).numpy())  # stored like the bank's W / sd (direction in z-space)
        s = Rc @ w
        held.append(auroc(s[torch.from_numpy(ownc == i)], s[torch.from_numpy(ownc != i)]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dirs.npy", np.stack(dirs))
    gate = [h >= 0.9 for h in held]
    g = {"concepts": names, "dirs": str(out / "dirs.npy"), "gate": gate, "layer": L,
         "calib_auroc": held, "format": "answer, no brief, mean-diff / SD"}
    (out / "geometry.json").write_text(json.dumps(g))
    rep = {"n": len(names), "gate_pass": sum(gate),
           "calib_auroc_median": float(np.median(held))}
    if a.split:  # does the split (chosen on repeat-format geometry) hold in the answer format?
        sp = json.loads(Path(a.split).read_text())
        D = np.stack(dirs)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        ix = {n: i for i, n in enumerate(names)}
        tr = [ix[n] for n in sp["train"] if n in ix]

        def near(n):
            return float((D[tr] @ D[ix[n]]).max()) if n in ix else None

        S = D @ D.T
        np.fill_diagonal(S, -1)
        rep["split_check"] = {
            "pool_nn_cos_p90": float(np.percentile(S.max(1), 90)),
            "heldout_cos_nearest_train": {n[:60]: near(n) for n in sp["heldout"]},
            "far_cos_nearest_train": {n[:60]: near(n) for n in sp["heldout_far"]},
            "math_cos": {n[:60]: float(D[ix[n]] @ D[ix["mathematical notation and symbols"]])
                         for n in sp["heldout"] + sp["heldout_far"] if n in ix},
        }
    (out / "geometry_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:3000], flush=True)


def probes(a, cfg, data, pol, opening):
    sp = json.loads(Path(a.split).read_text())
    want = list(dict.fromkeys(sp["train"] + sp["heldout"] + sp["heldout_far"]))
    i, n = (int(x) for x in a.shard.split("/"))
    want = want[i::n]
    held = [int(x) for x in cfg.probes.held_out_layers]
    layers = [cfg.primary_layer()] + [h for h in held if h != cfg.primary_layer()]
    rep = {}
    for c in want:
        t0 = time.time()
        try:
            _, q = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening)
            rep[c] = {"ok": True, "quality": q}
        except RuntimeError as e:  # gate failure: replace from the same cluster / type
            rep[c] = {"ok": False, "error": str(e)[:300]}
        rep[c]["sec"] = round(time.time() - t0, 1)
        print(c[:60], rep[c]["ok"], rep[c]["sec"], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"probes_shard{i}of{n}.json").write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--split")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--read-batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(None, False)
    pol.batch_size = a.read_batch  # no-grad reads only
    opening = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)[0]
    if a.geometry:
        geometry(a, cfg, data, pol, opening)
    if a.probes:
        probes(a, cfg, data, pol, opening)


if __name__ == "__main__":
    main()
