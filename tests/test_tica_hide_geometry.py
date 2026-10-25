"""tica_hide_geometry, synthetic paired dumps: a planted shared hide vector applied on hiding turns
only must be recovered (large shift, own-concept suppression, anti-alignment, pairing checks)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_hide_geometry import analyse, load


def _dump(path, x, meta, layers):
    torch.save(
        {"layers": layers, "x": x.half(), "meta": meta, "adapter": None, "base": False}, path
    )


def test_planted_hide_vector(tmp_path):
    g = torch.Generator().manual_seed(0)
    d, layers = 32, [8, 12]
    concepts = ["a", "b", "c", "e"]
    cdir = {c: torch.randn(d, generator=g) * 3 for c in concepts}
    wdir = {c: torch.randn(d, generator=g) * 3 for c in concepts}  # where the concept is re-encoded
    v = torch.randn(d, generator=g)
    v = 6 * v / v.norm()
    meta, xb = [], []
    for ep in range(16):
        c = concepts[ep % 4]
        cond = "real_score" if ep % 2 == 0 else "nobrief"
        for t in range(1, 7):
            meta.append(
                {
                    "ep": ep,
                    "turn": t,
                    "cond": cond,
                    "off": False,
                    "concept": c,
                    "text_concept": c,
                    "split": "train",
                    "detected": t % 3 == 0,
                }
            )
            xb.append(torch.stack([cdir[c] + torch.randn(d, generator=g) for _ in layers]))
    xb = torch.stack(xb)
    xt = xb.clone()
    for i, m in enumerate(meta):
        if m["cond"] == "real_score":
            c = m["concept"]  # push + remove base concept dir + re-encode elsewhere
            xt[i] += v - (xt[i] @ cdir[c] / cdir[c].norm() ** 2)[:, None] * cdir[c] + wdir[c]
    _dump(tmp_path / "base_train_s1.pt", xb, meta, layers)
    _dump(tmp_path / "m_train_s1.pt", xt, meta, layers)
    lb, Xb, Mb = load([str(tmp_path / "base_train_s1.pt")])
    _, Xt, Mt = load([str(tmp_path / "m_train_s1.pt")])
    r = analyse(lb, Xb, Mb, Xt, Mt, min_conc=3)
    assert r["n_paired"] == len(meta) and r["n_unpaired"] == 0
    L = r["layers"][8]
    assert L["shift"] > 0.3
    assert L["supp"] < 0.3  # own-concept signal mostly removed along the base direction
    assert L["anti"] > 0.2  # per-concept shifts point against the own concept direction
    assert L["fresh_auc"] > 0.9 and L["base_auc"] < L["fresh_auc"]  # rotation, not erasure
    # no-brief records untouched: D_N = 0
    assert abs(r["n_N"] - 48) == 0


def test_unpaired_dropped(tmp_path):
    layers = [4]
    meta = [
        {
            "ep": 0,
            "turn": t,
            "cond": "real_score",
            "off": False,
            "concept": "a",
            "text_concept": "a",
            "split": "train",
            "detected": False,
        }
        for t in range(1, 5)
    ]
    x = torch.randn(4, 1, 8)
    _dump(tmp_path / "b_s1.pt", x, meta, layers)
    meta2 = [dict(m, ep=1) for m in meta]
    _dump(tmp_path / "t_s1.pt", x, meta2, layers)
    lb, Xb, Mb = load([str(tmp_path / "b_s1.pt")])
    _, Xt, Mt = load([str(tmp_path / "t_s1.pt")])
    import pytest

    with pytest.raises(SystemExit):
        analyse(lb, Xb, Mb, Xt, Mt)
