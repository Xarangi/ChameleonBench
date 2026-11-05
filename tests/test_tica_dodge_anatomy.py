"""tica_dodge_anatomy, synthetic paired dumps: a planted escalation (more of the concept's own
hide edit after every caught turn, carried unchanged across layers) must show up as caught gain >
passed gain, esc aligned with the hide edit, cross-layer esc cosine ~1; records and dump rows must
align; the level-controlled feedback effect must not mistake regression to the mean for feedback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_dodge_anatomy import analyse, layer_of, load_records, ood
from tica_hide_geometry import load


def _write(tmp_path, stem, x, meta, layers, recs):
    torch.save({"layers": layers, "x": x.half(), "meta": meta, "adapter": None, "base": False},
               tmp_path / f"{stem}.pt")
    (tmp_path / f"{stem}.json").write_text(json.dumps({"records": recs}))


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    d, layers, concepts = 24, [8, 12], ["a", "b", "c", "e"]
    cdir = {c: torch.randn(d, generator=g) * 3 for c in concepts}
    hdir = {c: torch.randn(d, generator=g) for c in concepts}
    meta, recs, xb, xt = [], [], [], []
    for ep in range(16):
        c = concepts[ep % 4]
        cond = "real_score" if ep % 4 != 3 else "nobrief"
        level = 1.0
        for t in range(1, 9):
            caught = (t + ep) % 3 != 0
            m = {"ep": ep, "turn": t, "cond": cond, "off": False, "concept": c, "text_concept": c,
                 "split": "train", "detected": caught}
            meta.append(m)
            kind = "linear@8" if ep % 2 else "linear@26"
            recs.append({**m, "refits": {"r1": [1.0 if caught else -1.0, caught]},
                         "refit_meta": {"r1": {"deployed": 0, "beaten": 9, "kind": kind}}})
            b = torch.stack([cdir[c] + torch.randn(d, generator=g) for _ in layers])
            xb.append(b)
            edit = level * hdir[c] if cond == "real_score" else torch.zeros(d)
            xt.append(b + edit + 0.05 * torch.randn(len(layers), d, generator=g))
            if cond == "real_score" and caught:
                level += 0.5  # escalation: more of the same edit after a catch
    return layers, meta, recs, torch.stack(xb), torch.stack(xt)


def test_planted_escalation(tmp_path):
    layers, meta, recs, xb, xt = _data()
    _write(tmp_path, "base_train_s1", xb, meta, layers, recs)
    _write(tmp_path, "m_train_s1", xt, meta, layers, recs)
    lb, Xb, Mb = load([str(tmp_path / "base_train_s1.pt")])
    _, Xt, Mt = load([str(tmp_path / "m_train_s1.pt")])
    r = analyse(lb, Xb, Mb, Xt, Mt, load_records([str(tmp_path / "m_train_s1.pt")]))
    L = r["layers"][8]
    assert L["caught"]["gain"][0] > 0.2 > L["passed"]["gain"][0]
    assert L["caught"]["f_hide"][0] > 0.8
    assert L["esc_cos_hide"] > 0.9
    assert r["cross_layer"]["esc_cos"]["8-12"] > 0.9
    assert r["n_steps"]["shallow_catcher"] > 0 and r["n_steps"]["deep_catcher"] > 0
    o = ood(lb, Xb, Xb, Mb, Xt, Mt)
    assert set(o[8]) >= {"maha", "knn", "pcares", "norm"}  # xlayer: L + 4 must be dumped
    assert "xlayer" in o[8] and "xlayer" not in o[12]


def test_misaligned_records(tmp_path):
    layers, meta, recs, xb, xt = _data()
    _write(tmp_path, "base_train_s1", xb, meta, layers, recs)
    _write(tmp_path, "m_train_s1", xt, meta, layers, recs[1:] + recs[:1])
    lb, Xb, Mb = load([str(tmp_path / "base_train_s1.pt")])
    _, Xt, Mt = load([str(tmp_path / "m_train_s1.pt")])
    with pytest.raises(SystemExit):
        analyse(lb, Xb, Mb, Xt, Mt, load_records([str(tmp_path / "m_train_s1.pt")]))


def test_layer_of():
    assert layer_of("linear@22") == 22 and layer_of("maxpool@8") == 8
    assert layer_of("ens4_linear") is None and layer_of(None) is None


def test_feedback_effect_controls_regression_to_mean():
    import random as _r

    from tica_dodge_anatomy import ols_effect

    rng = _r.Random(0)
    # pure regression to the mean: y = -level + noise, caught turns have low levels; no feedback
    rows = []
    for g in range(40):
        for _ in range(6):
            lvl = rng.gauss(0, 1)
            rows.append((-lvl + rng.gauss(0, 0.1), lvl < 0, lvl, g))
    eff = ols_effect(rows)
    assert eff["beta"][1] < 0 < eff["beta"][2]  # caught adds nothing beyond the level
    rows = [(y + (0.5 if c else 0.0), c, lvl, g) for y, c, lvl, g in rows]
    assert ols_effect(rows)["beta"][1] > 0.3
