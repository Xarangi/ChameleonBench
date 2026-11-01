"""crypsis.probes.cache: explicit keys hit only on a full match; old cache locations still load."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import torch

from crypsis.probes.cache import CacheKey, ProbeCache
from crypsis.probes.kinds import zoo as nzoo
from next_chameleons.tica import monitors as omon


def _bank(name):
    d = 4
    return omon.ConceptProbeBank(name, 3, ["c"], W=torch.ones(1, d), b=torch.zeros(1),
                                 mu=torch.zeros(1, d), sd=torch.ones(1, d), thr=torch.ones(1),
                                 zero=torch.zeros(1), hundred=torch.ones(1))


def test_key_components_and_hits(tmp_path):
    cache = ProbeCache(tmp_path)
    key = CacheKey("Qwen3.5-2B", "abc", "attnpool", (12,), "v1", {"fpr": 0.01})
    mon = nzoo.MeanDiffMonitor("m", 12, "c", torch.arange(4.0), thr=0.5)
    assert cache.load(key) is None
    cache.save(key, mon, {"auroc": 1.0})
    got, info = cache.load(key)
    assert info == {"auroc": 1.0} and torch.equal(got.w, mon.w) and got.thr == 0.5
    for f, v in (("model_id", "x"), ("data_digest", "x"), ("kind", "x"), ("layers", (13,)),
                 ("code_version", "v2"), ("extra", {"fpr": 0.05})):
        assert cache.load(dataclasses.replace(key, **{f: v})) is None, f


def test_legacy_panel_and_answer_locations(tmp_path):
    # panel cache: <sha256(key)[:24]>.pt with {"bank", "info"} (snapshot _panel_cached)
    key_json = json.dumps({"v": 1, "code": "2a4396d1b602f34f", "kind": "linear@12"},
                          sort_keys=True)
    old = tmp_path / f"{hashlib.sha256(key_json.encode()).hexdigest()[:24]}.pt"
    torch.save({"bank": _bank("panel0_1"), "info": {"kept": True}}, old)
    bank, info = ProbeCache(tmp_path).load_digest(key_json)
    assert info == {"kept": True} and bank.name == "panel0_1"
    # answer-probe cache: <sha256(key)[:16]>/L{L}.pt + quality.json (snapshot fit_answer_probes)
    d = tmp_path / hashlib.sha256(key_json.encode()).hexdigest()[:16]
    d.mkdir()
    torch.save(_bank("answer_base_L3").state(), d / "L3.pt")
    (d / "quality.json").write_text(json.dumps({"3": {"auroc": 0.99}}))
    mons, q = ProbeCache(tmp_path).load_answer_dir(key_json, [3])
    assert q == {3: {"auroc": 0.99}} and type(mons[0]).__module__ == "crypsis.probes.kinds.linear"
    assert torch.equal(mons[0].W, torch.ones(1, 4))
