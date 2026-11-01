"""models.depth == probe_zoo depth logic, and reproduces every layer list of the v4d configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crypsis.models.depth import LayerSpec, ZooLayers, pool_layers, scale_kind, scale_layer

CONFIGS = sorted(Path("configs/tica").glob("final_v4d*.yaml"))
LIST_FIELDS = (("probes", "held_out_layers"), ("rl.adaptive", "kind_layers"),
               ("rl.adaptive", "panel_layers"), ("rl.adaptive", "stay_layers"),
               ("rl.adaptive", "act_anchor_layers"))
KIND_FIELDS = (("final", "train_refit_kinds"), ("final", "heldout_refit_kinds"),
               ("rl.adaptive", "episode_refit_kinds"), ("rl.adaptive", "first_probe_panel_kinds"))


def _get(cfg, sec, key):
    for part in sec.split("."):
        cfg = cfg.get(part) or {}
    return list(cfg.get(key) or [])


def test_scale_layer_and_zoo_match_probe_zoo():
    from next_chameleons.tica import probe_zoo as pz

    try:
        for depth in (24, 32, 48):
            assert [scale_layer(L, depth) for L in range(32)] == [
                pz.scale_layer(L, depth) for L in range(32)]
            assert [LayerSpec.parse(f"{L}@32").resolve(depth) for L in range(32)] == [
                pz.scale_layer(L, depth) for L in range(32)]
            pz.set_zoo_depth(depth)
            z = ZooLayers.for_depth(depth)
            for kind in (*pz.ENS_KINDS, *pz.TOKENS_ENS, "linear", "maxpool", "linear@7"):
                assert z.kind_layers(kind, 5) == pz.kind_layers(kind, 5), (depth, kind)
                assert scale_kind(kind, depth) == pz.scale_kind(kind, depth)
    finally:
        pz.set_zoo_depth(32)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_v4d_config_layers_resolve_from_32_layer_source(path):
    src = yaml.safe_load(Path("configs/tica/final_v4d_klx_tamia_4b.yaml").read_text())
    cfg = yaml.safe_load(path.read_text())
    depth = int((cfg.get("model") or {}).get("depth") or 32)
    primary = (cfg.get("probes") or {}).get("layer")
    assert (primary if primary is not None else 12) == LayerSpec.parse("12@32").resolve(depth)
    z = ZooLayers.for_depth(depth)
    for sec, key in LIST_FIELDS:
        got = _get(cfg, sec, key)
        assert got == [LayerSpec.parse(f"{L}@32").resolve(depth) for L in _get(src, sec, key)]
        assert [LayerSpec.parse(str(L)).resolve(depth) for L in got] == got  # in range
    for sec, key in KIND_FIELDS:
        kinds = _get(cfg, sec, key)
        assert kinds == [scale_kind(k, depth) for k in _get(src, sec, key)], key
        for k in kinds:
            assert all(0 <= L < depth for L in z.kind_layers(k, 0))


def test_layer_spec_forms_and_pool_layers():
    assert LayerSpec.parse("L12").resolve(24) == 12
    assert LayerSpec.parse(12).resolve(24) == 12
    assert LayerSpec.parse("0.375").resolve(24) == 9
    with pytest.raises(ValueError):
        LayerSpec.parse("30").resolve(24)
    assert pool_layers(9, "primary") == [9]
    assert pool_layers(12, "all", [20, 12, 8]) == [12, 8, 20]
    assert pool_layers(12, [24, 8]) == [12, 8, 24]
