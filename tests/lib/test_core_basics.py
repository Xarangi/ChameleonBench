"""core.types / registry / io / paths: text_id equivalence and minimal unit tests."""

from __future__ import annotations

import pytest

from crypsis.core import io
from crypsis.core.paths import Paths, PathsError
from crypsis.core.registry import Registry
from crypsis.core.types import MultiView, Readout, Text, Verdict, text_id


def test_text_id_equals_tica():
    from next_chameleons.tica.answer_format import text_id as old

    for s in ["", "hello", "naïve — ünïcode ✓", "x" * 5000]:
        assert text_id(s) == old(s) == Text(s).id


def test_types_roundtrip():
    r = {"raw": 1.0, "score": 60.0, "thr": 50.0, "p": 0.7, "detected": True}
    assert Readout.from_dict(r).to_dict() == r
    v = {"reward": 1.0, "passed": True, "novel": False, "next_monitor": "m1"}
    assert Verdict.from_dict(v).to_dict() == v
    mv = MultiView(list(range(10)), [(2, 4, 5), (7, 9, 10)])
    w = mv.view(1)
    assert (w.ids, w.start, w.end, w.n_content) == (list(range(10)), 7, 9, 2)


def test_registry():
    reg: Registry[type] = Registry("thing")

    @reg.register("a")
    class A:
        pass

    reg.register("b", int)
    assert reg.get("a") is A and reg.names() == ["a", "b"] and "b" in reg
    with pytest.raises(KeyError):
        reg.register("a", str)
    with pytest.raises(KeyError, match="available: a, b"):
        reg.get("zz")


def test_io(tmp_path):
    p = io.write_json(tmp_path / "d" / "r.json", {"x": 1})
    assert io.read_json(p) == {"x": 1} and io.is_done(p, require_valid=True)
    assert not list((tmp_path / "d").glob("*.tmp"))
    log = tmp_path / "t.jsonl"
    io.append_jsonl(log, [{"step": 1}, {"step": 2}])
    with open(log, "a") as fh:
        fh.write('{"step": 3, "trunc')
    assert io.read_jsonl(log) == [{"step": 1}, {"step": 2}]
    assert io.read_jsonl(tmp_path / "missing.jsonl") == []
    assert not io.stage_done(tmp_path, "eval")
    io.mark_done(tmp_path, "eval")
    assert io.stage_done(tmp_path, "eval")


def test_paths(tmp_path):
    env = {"CRYPSIS_RAW": str(tmp_path / "raw"), "CRYPSIS_ARTIFACTS": str(tmp_path / "art")}
    p = Paths.resolve(env=env)
    assert p.runs == tmp_path / "art" / "runs" and p.hf_cache is None
    prof = tmp_path / "site.yaml"
    prof.write_text(f"paths: {{raw: {tmp_path / 'praw'}}}\n")
    q = Paths.resolve(profile=prof, artifacts=tmp_path / "x", env=env)
    assert q.raw == tmp_path / "praw" and q.artifacts == tmp_path / "x"
    assert q.interpolate_tree({"a": ["${paths.raw}/f.json"]}) == {"a": [f"{tmp_path}/praw/f.json"]}
    with pytest.raises(PathsError):
        Paths.resolve(env={})
    with pytest.raises(PathsError):
        p.interpolate("${paths.hf_cache}/m")
