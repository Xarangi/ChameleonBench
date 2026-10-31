# ruff: noqa: E501
"""Equivalence of crypsis.core / models.depth against the tica code."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import yaml

from crypsis.core import metrics
from crypsis.core.rundir import RunDir
from crypsis.core.types import text_id
from crypsis.models import depth as D
from next_chameleons.incontext import stats
from next_chameleons.tica import answer_format, monitors, probe_zoo

ROOT = Path(__file__).resolve().parents[2]


def test_auroc_threshold():
    rng = np.random.default_rng(0)
    cases = [(rng.normal(size=50), rng.normal(0.5, 1, size=70)),
             (rng.integers(0, 4, 40).astype(float), rng.integers(0, 4, 30).astype(float))]
    for p, n in cases:
        old = monitors.auroc(torch.tensor(p), torch.tensor(n))
        assert metrics.auroc(p, n) == old
        assert metrics.auroc(p, n) == stats.auroc(p, n)
        for fpr in (0.01, 0.05, 0.3):
            assert metrics.quantile_threshold(n, fpr) == monitors.quantile_threshold(torch.tensor(n), fpr)


def test_text_id():
    for t in ("", "hello", "naïve é\n"):
        assert text_id(t) == answer_format.text_id(t)


def _old_rule(out: Path):
    """Selection rule of the heredoc in scripts/slurm/tica_resume.sh (checkpoint dirs only)."""
    best = (0, False, "")
    st = lambda d: Path(d, "trainer_state.pt").exists()  # noqa: E731
    for d in out.glob("adaptive/step_*"):
        best = max(best, (int(d.name[5:]), st(d), str(d)))
    for d in out.glob("resume_*/adaptive/step_*"):
        off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
        best = max(best, (off + int(d.name[5:]), st(d), str(d)))
    return best


def test_rundir_newest(tmp_path):
    out = tmp_path / "run"
    (out / "adaptive" / "step_10").mkdir(parents=True)
    (out / "adaptive" / "step_20").mkdir()
    (out / "resume_20" / "adaptive" / "step_5").mkdir(parents=True)
    (out / "resume_20" / "adaptive" / "step_10").mkdir()
    (out / "resume_20" / "adaptive" / "step_10" / "trainer_state.pt").write_bytes(b"x")
    real = tmp_path / "real_ckpt"
    real.mkdir()
    (out / "resume_30").mkdir()
    (out / "resume_30" / "adaptive").mkdir()
    (out / "resume_30" / "adaptive" / "step_2").symlink_to(real)
    for _ in range(2):
        want = _old_rule(out)
        got = RunDir(out).newest()
        assert got is not None and got.global_step == want[0] and Path(got.path).resolve() == Path(want[2]).resolve()
        (out / "resume_30" / "adaptive" / "step_2" / "trainer_state.pt").write_bytes(b"x")  # via symlink
        (out / "resume_30" / "adaptive" / "step_2").unlink()
        (out / "resume_30" / "adaptive" / "step_2").symlink_to(real)


def test_depth():
    for depth in (24, 32, 48):
        for L in range(32):
            assert D.scale_layer(L, depth) == probe_zoo.scale_layer(L, depth)
        assert D.scale_kind("maxpool@24", depth) == probe_zoo.scale_kind("maxpool@24", depth)
        assert D.split_kind("linear@22") == probe_zoo.split_kind("linear@22")
    assert D.scale_kind("linear", 48) == "linear"
    assert D.parse_layer("12", 24) == 12
    assert D.parse_layer("12@32", 48) == 18
    assert D.parse_layer("0.375", 32) == 12


def test_config_layers_resolve_to_themselves():
    cfg = yaml.safe_load((ROOT / "configs/tica/final_v4d_klx_tamia_2b.yaml").read_text())
    depth = cfg["model"]["depth"]

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("kind_layers", "held_out_layers") and isinstance(v, list):
                    yield from v
                else:
                    yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    layers = list(walk(cfg))
    assert layers
    for L in layers:
        assert D.parse_layer(L, depth) == L
