"""tica_plot: Kaplan-Meier on deploy_records + plots on synthetic logs (no GPU, no data texts)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "tica_plot", Path(__file__).resolve().parents[1] / "scripts" / "tica_plot.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def test_km_censoring():
    recs = [{"turns": 1, "beaten": True}, {"turns": 2, "beaten": False},
            {"turns": 2, "beaten": True}, {"turns": 3, "beaten": True}]
    xs, ys = tp.km(recs)
    assert xs == [0, 1, 2, 3]
    # S(1)=3/4, S(2)=3/4*(1-1/3)=1/2, S(3)=1/2*(1-1/1)=0
    assert ys == pytest.approx([0.0, 0.25, 0.5, 1.0])
    assert tp.km([]) == ([], [])


def test_get_prefix_and_nan_smoothing():
    rows = [{"step": 1, "rollout/beat_frac": 0.5}, {"step": 2, "beat_frac": float("nan")}]
    assert tp.series(rows, "beat_frac") == ([1], [0.5])
    _, ys = tp.smooth([1, 2], [1.0, float("nan")], 3)
    assert ys == [1.0, 1.0]
    assert math.isnan(tp.smooth([1], [float("nan")], 3)[1][0])


def test_plots_render(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = tmp_path / "run"
    (run / "adaptive").mkdir(parents=True)
    (run / "val").mkdir()
    with (run / "adaptive" / "train_log.jsonl").open("w") as f:
        for s in range(1, 31):
            f.write(json.dumps({
                "step": s, "rollout/pass_rate": s / 30, "rollout/beat_frac": 0.5,
                "rollout/beat_within1": 0.1, "rollout/beat_turns_mean": 2.0, "train/kl": 0.1,
                "rollout/task_family/topic/beaten": 0.6,
                "rollout/task_family/topic/turns_to_beat": float("nan") if s % 2 else 2.5,
                "rollout/task_family/topic/hazard": 0.3, "rollout/kind_p_linear": 0.5,
                "rollout/concept_p_c0": 0.2}) + "\n")
    for s in (10, 20):
        (run / "val" / f"step_{s}.json").write_text(json.dumps({"step": s, "val_score": 0.4}))
    (run / "adaptive" / "early_stop.json").write_text(json.dumps({"best_step": 20}))
    outs = tp.train_curves(run, tmp_path, plt)
    assert [p.exists() for p in outs] == [True, True]
    race = tmp_path / "a" / "r.json"
    race.parent.mkdir()
    race.write_text(json.dumps({
        "pass_rate_after_first_turn_by_topic": {"c0": 0.3},
        "deploy_records": [{"kind": "linear", "turns": 2, "beaten": True, "concept": "c0"}]}))
    old = tmp_path / "b" / "r.json"
    old.parent.mkdir()
    old.write_text(json.dumps({"beat_frac": 0.5}))
    outs = tp.race_plots([race, old], tmp_path, "t", plt)
    assert [p.name for p in outs] == ["t_km.png", "t_concepts.png"]
    assert tp.race_labels([race, old]) == ["a/r", "b/r"]
