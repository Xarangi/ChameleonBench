"""Probe-panel aggregation and training-run report parsing (REPLICATION 2026-10-18d)."""

import json

import torch

from next_chameleons.tica.panel import aggregate, cv_auroc, jaccard, refit_events
from next_chameleons.tica.run_report import load_run, probe_index, report, retry_pass


def _rec(ep, t, det, panel, off=False, concept="c", refits=None, meta=None):
    return {"ep": ep, "turn": t, "concept": concept, "off": off, "detected": det,
            "active": "a", "panel": {p: [1.0 if d else -1.0, d] for p, d in panel.items()},
            "refits": refits or {}, "refit_meta": meta or {}}


def test_jaccard():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard(set(), set()) is None


def test_aggregate_panel_and_strategy_change():
    P = ("base:linear@8", "base:linear@24")
    recs = [
        _rec(0, 0, True, {P[0]: True, P[1]: True}),
        _rec(0, 1, False, {P[0]: False, P[1]: True}),   # dodged L8 only
        _rec(0, 2, False, {P[0]: False, P[1]: True}),   # same dodge after a pass
        _rec(0, 3, True, {P[0]: True, P[1]: False}, off=True),
    ]
    agg = aggregate(recs, 4)
    assert agg["n_monitored"] == 3
    assert agg["panel"][P[0]]["det"] == 1 / 3
    assert agg["panel"][P[0]]["det_when_active_beaten"] == 0.0
    assert agg["panel"][P[1]]["det_off"] == 0.0
    assert agg["layer_profile_by_turn"]["8"][:3] == [1.0, 0.0, 0.0]
    # turn 0 -> 1 (after a catch): dodged {} -> {L8}: J = 0; turn 1 -> 2 (after a pass): J = 1
    assert agg["jaccard_dodged_next_after_catch"] == 0.0
    assert agg["jaccard_dodged_next_after_pass"] == 1.0
    assert agg["panel_flip_rate_after_catch"] == 0.5
    assert agg["active_pass_by_turn"][:3] == [0.0, 1.0, 1.0]


def test_refit_events_and_old_refit_redetection():
    results = [
        {"deployed": ["b"], "active_monitor": "b", "passed": True},
        {"deployed": ["b", "r1"], "active_monitor": "r1", "passed": False},
        {"deployed": ["b", "r1"], "active_monitor": "r1", "passed": True},
        {"deployed": ["b", "r1", "r2"], "active_monitor": "r2", "passed": False},
    ]
    ev = refit_events(results)
    assert ev["r1"] == {"deployed": 1, "beaten": 2}
    assert ev["r2"] == {"deployed": 3, "beaten": None}
    meta = {"r1": {**ev["r1"], "kind": "linear"}}
    recs = [_rec(0, t, False, {}, refits={"r1": [0.0, t == 3]}, meta=meta) for t in range(4)]
    old = aggregate(recs, 4)["old_refit"]
    assert old["redetect_after_beaten"] == 1.0  # turn 3 > beaten turn 2, detected
    assert old["retro_detect_before_deploy"] == 0.0  # turn 0 < deployed turn 1
    assert old["refits_beaten"] == 1


def test_cv_auroc_separable_vs_noise():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(80, 6, generator=g)
    y = torch.tensor([1] * 40 + [0] * 40)
    sep = x.clone()
    sep[:40, 0] += 6.0
    assert cv_auroc(sep, y) > 0.95
    assert 0.2 < cv_auroc(x, y) < 0.8


def _write(d, name, rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _turn(step, ep, t, det, nd=1, active="p0", concept="c", cond="real_score", off=False):
    return {"step": step, "ep": ep, "turn": t, "concept": concept, "cond": cond, "off": off,
            "detected": det, "active": active, "n_deployed": nd, "heldout_det": det}


def test_run_report_chunks_supersede_and_offsets(tmp_path):
    run = tmp_path / "run"
    # chunk 0 ran steps 1..3 (rows step 0..2); resumed from the checkpoint at step 2
    _write(run / "adaptive", "train_log.jsonl", [{"step": s, "sec": 1.0} for s in (1, 2, 3)])
    _write(run / "adaptive", "turns.jsonl", [_turn(s, 0, 0, True) for s in (0, 1, 2)])
    _write(run / "resume_2" / "adaptive", "train_log.jsonl", [{"step": 1, "sec": 2.0}])
    _write(run / "resume_2" / "adaptive", "turns.jsonl", [_turn(0, 0, 0, False)])
    d = load_run(run)
    assert [r["g"] for r in d["train"]] == [1, 2, 3]  # chunk-0 step 3 superseded by resume g=3
    assert [r["sec"] for r in d["train"]] == [1.0, 1.0, 2.0]
    assert sorted(r["g"] for r in d["turns"]) == [1, 2, 3]
    # a branch whose adaptive/ started at global 360
    br = tmp_path / "branch"
    _write(br / "adaptive", "train_log.jsonl", [{"step": 1, "sec": 1.0}])
    assert load_run(br, base_offset=360)["train"][0]["g"] == 361


def test_probe_index_hazard_and_retry(tmp_path):
    rows = [  # episode 0: bank 1 caught, caught, passed; then bank 2 (refit) passed first try
        _turn(0, 0, 0, True), _turn(0, 0, 1, True), _turn(0, 0, 2, False),
        _turn(0, 0, 3, False, nd=2, active="p1"),
    ]
    rows = [{**r, "g": 1} for r in rows]
    pi = probe_index(rows, window=20)
    w = pi["by_index"]["0-20"]
    assert w["k1"] == {"beaten": 1.0, "n": 1, "turns": 3.0}
    assert w["k1_first"]["beaten"] == 1.0
    assert w["k2"]["turns"] == 1.0
    assert pi["hazard"]["0-20"]["k1"][:3] == [0.0, 0.0, 1.0]
    rp = retry_pass(rows, window=20)
    assert rp["0-20"] == {"pass": 0.5, "n": 2}  # t1 after a catch: caught; t2: passed


def test_report_smoke(tmp_path):
    run = tmp_path / "run"
    _write(run / "adaptive", "train_log.jsonl",
           [{"step": 1, "sec": 1.0, "rollout/beat_frac_linear@8": 0.5,
             "rollout/beat_n_linear@8": 4}])
    _write(run / "adaptive", "turns.jsonl", [_turn(0, 0, 0, True), _turn(0, 0, 1, False)])
    _write(run / "adaptive", "refits.jsonl",
           [{"step": 0, "kind": "linear@8", "kept": True, "auroc": 0.98},
            {"step": 0, "kind": "attnpool", "kept": False, "auroc": 0.9, "why": "tpr_np"}])
    rep = report(run, families={"c": "topic"})
    assert rep["kind_beat"]["linear@8"]["0-80"] == {"beaten": 0.5, "n": 4}
    assert rep["refit_readability"]["by_window"]["0-80"]["frac_lt_099"] == 1.0
    assert rep["refit_readability"]["reject_reasons"] == {"tpr_np": 1}
    assert rep["per_concept"]["family:topic"]["0-80"]["k1_beaten"] == 1.0


def test_selectivity_hidden_when_other_monitored():
    from next_chameleons.tica.panel import selectivity

    P = "base:linear@24"
    recs = [
        {**_rec(0, 0, False, {P: False}), "cond": "real_score"},
        {**_rec(1, 0, False, {P: False}), "cond": "mismatch", "false_alarm": {P: [1.0, True]}},
        {**_rec(1, 1, False, {P: True}), "cond": "mismatch", "false_alarm": {P: [-1.0, False]}},
        {**_rec(2, 0, False, {P: True}), "cond": "nobrief"},
    ]
    sel = selectivity(recs, 2)
    row = sel["text_probe"][P]
    assert (row["real"], row["mismatch"], row["nobrief"]) == (0.0, 0.5, 1.0)
    assert row["hidden_when_other_monitored"] == 0.5
    assert row["mismatch_by_turn"] == [0.0, 1.0]
    assert sel["false_alarm"][P] == 0.5
    # the dodging aggregate ignores selectivity records
    assert aggregate(recs, 2)["n_monitored"] == 1
