"""Data-parallel episodes (tica/dp.py, REPLICATION 2026-10-19b): 2 ranks x 2 episodes on gloo
reproduce 1 process x 4 episodes with the same draws (adaptive.episode_rng), every rollout-coupled
feature on (tests/tica_dp_worker.py). Exact: trainer RNG, curriculum bandits, kind stats, the
turns-to-beat records, the sampler weights, every rank's weights equal. Float tolerance: losses,
LoRA weights, pursuit centroid (reads batch 2 vs 4 episodes; summed gradients in another order)."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, "tests")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(world: int, out: Path, steps: int, budget: bool = False) -> list[dict]:
    import torch.multiprocessing as mp

    out.mkdir(parents=True)
    (out / "anchor.json").write_text(json.dumps([{"prompt": f"p{k}", "answer": f"ans {k} ok"}
                                                 for k in range(6)]))
    env = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(Path("tests").resolve()), env])
    try:
        from tica_dp_worker import run

        mp.spawn(run, args=(world, str(out), _port(), steps, budget), nprocs=world, join=True)
    finally:
        os.environ["PYTHONPATH"] = env
    return [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(world)]


def _close(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_two_ranks_match_one_process(tmp_path):
    steps = 3
    (one,) = _spawn(1, tmp_path / "w1", steps)
    r0, r1 = _spawn(2, tmp_path / "w2", steps)
    # every rank holds the same weights and shared state
    for n in r0["lora"]:
        assert torch.equal(r0["lora"][n], r1["lora"][n]), n
    for k in ("rng", "kind_bandit", "concept_bandit", "concept_weights", "kind_stats",
              "deploy_log", "seg_log"):
        assert r0[k] == r1[k], k
    # ... and the same as one process running all 4 episodes
    for k in ("rng", "kind_bandit", "concept_bandit", "kind_stats", "deploy_log", "seg_log"):
        assert r0[k] == one[k], k
    for c, w in one["concept_weights"].items():
        assert _close(w, r0["concept_weights"][c]), c
    for n, v in one["lora"].items():
        assert torch.allclose(v, r0["lora"][n], atol=1e-5, rtol=1e-4), n
    for c, (cen, sc) in one["pursuit"].items():
        assert torch.allclose(cen, r0["pursuit"][c][0], atol=1e-5)
        assert _close(sc, r0["pursuit"][c][1])
    for a, b in zip(one["recs"], r0["recs"], strict=True):
        for k in ("train/loss", "grad_norm", "train/kl", "train/hide_mse", "train/match_mse",
                  "train/nll", "train/n_pursuit", "cap_gen/n_forks", "cap_gen/n_train",
                  "cap_gen/nll", "anchor/kl", "rollout/turn_move"):
            if k in a:
                assert _close(a[k], b[k], 1e-4), (k, a[k], b[k])
        for k in ("rollout/pass_rate", "rollout/ep_refits", "rollout/ep_refits_accepted",
                  "rollout/ep_first_probe_att", "rollout/ep_first_probe_acc",
                  "rollout/ep_first_probe_shared", "rollout/bank_size", "rollout/ep_off_turns",
                  "rollout/ep_censored", "rollout/beat_n", "rollout/beat_frac", "n_views",
                  "n_targets", "train/n_hide", "train/n_match"):
            assert a.get(k) == b.get(k), (k, a.get(k), b.get(k))
    # the run exercised what it is meant to (refits, shared turn-0 probes, cap_gen hide)
    assert sum(r["rollout/ep_refits"] for r in one["recs"]) > 0
    assert sum(r["rollout/ep_first_probe_shared"] for r in one["recs"]) > 0
    assert any(r.get("cap_gen/n_train", 0) > 0 for r in one["recs"])
    # rank 0 alone writes the logs; they hold every episode
    for name in ("train_log.jsonl", "turns.jsonl", "refits.jsonl"):
        a = (tmp_path / "w1" / "run" / name).read_text().splitlines()
        b = (tmp_path / "w2" / "run" / name).read_text().splitlines()
        assert len(a) == len(b), name
    turns = [json.loads(x) for x in
             (tmp_path / "w2" / "run" / "turns.jsonl").read_text().splitlines()]
    assert {r["ep"] for r in turns} == {0, 1, 2, 3}


def test_episode_rng_off_by_default():
    from next_chameleons.tica.adaptive import AdaptiveConfig

    assert AdaptiveConfig().episode_rng is False


def test_dp_rejects_unsupported_settings(tmp_path, monkeypatch):
    from next_chameleons.tica import dp
    from next_chameleons.tica.trainers import mathrl
    from tica_dp_worker import build

    monkeypatch.setattr(mathrl, "load_questions", mathrl.load_questions)  # build() replaces it
    monkeypatch.setattr(dp, "world", lambda: (2, 0))
    with pytest.raises(ValueError, match="episode_keep"):
        build(tmp_path / "run", episode_keep=True)


def test_two_ranks_match_one_process_with_turn_budget(tmp_path):
    """v4 runs the turn budget (10-19b) data-parallel: online switches / tails on each rank's own
    episodes give the same step as one process."""
    steps = 2
    (one,) = _spawn(1, tmp_path / "w1", steps, budget=True)
    r0, r1 = _spawn(2, tmp_path / "w2", steps, budget=True)
    for n in r0["lora"]:
        assert torch.equal(r0["lora"][n], r1["lora"][n]), n
        assert torch.allclose(one["lora"][n], r0["lora"][n], atol=1e-5, rtol=1e-4), n
    for k in ("rng", "kind_bandit", "concept_bandit", "kind_stats", "deploy_log", "seg_log"):
        assert r0[k] == one[k], k
    for a, b in zip(one["recs"], r0["recs"], strict=True):
        for k in ("rollout/pass_rate", "rollout/budget_tail_frac", "rollout/budget_switches",
                  "rollout/ep_refits", "n_targets"):
            assert a.get(k) == b.get(k), (k, a.get(k), b.get(k))
    assert sum(r["rollout/budget_switches"] for r in one["recs"]) > 0
