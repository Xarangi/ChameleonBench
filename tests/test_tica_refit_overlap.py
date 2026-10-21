"""adaptive.refit_overlap (10-21e): fits on a worker thread overlapping the next episodes' reads
must give the SAME rollout as the serial path (refit records, probe weights, verdicts, the global
torch RNG end state), including the switch fallback (rollback + redone reads)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

DROP = {"sec", "sec_read"}


def _run(tmp_path, monkeypatch, overlap: bool, kinds, switch: int):
    from next_chameleons.tica import trainers
    from next_chameleons.tica.adaptive import AdaptiveConfig, AdaptiveDefender
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base_name = pool.entries[0].monitor.name

    class FakeSampler:
        def sample(self, n):
            return [EpisodeSpec(f"real_score|cats|{k}", "real_score",
                                ["aa", "bb b", "cc", "dd d", "ee"], "cats", "cats", base_name)
                    for k in range(n)]

    cfg = AdaptiveSupervisedConfig(
        steps=1, episodes_per_step=4, turns=5, micro_batch=4, lr=1e-2, warmup_steps=1,
        save_every=0, seed=3,
        adaptive=AdaptiveConfig(
            refit_mode="episode", refit_min_auroc=0.0, refit_min_tpr_np=0.0, refit_C=1.0,
            refit_switch_max=switch, episode_refit_kind="mixed", episode_refit_kinds=kinds,
            episode_kind_sampling="uniform", episode_rng=True, episode_keep=True,
            refit_np_cache=True, refit_overlap=overlap,
            episode_fork={"pos": 8, "np": 8, "other": 8, "free": 3}))
    out = tmp_path / ("on" if overlap else "off")
    tr = AdaptiveSupervisedTrainer(
        pol, pool, FakeSampler(), cfg, out,
        branch_concepts=[ConceptTexts("cats", "train", [f"meow {k}" for k in range(12)]),
                         ConceptTexts("dogs", "train", [f"woof {k}" for k in range(12)])],
        branch_negatives=["a b", "c d", "e f", "g h"])
    (out / "bank").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TICA_DUMP_REFITS", str(out / "dump"))  # every refit's tensors (zoo too)
    orig = AdaptiveDefender.score
    monkeypatch.setattr(AdaptiveDefender, "score",
                        lambda self, n, p, r: {**orig(self, n, p, r), "detected": False})
    sup = trainers.supervised
    real_gate = sup.gate_fit

    def gate(kind, *a, **kw):  # meandiff always fails its gate -> the switch fallback runs
        mon, info = real_gate(kind, *a, **kw)
        return (None, {**info, "kept": False}) if kind == "meandiff" else (mon, info)

    monkeypatch.setattr(sup, "gate_fit", gate)
    torch.manual_seed(1234)
    trajs, m = tr.rollout(FakeSampler().sample(4))
    rng_end = torch.get_rng_state()
    recs = [{k: v for k, v in json.loads(x).items() if k not in DROP}
            for x in (tr.out / "refits.jsonl").read_text().splitlines()]
    banks = {f"{d}/{p.name}": torch.load(p, weights_only=False)
             for d in ("bank", "dump") for p in sorted((out / d).glob("*.pt"))}
    res = [[{k: v for k, v in r.items()} for r in t.results] for t in trajs]
    return recs, banks, res, m, rng_end


def _same_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.allclose(a.float(), b.float(), rtol=0, atol=0)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_state(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _check(tmp_path, monkeypatch, kinds, switch):
    off = _run(tmp_path, monkeypatch, False, kinds, switch)
    on = _run(tmp_path, monkeypatch, True, kinds, switch)
    assert off[0] == on[0] and len(off[0]) >= 8  # refit records: kinds, kept/why, auroc, order
    assert off[1].keys() == on[1].keys() and any(k.startswith("dump/") for k in off[1])
    for k in off[1]:
        assert _same_state(off[1][k], on[1][k]), k  # probe weights, bit-identical
    assert off[2] == on[2]  # every turn's verdicts / active probes / refit outcomes
    for k, v in off[3].items():
        if not k.startswith("sec") and "overlap" not in k:
            assert on[3][k] == v or (v != v and on[3][k] != on[3][k]), k
    assert torch.equal(off[4], on[4])  # global torch RNG ends where the serial fits leave it
    assert "sec_ep_refit_overlap_saved" in on[3] and "sec_ep_refit_overlap_saved" not in off[3]
    return on[3]


def test_refit_overlap_exact_with_switch_fallback(tmp_path, monkeypatch):
    m = _check(tmp_path, monkeypatch, ["meandiff", "linear", "rlobf_attn"], switch=2)
    assert m["ep_refit_switch_attempts"] > 0 and m["ep_refit_overlap_redo"] > 0


def test_refit_overlap_exact_no_switch(tmp_path, monkeypatch):
    _check(tmp_path, monkeypatch, ["linear", "rlobf_attn"], switch=0)
