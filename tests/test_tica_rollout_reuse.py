"""updates_per_rollout (REPLICATION 2026-10-18j): K optimizer updates on one rollout. Default 1 =
the old path (bit-identical, checked 10-18j); the first update's loss is the on-policy loss; later
updates move the weights further."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests")

from test_tica_kl_anchor import _setup


def _lora(pol):
    return {n: p.detach().clone() for n, p in pol.model.named_parameters() if "lora_" in n}


def _run(tmp_path, k, **adaptive):
    torch.manual_seed(0)
    tr, pol, _ = _setup(tmp_path, **adaptive)
    tr.cfg.updates_per_rollout = k
    before = _lora(pol)
    rec = tr.step()
    return rec, before, _lora(pol)


def test_default_is_one_update(tmp_path):
    from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedConfig

    assert AdaptiveSupervisedConfig().updates_per_rollout == 1
    rec, _, _ = _run(tmp_path, 1)
    assert "train/update_losses" not in rec


def test_k_updates_reuse_the_rollout(tmp_path):
    r1, b1, a1 = _run(tmp_path / "k1", 1)
    r3, b3, a3 = _run(tmp_path / "k3", 3, kl_weight=0.0)
    assert all(torch.equal(b1[n], b3[n]) for n in b1)  # same start
    assert abs(r1["train/loss"] - r3["train/loss"]) < 1e-6  # logged = first (on-policy) update
    assert len(r3["train/update_losses"]) == 3 and len(r3["train/update_gnorms"]) == 3
    assert abs(r3["train/update_losses"][0] - r3["train/loss"]) < 1e-6
    d1 = sum(float((a1[n] - b1[n]).norm()) for n in a1)
    d3 = sum(float((a3[n] - b3[n]).norm()) for n in a3)
    assert d3 > d1 * 1.2  # later updates keep moving the weights
    assert r3["train/update_losses"][-1] != r3["train/update_losses"][0]


def test_k_updates_with_kl_anchor(tmp_path):
    r, _, _ = _run(tmp_path, 2, kl_weight=1.0, kl_ref="nobrief", kl_topk=5)
    assert len(r["train/update_losses"]) == 2 and r["train/kl"] > 0
