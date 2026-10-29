"""SupervisedConfig.state_keep_last / state_keep_every (10-29): optimizer-state retention."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer


def test_prune_keeps_newest_and_round_global_steps(tmp_path: Path) -> None:
    for k in (20, 40, 60, 80, 100, 120):
        (tmp_path / f"step_{k}").mkdir()
        (tmp_path / f"step_{k}" / "trainer_state.pt").write_bytes(b"x")
        (tmp_path / f"step_{k}" / "adapter_model.safetensors").write_bytes(b"a")
    me = SimpleNamespace(cfg=SimpleNamespace(state_keep_last=2, state_keep_every=100),
                         out=tmp_path, _step_offset=lambda: 880)
    AdaptiveSupervisedTrainer._prune_states(me)
    kept = sorted(int(p.parent.name[5:]) for p in tmp_path.glob("step_*/trainer_state.pt"))
    assert kept == [20, 100, 120]  # global 900 (round) + the newest two (global 980, 1000)
    assert len(list(tmp_path.glob("step_*/adapter_model.safetensors"))) == 6
