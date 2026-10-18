"""adaptive.eval_every (10-18): evals at global steps divisible by eval_every, saved first."""

from types import SimpleNamespace

from next_chameleons.tica.adaptive import AdaptiveConfig
from next_chameleons.tica.trainers.supervised import AdaptiveSupervisedTrainer


def _run(monkeypatch, tmp_path, save_every, eval_every, offset, steps):
    monkeypatch.setenv("TICA_STEP_OFFSET", str(offset))
    tr = object.__new__(AdaptiveSupervisedTrainer)
    saves, evals = [], []
    tr.cfg = SimpleNamespace(steps=steps, save_every=save_every,
                             adaptive=AdaptiveConfig(eval_every=eval_every))
    tr.out, tr.step_k, tr._wb, tr._wb_off = tmp_path, 0, None, True

    def step():
        tr.step_k += 1
        return {}

    tr.step = step
    tr.policy = SimpleNamespace(save_adapter=lambda p: saves.append(p.name))
    tr._eval_on_save = lambda p: evals.append(p.name)
    tr._early_stop_check = lambda: False
    tr.train()
    return saves, evals


def test_default_evals_every_save(monkeypatch, tmp_path):
    saves, evals = _run(monkeypatch, tmp_path, 20, 0, 0, 60)
    assert evals == ["step_20", "step_40", "step_60"]
    assert saves == ["step_20", "step_40", "step_60", "final"]


def test_eval_every_global_steps(monkeypatch, tmp_path):
    # resume chunk at global 420: evals at global 450 / 500 (local 30 / 80), saved first
    saves, evals = _run(monkeypatch, tmp_path, 20, 50, 420, 100)
    assert evals == ["step_30", "step_80"]
    assert "step_30" in saves and "step_80" in saves and "step_20" in saves
