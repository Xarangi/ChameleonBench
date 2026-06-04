from pathlib import Path

from next_chameleons.config import ConfigSource
from next_chameleons.training.runner import TrainingRegimeRunner, TrainingRunRequest


def test_training_regime_runner_dispatches_synthetic_smoke(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_run_experiment_smoke(experiment, output_dir, *, paths=None):
        called["experiment"] = experiment
        called["output_dir"] = output_dir
        called["paths"] = paths
        return output_dir / "report.json"

    monkeypatch.setattr(
        "next_chameleons.pipeline.run_experiment_smoke",
        fake_run_experiment_smoke,
    )
    paths = ConfigSource.builtin()
    runner = TrainingRegimeRunner(paths=paths)
    request = TrainingRunRequest(
        experiment="multi_probe_evasion",
        mode="synthetic_smoke",
        output_dir=tmp_path / "out",
    )

    report = runner.run(request)

    assert report == tmp_path / "out" / "report.json"
    assert called == {
        "experiment": "multi_probe_evasion",
        "output_dir": tmp_path / "out",
        "paths": paths,
    }
