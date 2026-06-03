# Library API

This page shows the public surfaces that are safe to import when designing new
Neural Chameleon experiments. The package is meant to be used both from the CLI
and from Python.

## Install And Import

From a source checkout:

```bash
uv sync --extra dev
uv run python -c "import next_chameleons; print(next_chameleons.__all__)"
```

In Python:

```python
from next_chameleons import (
    ActivationBatch,
    DatasetAdapter,
    EnsembleJudge,
    Probe,
    ProjectPaths,
    TrainingRegime,
    load_experiment,
)
```

## Config-Driven Experiment Loading

Current config loading expects a project-style config tree. For this repository
that is `configs/`; for your own experiment pack, keep the same group layout:

```text
configs/
  dataset/
  experiment/
  judge/
  model/
  train/
```

Load a preset from Python:

```python
from pathlib import Path
from next_chameleons import ProjectPaths, load_experiment

paths = ProjectPaths(root=Path("."), configs=Path("configs"))
experiment = load_experiment("paper_gemma2_2b_real", paths=paths)
print(experiment["id"])
```

The intended future package shape is a config-source interface where user config
directories can be supplied explicitly and built-in reference configs are just
examples. For now, source-checkout configs remain the supported path.

## Custom Probe

```python
from dataclasses import dataclass
import numpy as np

from next_chameleons import ActivationBatch, ProbePrediction, ProbeReport
from next_chameleons.probes.base import auc_rank, calibrate_threshold
from next_chameleons.registry import PROBES


@dataclass
class MeanActivationProbe:
    probe_id: str = "mean_activation"
    family: str = "custom"
    threshold: float = 0.0

    def fit(self, batch: ActivationBatch) -> "MeanActivationProbe":
        scores = self._scores(batch)
        self.threshold = calibrate_threshold(scores, batch.labels)
        return self

    def predict(self, batch: ActivationBatch) -> ProbePrediction:
        return ProbePrediction(
            scores=self._scores(batch),
            labels=batch.labels,
            threshold=self.threshold,
        )

    def report(self, batch: ActivationBatch) -> ProbeReport:
        prediction = self.predict(batch)
        return ProbeReport(
            probe_id=self.probe_id,
            family=self.family,
            threshold=self.threshold,
            accuracy=float(np.mean(prediction.predicted == batch.labels)),
            detection_rate=float(np.mean(prediction.predicted)),
            auc=auc_rank(prediction.scores, batch.labels),
            metadata={},
        )

    def _scores(self, batch: ActivationBatch) -> np.ndarray:
        return batch.activations.mean(axis=(1, 2))


PROBES.register("mean_activation", MeanActivationProbe)
```

For in-repo built-ins, add the module import to
`next_chameleons.plugins.load_builtin_plugins()`. For external experiment packs,
import your registration module before loading the config that refers to it.

## Custom Judge

A Judge aggregates one or more Probes. The default `EnsembleJudge` is enough for
many experiments:

```python
from next_chameleons import EnsembleJudge
from next_chameleons.probes import LinearProbe, MLPProbe

judge = EnsembleJudge(
    judge_id="linear_mlp_mean",
    probes=[LinearProbe(), MLPProbe()],
    aggregation="mean",
    threshold=0.55,
)
```

## Custom Training Regime

Training Regimes describe the research objective. Training Backends execute the
mechanics.

```python
from next_chameleons import TrainingRegimeSpec
from next_chameleons.registry import TRAINING_REGIMES


class MyCurriculumRegime:
    regime_id = "my_curriculum"

    def spec(self) -> TrainingRegimeSpec:
        return TrainingRegimeSpec(
            regime_id=self.regime_id,
            objective="my_probe_curriculum",
            feedback_channels=["binary_verdict"],
            probe_suite="strong_starters",
            supports_probe_discovery=True,
        )


TRAINING_REGIMES.register("my_curriculum", MyCurriculumRegime)
```

## CLI First, Python When Needed

Most experiments should still start with configs and the CLI:

```bash
uv run next-chameleons registry
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons experiment-smoke multi_probe_evasion
```

Drop into Python when you are adding a new Probe, Judge, DatasetAdapter,
Training Regime, or analysis loop that the existing configs cannot express.

