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
    ConfigSource,
    DatasetAdapter,
    EnsembleJudge,
    MeanDifferenceProbe,
    Probe,
    ProjectPaths,
    QuadraticProbe,
    TrainingRegime,
    load_experiment,
)
```

For the most stable public import seam, use `next_chameleons.api`:

```python
from next_chameleons.api import Probe, TrainingRegimeRunner, TrainingRunRequest
```

## Config-Driven Experiment Loading

Config loading is source-driven. The package ships reference configs as
packaged presets, and the source checkout keeps editable copies in `configs/`.
External projects can bring their own config tree with the same group layout:

```text
configs/
  dataset/
  experiment/
  judge/
  model/
  train/
```

Load a preset from this repository:

```python
from pathlib import Path
from next_chameleons import ProjectPaths, load_experiment

paths = ProjectPaths(root=Path("."), configs=Path("configs"))
experiment = load_experiment("paper_gemma2_2b_real", paths=paths)
print(experiment["id"])
```

Load an external experiment pack:

```python
from pathlib import Path
from next_chameleons import ConfigSource, load_experiment

source = ConfigSource.from_config_dir(Path("/path/to/my_next_chameleon_configs"))
experiment = load_experiment("my_probe_curriculum", paths=source)
```

For CLI usage:

```bash
next-chameleons --config-dir /path/to/my/configs config-check my_probe_curriculum
```

You can also set `NEXT_CHAMELEONS_CONFIG_DIR`. If unset, the CLI discovers the
nearest source checkout with `configs/`; if no checkout is found, it falls back
to packaged reference configs. User configs are searched before packaged
reference configs, so local presets can override built-in names.

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

External packages can also expose an entry point so `next-chameleons registry`
loads them automatically:

```toml
[project.entry-points."next_chameleons.plugins"]
my_lab = "my_lab_next_chameleons_plugin"
```

The target module should register Probes, Judges, DatasetAdapters, or Training
Regimes as an import side effect.

## Custom Judge

A Judge aggregates one or more Probes. The default `EnsembleJudge` is enough for
many experiments:

```python
from next_chameleons import EnsembleJudge
from next_chameleons.probes import LinearProbe, MLPProbe, QuadraticProbe

judge = EnsembleJudge(
    judge_id="linear_mlp_quadratic_mean",
    probes=[LinearProbe(), MLPProbe(), QuadraticProbe()],
    aggregation="mean",
    threshold=0.55,
)
```

For stronger Atlas-inspired starter Judges, use the `strong_starters` config or
compose its families manually:

```python
from next_chameleons.probes import (
    AttentionProbe,
    GeometryProbe,
    LinearProbe,
    MeanDifferenceProbe,
    MLPProbe,
    QuadraticProbe,
)

probe_pool = [
    LinearProbe(probe_id="linear_l12", layers=[12]),
    MeanDifferenceProbe(probe_id="mean_diff_l12", layers=[12]),
    QuadraticProbe(probe_id="quadratic_l12", layers=[12]),
    MLPProbe(probe_id="mlp64_l12", layers=[12], hidden_units=64),
    AttentionProbe(probe_id="attention4_l12", layers=[12], heads=4),
    GeometryProbe(probe_id="centroid_l12", layers=[12]),
]
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

Then point a user config at the regime:

```yaml
id: my_probe_curriculum
track: extension
dataset: my_dataset
judge: strong_starters
train_backend: qlora
training_regime: my_curriculum
mechanisms:
  probe_discovery:
    config: configs/mechanism/probe_discovery.yaml
    role: promote stronger held-out probes after each round
adaptive:
  rounds: 5
  feedback_channel: binary_verdict
  heldout_probe_families:
    - quadratic
    - attention
```

The regime object describes the research objective. The backend config still
controls execution details such as LoRA/QLoRA/full-finetune/GRPO, learning rate,
batching, KL weight, checkpoint cadence, and W&B mode.

## Bring Your Own Experiment Pack

External users should not edit this repository to run new settings. Create a
separate config tree:

```text
my_next_chameleon_pack/
  dataset/my_dataset.yaml
  experiment/my_probe_curriculum.yaml
  judge/my_judge.yaml
  model/my_model.yaml
  train/my_backend.yaml
```

Run it directly:

```bash
next-chameleons --config-dir my_next_chameleon_pack \
  config-check my_probe_curriculum

next-chameleons --config-dir my_next_chameleon_pack \
  real-run-plan my_probe_curriculum \
  --output-path runs/my_probe_curriculum_plan.json
```

If the pack has custom Python components, expose a plugin entry point or import
the registration module before running from Python.

## CLI First, Python When Needed

Most experiments should still start with configs and the CLI:

```bash
uv run next-chameleons registry
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons experiment-smoke multi_probe_evasion
```

Drop into Python when you are adding a new Probe, Judge, DatasetAdapter,
Training Regime, or analysis loop that the existing configs cannot express.

## Training Regime Runner

Use `TrainingRegimeRunner` when you want one Python seam for synthetic smoke,
real training, real evaluation, and adaptive runs:

```python
from pathlib import Path
from next_chameleons.api import ConfigSource, TrainingRegimeRunner, TrainingRunRequest

source = ConfigSource.from_config_dir(Path("my_configs"))
runner = TrainingRegimeRunner(paths=source)
report = runner.run(
    TrainingRunRequest(
        experiment="my_probe_curriculum",
        mode="synthetic_smoke",
        output_dir=Path("runs/my_probe_curriculum"),
    )
)
print(report)
```
