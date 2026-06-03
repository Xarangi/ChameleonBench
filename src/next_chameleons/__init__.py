"""Next Chameleons benchmark and training kit.

The top-level package exports the stable lightweight interfaces users need to
design datasets, activation caches, Probes, Judges, Training Regimes, and config
driven experiments. Concrete built-ins still live in their subpackages.
"""

from next_chameleons.activations import (
    ActivationBatch,
    ActivationExtractor,
    load_activation_cache,
    save_activation_cache,
)
from next_chameleons.config import ProjectPaths, load_config_group, load_experiment, load_yaml
from next_chameleons.datasets import DatasetAdapter, DatasetBundle, Example, paper_trigger
from next_chameleons.judges import EnsembleJudge, JudgeResult
from next_chameleons.probes.base import Probe, ProbePrediction, ProbeReport
from next_chameleons.registry import (
    ACTIVATION_EXTRACTORS,
    CURRICULA,
    DATASET_ADAPTERS,
    FEEDBACK_CHANNELS,
    JUDGES,
    METRICS,
    PROBE_SELECTORS,
    PROBES,
    REPORTS,
    TRAINING_BACKENDS,
    TRAINING_REGIMES,
    Registry,
)
from next_chameleons.training.backends import TrainingBackend, TrainingResult
from next_chameleons.training.regimes import TrainingRegime, TrainingRegimeSpec

__all__ = [
    "ACTIVATION_EXTRACTORS",
    "CURRICULA",
    "DATASET_ADAPTERS",
    "FEEDBACK_CHANNELS",
    "JUDGES",
    "METRICS",
    "PROBES",
    "PROBE_SELECTORS",
    "REPORTS",
    "TRAINING_BACKENDS",
    "TRAINING_REGIMES",
    "ActivationBatch",
    "ActivationExtractor",
    "DatasetAdapter",
    "DatasetBundle",
    "EnsembleJudge",
    "Example",
    "JudgeResult",
    "Probe",
    "ProbePrediction",
    "ProbeReport",
    "ProjectPaths",
    "Registry",
    "TrainingBackend",
    "TrainingRegime",
    "TrainingRegimeSpec",
    "TrainingResult",
    "load_activation_cache",
    "load_config_group",
    "load_experiment",
    "load_yaml",
    "paper_trigger",
    "save_activation_cache",
]
