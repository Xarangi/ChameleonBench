"""Stable public API for designing Next Chameleons experiments."""

from __future__ import annotations

from next_chameleons.activations import (
    ActivationBatch,
    ActivationExtractor,
    load_activation_cache,
    save_activation_cache,
)
from next_chameleons.config import (
    ConfigSource,
    ProjectPaths,
    find_config_path,
    load_config_group,
    load_experiment,
    load_yaml,
)
from next_chameleons.datasets import DatasetAdapter, DatasetBundle, Example, paper_trigger
from next_chameleons.judges import EnsembleJudge, JudgeResult
from next_chameleons.plugins import load_builtin_plugins, load_entrypoint_plugins, load_plugins
from next_chameleons.probes.atlas import MeanDifferenceProbe, QuadraticProbe
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
from next_chameleons.training.runner import (
    TrainingRegimeRunner,
    TrainingRunMode,
    TrainingRunRequest,
)


def run_adaptive_smoke(*args, **kwargs):
    """Run the synthetic adaptive smoke experiment.

    Imported lazily so the stable public API remains cheap to import.
    """

    from next_chameleons.pipeline import run_adaptive_smoke as run

    return run(*args, **kwargs)


def run_experiment_smoke(*args, **kwargs):
    """Run a synthetic experiment smoke config."""

    from next_chameleons.pipeline import run_experiment_smoke as run

    return run(*args, **kwargs)


def run_paper_dry_run(*args, **kwargs):
    """Validate and report the paper dry-run plan."""

    from next_chameleons.pipeline import run_paper_dry_run as run

    return run(*args, **kwargs)


def validate_paper_configs(*args, **kwargs):
    """Validate paper replication configs."""

    from next_chameleons.pipeline import validate_paper_configs as run

    return run(*args, **kwargs)


def run_paper_data_check(*args, **kwargs):
    """Check paper data manifests and redaction invariants."""

    from next_chameleons.real_pipeline import run_paper_data_check as run

    return run(*args, **kwargs)


def run_paper_materialize_data(*args, **kwargs):
    """Materialize paper replication data under the configured raw cache."""

    from next_chameleons.real_pipeline import run_paper_materialize_data as run

    return run(*args, **kwargs)


def run_paper_materialize_safety_data(*args, **kwargs):
    """Materialize paper safety eval data under the configured raw cache."""

    from next_chameleons.real_pipeline import run_paper_materialize_safety_data as run

    return run(*args, **kwargs)


def run_paper_readiness_check(*args, **kwargs):
    """Run paper-readiness validation."""

    from next_chameleons.real_pipeline import run_paper_readiness_check as run

    return run(*args, **kwargs)


def run_real_adaptive(*args, **kwargs):
    """Run a real adaptive chameleon experiment."""

    from next_chameleons.real_pipeline import run_real_adaptive as run

    return run(*args, **kwargs)


def run_real_eval(*args, **kwargs):
    """Run real post-hoc probe evaluation."""

    from next_chameleons.real_pipeline import run_real_eval as run

    return run(*args, **kwargs)


def run_real_train(*args, **kwargs):
    """Run real chameleon fine-tuning."""

    from next_chameleons.real_pipeline import run_real_train as run

    return run(*args, **kwargs)

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
    "ConfigSource",
    "DatasetAdapter",
    "DatasetBundle",
    "EnsembleJudge",
    "Example",
    "JudgeResult",
    "MeanDifferenceProbe",
    "Probe",
    "ProbePrediction",
    "ProbeReport",
    "ProjectPaths",
    "QuadraticProbe",
    "Registry",
    "TrainingBackend",
    "TrainingRegime",
    "TrainingRegimeRunner",
    "TrainingRegimeSpec",
    "TrainingResult",
    "TrainingRunMode",
    "TrainingRunRequest",
    "find_config_path",
    "load_activation_cache",
    "load_builtin_plugins",
    "load_config_group",
    "load_entrypoint_plugins",
    "load_experiment",
    "load_plugins",
    "load_yaml",
    "paper_trigger",
    "run_adaptive_smoke",
    "run_experiment_smoke",
    "run_paper_data_check",
    "run_paper_dry_run",
    "run_paper_materialize_data",
    "run_paper_materialize_safety_data",
    "run_paper_readiness_check",
    "run_real_adaptive",
    "run_real_eval",
    "run_real_train",
    "save_activation_cache",
    "validate_paper_configs",
]
