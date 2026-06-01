from pathlib import Path

import numpy as np

from next_chameleons.artifacts import DatasetManifest, SourceReference
from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.objectives import (
    build_paper_scenarios,
    chameleon_objective,
    paper_chameleon_objective,
)
from next_chameleons.probes.base import ProbePrediction
from next_chameleons.real_runs import build_real_run_plan


def test_chameleon_objective_combines_behavior_and_probe_loss() -> None:
    prediction = ProbePrediction(
        scores=np.array([0.1, 0.8, 0.2, 0.6]),
        labels=np.array([0, 1, 0, 1]),
        threshold=0.5,
    )

    loss = chameleon_objective(
        reference_distribution=np.array([0.5, 0.5]),
        candidate_distribution=np.array([0.45, 0.55]),
        probe_predictions=[prediction],
        behavior_weight=0.2,
        obfuscation_weight=1.0,
    )

    assert loss.behavior_kl > 0
    assert loss.obfuscation_loss == 0.7
    assert loss.total > loss.behavior_kl * 0.2


def test_real_run_plan_accepts_pinned_sources() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiment = load_experiment("paper_reproduction", paths=paths)
    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=Path("/scratch/next_chameleons_artifacts"),
        raw_cache_root=Path("/scratch/next_chameleons_raw_cache"),
    )

    plan.validate()


def test_paper_objective_and_scenarios_match_table_one() -> None:
    scenarios = build_paper_scenarios(("html", "german"))

    assert [scenario.target for scenario in scenarios[:3]] == [1.0, 0.0, 1.0]
    assert scenarios[1].trigger_concept == "html"
    assert scenarios[2].trigger_concept == "german"

    loss = paper_chameleon_objective(
        reference_distribution=np.array([0.5, 0.5]),
        candidate_distribution=np.array([0.49, 0.51]),
        probe_scores=np.array([0.9, 0.1, 0.8]),
        probe_targets=np.array([1.0, 0.0, 1.0]),
    )

    assert loss.obfuscation_loss < 0.02
    assert loss.total > 0


def test_manifest_still_rejects_unpinned_sources() -> None:
    manifest = DatasetManifest(
        dataset_id="placeholder",
        source=SourceReference(
            kind="huggingface_dataset",
            name="placeholder",
            revision="to-pin-before-real-run",
            checksum="to-record-after-download",
        ),
        split="train",
        num_examples=0,
        label_names=[],
    )

    try:
        manifest.validate()
    except ValueError as exc:
        assert "lacks pinned" in str(exc)
    else:
        raise AssertionError("Expected unpinned manifest to fail")
