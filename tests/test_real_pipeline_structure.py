from pathlib import Path

from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.real_pipeline import _resolve_real_experiment
from next_chameleons.real_runs import build_real_run_plan


def test_paper_sources_are_pinned_for_real_runs() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiment = load_experiment("paper_gemma2_2b_real", paths=paths)
    if "model_config" not in experiment:
        experiment = _resolve_real_experiment("paper_gemma2_2b_real", paths)

    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=Path("/tmp/next_chameleons_artifacts"),
        raw_cache_root=Path("/tmp/next_chameleons_raw_cache"),
    )

    plan.validate()
    assert {
        "ultrachat",
        "benign_synthetic_4697",
        "dolus_deception",
        "apollo_repe_deception",
        "circuit_breakers_harmful",
        "synthetic_harmful",
        "jailbreakbench_behaviors",
    } == set(plan.dataset_sources)
    assert "google/gemma-2-2b-it" in plan.model_ids
    assert experiment["train_backend_config"]["max_steps"] == 100


def test_adaptive_real_config_resolves_model_and_train_backend() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    experiment = _resolve_real_experiment("adaptive_gemma2_2b_real", paths)

    assert experiment["model_config"]["id"] == "gemma_2_2b_it"
    assert experiment["train_backend_config"]["id"] == "qlora"
    assert experiment["train_backend_config"]["max_steps"] == 100
