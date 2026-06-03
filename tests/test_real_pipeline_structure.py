import json
from pathlib import Path

import pytest

from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.hf.paper_data import materialize_smoke_paper_data
from next_chameleons.prefetch import collect_prefetch_assets
from next_chameleons.real_pipeline import (
    _resolve_real_experiment,
    run_paper_data_check,
    run_paper_readiness_check,
)
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
    assert experiment["real_run"]["train_source"] == "benign_synthetic_4697"
    assert experiment["real_run"]["activation_pooling"] == "mean_generation_tokens"


def test_adaptive_real_config_resolves_model_and_train_backend() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    experiment = _resolve_real_experiment("adaptive_gemma2_2b_real", paths)

    assert experiment["model_config"]["id"] == "gemma_2_2b_it"
    assert experiment["train_backend_config"]["id"] == "qlora"
    assert experiment["train_backend_config"]["max_steps"] == 100


def test_all_paper_real_configs_resolve_and_are_readiness_clean() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiments = [
        "paper_minimal_gemma2_2b_real",
        "paper_gemma2_9b_real",
        "paper_gemma2_2b_real",
        "paper_llama31_8b_real",
        "paper_qwen25_7b_real",
        "language_generalization_real",
    ]

    for name in experiments:
        experiment = _resolve_real_experiment(name, paths)
        assert experiment["track"] == "paper_real"
        if name == "paper_minimal_gemma2_2b_real":
            assert set(experiment["seeds"]) == {17}
        else:
            assert set(experiment["seeds"]) == {17, 23, 41}
        assert experiment["real_run"]["default_probe_layer"] == 12
        assert experiment["train_backend_config"]["objective"] == "kl_plus_frozen_probe_mse"
        report_path = run_paper_readiness_check(name, paths=paths)
        assert json.loads(report_path.read_text())["ready"] is True


def test_minimal_paper_data_check_validates_concept_probe_balance(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    raw_cache = tmp_path / "raw_cache"
    artifact_root = tmp_path / "artifacts"

    materialize_smoke_paper_data(
        raw_cache_root=raw_cache,
        artifact_root=artifact_root,
        examples_per_concept=1,
    )
    report_path = run_paper_data_check(
        "paper_minimal_gemma2_2b_real",
        artifact_root=artifact_root,
        raw_cache_root=raw_cache,
        paths=paths,
    )

    report = json.loads(report_path.read_text())
    assert report["ready"] is True
    assert report["train_concepts"] == [
        "html",
        "biology-focused",
        "mathematical",
        "comforting",
    ]
    for counts in report["concept_probe_trainability"].values():
        assert counts["trainable"] is True


def test_paper_materialization_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    from next_chameleons.real_pipeline import run_paper_materialize_data

    with pytest.raises(ValueError, match="smoke=True or generate=True"):
        run_paper_materialize_data(
            artifact_root=tmp_path / "artifacts",
            raw_cache_root=tmp_path / "raw",
            smoke=True,
            generate=True,
        )


def test_paper_data_check_can_require_locally_rated_examples(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    raw_cache = tmp_path / "raw_cache"
    artifact_root = tmp_path / "artifacts"
    materialize_smoke_paper_data(
        raw_cache_root=raw_cache,
        artifact_root=artifact_root,
        examples_per_concept=1,
    )

    with pytest.raises(ValueError, match="require locally rated examples"):
        run_paper_data_check(
            "paper_minimal_gemma2_2b_real",
            artifact_root=artifact_root,
            raw_cache_root=raw_cache,
            require_rated=True,
            paths=paths,
        )


def test_prefetch_collects_models_datasets_and_generator() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    assets = collect_prefetch_assets(
        ["paper_minimal_gemma2_2b_real"],
        include_generator=True,
        paths=paths,
    )

    model_ids = {asset["repo_id"] for asset in assets["models"]}
    dataset_ids = {asset["repo_id"] for asset in assets["datasets"]}
    assert "google/gemma-2-2b-it" in model_ids
    assert "google/gemma-2-27b-it" in model_ids
    assert "AlignmentResearch/DolusChat" in dataset_ids
    assert "AISC-Linear-Probe-Gen/obfuscated_activations" in dataset_ids
