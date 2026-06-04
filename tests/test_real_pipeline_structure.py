import json
from pathlib import Path

import pytest

from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.hf.data import HFTextExample, local_dataset_snapshot_path
from next_chameleons.hf.paper_data import materialize_smoke_paper_data
from next_chameleons.hf.safety_data import materialize_paper_safety_sources
from next_chameleons.prefetch import collect_prefetch_assets
from next_chameleons.real_pipeline import (
    _load_examples_for_source,
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
        "apollo_roleplaying",
        "apollo_insider_trading",
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
    assert "scale-safety-research/roleplaying" in dataset_ids
    assert "scale-safety-research/insider_trading" in dataset_ids


def test_offline_dataset_snapshot_resolves_prefetched_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path
        / "datasets--AlignmentResearch--DolusChat"
        / "snapshots"
        / "830e454b54349f8c923a7377e28c17d1796892cb"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = local_dataset_snapshot_path(
        "AlignmentResearch/DolusChat",
        "830e454b54349f8c923a7377e28c17d1796892cb",
    )

    assert resolved == snapshot


def test_text_eval_falls_back_from_activation_artifact_source(monkeypatch) -> None:
    dataset_cfg = {
        "sources": {
            "circuit_breakers_harmful": {
                "kind": "huggingface_dataset",
                "name": "AISC-Linear-Probe-Gen/obfuscated_activations",
                "revision": "abc",
                "text_fallback_source": "jailbreakbench_behaviors",
            },
            "jailbreakbench_behaviors": {
                "kind": "huggingface_dataset",
                "name": "JailbreakBench/JBB-Behaviors",
                "config_name": "behaviors",
                "split": "harmful",
                "revision": "def",
            },
        }
    }
    calls = []

    def fake_load_hf_text_examples(
        *,
        dataset_id,
        revision,
        domain,
        dataset_config_name,
        split,
        **kwargs,
    ):
        calls.append((dataset_id, revision, domain, dataset_config_name, split))
        if dataset_id == "AISC-Linear-Probe-Gen/obfuscated_activations":
            raise ValueError("not a text dataset")
        return [
            HFTextExample(
                example_id="fallback:0",
                text="fallback harmful behavior",
                label=1,
                domain=domain,
                source=dataset_id,
                text_hash="hash",
            )
        ]

    monkeypatch.setattr(
        "next_chameleons.real_pipeline.load_hf_text_examples",
        fake_load_hf_text_examples,
    )

    examples = _load_examples_for_source(
        dataset_cfg,
        "circuit_breakers_harmful",
        split="train",
        max_examples=1,
        real_cfg={},
    )

    assert calls == [
        (
            "AISC-Linear-Probe-Gen/obfuscated_activations",
            "abc",
            "circuit_breakers_harmful",
            None,
            "train",
        ),
        (
            "JailbreakBench/JBB-Behaviors",
            "def",
            "jailbreakbench_behaviors",
            "behaviors",
            "harmful",
        ),
    ]
    assert examples[0].domain == "circuit_breakers_harmful"
    assert examples[0].source == "JailbreakBench/JBB-Behaviors"


def test_fallback_only_text_eval_skips_non_text_source(monkeypatch) -> None:
    dataset_cfg = {
        "sources": {
            "circuit_breakers_harmful": {
                "kind": "huggingface_dataset",
                "name": "AISC-Linear-Probe-Gen/obfuscated_activations",
                "revision": "abc",
                "text_fallback_source": "jailbreakbench_behaviors",
                "text_eval_mode": "fallback_only",
            },
            "jailbreakbench_behaviors": {
                "kind": "huggingface_dataset",
                "name": "JailbreakBench/JBB-Behaviors",
                "config_name": "behaviors",
                "split": "harmful",
                "revision": "def",
            },
        }
    }
    calls = []

    def fake_load_hf_text_examples(
        *,
        dataset_id,
        revision,
        domain,
        dataset_config_name,
        split,
        **kwargs,
    ):
        calls.append((dataset_id, revision, domain, dataset_config_name, split))
        return [
            HFTextExample(
                example_id="fallback:0",
                text="fallback harmful behavior",
                label=1,
                domain=domain,
                source=dataset_id,
                text_hash="hash",
            )
        ]

    monkeypatch.setattr(
        "next_chameleons.real_pipeline.load_hf_text_examples",
        fake_load_hf_text_examples,
    )

    examples = _load_examples_for_source(
        dataset_cfg,
        "circuit_breakers_harmful",
        split="train",
        max_examples=1,
        real_cfg={},
    )

    assert calls == [
        (
            "JailbreakBench/JBB-Behaviors",
            "def",
            "jailbreakbench_behaviors",
            "behaviors",
            "harmful",
        )
    ]
    assert examples[0].domain == "circuit_breakers_harmful"


def test_paper_safety_source_materializer_writes_balanced_local_manifests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeSplit(list):
        def select(self, indices):
            return FakeSplit([self[index] for index in indices])

    class FakeDatasets:
        @staticmethod
        def load_dataset(dataset_id, *args, **kwargs):
            if dataset_id == "scale-safety-research/roleplaying":
                return {
                    "honest": FakeSplit([{"text": "honest roleplay"}]),
                    "deceptive": FakeSplit([{"text": "deceptive roleplay"}]),
                }
            if dataset_id == "scale-safety-research/insider_trading":
                return {
                    "honest": FakeSplit([{"text": "honest insider"}]),
                    "deceptive": FakeSplit([{"text": "deceptive insider"}]),
                }
            if dataset_id == "JailbreakBench/JBB-Behaviors":
                return {
                    "benign": FakeSplit([{"Goal": "benign goal"}]),
                    "harmful": FakeSplit([{"Goal": "harmful goal"}]),
                }
            raise AssertionError(dataset_id)

    monkeypatch.setattr(
        "next_chameleons.hf.safety_data.require_datasets",
        lambda: FakeDatasets,
    )

    report_path = materialize_paper_safety_sources(
        raw_cache_root=tmp_path / "raw",
        artifact_root=tmp_path / "artifacts",
        dataset_cfg={
            "sources": {
                "apollo_roleplaying": {"revision": "role-sha"},
                "apollo_insider_trading": {"revision": "insider-sha"},
                "jailbreakbench_behaviors": {"revision": "jbb-sha"},
            }
        },
    )

    assert report_path.exists()
    apollo = tmp_path / "raw" / "apollo_repe_deception" / "examples.jsonl"
    synthetic = tmp_path / "raw" / "synthetic_harmful" / "examples.jsonl"
    assert apollo.exists()
    assert synthetic.exists()
    assert sum(1 for _ in apollo.open()) == 4
    assert sum(1 for _ in synthetic.open()) == 2
    report = json.loads(report_path.read_text())
    assert {source["source_name"] for source in report["sources"]} == {
        "apollo_repe_deception",
        "synthetic_harmful",
    }
