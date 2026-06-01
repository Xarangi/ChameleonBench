from pathlib import Path

from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.plugins import load_builtin_plugins
from next_chameleons.registry import (
    ACTIVATION_EXTRACTORS,
    CURRICULA,
    DATASET_ADAPTERS,
    FEEDBACK_CHANNELS,
    JUDGES,
    PROBE_SELECTORS,
    PROBES,
    TRAINING_BACKENDS,
    TRAINING_REGIMES,
)


def test_paper_config_loads_all_reported_models() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiment = load_experiment("paper_reproduction", paths=paths)

    assert experiment["id"] == "paper_reproduction"
    assert {model["id"] for model in experiment["model_configs"]} == {
        "gemma_2_9b_it",
        "gemma_2_2b_it",
        "llama_3_1_8b_instruct",
        "qwen_2_5_7b_instruct",
    }
    assert experiment["dataset_config"]["benign_concepts"] == [
        "german",
        "comforting",
        "confused",
        "html",
        "jokey",
        "biology-focused",
        "all-caps",
        "literature-focused",
        "finnish",
        "mathematical",
        "chemistry-based",
    ]


def test_sweep_configs_exist_for_fast_iteration() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    assert (paths.configs / "sweep" / "quick_iteration.yaml").exists()
    assert (paths.configs / "sweep" / "adaptive_curriculum.yaml").exists()


def test_probe_discovery_is_a_mechanism_not_experiment() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    mechanism = load_config_group("mechanism", "probe_discovery", paths=paths)

    assert mechanism["kind"] == "mechanism"
    assert "curriculum_probe_escalation" in mechanism["used_by"]
    assert not (paths.configs / "experiment" / "probe_discovery.yaml").exists()


def test_integrated_mechanism_configs_are_not_standalone_experiments() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    mechanisms = {
        "probe_discovery",
        "feedback_channels",
        "inference_monitoring",
        "capability_retention",
        "source_redaction_policy",
    }

    for name in mechanisms:
        mechanism = load_config_group("mechanism", name, paths=paths)
        assert mechanism["kind"] == "mechanism"
        assert mechanism["used_by"]
        assert not (paths.configs / "experiment" / f"{name}.yaml").exists()


def test_experiment_mechanism_references_resolve() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    for config_path in sorted((paths.configs / "experiment").glob("*.yaml")):
        experiment = load_config_group("experiment", config_path.stem, paths=paths)
        for ref in experiment.get("mechanisms", {}).values():
            expected = paths.root / ref["config"]
            assert expected.exists(), f"{config_path.name} references missing {expected}"


def test_builtin_registries_are_populated() -> None:
    load_builtin_plugins()

    assert "synthetic" in DATASET_ADAPTERS.names()
    assert "synthetic" in ACTIVATION_EXTRACTORS.names()
    assert {"linear", "mlp", "attention", "geometry", "oracle"}.issubset(set(PROBES.names()))
    assert "ensemble" in JUDGES.names()
    assert "active_probe_pool" in PROBE_SELECTORS.names()
    assert {"dry_run", "lora", "qlora", "full_finetune", "grpo"}.issubset(
        set(TRAINING_BACKENDS.names())
    )
    assert {
        "paper_sft_kl_probe_mse",
        "multi_probe_evasion",
        "curriculum_probe_escalation",
        "multiturn_self_redteam",
        "grpo_probe_reward",
    }.issubset(set(TRAINING_REGIMES.names()))
    assert {"binary_verdict", "highest_feedback"}.issubset(set(FEEDBACK_CHANNELS.names()))
    assert "probe_escalation" in CURRICULA.names()
