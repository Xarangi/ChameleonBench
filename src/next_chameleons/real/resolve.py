"""Experiment resolution and readiness gates for real runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.reports.json_report import JsonReport

SUPPORTED_BEHAVIOR_LOSS_MODES = (
    "teacher_forced_ce_on_behavior_samples",
    "base_on_policy_completions",
    "teacher_forced_text",
    "scenario_text",
)

PAPER_BENIGN_TRAIN_SOURCES = (
    "benign_preset_official",
    "benign_synthetic_4697",
)


def resolve_real_experiment(experiment_name: str, paths: ProjectPaths) -> dict[str, Any]:
    experiment = load_experiment(experiment_name, paths=paths)
    if "model_config" not in experiment and isinstance(experiment.get("model"), str):
        experiment["model_config"] = load_config_group("model", experiment["model"], paths=paths)
    if "model_config" not in experiment and "base_model_config" in experiment:
        experiment["model_config"] = experiment["base_model_config"]
    if "train_backend_config" not in experiment:
        train_name = experiment.get("train_backend")
        if train_name is None:
            train_name = experiment.get("train_backends", ["qlora"])[0]
        experiment["train_backend_config"] = load_config_group(
            "train",
            str(train_name),
            paths=paths,
        )
    experiment["train_backend_config"] = dict(experiment["train_backend_config"])
    if experiment.get("real_run", {}).get("max_steps") is not None:
        experiment["train_backend_config"]["max_steps"] = int(experiment["real_run"]["max_steps"])
    if experiment.get("track") == "adaptive_real":
        experiment["train_backend_config"]["max_steps"] = int(
            experiment.get("real_run", {}).get(
                "max_steps_per_round",
                experiment["train_backend_config"].get("max_steps", 100),
            )
        )
    return experiment


def expected_probe_layer(experiment: dict[str, Any]) -> int:
    """Primary probe layer for an experiment, honoring per-model paper layers."""

    model_cfg = experiment.get("model_config", {})
    if model_cfg.get("paper_target_layer") is not None:
        return int(model_cfg["paper_target_layer"])
    return int(experiment.get("real_run", {}).get("default_probe_layer", 12))


def run_paper_readiness_check(
    experiment_name: str,
    *,
    paths: ProjectPaths | None = None,
) -> Path:
    """Write a paper-readiness report for a real replication config.

    Protocol-shaped requirements (behavior loss mode, probe layer, train source)
    are read from the experiment's own configuration so versioned protocol
    variants stay submittable; only structurally required invariants fail hard.
    """

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment.get("real_run", {}))
    train_cfg = experiment.get("train_backend_config", {})
    failures: list[str] = []
    if experiment.get("track") != "paper_real":
        failures.append("experiment track must be paper_real")
    if train_cfg.get("objective") != "kl_plus_frozen_probe_mse":
        failures.append("train backend objective must be kl_plus_frozen_probe_mse")
    behavior_loss_mode = str(
        train_cfg.get("behavior_loss_mode", train_cfg.get("behavior_kl_mode", ""))
    )
    if behavior_loss_mode not in SUPPORTED_BEHAVIOR_LOSS_MODES:
        failures.append(
            "train backend behavior_loss_mode must be one of "
            f"{sorted(SUPPORTED_BEHAVIOR_LOSS_MODES)}"
        )
    if str(real_cfg.get("train_source")) not in PAPER_BENIGN_TRAIN_SOURCES:
        failures.append(
            f"paper_real train_source must be one of {sorted(PAPER_BENIGN_TRAIN_SOURCES)}"
        )
    configured_layer = int(real_cfg.get("default_probe_layer", -1))
    if configured_layer != expected_probe_layer(experiment):
        failures.append(
            "default_probe_layer must match the model paper_target_layer "
            f"({expected_probe_layer(experiment)})"
        )
    if str(real_cfg.get("activation_pooling")) != "mean_generation_tokens":
        failures.append("activation_pooling must be mean_generation_tokens")
    if float(real_cfg.get("eval_holdout_fraction", 0.0)) <= 0.0:
        failures.append("eval_holdout_fraction must be positive for post-hoc probe reporting")
    single_seed_experiments = {
        "paper_minimal_gemma2_2b_real",
        "paper_golden_serteal_gemma2_9b",
    }
    expected_seed_set = (
        {17} if experiment.get("id") in single_seed_experiments else {17, 23, 41}
    )
    if set(experiment.get("seeds", [])) != expected_seed_set:
        failures.append(f"seeds must be {sorted(expected_seed_set)}")
    controls = set(experiment.get("evaluation", {}).get("controls", []))
    if "base_model_trigger_control" not in controls:
        failures.append("missing base_model_trigger_control")
    if "selectivity_heatmap" not in controls:
        failures.append("missing selectivity_heatmap")
    report = {
        "track": "paper_readiness_check",
        "experiment_id": experiment["id"],
        "ready": not failures,
        "failures": failures,
        "behavior_loss_mode": behavior_loss_mode,
        "probe_layer": configured_layer,
        "checked_requirements": [
            "paper_real_track",
            "frozen_probe_mse_objective",
            "supported_behavior_loss_mode",
            "benign_concept_train_source",
            "model_matched_primary_probe_layer",
            "mean_generation_token_pooling",
            "heldout_posthoc_probe_eval",
            "configured_training_seeds",
            "base_trigger_and_selectivity_controls",
        ],
    }
    output = Path("runs") / "readiness" / f"{experiment_name}.json"
    return JsonReport().write(output, report)
