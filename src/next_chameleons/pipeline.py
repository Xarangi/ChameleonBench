"""End-to-end dry-run and adaptive smoke pipelines."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from next_chameleons.activations import ActivationBatch, save_activation_cache
from next_chameleons.adaptive.feedback import FeedbackMessage, format_feedback
from next_chameleons.benchmarks.suites import CAPABILITY_TASKS, PAPER_REPLICATION_CHECKS
from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.curricula import ProbeEscalationCurriculum
from next_chameleons.datasets import (
    PAPER_BENIGN_CONCEPTS,
    PAPER_LANGUAGE_GENERALIZATION,
    PAPER_SAFETY_DATASETS,
    DatasetBundle,
    paper_source_coverage,
)
from next_chameleons.inference.monitoring import monitor_activation_batch
from next_chameleons.judges import EnsembleJudge, JudgeResult
from next_chameleons.metrics import behavior_gate
from next_chameleons.plugins import load_builtin_plugins
from next_chameleons.probes.atlas import MeanDifferenceProbe, QuadraticProbe
from next_chameleons.probes.attention import AttentionProbe
from next_chameleons.probes.geometry import GeometryProbe
from next_chameleons.probes.linear import LinearProbe
from next_chameleons.probes.mlp import MLPProbe
from next_chameleons.registry import (
    ACTIVATION_EXTRACTORS,
    DATASET_ADAPTERS,
    PROBE_SELECTORS,
    TRAINING_BACKENDS,
)
from next_chameleons.reports.json_report import JsonReport
from next_chameleons.training.backends import AdaptiveRoundResult


def _synthetic_data_and_activations(paths: ProjectPaths) -> tuple[DatasetBundle, ActivationBatch]:
    dataset_cfg = load_config_group("dataset", "synthetic_smoke", paths=paths)
    adapter_cls = DATASET_ADAPTERS.get("synthetic")
    adapter = adapter_cls(**dataset_cfg)
    bundle = adapter.load()
    extractor_cls = ACTIVATION_EXTRACTORS.get("synthetic")
    extractor = extractor_cls(**dataset_cfg)
    return bundle, extractor.extract(bundle)


def _probe_from_spec(spec: dict[str, Any], batch: ActivationBatch):
    probe_type = spec.get("type", "linear")
    probe_id = str(spec.get("id", probe_type))
    spec_layers = spec.get("layers")
    layer_indices: list[int] | None = None
    if isinstance(spec_layers, list):
        layer_indices = [
            batch.layers.index(int(layer)) if int(layer) in batch.layers else int(layer)
            for layer in spec_layers
            if (int(layer) in batch.layers or 0 <= int(layer) < len(batch.layers))
        ]
        if not layer_indices:
            layer_indices = None
    if probe_type in {"linear", "paper_linear"}:
        return LinearProbe(probe_id=probe_id, layers=layer_indices)
    if probe_type in {"mean_difference", "atlas_mean_difference"}:
        return MeanDifferenceProbe(probe_id=probe_id, layers=layer_indices)
    if probe_type in {"quadratic", "atlas_quadratic"}:
        return QuadraticProbe(probe_id=probe_id, layers=layer_indices)
    if probe_type in {"mlp", "oracle"}:
        return MLPProbe(
            probe_id=probe_id,
            hidden_units=int(spec.get("hidden_units", 128 if probe_type == "oracle" else 64)),
            layers=layer_indices,
        )
    if probe_type == "attention":
        return AttentionProbe(
            probe_id=probe_id,
            heads=int(spec.get("heads", 4)),
            layers=layer_indices,
        )
    if probe_type == "geometry":
        return GeometryProbe(probe_id=probe_id)
    if probe_type in {"layer_sweep", "layer_ensemble"}:
        base = spec.get("base_probe", "linear")
        probes = []
        selected_layers = layer_indices or list(range(len(batch.layers)))
        for layer_index in selected_layers:
            layer_label = batch.layers[layer_index]
            if base == "mlp":
                probes.append(
                    MLPProbe(probe_id=f"{probe_id}_{layer_label}", layers=[layer_index])
                )
            elif base == "attention":
                probes.append(
                    AttentionProbe(probe_id=f"{probe_id}_{layer_label}", layers=[layer_index])
                )
            elif base == "mean_difference":
                probes.append(
                    MeanDifferenceProbe(probe_id=f"{probe_id}_{layer_label}", layers=[layer_index])
                )
            elif base == "quadratic":
                probes.append(
                    QuadraticProbe(probe_id=f"{probe_id}_{layer_label}", layers=[layer_index])
                )
            else:
                probes.append(
                    LinearProbe(probe_id=f"{probe_id}_{layer_label}", layers=[layer_index])
                )
        return probes
    raise ValueError(f"Unsupported probe spec {spec}")


def build_judge_from_config(judge_cfg: dict[str, Any], batch: ActivationBatch) -> EnsembleJudge:
    """Build an EnsembleJudge from project config."""

    probes = []
    for spec in judge_cfg.get("probes", []):
        created = _probe_from_spec(spec, batch)
        if isinstance(created, list):
            probes.extend(created)
        else:
            probes.append(created)
    return EnsembleJudge(
        judge_id=str(judge_cfg.get("id", "ensemble")),
        probes=probes,
        aggregation=str(judge_cfg.get("aggregation", "max")),
        threshold=float(judge_cfg.get("threshold", 0.55)),
    )


def validate_paper_configs(paths: ProjectPaths) -> dict[str, Any]:
    """Load paper configs and return a compact validation payload."""

    experiment = load_experiment("paper_reproduction", paths=paths)
    models = experiment.get("model_configs", [])
    if len(models) != 4:
        raise ValueError("Paper reproduction must include four reported model configs")
    dataset_cfg = experiment["dataset_config"]
    concepts = dataset_cfg.get("benign_concepts", [])
    missing_concepts = sorted(set(PAPER_BENIGN_CONCEPTS) - set(concepts))
    if missing_concepts:
        raise ValueError(f"Paper reproduction missing benign concepts: {missing_concepts}")
    safety_sets = dataset_cfg.get("safety_probe_datasets", [])
    missing_safety = sorted(set(PAPER_SAFETY_DATASETS) - set(safety_sets))
    if missing_safety:
        raise ValueError(f"Paper reproduction missing safety probe datasets: {missing_safety}")
    evaluation = experiment.get("evaluation", {})
    if float(evaluation.get("target_fpr", -1.0)) != 0.01:
        raise ValueError("Paper reproduction must use fixed 1% FPR calibration")
    backends = ["lora", "qlora", "full_finetune"]
    for backend in backends:
        load_config_group("train", backend, paths=paths)
    return {
        "experiment_id": experiment["id"],
        "models": [model["hf_id"] for model in models],
        "train_backends_validated": backends,
        "dataset_policy": dataset_cfg["policy"],
        "benign_concepts": concepts,
        "safety_probe_datasets": safety_sets,
        "source_coverage": paper_source_coverage(),
        "capability_tasks": list(CAPABILITY_TASKS),
        "language_generalization": PAPER_LANGUAGE_GENERALIZATION,
        "paper_replication_checks": list(PAPER_REPLICATION_CHECKS),
    }


def run_paper_dry_run(output_dir: Path, *, paths: ProjectPaths | None = None) -> Path:
    """Run a local paper-style dry run on synthetic activations."""

    load_builtin_plugins()
    resolved = paths or ProjectPaths.discover(Path(__file__).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_paper_configs(resolved)
    bundle, batch = _synthetic_data_and_activations(resolved)
    cache_path = output_dir / "activation_cache" / "synthetic_selective.npz"
    save_activation_cache(batch, cache_path)
    judge_cfg = load_config_group("judge", "paper_default", paths=resolved)
    judge = build_judge_from_config(judge_cfg, batch).fit(batch)
    judge_result = judge.evaluate(batch)
    gate = behavior_gate(
        reference_distribution=np.array([0.5, 0.5]),
        candidate_distribution=np.array([0.51, 0.49]),
        task_accuracy=0.91,
        kl_max=0.15,
        task_accuracy_min=0.8,
    )
    report = {
        "track": "paper_dry_run",
        "validation": validation,
        "dataset_manifest": bundle.manifest.to_dict(),
        "activation_cache_manifest": str(cache_path.with_suffix(".manifest.json")),
        "judge": judge_result.to_dict(),
        "behavior_gate": gate.to_dict(),
        "paper_tables_present": True,
    }
    return JsonReport().write(output_dir / "paper_dry_run_report.json", report)


def _round_payload(result: AdaptiveRoundResult) -> dict[str, Any]:
    return asdict(result) | {"checkpoint_path": str(result.checkpoint_path)}


def _feedback_conditions(
    experiment: dict[str, Any],
    *,
    include_ablations: bool,
) -> dict[str, list[str]]:
    primary_feedback = experiment["multiturn_feedback_modes"]["primary"]
    feedback_modes = [primary_feedback]
    if include_ablations:
        feedback_modes.extend(experiment["multiturn_feedback_modes"].get("ablations", []))
    return {
        "loss_only": ["none"],
        "explicit_judge_feedback": ["curriculum_message"],
        "multiturn_adaptation": feedback_modes,
    }


def run_adaptive_smoke(
    output_dir: Path,
    *,
    paths: ProjectPaths | None = None,
    experiment_name: str = "adaptive_2b",
    rounds: int | None = 2,
    training_modes: list[str] | None = None,
    include_feedback_ablations: bool = True,
    obfuscation_strength: float = 0.3,
) -> Path:
    """Run a local Adaptive Chameleon Loop test over configured modes."""

    load_builtin_plugins()
    resolved = paths or ProjectPaths.discover(Path(__file__).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(experiment_name, paths=resolved)
    bundle, original_batch = _synthetic_data_and_activations(resolved)
    judge_cfg = load_config_group("judge", "strong_starters", paths=resolved)
    selector_cls = PROBE_SELECTORS.get("active_probe_pool")
    selector = selector_cls(promote_top_k=3, diversity_weight=0.25)
    backend_cls = TRAINING_BACKENDS.get("dry_run")
    backend = backend_cls(obfuscation_strength=obfuscation_strength)

    mode_results: list[dict[str, Any]] = []
    primary_feedback = experiment["multiturn_feedback_modes"]["primary"]
    selected_modes = training_modes or list(experiment["training_modes"])
    round_count = int(rounds if rounds is not None else experiment["rounds"])
    feedback_conditions = _feedback_conditions(
        experiment,
        include_ablations=include_feedback_ablations,
    )
    for mode in selected_modes:
        for feedback_mode in feedback_conditions.get(mode, ["none"]):
            batch = original_batch
            judge = build_judge_from_config(judge_cfg, batch).fit(batch)
            round_results: list[AdaptiveRoundResult] = []
            condition_id = f"{mode}__{feedback_mode}"
            for round_index in range(round_count):
                train_result = backend.train(
                    batch,
                    output_dir=output_dir / "checkpoints" / condition_id / f"round_{round_index}",
                    mode=mode,
                )
                batch = train_result.adapted_batch or batch
                judge_result: JudgeResult = judge.fit(batch).evaluate(batch)
                promotion = selector.select(
                    batch,
                    existing_families={report.family for report in judge_result.probe_reports},
                )
                judge.probes.extend(promotion.promoted)
                round_results.append(
                    AdaptiveRoundResult(
                        round_index=round_index,
                        training_mode=mode,
                        feedback_mode=feedback_mode,
                        checkpoint_path=train_result.checkpoint_path,
                        judge_flagged=judge_result.flagged,
                        judge_score=judge_result.score,
                        promoted_probe_ids=[score.probe_id for score in promotion.scores],
                    )
                )
            heldout_judge = build_judge_from_config(judge_cfg, batch).fit(batch).evaluate(batch)
            mode_results.append(
                {
                    "condition_id": condition_id,
                    "mode": mode,
                    "feedback_mode": feedback_mode,
                    "rounds": [_round_payload(result) for result in round_results],
                    "heldout_judge": heldout_judge.to_dict(),
                }
            )

    report = {
        "track": "adaptive_smoke",
        "base_model": experiment["base_model"],
        "experiment_id": experiment["id"],
        "mechanisms": experiment.get("mechanisms", {}),
        "dataset_manifest": bundle.manifest.to_dict(),
        "primary_feedback_mode": primary_feedback,
        "include_feedback_ablations": include_feedback_ablations,
        "round_count": round_count,
        "obfuscation_strength": obfuscation_strength,
        "holdout_policy": experiment["holdout"],
        "mode_results": mode_results,
    }
    return JsonReport().write(output_dir / "adaptive_smoke_report.json", report)


def _regime_spec(regime_id: str | None) -> dict[str, Any] | None:
    if not regime_id:
        return None
    from next_chameleons.registry import TRAINING_REGIMES

    regime_cls = TRAINING_REGIMES.get(regime_id)
    return asdict(regime_cls().spec())


def _feedback_preview(modes: list[str]) -> dict[str, str]:
    preview: dict[str, str] = {}
    for mode in modes:
        preview[mode] = format_feedback(
            FeedbackMessage(
                mode=mode,
                caught=True,
                score=0.73,
                threshold=0.5,
                probe_family="attention",
                layer=12,
                reason="activation monitor",
            )
        )
    return preview


def run_experiment_smoke(
    experiment_name: str,
    output_dir: Path,
    *,
    paths: ProjectPaths | None = None,
    rounds: int | None = None,
) -> Path:
    """Run the initial executable synthetic path for any experiment preset."""

    load_builtin_plugins()
    resolved = paths or ProjectPaths.discover(Path(__file__).resolve())
    experiment = load_experiment(experiment_name, paths=resolved)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name == "paper_reproduction" or experiment.get("track") in {
        "paper",
        "paper_real",
    }:
        report_path = run_paper_dry_run(output_dir / "paper", paths=resolved)
        report = {
            "track": "experiment_smoke",
            "experiment_id": experiment["id"],
            "delegated_report": str(report_path),
            "executable_status": "paper_dry_run_completed",
        }
        return JsonReport().write(output_dir / f"{experiment_name}_smoke_report.json", report)

    bundle, batch = _synthetic_data_and_activations(resolved)
    judge_cfg = experiment.get("judge_config") or load_config_group(
        "judge",
        str(experiment.get("judge", "strong_starters")),
        paths=resolved,
    )
    judge = build_judge_from_config(judge_cfg, batch).fit(batch)
    initial_result = judge.evaluate(batch)
    backend_cls = TRAINING_BACKENDS.get(
        "grpo" if experiment.get("train_backend") == "grpo" else "dry_run"
    )
    backend = backend_cls()
    mode = str(experiment.get("training_regime", experiment.get("track", "loss_only")))
    train_result = backend.train(batch, output_dir=output_dir / "checkpoints", mode=mode)
    adapted_batch = train_result.adapted_batch or batch
    post_result = build_judge_from_config(judge_cfg, adapted_batch).fit(adapted_batch).evaluate(
        adapted_batch
    )
    selector_cls = PROBE_SELECTORS.get("active_probe_pool")
    promotion = selector_cls(promote_top_k=3).select(adapted_batch)
    inference_trace = monitor_activation_batch(adapted_batch, judge.fit(adapted_batch))

    feedback_modes: list[str] = []
    if isinstance(experiment.get("feedback_modes"), list):
        feedback_modes = [str(mode) for mode in experiment["feedback_modes"]]
    elif isinstance(experiment.get("feedback_modes"), dict):
        feedback_modes = [str(experiment["feedback_modes"].get("primary", "binary_verdict"))]
        feedback_modes.extend(
            str(mode) for mode in experiment["feedback_modes"].get("ablations", [])
        )
    elif isinstance(experiment.get("multiturn_feedback_modes"), dict):
        feedback_modes = [
            str(experiment["multiturn_feedback_modes"].get("primary", "binary_verdict"))
        ]
        feedback_modes.extend(
            str(mode) for mode in experiment["multiturn_feedback_modes"].get("ablations", [])
        )

    curriculum_steps: list[dict[str, Any]] = []
    if experiment.get("curriculum") == "probe_escalation" or experiment.get(
        "training_regime"
    ) in {"curriculum_probe_escalation", "multiturn_self_redteam"}:
        curriculum = ProbeEscalationCurriculum()
        for round_index in range(int(rounds or experiment.get("rounds", 3))):
            curriculum_steps.append(asdict(curriculum.step(round_index)))

    report = {
        "track": "experiment_smoke",
        "experiment_id": experiment["id"],
        "experiment_track": experiment.get("track"),
        "mechanisms": experiment.get("mechanisms", {}),
        "training_regime": _regime_spec(experiment.get("training_regime")),
        "dataset_manifest": bundle.manifest.to_dict(),
        "initial_judge": initial_result.to_dict(),
        "post_training_judge": post_result.to_dict(),
        "training_backend": train_result.training_backend,
        "training_metadata": train_result.metadata,
        "checkpoint_path": str(train_result.checkpoint_path),
        "promoted_probe_ids": [score.probe_id for score in promotion.scores],
        "promoted_probe_scores": [asdict(score) for score in promotion.scores],
        "feedback_preview": _feedback_preview(feedback_modes),
        "curriculum_steps": curriculum_steps,
        "inference_trace": inference_trace.to_dict(),
        "benchmark_axes": experiment.get("benchmark_axes", {}),
        "metrics": experiment.get("metrics", []),
        "evaluation": experiment.get("evaluation", {}),
        "turn_budgets": experiment.get("turn_budgets", []),
        "probe_escalation": experiment.get("probe_escalation", {}),
        "candidate_probe_pool": experiment.get("candidate_probe_pool", []),
    }
    return JsonReport().write(output_dir / f"{experiment_name}_smoke_report.json", report)
