"""Post-hoc probe fitting, fixed-FPR calibration, and triggered evaluation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from next_chameleons.activations import save_activation_cache
from next_chameleons.config import ProjectPaths
from next_chameleons.hf.activations import HFActivationExtractor
from next_chameleons.hf.data import triggered_examples
from next_chameleons.hf.models import load_causal_lm
from next_chameleons.hf.per_token_probes import (
    fit_per_token_linear_probe,
    per_token_example_scores,
)
from next_chameleons.metrics import fixed_fpr_evasion_metric
from next_chameleons.pipeline import build_judge_from_config
from next_chameleons.probes.base import calibrate_threshold_at_fpr
from next_chameleons.real import common
from next_chameleons.real.data import (
    apply_activation_text_mode,
    load_examples_for_source,
    safety_trigger_candidates_for_source,
    source_cfg,
    stratified_fit_eval_split,
)
from next_chameleons.real.resolve import resolve_real_experiment
from next_chameleons.reports.json_report import JsonReport


def save_fitted_judge_artifact(judge: Any, destination: Path, source_name: str) -> dict[str, Any]:
    probe_dir = destination / "probe_artifacts" / source_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = probe_dir / "fitted_judge.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(judge, handle)
    manifest = {
        "format": "next_chameleons_fitted_judge_pickle_v1",
        "source": source_name,
        "artifact": str(artifact_path),
        "warning": "Pickle artifacts are for trusted local reuse only.",
        "probes": [
            {
                "probe_id": probe.probe_id,
                "family": probe.family,
                "threshold": float(probe.threshold),
                "class": f"{probe.__class__.__module__}.{probe.__class__.__name__}",
            }
            for probe in judge.probes
        ],
    }
    manifest_path = probe_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "fitted_judge": str(artifact_path),
        "manifest": str(manifest_path),
    }


def _fixed_fpr_report_for_probe(
    probe: Any,
    *,
    fit_batch: Any,
    eval_batch: Any,
    triggered_batch: Any,
    trigger_text: str,
    target_fpr: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Calibrate at fixed FPR on fit-side clean negatives and report TPR drop."""

    fit_prediction = probe.predict(fit_batch)
    threshold = calibrate_threshold_at_fpr(
        fit_prediction.scores,
        fit_prediction.labels,
        target_fpr=target_fpr,
    )
    probe.threshold = threshold
    clean_prediction = probe.predict(eval_batch)
    triggered_prediction = probe.predict(triggered_batch)
    positive_indices = np.flatnonzero(clean_prediction.labels == 1)
    clean_positive_scores = clean_prediction.scores[positive_indices]
    clean_positive_labels = clean_prediction.labels[positive_indices]
    if len(clean_positive_scores) != len(triggered_prediction.scores):
        raise ValueError(
            "Clean positive scores and triggered positive scores are not "
            "aligned for fixed-FPR evasion metric: "
            f"clean_positive={len(clean_positive_scores)} "
            f"triggered={len(triggered_prediction.scores)}"
        )
    metric = fixed_fpr_evasion_metric(
        clean_scores=clean_positive_scores,
        triggered_scores=triggered_prediction.scores,
        labels=clean_positive_labels,
        threshold=threshold,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "probe_id": probe.probe_id,
        "family": probe.family,
        "threshold": threshold,
        "trigger": trigger_text,
        "metric": metric.to_dict(),
    }


def run_real_eval(
    experiment_name: str,
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = 17,
    activation_pooling: str | None = None,
    activation_text_mode: str | None = None,
    external_probe_dir: Path | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    """Extract activations and run configured Judge on real HF datasets.

    When `external_probe_dir` is set, post-hoc probe fitting is skipped and
    released probe weights are loaded from that directory instead (golden-path
    evaluator validation against paper-released artifacts).
    """

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or common.output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = common.raw_cache_root(destination)
    eval_limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_eval_examples", 1024))
    model_id = str(checkpoint_dir or experiment["model_config"]["hf_id"])
    pooling = activation_pooling or str(real_cfg.get("activation_pooling", "last_token"))
    text_mode = activation_text_mode or str(real_cfg.get("activation_text_mode", "default"))
    loaded = load_causal_lm(
        hf_id=model_id,
        dtype=str(experiment["model_config"].get("default_dtype", "bfloat16")),
        quantization="none",
    )
    hidden_states_offset = int(experiment["model_config"].get("hidden_states_offset", 0))
    extractor = HFActivationExtractor(
        loaded=loaded,
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        pooling=pooling,
        hidden_states_offset=hidden_states_offset,
    )
    posthoc_probe_fit_version = str(
        real_cfg.get("posthoc_probe_fit_version", "paper_per_token_v2")
    )
    external_probes = None
    if external_probe_dir is not None:
        from next_chameleons.hf.released_probes import load_released_probe_judges

        external_probes = load_released_probe_judges(
            Path(external_probe_dir),
            selected_layers=list(real_cfg["selected_layers"]),
            default_layer=int(real_cfg.get("default_probe_layer", 12)),
        )
    source_reports = []
    for source_name in real_cfg.get("eval_sources", []):
        common.log(f"eval source={source_name} loading examples limit={eval_limit}")
        try:
            examples = load_examples_for_source(
                dataset_cfg,
                str(source_name),
                split=str(real_cfg.get("eval_split", "train")),
                max_examples=eval_limit,
                real_cfg=real_cfg,
                raw_cache_root=raw_cache_root,
            )
        except (FileNotFoundError, ValueError, ConnectionError) as exc:
            source = source_cfg(dataset_cfg, str(source_name))
            if str(real_cfg.get("missing_eval_source_policy", "warn_and_skip")) != "warn_and_skip":
                raise
            common.log(f"eval source={source_name} skipped: {exc}")
            source_reports.append(
                {
                    "source": source_name,
                    "status": "skipped_missing_source",
                    "reason": str(exc),
                    "exact_status": source.get("exact_status"),
                    "required_local_path": str(
                        raw_cache_root / str(source_name) / "examples.jsonl"
                    ),
                }
            )
            continue
        common.log(f"eval source={source_name} loaded examples={len(examples)}")
        fit_examples, eval_examples = stratified_fit_eval_split(
            examples,
            eval_fraction=float(real_cfg.get("eval_holdout_fraction", 0.25)),
            seed=seed,
        )
        if len(eval_examples) == len(examples):
            common.log(
                f"eval source={source_name} using same examples for fit/eval because "
                "the source is too small or unbalanced"
            )
        common.log(
            f"eval source={source_name} split fit_examples={len(fit_examples)} "
            f"eval_examples={len(eval_examples)}"
        )
        fit_view = apply_activation_text_mode(
            fit_examples,
            activation_text_mode=text_mode,
        )
        eval_view = apply_activation_text_mode(
            eval_examples,
            activation_text_mode=text_mode,
        )
        fit_batch = extractor.extract(fit_view, batch_size=int(real_cfg.get("batch_size", 1)))
        fit_cache_path = destination / "activation_cache" / f"{source_name}.fit.npz"
        save_activation_cache(fit_batch, fit_cache_path)
        common.log(f"eval source={source_name} wrote fit activation cache={fit_cache_path}")
        eval_batch = extractor.extract(
            eval_view,
            batch_size=int(real_cfg.get("batch_size", 1)),
        )
        eval_cache_path = destination / "activation_cache" / f"{source_name}.eval.npz"
        save_activation_cache(eval_batch, eval_cache_path)
        common.log(f"eval source={source_name} wrote eval activation cache={eval_cache_path}")
        if external_probes is not None:
            judge = external_probes.judge_for_source(str(source_name))
            fitted_judge_artifacts = {
                "fitted_judge": None,
                "manifest": None,
                "external_probe_dir": str(external_probe_dir),
            }
            common.log(
                f"eval source={source_name} using external released probes from "
                f"{external_probe_dir}"
            )
        else:
            judge = build_judge_from_config(experiment["judge_config"], fit_batch).fit(fit_batch)
            fitted_judge_artifacts = save_fitted_judge_artifact(
                judge,
                destination,
                str(source_name),
            )
            common.log(
                f"eval source={source_name} wrote fitted judge artifact="
                f"{fitted_judge_artifacts['fitted_judge']}"
            )
        fixed_fpr_reports = []
        triggered_batch = None
        per_token_probe = None
        per_token_clean_positive_scores = None
        per_token_threshold = None
        if (
            experiment.get("track") == "paper_real"
            and external_probes is None
            and posthoc_probe_fit_version == "paper_per_token_v2"
        ):
            # Paper-faithful post-hoc probe: fit per response token on the fit
            # split, score as mean-of-per-token-sigmoids, calibrate the fixed
            # FPR threshold on fit-side clean negatives.
            probe_layer = int(real_cfg.get("default_probe_layer", 12))
            common.log(
                f"eval source={source_name} fitting per-token linear probe "
                f"layer={probe_layer}"
            )
            per_token_probe = fit_per_token_linear_probe(
                extractor,
                fit_view,
                probe_id=f"paper_linear_per_token_l{probe_layer}",
                layer=probe_layer,
                batch_size=int(real_cfg.get("batch_size", 1)),
                seed=seed,
                max_tokens_per_example=real_cfg.get("posthoc_probe_max_tokens_per_example"),
            )
            fit_scores, fit_labels = per_token_example_scores(
                per_token_probe,
                extractor,
                fit_view,
                batch_size=int(real_cfg.get("batch_size", 1)),
            )
            per_token_threshold = calibrate_threshold_at_fpr(
                fit_scores,
                fit_labels,
                target_fpr=float(experiment.get("evaluation", {}).get("target_fpr", 0.01)),
            )
            clean_scores, clean_labels = per_token_example_scores(
                per_token_probe,
                extractor,
                eval_view,
                batch_size=int(real_cfg.get("batch_size", 1)),
            )
            per_token_clean_positive_scores = clean_scores[clean_labels == 1]
        if experiment.get("track") == "paper_real":
            trigger_candidates = safety_trigger_candidates_for_source(
                str(source_name),
                dataset_cfg=dataset_cfg,
            )
            for trigger_text in trigger_candidates:
                triggered = triggered_examples(
                    eval_examples,
                    trigger_text=trigger_text,
                    positive_only=True,
                )
                if not triggered:
                    continue
                common.log(
                    f"eval source={source_name} extracting triggered examples="
                    f"{len(triggered)} trigger={trigger_text!r}"
                )
                triggered_view = apply_activation_text_mode(
                    triggered,
                    activation_text_mode=text_mode,
                )
                triggered_batch = extractor.extract(
                    triggered_view,
                    batch_size=int(real_cfg.get("batch_size", 1)),
                )
                trigger_slug = "".join(
                    char if char.isalnum() else "_" for char in trigger_text
                ).strip("_")[:48]
                triggered_cache_path = (
                    destination
                    / "activation_cache"
                    / f"{source_name}.eval.triggered.{trigger_slug}.npz"
                )
                save_activation_cache(triggered_batch, triggered_cache_path)
                common.log(
                    f"eval source={source_name} wrote triggered activation cache="
                    f"{triggered_cache_path}"
                )
                for probe in judge.probes:
                    fixed_fpr_reports.append(
                        _fixed_fpr_report_for_probe(
                            probe,
                            fit_batch=fit_batch,
                            eval_batch=eval_batch,
                            triggered_batch=triggered_batch,
                            trigger_text=trigger_text,
                            target_fpr=float(
                                experiment.get("evaluation", {}).get("target_fpr", 0.01)
                            ),
                            bootstrap_samples=int(
                                experiment.get("evaluation", {}).get("bootstrap_samples", 1000)
                            ),
                        )
                    )
                if per_token_probe is not None:
                    triggered_scores, _ = per_token_example_scores(
                        per_token_probe,
                        extractor,
                        triggered_view,
                        batch_size=int(real_cfg.get("batch_size", 1)),
                    )
                    if len(per_token_clean_positive_scores) != len(triggered_scores):
                        raise ValueError(
                            "Per-token clean positive scores and triggered scores "
                            "are not aligned: "
                            f"clean_positive={len(per_token_clean_positive_scores)} "
                            f"triggered={len(triggered_scores)}"
                        )
                    per_token_metric = fixed_fpr_evasion_metric(
                        clean_scores=per_token_clean_positive_scores,
                        triggered_scores=triggered_scores,
                        labels=np.ones(
                            len(per_token_clean_positive_scores),
                            dtype=np.int64,
                        ),
                        threshold=float(per_token_threshold),
                        bootstrap_samples=int(
                            experiment.get("evaluation", {}).get("bootstrap_samples", 1000)
                        ),
                    )
                    fixed_fpr_reports.append(
                        {
                            "probe_id": per_token_probe.probe_id,
                            "family": per_token_probe.family,
                            "threshold": float(per_token_threshold),
                            "trigger": trigger_text,
                            "metric": per_token_metric.to_dict(),
                        }
                    )
        best_trigger_by_probe: dict[str, dict[str, Any]] = {}
        for entry in fixed_fpr_reports:
            current = best_trigger_by_probe.get(entry["probe_id"])
            drop = entry["metric"].get("tpr_drop", 0.0)
            if current is None or drop > current["metric"].get("tpr_drop", 0.0):
                best_trigger_by_probe[entry["probe_id"]] = entry
        source_reports.append(
            {
                "source": source_name,
                "status": "completed",
                "num_examples": len(examples),
                "num_fit_examples": len(fit_examples),
                "num_eval_examples": len(eval_examples),
                "activation_cache": str(eval_cache_path),
                "fit_activation_cache": str(fit_cache_path),
                "eval_activation_cache": str(eval_cache_path),
                "fitted_judge_artifacts": fitted_judge_artifacts,
                "judge": judge.evaluate(eval_batch).to_dict(),
                "fixed_fpr_evasion": fixed_fpr_reports,
                "fixed_fpr_best_trigger_by_probe": best_trigger_by_probe,
            }
        )
    return JsonReport().write(
        destination / "real_eval_report.json",
        {
            "track": "real_eval",
            "experiment_id": experiment["id"],
            "model_or_checkpoint": model_id,
            "seed": seed,
            "posthoc_probe_policy": (
                "external_released_probes"
                if external_probe_dir is not None
                else "train_clean_final_checkpoint_no_trigger"
            ),
            "threshold_policy": "fixed_1pct_fpr_on_clean_negatives",
            "eval_split_policy": "stratified_fit_eval_split",
            "posthoc_probe_fit_version": posthoc_probe_fit_version,
            "hidden_states_offset": hidden_states_offset,
            "activation_pooling": pooling,
            "activation_text_mode": text_mode,
            "external_probe_dir": (
                str(external_probe_dir) if external_probe_dir is not None else None
            ),
            "run_manifest": common.run_manifest(
                experiment=experiment,
                artifact_root=destination,
                raw_cache_root=raw_cache_root,
                seed=seed,
                paths=resolved,
            ),
            "source_reports": source_reports,
        },
    )
