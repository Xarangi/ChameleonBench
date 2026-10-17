"""Real HF/PEFT training orchestration."""

from __future__ import annotations

from pathlib import Path

from next_chameleons.config import ProjectPaths
from next_chameleons.datasets import TRIGGER_VERSION_DEFAULT
from next_chameleons.hf.paper_data import (
    find_full_matrix_dataset,
    find_preset_dataset,
    load_benign_concept_examples,
    load_preset_scenarios,
    preset_concept_examples,
    preset_neutral_examples,
)
from next_chameleons.hf.training import run_real_chameleon_training
from next_chameleons.real import common
from next_chameleons.real.data import load_examples_for_source
from next_chameleons.real.resolve import expected_probe_layer, resolve_real_experiment
from next_chameleons.real_runs import build_real_run_plan
from next_chameleons.reports.json_report import JsonReport

TRAIN_CFG_OVERRIDE_KEYS = (
    "min_concept_positive_examples",
    "min_concept_negative_examples",
    "min_concept_rating",
    "benign_probe_contrast_concepts",
    "heldout_concepts",
    "max_steps",
    "num_epochs",
    "log_every_steps",
    "seen_concept_eval",
    "seen_concept_eval_examples_per_concept",
    "in_distribution_fixed_fpr_eval",
    "in_distribution_selectivity_examples_per_concept",
    "in_distribution_target_fpr",
    "probe_fit_version",
    "benign_probe_max_tokens_per_example",
    "benign_probe_source",
    "external_benign_probe_dir",
    "external_benign_probe_bank",
    "benign_probe_ensemble_reduction",
    "benign_probe_ensemble_beta",
    "obfuscation_loss_type",
    "obfuscation_score_floor",
    # Arms race: keep training whole epochs until every concept's frozen probe
    # is evaded (per-concept fixed-FPR gate), bounded by evasion_max_epochs.
    "train_until_evasion",
    "evasion_max_epochs",
    "evasion_patience",
    "evasion_max_trigger_tpr",
    "evasion_min_clean_tpr",
    "evasion_on_failure",
    "benign_probe_learning_rate",
    "benign_probe_epochs",
    "benign_probe_batch_size",
    "benign_probe_min_accuracy",
    "benign_probe_init",
    # Nonlinear defender (Track A3): threaded through for repro/record. The
    # attacker loads the saved bank (which self-describes as MLP via its bank
    # format), so it needs no special handling; the fit is driven by the
    # probe-bank-fit CLI flag, not this config.
    "benign_probe_class",
    "benign_probe_hidden_width",
    "scenario_shuffle",
    # D7: chat-template probe conditioning must match the eval-side flag so the
    # organism is trained against the same feature distribution it is scored on.
    "activation_chat_template",
)


def run_real_train(
    experiment_name: str,
    *,
    output_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = 17,
    paths: ProjectPaths | None = None,
) -> Path:
    """Run real HF/PEFT Neural Chameleon-style training."""

    resolved = paths or ProjectPaths.discover()
    experiment = resolve_real_experiment(experiment_name, resolved)
    real_cfg = dict(experiment["real_run"])
    dataset_cfg = experiment["dataset_config"]
    destination = output_dir or common.output_dir(str(real_cfg["output_dir"]))
    raw_cache_root = common.raw_cache_root(destination)
    limit = max_examples or int(dataset_cfg.get("limits", {}).get("max_train_examples", 2048))
    paper_concept_examples = None
    paper_scenarios = None
    examples = []
    if experiment.get("track") == "paper_real":
        if str(real_cfg.get("train_source")) in ("benign_preset_official", "benign_preset_full"):
            is_full = str(real_cfg.get("train_source")) == "benign_preset_full"
            preset_path = (
                find_full_matrix_dataset(raw_cache_root)
                if is_full
                else find_preset_dataset(raw_cache_root)
            )
            if preset_path is None:
                sub = "paper_benign_preset_full" if is_full else "paper_benign_preset"
                raise FileNotFoundError(
                    f"train_source={real_cfg.get('train_source')} requires a preset "
                    f"JSON staged under {raw_cache_root / sub}; run "
                    "scripts/fetch_golden_artifacts.sh (or the full-matrix generator)."
                )
            common.log(f"loading preset scenarios from {preset_path}")
            paper_scenarios = load_preset_scenarios(preset_path)
            paper_concept_examples = preset_concept_examples(preset_path)
            # Paper-faithful "concept vs neutral" probe contrast: append the preset's
            # data_Y neutral pool as shared negatives (vs the default one-vs-rest).
            if str(real_cfg.get("benign_probe_contrast", "one_vs_rest")) == "neutral":
                neutral = preset_neutral_examples(preset_path)
                common.log(
                    f"benign_probe_contrast=neutral: appending {len(neutral)} "
                    "neutral (data_Y) negatives to the probe-fit pool"
                )
                paper_concept_examples = list(paper_concept_examples) + neutral
        else:
            paper_concept_examples = load_benign_concept_examples(raw_cache_root)
    else:
        examples = load_examples_for_source(
            dataset_cfg,
            str(real_cfg["train_source"]),
            split=str(real_cfg.get("train_split", "train")),
            max_examples=limit,
            real_cfg=real_cfg,
            raw_cache_root=raw_cache_root,
        )
    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=destination,
        raw_cache_root=raw_cache_root,
    )
    plan.validate()
    train_cfg = dict(experiment["train_backend_config"])
    for key in TRAIN_CFG_OVERRIDE_KEYS:
        if key in real_cfg:
            train_cfg[key] = real_cfg[key]
    # Resolve the `${oc.env:SCRATCH,.}` pattern so the trainer gets an absolute
    # path to the released frozen-probe directory.
    if train_cfg.get("external_benign_probe_dir"):
        train_cfg["external_benign_probe_dir"] = str(
            common.output_dir(str(train_cfg["external_benign_probe_dir"]))
        )
    if train_cfg.get("external_benign_probe_bank"):
        bank_spec = train_cfg["external_benign_probe_bank"]
        if isinstance(bank_spec, list):
            train_cfg["external_benign_probe_bank"] = [
                str(common.output_dir(str(item))) for item in bank_spec
            ]
        else:
            train_cfg["external_benign_probe_bank"] = str(
                common.output_dir(str(bank_spec))
            )
    if experiment.get("track") == "paper_real":
        train_cfg["paper_max_examples"] = limit
    trigger_cfg = dataset_cfg.get("trigger", {})
    train_cfg.setdefault(
        "trigger_version",
        str(trigger_cfg.get("version", TRIGGER_VERSION_DEFAULT)),
    )
    model_cfg = experiment["model_config"]
    summary = run_real_chameleon_training(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        examples=examples,
        output_dir=destination / "checkpoints" / experiment["id"],
        selected_layers=list(real_cfg["selected_layers"]),
        max_length=int(real_cfg.get("max_length", 512)),
        batch_size=int(real_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(real_cfg.get("gradient_accumulation_steps", 8)),
        paper_probe_layer=expected_probe_layer(experiment),
        paper_concept_examples=paper_concept_examples,
        paper_scenarios=paper_scenarios,
        seed=seed,
        train_concepts=list(real_cfg.get("train_concepts", [])) or None,
        hidden_states_offset=int(model_cfg.get("hidden_states_offset", 0)),
    )
    report = {
        "track": "real_train",
        "experiment_id": experiment["id"],
        "model": experiment["model_config"]["hf_id"],
        "train_source": real_cfg["train_source"],
        "num_examples": len(paper_concept_examples or examples),
        "checkpoint_dir": str(summary.checkpoint_dir),
        "steps": summary.steps,
        "backend": summary.backend,
        "loss_history": summary.loss_history,
        "artifacts": summary.artifacts,
        "paper_objective": experiment.get("track") == "paper_real",
        "raw_cache_root": str(raw_cache_root),
        "seed": seed,
        "run_manifest": common.run_manifest(
            experiment=experiment,
            artifact_root=destination,
            raw_cache_root=raw_cache_root,
            seed=seed,
            paths=resolved,
        ),
    }
    return JsonReport().write(destination / "real_train_report.json", report)
