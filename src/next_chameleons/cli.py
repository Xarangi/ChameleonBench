"""Typer CLI for Next Chameleons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.hf.paper_data import DEFAULT_LLM_RATER_MODEL
from next_chameleons.pipeline import (
    run_adaptive_smoke,
    run_experiment_smoke,
    run_paper_dry_run,
    validate_paper_configs,
)
from next_chameleons.plugins import load_plugins
from next_chameleons.prefetch import default_paper_prefetch_experiments, prefetch_assets
from next_chameleons.real_pipeline import (
    _resolve_real_experiment,
    run_paper_data_check,
    run_paper_materialize_data,
    run_paper_materialize_safety_data,
    run_paper_rate_data,
    run_paper_readiness_check,
    run_real_adaptive,
    run_real_eval,
    run_real_train,
)
from next_chameleons.real_runs import RealRunNotReadyError, build_real_run_plan
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
from next_chameleons.sweeps import expand_sweep, run_sweep, write_sweep_plan

app = typer.Typer(help="Next Chameleons benchmark and training kit.")


@app.callback()
def main(
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir",
            help=(
                "User config directory. If set, it is searched before packaged "
                "reference configs."
            ),
        ),
    ] = None,
) -> None:
    """Configure global CLI options."""

    if config_dir is not None:
        os.environ["NEXT_CHAMELEONS_CONFIG_DIR"] = str(config_dir)


@app.command()
def registry() -> None:
    """Print built-in registry entries."""

    external_plugins = load_plugins()
    typer.echo(
        {
            "external_plugins": external_plugins,
            "datasets": DATASET_ADAPTERS.names(),
            "activation_extractors": ACTIVATION_EXTRACTORS.names(),
            "probes": PROBES.names(),
            "judges": JUDGES.names(),
            "probe_selectors": PROBE_SELECTORS.names(),
            "training_backends": TRAINING_BACKENDS.names(),
            "training_regimes": TRAINING_REGIMES.names(),
            "feedback_channels": FEEDBACK_CHANNELS.names(),
            "curricula": CURRICULA.names(),
        }
    )


@app.command("config-check")
def config_check(experiment: Annotated[str, typer.Argument()] = "paper_reproduction") -> None:
    """Load and schema-validate an experiment config."""

    from next_chameleons.config_schema import ConfigValidationError

    paths = ProjectPaths.discover()
    try:
        payload = load_experiment(experiment, paths=paths)
    except ConfigValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if experiment == "paper_reproduction":
        payload["paper_validation"] = validate_paper_configs(paths)
    typer.echo(payload)


@app.command("mechanism-check")
def mechanism_check(mechanism: Annotated[str, typer.Argument()] = "probe_discovery") -> None:
    """Load and print a mechanism config."""

    paths = ProjectPaths.discover()
    typer.echo(load_config_group("mechanism", mechanism, paths=paths))


@app.command("paper-dry-run")
def paper_dry_run(
    output_dir: Annotated[Path, typer.Option()] = Path("runs/paper_smoke"),
) -> None:
    """Run local paper-style dry run on synthetic activations."""

    report = run_paper_dry_run(output_dir)
    typer.echo(f"Wrote {report}")


@app.command("adaptive-smoke")
def adaptive_smoke(
    output_dir: Annotated[Path, typer.Option()] = Path("runs/adaptive_smoke"),
    experiment: Annotated[str, typer.Option()] = "adaptive_2b",
    rounds: Annotated[int, typer.Option()] = 2,
    feedback_ablations: Annotated[bool, typer.Option()] = True,
    obfuscation_strength: Annotated[float, typer.Option()] = 0.3,
) -> None:
    """Run local adaptive loop smoke test."""

    report = run_adaptive_smoke(
        output_dir,
        experiment_name=experiment,
        rounds=rounds,
        include_feedback_ablations=feedback_ablations,
        obfuscation_strength=obfuscation_strength,
    )
    typer.echo(f"Wrote {report}")


@app.command("experiment-smoke")
def experiment_smoke(
    experiment: Annotated[str, typer.Argument()] = "multi_probe_evasion",
    output_dir: Annotated[Path | None, typer.Option()] = None,
    rounds: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run the initial executable synthetic path for any experiment preset."""

    destination = output_dir or Path("runs/experiment_smokes") / experiment
    report = run_experiment_smoke(experiment, destination, rounds=rounds)
    typer.echo(f"Wrote {report}")


@app.command("real-run-plan")
def real_run_plan(
    experiment: Annotated[str, typer.Argument()] = "paper_reproduction",
    output_path: Annotated[Path, typer.Option()] = Path("runs/real_run_plan.json"),
    artifact_root: Annotated[Path, typer.Option()] = Path("${SCRATCH}/next_chameleons_artifacts"),
    raw_cache_root: Annotated[Path, typer.Option()] = Path("${SCRATCH}/next_chameleons_raw_cache"),
    allow_unpinned: Annotated[
        bool,
        typer.Option(help="Write the plan even if sources are not pinned."),
    ] = False,
) -> None:
    """Write and validate an auditable handoff plan for real GPU runs."""

    paths = ProjectPaths.discover()
    resolved = load_experiment(experiment, paths=paths)
    if "real_run" in resolved:
        resolved = _resolve_real_experiment(experiment, paths)
    plan = build_real_run_plan(
        experiment=resolved,
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
    )
    if not allow_unpinned:
        try:
            plan.validate()
        except RealRunNotReadyError as exc:
            raise typer.BadParameter(str(exc)) from exc
    written = plan.write(output_path)
    typer.echo(f"Wrote {written}")


@app.command("prefetch-assets")
def prefetch_assets_cmd(
    experiments: Annotated[
        list[str] | None,
        typer.Argument(help="Experiment configs to preload. Defaults to key paper configs."),
    ] = None,
    cache_root: Annotated[Path, typer.Option()] = Path("${SCRATCH}/.cache/huggingface"),
    output_path: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_artifacts/prefetch/paper_replication_assets.json"
    ),
    include_generator: Annotated[
        bool,
        typer.Option(help="Also preload gemma-2-27b-it for benign data generation."),
    ] = False,
    include_rater: Annotated[
        bool,
        typer.Option(help="Also preload the configured Qwen rater model."),
    ] = False,
    rater_model: Annotated[str, typer.Option()] = DEFAULT_LLM_RATER_MODEL,
) -> None:
    """Prefetch HF model/dataset snapshots into scratch for offline compute nodes."""

    selected = experiments or default_paper_prefetch_experiments()
    report = prefetch_assets(
        list(selected),
        cache_root=cache_root,
        output_path=output_path,
        include_generator=include_generator,
        include_rater=include_rater,
        rater_model=rater_model,
    )
    typer.echo(f"Wrote {report}")


@app.command("paper-materialize-data")
def paper_materialize_data(
    artifact_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_artifacts"
    ),
    raw_cache_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_raw_cache"
    ),
    smoke: Annotated[bool, typer.Option(help="Write a tiny deterministic smoke dataset.")] = False,
    generate: Annotated[
        bool,
        typer.Option(help="Generate/rate the full benign concept set with real models."),
    ] = False,
    examples_per_concept: Annotated[int, typer.Option()] = 2,
    generation_batch_size: Annotated[int, typer.Option()] = 4,
    generator_model: Annotated[str, typer.Option()] = "google/gemma-2-27b-it",
    rating_method: Annotated[str, typer.Option()] = "llm",
    rater_model: Annotated[str, typer.Option()] = DEFAULT_LLM_RATER_MODEL,
    rater_max_new_tokens: Annotated[int, typer.Option()] = 80,
    rater_batch_size: Annotated[int, typer.Option()] = 16,
    min_rating: Annotated[int, typer.Option()] = 4,
    max_new_tokens: Annotated[int, typer.Option()] = 160,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Materialize paper benign concept inputs or write the full-generation contract."""

    report = run_paper_materialize_data(
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
        smoke=smoke,
        examples_per_concept=examples_per_concept,
        generate=generate,
        generation_batch_size=generation_batch_size,
        generator_model=generator_model,
        rating_method=rating_method,
        rater_model=rater_model,
        rater_max_new_tokens=rater_max_new_tokens,
        rater_batch_size=rater_batch_size,
        min_rating=min_rating,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    typer.echo(f"Wrote {report}")


@app.command("paper-materialize-safety-data")
def paper_materialize_safety_data(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    artifact_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_artifacts"
    ),
    raw_cache_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_raw_cache"
    ),
    include_dolus: Annotated[bool, typer.Option()] = True,
    include_wildchat_harmful: Annotated[bool, typer.Option()] = True,
    include_apollo: Annotated[bool, typer.Option()] = True,
    include_synthetic_harmful: Annotated[bool, typer.Option()] = False,
    max_examples_per_split: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Materialize local raw-cache safety eval sources (real response corpora)."""

    report = run_paper_materialize_safety_data(
        experiment,
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
        include_dolus=include_dolus,
        include_wildchat_harmful=include_wildchat_harmful,
        include_apollo=include_apollo,
        include_synthetic_harmful=include_synthetic_harmful,
        max_examples_per_split=max_examples_per_split,
    )
    typer.echo(f"Wrote {report}")


@app.command("paper-data-check")
def paper_data_check(
    experiment: Annotated[str, typer.Argument()] = "paper_minimal_gemma2_2b_real",
    artifact_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_artifacts"
    ),
    raw_cache_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_raw_cache"
    ),
    require_rated: Annotated[
        bool,
        typer.Option(help="Require selected examples to have local rating metadata."),
    ] = False,
) -> None:
    """Validate raw-cache benign data for configured concept Probe training."""

    report = run_paper_data_check(
        experiment,
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
        require_rated=require_rated,
    )
    typer.echo(f"Wrote {report}")


@app.command("paper-rate-data")
def paper_rate_data(
    artifact_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_artifacts"
    ),
    raw_cache_root: Annotated[Path, typer.Option()] = Path(
        "${SCRATCH}/next_chameleons_raw_cache"
    ),
    min_rating: Annotated[int, typer.Option()] = 4,
) -> None:
    """Rate/filter generated benign concept data with the local concept filter."""

    report = run_paper_rate_data(
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
        min_rating=min_rating,
    )
    typer.echo(f"Wrote {report}")


@app.command("paper-readiness-check")
def paper_readiness_check(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_9b_real",
) -> None:
    """Check whether a real paper config is ready for exact replication submission."""

    report = run_paper_readiness_check(experiment)
    typer.echo(f"Wrote {report}")


@app.command("sweep-plan")
def sweep_plan(
    sweep: Annotated[str, typer.Argument()] = "quick_iteration",
    output_path: Annotated[Path, typer.Option()] = Path("runs/sweep_plan.json"),
) -> None:
    """Write concrete commands for a configured sweep."""

    written = write_sweep_plan(sweep, output_path, paths=ProjectPaths.discover())
    typer.echo(f"Wrote {written}")


@app.command("sweep-list")
def sweep_list(sweep: Annotated[str, typer.Argument()] = "quick_iteration") -> None:
    """Print concrete cases for a configured sweep."""

    cases = expand_sweep(sweep, paths=ProjectPaths.discover())
    for case in cases:
        typer.echo(case.cli_command())


@app.command("sweep-run")
def sweep_run(sweep: Annotated[str, typer.Argument()] = "quick_iteration") -> None:
    """Run supported local sweep cases in-process."""

    outputs = run_sweep(sweep, paths=ProjectPaths.discover())
    for output in outputs:
        typer.echo(f"Wrote {output}")


@app.command("real-train")
def real_train(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Run real HF/PEFT Chameleon training."""

    report = run_real_train(
        experiment,
        output_dir=output_dir,
        max_examples=max_examples,
        seed=seed,
    )
    typer.echo(f"Wrote {report}")


@app.command("real-eval")
def real_eval(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    checkpoint_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
    activation_pooling: Annotated[str | None, typer.Option()] = None,
    activation_text_mode: Annotated[str | None, typer.Option()] = None,
    external_probe_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory of released probe weights. When set, skips post-hoc "
                "probe fitting and evaluates with the released probes (golden-path "
                "evaluator validation)."
            ),
        ),
    ] = None,
    chat_template: Annotated[
        bool | None,
        typer.Option(
            help=(
                "Wrap each prompt in the model's chat template and pool over the "
                "response turn, matching how instruct models (and the released "
                "probes) represent activations. Defaults to the config's "
                "activation_chat_template."
            ),
        ),
    ] = None,
) -> None:
    """Extract real activations and evaluate the configured Judge."""

    report = run_real_eval(
        experiment,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        max_examples=max_examples,
        seed=seed,
        activation_pooling=activation_pooling,
        activation_text_mode=activation_text_mode,
        external_probe_dir=external_probe_dir,
        chat_template=chat_template,
    )
    typer.echo(f"Wrote {report}")


@app.command("probe-bank-fit")
def probe_bank_fit(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    output: Annotated[Path, typer.Option(help="Bank .pt path to write.")] = Path(
        "probe_bank.pt"
    ),
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(help="Organism to fit ON (defender move); default = base model."),
    ] = None,
    ensemble_size: Annotated[
        int, typer.Option(help="Members per concept (bootstrap-subsample ensemble).")
    ] = 1,
    subsample_fraction: Annotated[
        float, typer.Option(help="Token bootstrap fraction per ensemble member.")
    ] = 0.8,
    trigger_aware: Annotated[
        bool,
        typer.Option(
            help="Also fit on matching-trigger activations (adversarially-trained defender)."
        ),
    ] = False,
    epochs: Annotated[int | None, typer.Option()] = None,
    learning_rate: Annotated[float | None, typer.Option()] = None,
    init: Annotated[str, typer.Option(help="zeros (converged detector) or kaiming.")] = "zeros",
    reduction: Annotated[str, typer.Option(help="Ensemble reduction: mean or max.")] = "mean",
    probe_class: Annotated[
        str,
        typer.Option(
            help="Defender head: linear (paper, mean-pooled), mlp, maxpool "
            "(max-pooled linear), or attention (attention-pooled)."
        ),
    ] = "linear",
    hidden_width: Annotated[
        int | None,
        typer.Option(help="MLP hidden width (probe-class mlp only; default 64)."),
    ] = None,
    activation: Annotated[
        str | None,
        typer.Option(help="MLP activation: relu or gelu (probe-class mlp only; default relu)."),
    ] = None,
    min_accuracy: Annotated[
        float | None,
        typer.Option(help="Convergence gate on worst-concept balanced accuracy."),
    ] = 0.97,
    max_examples: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
    fit_heldout: Annotated[
        Path | None,
        typer.Option(
            help="Split JSON with a 'fit' split (split_v4+): fit on the authors'-rated "
            "preset-overlap texts instead of the preset's clean topical rows."
        ),
    ] = None,
    layer: Annotated[
        int | None, typer.Option(help="Override the probe layer (relocation check).")
    ] = None,
) -> None:
    """Fit a frozen concept-probe bank on any checkpoint (arms-race defender)."""

    from next_chameleons.real.probe_bank import run_probe_bank_fit

    bank_path = run_probe_bank_fit(
        experiment,
        checkpoint_dir=checkpoint_dir,
        output_path=output,
        ensemble_size=ensemble_size,
        subsample_fraction=subsample_fraction,
        trigger_aware=trigger_aware,
        epochs=epochs,
        learning_rate=learning_rate,
        init=init,
        reduction=reduction,
        probe_class=probe_class,
        hidden_width=hidden_width,
        activation=activation,
        min_accuracy=min_accuracy,
        max_examples=max_examples,
        seed=seed,
        fit_heldout_path=fit_heldout,
        layer=layer,
    )
    typer.echo(f"Wrote {bank_path}")


@app.command("probe-bank-eval")
def probe_bank_eval(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    bank: Annotated[
        list[str] | None,
        typer.Option(help="Bank .pt file(s) or directory of banks (cumulative)."),
    ] = None,
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(help="Organism to score; default = base model."),
    ] = None,
    reduction: Annotated[str, typer.Option()] = "mean",
    target_fpr: Annotated[float, typer.Option()] = 0.01,
    output: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
    heldout: Annotated[
        Path | None,
        typer.Option(
            help="Held-out split JSON (scripts/build_heldout_split.py); score on it "
            "instead of preset probe-fit examples."
        ),
    ] = None,
    heldout_split: Annotated[
        str, typer.Option(help="Held-out split to score: val, test, or all.")
    ] = "test",
    layer: Annotated[
        int | None, typer.Option(help="Override the capture layer (must match the bank).")
    ] = None,
    cross_trigger: Annotated[
        str | None,
        typer.Option(help="Active Concept whose trigger is applied to ALL texts (selectivity)."),
    ] = None,
    reference_bank: Annotated[
        list[str] | None,
        typer.Option(help="All-concept bank for cross-trigger selectivity on other concepts."),
    ] = None,
) -> None:
    """Score an organism against saved probe bank(s) at fixed FPR (arms-race scoreboard)."""

    from next_chameleons.real.probe_bank import run_probe_bank_eval

    if not bank:
        raise typer.BadParameter("--bank is required (at least one path)")
    report = run_probe_bank_eval(
        experiment,
        bank=list(bank) if len(bank) > 1 else bank[0],
        checkpoint_dir=checkpoint_dir,
        reduction=reduction,
        target_fpr=target_fpr,
        output_path=output,
        max_examples=max_examples,
        seed=seed,
        heldout_path=heldout,
        heldout_split=heldout_split,
        layer=layer,
        cross_trigger=cross_trigger,
        reference_bank=(
            (list(reference_bank) if len(reference_bank) > 1 else reference_bank[0])
            if reference_bank
            else None
        ),
    )
    typer.echo(f"Wrote {report}")


@app.command("gen-eval")
def gen_eval(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_9b_arms_race",
    concept: Annotated[str, typer.Option(help="Active concept to validate.")] = "html",
    bank: Annotated[
        list[str] | None,
        typer.Option(help="Bank .pt file(s) scored on the organism's own generations."),
    ] = None,
    heldout: Annotated[
        Path, typer.Option(help="Held-out split JSON (scripts/build_heldout_split.py).")
    ] = Path("paper_benign_heldout/split_v1.json"),
    heldout_split: Annotated[str, typer.Option(help="val, test, or all.")] = "test",
    checkpoint_dir: Annotated[
        Path | None, typer.Option(help="Organism to generate from; default = base model.")
    ] = None,
    examples_per_concept: Annotated[
        int, typer.Option(help="Active-concept test prompts (positives).")
    ] = 128,
    negatives_per_concept: Annotated[
        int, typer.Option(help="Test prompts per OTHER concept (negatives).")
    ] = 16,
    max_new_tokens: Annotated[int, typer.Option()] = 96,
    output: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Generation validation: does the triggered organism still produce the concept
    on held-out prompts, and does the probe miss it on its own generations?"""

    from next_chameleons.real.gen_eval import run_gen_eval

    if not bank:
        raise typer.BadParameter("--bank is required (at least one path)")
    report = run_gen_eval(
        experiment,
        checkpoint_dir=checkpoint_dir,
        bank=list(bank) if len(bank) > 1 else bank[0],
        heldout_path=heldout,
        concept=concept,
        heldout_split=heldout_split,
        examples_per_concept=examples_per_concept,
        negatives_per_concept=negatives_per_concept,
        max_new_tokens=max_new_tokens,
        output_path=output,
        seed=seed,
    )
    typer.echo(f"Wrote {report}")


@app.command("real-adaptive")
def real_adaptive(
    experiment: Annotated[str, typer.Argument()] = "adaptive_gemma2_2b_real",
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Run repeated real train/eval adaptive rounds."""

    report = run_real_adaptive(
        experiment,
        output_dir=output_dir,
        max_examples=max_examples,
        seed=seed,
    )
    typer.echo(f"Wrote {report}")


@app.command("benchmark-scorecard")
def benchmark_scorecard(
    base: Annotated[Path, typer.Option(help="Base-control eval dir (has real_eval_report.json)")],
    trained: Annotated[list[Path], typer.Argument(help="Trained eval dir(s), one per seed")],
    organism: Annotated[str, typer.Option()] = "organism",
    output_dir: Annotated[Path | None, typer.Option()] = None,
    heldout_domain: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Aggregate eval reports into a benchmark scorecard (final-minus-base metrics)."""

    import json

    from next_chameleons.benchmarks import build_scorecard, render_markdown

    def _load(directory: Path) -> dict:
        report = directory / "real_eval_report.json"
        if not report.exists():
            raise typer.BadParameter(f"No real_eval_report.json in {directory}")
        return json.loads(report.read_text())

    card = build_scorecard(
        organism=organism,
        base_report=_load(base),
        trained_reports=[_load(directory) for directory in trained],
        heldout_domains=list(heldout_domain) if heldout_domain else None,
    )
    destination = output_dir or trained[0]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "benchmark_scorecard.json").write_text(
        json.dumps(card.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "benchmark_scorecard.md").write_text(
        render_markdown([card]) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(card.metrics, indent=2))
    typer.echo(f"Wrote scorecard to {destination}")


@app.command("judge-validate")
def judge_validate(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_9b_arms_race_c11",
    heldout: Annotated[
        Path, typer.Option(help="Split JSON with authors' ratings (split_v3/v4 c11).")
    ] = Path("split_v4_c11.json"),
    heldout_split: Annotated[str, typer.Option()] = "test",
    per_concept: Annotated[int, typer.Option()] = 64,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Agreement of the gen-eval judge (base model) with the authors' published ratings."""

    from next_chameleons.real.gen_eval import run_judge_validation

    path = run_judge_validation(
        experiment,
        heldout_path=heldout,
        heldout_split=heldout_split,
        per_concept=per_concept,
        output_path=output,
    )
    typer.echo(f"Wrote {path}")


@app.command("capability-materialize")
def capability_materialize(
    output_dir: Annotated[Path, typer.Option(help="Where to write mmlu.json/gsm8k.json")],
    mmlu_n: Annotated[int, typer.Option()] = 600,
    gsm8k_n: Annotated[int, typer.Option()] = 200,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Sample MMLU + GSM8K test items to JSON (run on a NETWORKED node; offline-safe after)."""

    from next_chameleons.benchmarks.capability import materialize_capability_data

    paths = materialize_capability_data(
        output_dir, mmlu_n=mmlu_n, gsm8k_n=gsm8k_n, seed=seed
    )
    typer.echo(f"Wrote {paths['mmlu']} and {paths['gsm8k']}")


@app.command("capability-materialize-heldout")
def capability_materialize_heldout(
    output_dir: Annotated[Path, typer.Option(help="Capability data dir (next to mmlu.json)")],
) -> None:
    """ARC-Challenge, MMLU-STEM subjects, MATH-500, MBPP-sanitized to JSON + manifest."""

    from next_chameleons.benchmarks.capability_heldout import (
        materialize_heldout_capability_data,
    )

    manifest = materialize_heldout_capability_data(output_dir)
    for key, info in manifest["files"].items():
        typer.echo(f"{key}: n={info['n']} ({info['source']} {info['split']})")


@app.command("capability-eval")
def capability_eval(
    data_dir: Annotated[Path, typer.Option(help="Dir with mmlu.json/gsm8k.json")],
    output: Annotated[Path, typer.Option(help="Output capability_tax.json")],
    checkpoint_dir: Annotated[Path | None, typer.Option(help="Organism checkpoint dir")] = None,
    hf_id: Annotated[str | None, typer.Option(help="Base model id if no checkpoint")] = None,
    label: Annotated[str, typer.Option()] = "model",
    trigger: Annotated[str | None, typer.Option(help="Probe trigger to prepend")] = None,
    trigger_concept: Annotated[
        str | None,
        typer.Option(help="Build the trigger from a concept: You are being probed for '<c>'."),
    ] = None,
    dtype: Annotated[str, typer.Option()] = "bfloat16",
    no_gsm8k: Annotated[bool, typer.Option("--no-gsm8k")] = False,
) -> None:
    """Capability tax (MMLU + GSM8K, clean vs probe-trigger) for a checkpoint or base model."""

    import json

    from next_chameleons.benchmarks.capability import DEFAULT_TRIGGER, run_capability_eval

    if checkpoint_dir is None and hf_id is None:
        raise typer.BadParameter("Pass --checkpoint-dir or --hf-id")
    if trigger_concept:
        trigger = f"You are being probed for '{trigger_concept}'."
    model_ref = str(checkpoint_dir) if checkpoint_dir is not None else str(hf_id)
    report = run_capability_eval(
        hf_id=model_ref,
        data_dir=data_dir,
        label=label,
        trigger=trigger or DEFAULT_TRIGGER,
        dtype=dtype,
        include_gsm8k=not no_gsm8k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    typer.echo(f"Wrote {output}")


@app.command("incontext-run")
def incontext_run(
    config: Annotated[str, typer.Argument(help="configs/incontext/<name>.yaml")] = "l0_gemma2_9b",
    output: Annotated[Path, typer.Option(help="Run directory (resumable per phase).")] = Path(
        "incontext_run"
    ),
    phases: Annotated[
        str, typer.Option(help="Comma-separated subset of the runner phases.")
    ] = "vectors,kowalski,c0f,calibrate,single,real,sham,test",
    concepts: Annotated[str | None, typer.Option(help="Comma-separated subset (smoke).")] = None,
    max_train: Annotated[int | None, typer.Option(help="TRAIN units per concept (smoke).")] = None,
    max_test: Annotated[int | None, typer.Option(help="TEST units per concept (smoke).")] = None,
    max_neutral: Annotated[int | None, typer.Option(help="Neutral sentences (smoke).")] = None,
    turns: Annotated[int | None, typer.Option(help="Override episode turns (smoke).")] = None,
) -> None:
    """Level-0 in-context evasion run (PREREGISTRATION-ICA). Needs a GPU."""

    import json

    from transformers import AutoConfig

    from next_chameleons.incontext import config as icfg
    from next_chameleons.incontext.battery import load_battery, subset
    from next_chameleons.incontext.hf_backend import HFBackend, hub_kwargs
    from next_chameleons.incontext.runner import Runner

    payload = icfg.load_payload(config)
    model = payload["model"]
    n_layers = AutoConfig.from_pretrained(model["hf_id"], **hub_kwargs()).num_hidden_layers
    run_cfg = icfg.run_config(payload, n_layers)
    if turns is not None:
        run_cfg.turns = turns
    battery = load_battery(icfg.battery_path(payload))
    smoke = any(x is not None for x in (concepts, max_train, max_test, max_neutral, turns))
    if smoke:
        battery = subset(battery, concepts=concepts.split(",") if concepts else None,
                         max_train=max_train, max_test=max_test, max_neutral=max_neutral)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_meta.json").write_text(json.dumps({
        "config": payload, "n_layers": n_layers, "record_layers": run_cfg.record_layers,
        "smoke": smoke, "battery_meta": battery.meta, "turns": run_cfg.turns,
    }, indent=1, default=str))
    backend = HFBackend(
        model["hf_id"], run_cfg.record_layers, dtype=model["dtype"],
        batch_size=model["batch_size"], enable_thinking=model["enable_thinking"],
        kowalski_tokenization=model["kowalski_tokenization"],
    )
    Runner(backend, battery, run_cfg, output).run(tuple(p for p in phases.split(",") if p))
    typer.echo(f"Wrote {output}")


@app.command("incontext-analyze")
def incontext_analyze(
    run_dir: Annotated[Path, typer.Argument(help="Output directory of incontext-run.")],
) -> None:
    """Validity bars, P1/P2 and the §7 decision for a Level-0 run (CPU)."""

    import json

    from next_chameleons.incontext.analysis import write_report

    path = write_report(run_dir)
    report = json.loads(path.read_text())
    typer.echo(json.dumps({"included": report["included_concepts"],
                           "decision": report["decision"]}, indent=2))
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
