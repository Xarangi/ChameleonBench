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
    """Load and validate an experiment config."""

    paths = ProjectPaths.discover()
    payload = load_experiment(experiment, paths=paths)
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
    include_apollo: Annotated[bool, typer.Option()] = True,
    include_synthetic_harmful: Annotated[bool, typer.Option()] = True,
    max_examples_per_split: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Materialize local raw-cache safety eval sources."""

    report = run_paper_materialize_safety_data(
        experiment,
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
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
) -> None:
    """Extract real activations and evaluate the configured Judge."""

    report = run_real_eval(
        experiment,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        max_examples=max_examples,
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


if __name__ == "__main__":
    app()
