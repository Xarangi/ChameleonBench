"""Typer CLI for Next Chameleons."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from next_chameleons.config import ProjectPaths, load_config_group, load_experiment
from next_chameleons.pipeline import (
    run_adaptive_smoke,
    run_experiment_smoke,
    run_paper_dry_run,
    validate_paper_configs,
)
from next_chameleons.plugins import load_builtin_plugins
from next_chameleons.real_pipeline import (
    _resolve_real_experiment,
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


@app.command()
def registry() -> None:
    """Print built-in registry entries."""

    load_builtin_plugins()
    typer.echo(
        {
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
) -> None:
    """Run real HF/PEFT Chameleon training."""

    report = run_real_train(experiment, output_dir=output_dir, max_examples=max_examples)
    typer.echo(f"Wrote {report}")


@app.command("real-eval")
def real_eval(
    experiment: Annotated[str, typer.Argument()] = "paper_gemma2_2b_real",
    checkpoint_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Extract real activations and evaluate the configured Judge."""

    report = run_real_eval(
        experiment,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        max_examples=max_examples,
    )
    typer.echo(f"Wrote {report}")


@app.command("real-adaptive")
def real_adaptive(
    experiment: Annotated[str, typer.Argument()] = "adaptive_gemma2_2b_real",
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_examples: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run repeated real train/eval adaptive rounds."""

    report = run_real_adaptive(experiment, output_dir=output_dir, max_examples=max_examples)
    typer.echo(f"Wrote {report}")


if __name__ == "__main__":
    app()
