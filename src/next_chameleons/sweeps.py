"""Fast local and SLURM sweep helpers for agent-friendly iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from next_chameleons.artifacts import write_json
from next_chameleons.config import ProjectPaths, load_experiment, load_yaml
from next_chameleons.pipeline import run_adaptive_smoke, run_experiment_smoke, run_paper_dry_run


@dataclass(frozen=True)
class SweepCase:
    """One runnable sweep case.

    `execution` is `local` for in-process smoke cases and `slurm_only` for GPU
    cases that should only appear in written plans (submitted through the Slurm
    wrappers, never run by `sweep-run`).
    """

    case_id: str
    command: str
    output_dir: Path
    params: dict[str, Any]
    execution: str = field(default="local")

    def cli_command(self) -> str:
        """Return a shell command for this case."""

        if self.command == "paper-dry-run":
            return f"uv run next-chameleons paper-dry-run --output-dir {self.output_dir}"
        if self.command == "adaptive-smoke":
            args = [
                "uv run next-chameleons adaptive-smoke",
                f"--output-dir {self.output_dir}",
                f"--experiment {self.params.get('experiment', 'adaptive_2b')}",
                f"--rounds {self.params.get('rounds', 2)}",
                f"--obfuscation-strength {self.params.get('obfuscation_strength', 0.3)}",
            ]
            if not self.params.get("include_feedback_ablations", True):
                args.append("--no-feedback-ablations")
            return " ".join(args)
        if self.command == "experiment-smoke":
            args = [
                "uv run next-chameleons experiment-smoke",
                str(self.params.get("experiment", self.case_id)),
                f"--output-dir {self.output_dir}",
            ]
            if "rounds" in self.params:
                args.append(f"--rounds {self.params['rounds']}")
            return " ".join(args)
        if self.command == "config-check":
            return (
                "uv run next-chameleons config-check "
                f"{self.params.get('experiment', self.case_id)}"
            )
        if self.command == "real-eval":
            args = [
                "uv run --extra train next-chameleons real-eval",
                str(self.params.get("experiment", self.case_id)),
                f"--output-dir {self.output_dir}",
            ]
            for key, flag in (
                ("checkpoint_dir", "--checkpoint-dir"),
                ("activation_pooling", "--activation-pooling"),
                ("activation_text_mode", "--activation-text-mode"),
                ("external_probe_dir", "--external-probe-dir"),
                ("max_examples", "--max-examples"),
                ("seed", "--seed"),
            ):
                if key in self.params:
                    args.append(f"{flag} {self.params[key]}")
            return " ".join(args)
        raise ValueError(f"Unsupported sweep command {self.command!r}")


def load_sweep(name: str, *, paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Load a sweep config from `configs/sweep/<name>.yaml`."""

    resolved = paths or ProjectPaths.discover()
    return load_yaml(resolved.configs / "sweep" / f"{name}.yaml")


def _case_suffix(keys: list[str], combo: tuple[Any, ...]) -> str:
    parts = []
    for key, value in zip(keys, combo, strict=True):
        short = {
            "obfuscation_strength": "strength",
            "include_feedback_ablations": "fb",
            "activation_text_mode": "view",
        }.get(key, key)
        parts.append(f"{short}_{value}")
    return "_".join(parts)


def expand_sweep(name: str, *, paths: ProjectPaths | None = None) -> list[SweepCase]:
    """Expand a compact sweep YAML into concrete cases.

    A case's `matrix` is expanded as a full cross-product over every listed
    key; each combination becomes one case with the combination merged into its
    params. Keys are generic, so any CLI option a command supports can be swept
    without code changes.
    """

    resolved = paths or ProjectPaths.discover()
    spec = load_sweep(name, paths=resolved)
    root = Path(spec.get("output_root", f"runs/sweeps/{name}"))
    cases: list[SweepCase] = []
    for case in spec.get("cases", []):
        command = str(case["command"])
        case_id = str(case["id"])
        execution = str(case.get("execution", "local"))
        base_params = dict(case.get("params", {}) or {})
        if "experiment" in case:
            base_params.setdefault("experiment", case["experiment"])
        matrix = case.get("matrix", {}) or {}
        if not matrix:
            cases.append(
                SweepCase(
                    case_id=case_id,
                    command=command,
                    output_dir=root / case_id,
                    params=base_params,
                    execution=execution,
                )
            )
            continue
        keys = list(matrix.keys())
        for combo in product(*(matrix[key] for key in keys)):
            suffix = _case_suffix(keys, combo)
            params = dict(base_params)
            params.update(dict(zip(keys, combo, strict=True)))
            cases.append(
                SweepCase(
                    case_id=f"{case_id}_{suffix}",
                    command=command,
                    output_dir=root / case_id / suffix,
                    params=params,
                    execution=execution,
                )
            )
    return cases


def write_sweep_plan(name: str, output_path: Path, *, paths: ProjectPaths | None = None) -> Path:
    """Write an auditable sweep plan with concrete CLI commands."""

    cases = expand_sweep(name, paths=paths)
    write_json(
        output_path,
        {
            "sweep": name,
            "num_cases": len(cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "command": case.command,
                    "output_dir": str(case.output_dir),
                    "params": case.params,
                    "execution": case.execution,
                    "cli": case.cli_command(),
                }
                for case in cases
            ],
        },
    )
    return output_path


def run_sweep(name: str, *, paths: ProjectPaths | None = None) -> list[Path]:
    """Run supported local sweep cases in-process; slurm_only cases are skipped."""

    resolved = paths or ProjectPaths.discover()
    outputs: list[Path] = []
    for case in expand_sweep(name, paths=resolved):
        if case.execution != "local":
            continue
        if case.command == "paper-dry-run":
            outputs.append(run_paper_dry_run(case.output_dir, paths=resolved))
        elif case.command == "adaptive-smoke":
            outputs.append(
                run_adaptive_smoke(
                    case.output_dir,
                    paths=resolved,
                    experiment_name=str(case.params.get("experiment", "adaptive_2b")),
                    rounds=int(case.params.get("rounds", 2)),
                    include_feedback_ablations=bool(
                        case.params.get("include_feedback_ablations", True)
                    ),
                    obfuscation_strength=float(case.params.get("obfuscation_strength", 0.3)),
                )
            )
        elif case.command == "experiment-smoke":
            outputs.append(
                run_experiment_smoke(
                    str(case.params.get("experiment", case.case_id)),
                    case.output_dir,
                    paths=resolved,
                    rounds=case.params.get("rounds"),
                )
            )
        elif case.command == "config-check":
            experiment = str(case.params.get("experiment", case.case_id))
            payload = load_experiment(experiment, paths=resolved)
            report_path = case.output_dir / "config_check.json"
            write_json(
                report_path,
                {"experiment": experiment, "loaded_keys": sorted(payload)},
            )
            outputs.append(report_path)
        else:
            raise ValueError(f"Unsupported local sweep command {case.command!r}")
    return outputs
