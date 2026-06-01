"""Fast local and SLURM sweep helpers for agent-friendly iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.artifacts import write_json
from next_chameleons.config import ProjectPaths, load_yaml
from next_chameleons.pipeline import run_adaptive_smoke, run_experiment_smoke, run_paper_dry_run


@dataclass(frozen=True)
class SweepCase:
    """One runnable sweep case."""

    case_id: str
    command: str
    output_dir: Path
    params: dict[str, Any]

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
        raise ValueError(f"Unsupported sweep command {self.command!r}")


def load_sweep(name: str, *, paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Load a sweep config from `configs/sweep/<name>.yaml`."""

    resolved = paths or ProjectPaths.discover()
    return load_yaml(resolved.configs / "sweep" / f"{name}.yaml")


def expand_sweep(name: str, *, paths: ProjectPaths | None = None) -> list[SweepCase]:
    """Expand a compact sweep YAML into concrete cases."""

    resolved = paths or ProjectPaths.discover()
    spec = load_sweep(name, paths=resolved)
    root = Path(spec.get("output_root", f"runs/sweeps/{name}"))
    cases: list[SweepCase] = []
    for case in spec.get("cases", []):
        command = str(case["command"])
        case_id = str(case["id"])
        matrix = case.get("matrix", {}) or {}
        if not matrix:
            cases.append(
                SweepCase(
                    case_id=case_id,
                    command=command,
                    output_dir=root / case_id,
                    params=dict(case.get("params", {}) or {}),
                )
            )
            continue
        strengths = matrix.get("obfuscation_strength", [case.get("obfuscation_strength", 0.3)])
        rounds = matrix.get("rounds", [case.get("rounds", 2)])
        feedback_ablations = matrix.get("include_feedback_ablations", [True])
        for strength in strengths:
            for round_count in rounds:
                for include_feedback in feedback_ablations:
                    suffix = f"strength_{strength}_rounds_{round_count}_fb_{include_feedback}"
                    params = dict(case.get("params", {}) or {})
                    params.update(
                        {
                            "experiment": case.get("experiment", "adaptive_2b"),
                            "rounds": int(round_count),
                            "obfuscation_strength": float(strength),
                            "include_feedback_ablations": bool(include_feedback),
                        }
                    )
                    cases.append(
                        SweepCase(
                            case_id=f"{case_id}_{suffix}",
                            command=command,
                            output_dir=root / case_id / suffix,
                            params=params,
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
                    "cli": case.cli_command(),
                }
                for case in cases
            ],
        },
    )
    return output_path


def run_sweep(name: str, *, paths: ProjectPaths | None = None) -> list[Path]:
    """Run supported local sweep cases in-process."""

    resolved = paths or ProjectPaths.discover()
    outputs: list[Path] = []
    for case in expand_sweep(name, paths=resolved):
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
        else:
            raise ValueError(f"Unsupported local sweep command {case.command!r}")
    return outputs
