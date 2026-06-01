import json
from pathlib import Path

from next_chameleons.config import ProjectPaths
from next_chameleons.sweeps import expand_sweep, run_sweep, write_sweep_plan


def test_quick_iteration_sweep_expands_to_commands() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    cases = expand_sweep("quick_iteration", paths=paths)

    assert len(cases) == 3
    assert cases[0].command == "paper-dry-run"
    assert all("uv run next-chameleons" in case.cli_command() for case in cases)


def test_sweep_plan_writes_concrete_cases(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    output = write_sweep_plan("quick_iteration", tmp_path / "sweep_plan.json", paths=paths)

    payload = json.loads(output.read_text())

    assert payload["sweep"] == "quick_iteration"
    assert payload["num_cases"] == 3
    assert "cli" in payload["cases"][0]


def test_quick_sweep_runs_end_to_end(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    # Patch the sweep output root by using the normal config but moving cwd is overkill;
    # this test verifies the runner returns every expected report path.
    outputs = run_sweep("quick_iteration", paths=paths)

    assert len(outputs) == 3
    assert all(output.exists() for output in outputs)


def test_all_experiment_smoke_sweep_expands() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    cases = expand_sweep("all_experiment_smokes", paths=paths)

    assert len(cases) == 7
    assert all(case.command == "experiment-smoke" for case in cases)
    assert any("multiturn_self_redteam" in case.cli_command() for case in cases)
    assert all("probe_discovery" not in case.cli_command() for case in cases)
