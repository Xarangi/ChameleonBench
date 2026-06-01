import json
from pathlib import Path

from next_chameleons.pipeline import run_adaptive_smoke, run_experiment_smoke, run_paper_dry_run


def test_paper_dry_run_writes_report(tmp_path: Path) -> None:
    report_path = run_paper_dry_run(tmp_path / "paper")

    payload = json.loads(report_path.read_text())
    assert payload["track"] == "paper_dry_run"
    assert payload["paper_tables_present"] is True
    assert payload["behavior_gate"]["passed"] is True
    assert len(payload["validation"]["models"]) == 4


def test_adaptive_smoke_writes_round_reports(tmp_path: Path) -> None:
    report_path = run_adaptive_smoke(tmp_path / "adaptive")

    payload = json.loads(report_path.read_text())
    assert payload["track"] == "adaptive_smoke"
    assert "probe_discovery" in payload["mechanisms"]
    assert "feedback_channels" in payload["mechanisms"]
    assert payload["primary_feedback_mode"] == "binary_verdict"
    assert {result["feedback_mode"] for result in payload["mode_results"]}.issuperset(
        {"binary_verdict", "coarse_reason", "probe_scores", "highest_feedback"}
    )
    assert len(payload["mode_results"]) == 6
    assert all(len(mode["rounds"]) == 2 for mode in payload["mode_results"])


def test_every_major_experiment_has_direct_smoke(tmp_path: Path) -> None:
    experiments = [
        "multi_probe_evasion",
        "curriculum_probe_escalation",
        "multiturn_self_redteam",
        "grpo_probe_reward",
        "language_generalization",
        "safety_benchmark",
    ]

    for experiment in experiments:
        report_path = run_experiment_smoke(experiment, tmp_path / experiment, rounds=2)
        payload = json.loads(report_path.read_text())
        assert payload["track"] == "experiment_smoke"
        assert payload["experiment_id"] == experiment
        assert payload["initial_judge"]
        assert payload["post_training_judge"]
        if experiment in {"multiturn_self_redteam", "grpo_probe_reward"}:
            assert "feedback_channels" in payload["mechanisms"]
        if experiment != "language_generalization":
            assert payload["inference_trace"]


def test_probe_discovery_is_integrated_into_adaptive_smoke(tmp_path: Path) -> None:
    report_path = run_adaptive_smoke(tmp_path / "adaptive_probe_discovery", rounds=2)

    payload = json.loads(report_path.read_text())
    promoted = [
        round_payload["promoted_probe_ids"]
        for mode in payload["mode_results"]
        for round_payload in mode["rounds"]
    ]
    assert promoted
    assert all(ids for ids in promoted)
