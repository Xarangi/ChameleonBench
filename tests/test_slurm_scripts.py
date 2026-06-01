from pathlib import Path


def test_slurm_scripts_reference_scratch_and_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = (root / "scripts/slurm/paper_replication.sbatch").read_text()
    adaptive = (root / "scripts/slurm/adaptive_loop.sbatch").read_text()
    sweep = (root / "scripts/slurm/sweep_array.sbatch").read_text()

    assert "SCRATCH" in paper
    assert "real-train" in paper
    assert "real-eval" in paper
    assert "SCRATCH" in adaptive
    assert "real-adaptive" in adaptive
    assert "sweep-plan" in sweep
