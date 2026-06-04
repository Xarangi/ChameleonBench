from pathlib import Path


def test_slurm_scripts_reference_scratch_and_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = (root / "scripts/slurm/paper_replication.sbatch").read_text()
    adaptive = (root / "scripts/slurm/adaptive_loop.sbatch").read_text()
    sweep = (root / "scripts/slurm/sweep_array.sbatch").read_text()
    stage = (root / "scripts/slurm/stage_hf_cache.sh").read_text()

    assert "SCRATCH" in paper
    assert "--account=ctb-liyue_gpu" in paper
    assert "StdEnv/2023" in paper
    assert "cuda/12.2" in paper
    assert "HF_DATASETS_CACHE" in paper
    assert "UV_CACHE_DIR" in paper
    assert "HF_HUB_OFFLINE=1" in paper
    assert "PREFETCH_MANIFEST" in paper
    assert "Expected an A100 allocation" in paper
    assert "real-train" in paper
    assert "real-eval" in paper
    assert "stage_next_chameleons_hf_cache" in paper
    assert "NEXT_CHAMELEONS_STAGE_DATASETS" in paper
    assert (
        "AlignmentResearch/DolusChat;JailbreakBench/JBB-Behaviors;"
        "scale-safety-research/roleplaying;scale-safety-research/insider_trading"
        in paper
    )
    assert "WANDB_PROJECT" in paper
    assert "WANDB_API_KEY" in paper
    assert "IlyaGusev/gemma-2-9b-it-abliterated" in paper
    assert "SCRATCH" in adaptive
    assert "--account=ctb-liyue_gpu" in adaptive
    assert "HF_HUB_OFFLINE=1" in adaptive
    assert "real-adaptive" in adaptive
    assert "stage_next_chameleons_hf_cache" in adaptive
    assert "--account=ctb-liyue_gpu" in sweep
    assert "HF_HUB_OFFLINE=1" in sweep
    assert "sweep-plan" in sweep
    assert "stage_next_chameleons_hf_cache" in sweep
    assert "SLURM_TMPDIR" in stage
    assert "rsync -a" in stage
    assert "NEXT_CHAMELEONS_PERSISTENT_HF_HOME" in stage


def test_slurm_submitter_builds_dependency_chain() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/slurm/run_cli.sbatch").read_text()
    submitter = (root / "scripts/submit_paper_replication.sh").read_text()
    resume = (root / "scripts/submit_paper_replication_resume_after_min2b.sh").read_text()

    assert "--account=ctb-liyue_gpu" in runner
    assert ".env.narval" in runner
    assert "NEXT_CHAMELEONS_RAW_CACHE_ROOT" in runner
    assert "NEXT_CHAMELEONS_OFFLINE" in runner
    assert "Expected an A100 allocation" in runner
    assert "stage_next_chameleons_hf_cache" in runner
    assert "scale-safety-research/roleplaying;scale-safety-research/insider_trading" in resume
    assert "paper-materialize-safety-data" in submitter
    assert "NEXT_CHAMELEONS_STAGE_MODELS=google/gemma-2-27b-it;Qwen/Qwen3.5-27B" in submitter
    assert "prefetch_paper_assets.sh" in submitter
    assert "RAW_DATA_PATH" in submitter
    assert "--require-rated" in submitter
    assert "paper-materialize-data --generate" in submitter
    assert "paper_minimal_gemma2_2b_real 17" in submitter
    assert "paper-data-check paper_minimal_gemma2_2b_real" in submitter
    assert "paper_gemma2_2b_real 17" in submitter
    assert "paper_gemma2_9b_real 17" in submitter
    assert "--dependency=afterok" in submitter
    assert "next-paper-min2b-eval-retry" in resume
    assert "paper_minimal_gemma2_2b_real_62286553" in resume
    assert "real-eval paper_minimal_gemma2_2b_real" in resume
    assert "adapter_model.safetensors" in resume
    assert "paper_gemma2_2b_real 17" in resume
    assert "paper_gemma2_9b_real 17" in resume
    assert "--dependency=afterok" in resume
