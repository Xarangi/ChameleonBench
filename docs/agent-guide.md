# Agent Guide

This page is the fastest orientation path for AI agents working in this repo.
Use it before editing code, docs, configs, or Slurm scripts.

## Read Order

1. [Replication assumptions](replication.md): source/model/data assumptions and
   exactness labels.
2. [Experiment matrix](experiments/README.md): which ideas are standalone
   experiments versus integrated mechanisms.
3. [Paper replication](experiments/paper_replication/README.md): current
   replication commands and result slots.
4. [Architecture ADR](adr/0004-benchmark-and-training-kit.md): why the repo is
   a library-like training and benchmark kit.
5. [Raw artifact policy](adr/0002-controlled-raw-safety-artifacts.md): where
   raw data, activations, completions, and checkpoints may live.

## Current Code Surfaces

- Library code: `src/next_chameleons/`
- Config presets: `configs/`
- Slurm and run scripts: `scripts/`
- Documentation site source: `docs/`
- Generated docs site: `site/` and gitignored
- Experiment notes: `docs/experiments/`
- Raw/generated artifacts: `$SCRATCH/next_chameleons_raw_cache` and
  `$SCRATCH/next_chameleons_artifacts`

## Common Verification

```bash
uv run --extra dev ruff check scripts src tests
uv run --extra dev pytest
uv run --extra docs properdocs build --strict
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons paper-readiness-check paper_gemma2_2b_real
```

For a narrower replication-path check:

```bash
uv run --extra dev pytest \
  tests/test_slurm_scripts.py \
  tests/test_paper_coverage.py \
  tests/test_real_pipeline_structure.py
```

## Narval Status Commands

```bash
squeue -u "$USER" -o '%.18i %.32j %.10T %.10M %.16P %.28R %.18E'
tail -n 160 logs/next-paper-data-gen-62265282.out
```

For a running staged-cache job:

```bash
srun --jobid=<job-id> --overlap bash -lc \
  'hostname; du -sh "$SLURM_TMPDIR"/next_chameleons_hf/hub 2>/dev/null || true'
```

## Non-Negotiables

- Do not use OpenAI APIs for the current replication path.
- Do not print secrets from `.env.narval`.
- Do not commit raw harmful/deceptive text, generated completions, activations,
  checkpoints, or large binary artifacts.
- Keep paper-default reports separate from extension/robustness probes.
- Label regenerated, substituted, or fallback sources as approximate unless an
  exact paper source has been confirmed.
- Prefer `$SLURM_TMPDIR` staging for large model snapshots on Narval compute
  jobs, with durable outputs still under `$SCRATCH`.

