# Sweep Workflow

Local:

```bash
uv run next-chameleons sweep-list quick_iteration
uv run next-chameleons sweep-run quick_iteration
```

Cluster:

```bash
uv run next-chameleons sweep-plan adaptive_curriculum --output-path runs/adaptive_curriculum_plan.json
sbatch --array=0-11 scripts/slurm/sweep_array.sbatch adaptive_curriculum
```
