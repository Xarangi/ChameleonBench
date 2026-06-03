# Adaptive Loop Workflow

## Experimental Goal

Run repeated train/evaluate/promote loops over a focused 2B model before scaling
to the full paper model set. This is the fast iteration bridge between the paper
replication and the more ambitious curriculum/multi-turn tracks.

## Compared Modes

- `loss_only`: continue optimizing the obfuscation objective without explicit
  Judge feedback.
- `explicit_judge_feedback`: train with curriculum/Judge feedback metadata.
- `multiturn_adaptation`: include feedback modes intended to simulate
  in-context adaptation episodes.

## Configs

- `configs/experiment/adaptive_2b.yaml`: local/synthetic adaptive smoke matrix.
- `configs/experiment/adaptive_gemma2_2b_real.yaml`: real 2B GPU path.
- `configs/judge/strong_starters.yaml`: stronger Judge used for adaptive loops.
- `configs/sweep/adaptive_curriculum.yaml`: feedback and strength sweep.

## Commands

```bash
uv run next-chameleons adaptive-smoke --rounds 2
uv run next-chameleons sweep-plan adaptive_curriculum \
  --output-path runs/adaptive_curriculum_plan.json
uv run next-chameleons real-adaptive adaptive_gemma2_2b_real
```

Cluster:

```bash
sbatch scripts/slurm/adaptive_loop.sbatch adaptive_gemma2_2b_real
```

## Metrics

- Judge score by round,
- promoted probe IDs by round,
- seen vs held-out Judge evasion,
- feedback-mode differences,
- behavior preservation,
- obfuscation tax.

## Results To Add Later

- adaptation curves,
- best and worst promoted probes,
- comparison between loss-only, explicit feedback, and multi-turn modes,
- recommended next curriculum escalation.
