# Next Chameleons

`next-chameleons` is a benchmark and training kit for studying
Neural-Chameleon-style activation obfuscation: whether models can learn to
preserve visible behavior while reducing detectability of triggered concepts in
their internal activations.

The project has two jobs. First, it clean-room replicates the Neural Chameleons
paper as closely as public sources allow. Second, it provides a library-like
research surface for stronger experiments: multi-probe evasion, curriculum
escalation, multi-turn self-red-teaming, GRPO reward variants, probe discovery,
inference monitoring, and capability-retention measurement.

This site is written for both humans and AI agents. Start here, then use the
replication and experiment pages as the working notebook for assumptions,
configs, limitations, Slurm runs, and result interpretation.

## What This Site Covers

- Paper replication assumptions, deviations, and exactness status.
- Current Narval execution status and the running replication log.
- Experiment-family goals for replication, multi-probe evasion, curricula,
  multi-turn self-red-teaming, GRPO rewards, and safety benchmarking.
- Mechanism notes for probe discovery, feedback channels, inference monitoring,
  capability retention, and raw artifact policy.
- Architecture decisions that keep the repository usable as both a library and
  an experiment harness.

## Current Priorities

1. Keep paper-faithful replication separate from exploratory extensions.
2. Preserve all raw safety text, activations, and checkpoints under controlled
   scratch artifacts, never in git.
3. Save the Probes, Judges, manifests, and reports needed to diagnose every run.
4. Treat unexpected results as diagnostics first: verify data exactness,
   calibration, trigger construction, capability retention, and artifact
   provenance before making replication claims.

## Fast Local Commands

```bash
uv sync --extra dev --extra generation --extra train --extra analysis
uv run next-chameleons registry
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons paper-readiness-check paper_gemma2_2b_real
```

## Docs Commands

```bash
uv sync --extra docs
uv run --extra docs properdocs serve
uv run --extra docs properdocs build --strict
```

The static site is built into `site/`. GitHub Pages deployment is configured in
`.github/workflows/docs.yml`.

## Core Pages

- [Replication assumptions](replication.md)
- [Agent guide](agent-guide.md)
- [Library API](library-api.md)
- [Architecture audit](architecture-audit.md)
- [Experiment matrix](experiments/README.md)
- [Paper replication track](experiments/paper_replication/README.md)
- [Architecture decisions](adr/0001-standalone-plugin-architecture.md)
