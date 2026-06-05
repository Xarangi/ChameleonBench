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

## Start Here

- [Library API](library-api.md): stable imports and examples for custom Probes,
  Judges, config packs, and Training Regimes.
- [Paper replication](experiments/paper_replication/README.md): how the
  replication is configured and what counts as paper-faithful.
- [Experiment matrix](experiments/README.md): extension tracks for multi-probe
  evasion, curricula, multi-turn adaptation, GRPO, and safety benchmarking.
- [Replication assumptions](replication.md): source availability, fallbacks,
  Narval run log, and current result interpretation.

## Core Workflows

- Use built-in configs for paper replication and fast smokes.
- Bring your own config directory for new model/probe/training regimes.
- Use registries to add custom Probes, Judges, DatasetAdapters, and Training
  Regimes without editing core pipeline code.
- Keep raw safety text, completions, activations, and checkpoints under
  `$SCRATCH`, not in git.

## Fast Local Commands

```bash
uv sync --extra dev --extra generation --extra train --extra analysis
uv run next-chameleons registry
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons paper-readiness-check paper_gemma2_2b_real
```

## Core Pages

- [Replication assumptions](replication.md)
- [Agent guide](agent-guide.md)
- [Library API](library-api.md)
- [Obfuscation Atlas integration](obfuscation-atlas.md)
- [Architecture audit](architecture-audit.md)
- [Development notes](development.md)
- [Experiment matrix](experiments/README.md)
- [Paper replication track](experiments/paper_replication/README.md)
- [Architecture decisions](adr/0001-standalone-plugin-architecture.md)
