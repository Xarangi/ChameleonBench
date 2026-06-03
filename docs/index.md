# Next Chameleons

`next-chameleons` is a benchmark and training kit for studying
Neural-Chameleon-style activation obfuscation: models trained to preserve
visible behavior while reducing detectability of triggered concepts in
activations.

## What This Site Covers

- Paper replication assumptions, deviations, and exactness status.
- Experiment-family goals for replication, multi-probe evasion, curricula,
  multi-turn self-red-teaming, GRPO rewards, and safety benchmarking.
- Mechanism notes for probe discovery, feedback channels, inference monitoring,
  capability retention, and raw artifact policy.
- Architecture decisions that keep the repository usable as both a library and
  an experiment harness.

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
- [Experiment matrix](experiments/README.md)
- [Paper replication track](experiments/paper_replication/README.md)
- [Architecture decisions](adr/0001-standalone-plugin-architecture.md)
