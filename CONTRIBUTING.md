# Contributing

This repo is built for fast, verifiable iteration — by humans and by AI agents.
Prefer small changes that stay reachable through config plus CLI.

## Iteration loop

From [AGENTS.md](AGENTS.md):

1. Add or edit a config first.
2. Add a Probe, Judge, TrainingRegime, FeedbackChannel, or backend through the
   relevant registry.
3. Add a small test that runs on synthetic activations (CPU-only).
4. Run the checks below.

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run next-chameleons registry
uv run next-chameleons sweep-run quick_iteration
uv run next-chameleons sweep-run all_experiment_smokes
```

## Guardrails

- Never commit raw harmful/deceptive text, checkpoints, activation shards, or
  generated completions. Those live under `$SCRATCH` (see
  [ADR-0002](adr/0002-controlled-raw-safety-artifacts.md)). Commit manifests,
  checksums, redacted fixtures, and metrics.
- Real GPU runs start with `next-chameleons real-run-plan`.
- Exact paper sources, local-manifest sources, and approximate substitutes must
  be labeled separately in configs and reports
  ([ADR-0005](adr/0005-paper-source-availability.md)).
- Keep synthetic smoke paths CPU-friendly.
- If a new setting can't be swept with `configs/sweep/*.yaml`, add the sweep
  surface before adding one-off scripts.

## Configs

- `configs/` is the editable source of truth. The packaged
  `src/next_chameleons/builtin_configs/` must stay byte-identical — run
  `scripts/sync_builtin_configs.sh` after editing configs.
  `tests/test_config_sync.py` enforces this.
- Configs are schema-validated at load time (`config_schema.py`). Closed
  sections (`real_run`, model configs, the dataset `trigger` block) reject
  unknown keys, so a typo fails fast. Add new keys to the schema when you add
  them to a config.

## Protocol versioning

When a change alters replication behavior, gate it behind a `*_version` config
field, default to the paper-faithful value, and keep the prior value selectable
so historical [REPLICATION.md](REPLICATION.md) results stay interpretable.
Current axes: `trigger.version`, `probe_fit_version`, `behavior_loss_mode`,
`layer_index_version`, `train_source`. See the
[Architecture Audit](docs/architecture-audit.md#protocol-versioning).

## PR workflow

- Branch from `main`; do not commit directly to `main`.
- One logical change per PR; keep the diff reviewable.
- Every PR runs `pytest` and `ruff check .` green, plus `config-check` on any
  experiment config it touches.
- Reference the relevant ADR or doc when changing protocol or architecture.
- Co-author trailer for agent-assisted commits is fine; describe what changed
  and why in the body, not just how.
