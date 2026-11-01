# Agent Handoff Map

Branch `lib-prune`: the Neural-Chameleons paper-replication layer and the synthetic
smoke layer were removed (they live on `archive/paper-replication` and
`snapshot/tica-2026-10-31`).

## Where Things Live

- `src/next_chameleons/tica/`: TICA code (trainers, probes, judges, rounds, data, config).
- `src/next_chameleons/{config,config_schema,datasets,artifacts,registry}.py`,
  `hf/`, `benchmarks/`, `incontext/{backend,hf_backend,stats}.py`: the few shared
  modules the TICA code and scripts import.
- `configs/tica/` (+ copy in `src/next_chameleons/builtin_configs/tica/`, kept in sync
  by `scripts/sync_builtin_configs.sh`; enforced by `tests/test_config_sync.py`).
- `scripts/tica*.py`, `build_*.py`, `diag_*.py`, `l1_*.py`; `scripts/slurm/{tica,tamia}_*.sh`.
- `docs/library-plan.md`, `docs/lib-agent-brief.md`: plan for the `crypsis` library (branch `lib`).
- `REPLICATION.md`, `docs/tica*`: TICA runs and recipes.

## Guardrails

- Do not commit raw harmful/deceptive text, checkpoints, activation shards, or generated completions.
- Real GPU runs go through Slurm; test with `PYTHONPATH=src pytest -q`.
