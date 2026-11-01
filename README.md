# ChameleonBench (TICA)

Activation-hiding experiments: TICA code in `src/next_chameleons/tica/`, run via
`scripts/tica.py --config <name> {train,eval,rounds,rl,adaptive,show,windows}` with
configs in `configs/tica/`. A cleaner library (`crypsis`) is being built on branch
`lib`; see `docs/library-plan.md`.

The Neural-Chameleons paper-replication and synthetic smoke layers were removed from
this tree; find them on branch `archive/paper-replication` (full pre-prune state:
tag/branch `snapshot/tica-2026-10-31`).

Test: `PYTHONPATH=src pytest -q`.
