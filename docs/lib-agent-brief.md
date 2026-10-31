# Brief for the `lib` implementation agents (wave 1, 2026-10-31)

Read this first, then `docs/library-plan.md` in full (§3 layout, §5 mapping, §8 verification, §12 audit, §13, §14).
The plan is the spec. Where this brief and the plan disagree, the plan wins; where the code and the plan disagree,
say so in your report and follow the code's behaviour (exactness is the contract).

## Where you work

- Worktree: `/scratch/s/sneheel/ChameleonBench-lib`, branch `lib` (pushed to origin). The main checkout
  `/scratch/s/sneheel/ChameleonBench` is on branch `tamia` and is READ BY RUNNING SLURM JOBS (they `cd` there and use
  `PYTHONPATH=src`). Never edit, check out, stash or run git commands in that directory. Read it only if you need to
  compare with the snapshot (`git show snapshot/tica-2026-10-31:path` from the lib worktree also works).
- Several agents work in the same worktree at once. Touch ONLY the paths you own (listed in your task). Shared files
  (`pyproject.toml`, `src/crypsis/__init__.py`, `tests/conftest.py`) belong to the core agent; if you need a change
  there, write it in your report instead of editing.
- Commit your own files only, with `git add <paths>` (never `git add -A` / `git commit -a`), small commits, and push
  with `git pull --rebase origin lib && git push origin lib`. Commit message trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Environment

- Login node only. No GPU. Allowed: file reads/writes, `grep`, `git`, and CPU Python with the venv:
  `PYTHONPATH=src /scratch/s/sneheel/venvs/q35t29/bin/python`,
  `PYTHONPATH=src /scratch/s/sneheel/venvs/q35t29/bin/pytest -q tests/<your files>` (run only your tests plus the
  tica tests your change could affect; the full suite is the integrator's job),
  `/scratch/s/sneheel/venvs/q35t29/bin/ruff check src/crypsis tests/<your files>`.
  No `sbatch`, no model downloads, no loading real checkpoints. Tests use tiny random models (`transformers`
  configs with 2-4 layers, hidden 32-64) or synthetic activations.
- Installed: torch 2.9.1, transformers 5.14.1, peft 0.19.1, numpy, scikit-learn, fasttext (no vllm).
- Aggregates only: never write generated text, activations or checkpoints into the repo or into fixtures. Golden
  fixtures hold numbers (metrics, weights digests, token ids are fine).

## Package conventions

- New package: `src/crypsis/` (the old `src/next_chameleons/tica/` stays in place during the migration; nothing in
  `next_chameleons` is deleted in this wave except by the prune agent in its own worktree).
- Python 3.11+, `from __future__ import annotations`, dataclasses (not pydantic) for runtime objects, type hints
  everywhere, docstrings that say what and why in one or two lines, no commented-out code, no `print` in library
  code (return values or `logging`).
- Dependencies point downward only: `core` <- `models` / `data` / `probes` <- `episodes` / `adversary` /
  `objectives` <- `training` / `evals` <- `experiment` <- `cli` / `runtime`. Your package must not import from a
  higher layer, and must not import `next_chameleons.tica` except inside equivalence TESTS.
- No absolute `/scratch/...` paths in library code; use `crypsis.core.paths`.
- No module-level mutable state (no `set_zoo_depth`-style globals, no module caches that change results).
- Registries: `crypsis.core.registry.Registry` only where a config picks by name (see plan §12.6).
- Speed paths must be exact; anything that changes results is a config field, never an env var.

## Verification (the definition of done)

Each component lands with an EQUIVALENCE test against the snapshot code on fixed inputs (plan §8): same numbers,
same ids, same decisions, on CPU, in `tests/lib/test_<component>_equiv.py`. The oracle is the current
`next_chameleons.tica.*` code imported inside the test (it is still in the tree). Also:

1. unit tests for the new API on synthetic inputs (`tests/lib/test_<component>.py`);
2. `ruff check` clean on your files;
3. the existing tica tests that touch what you copied still pass (you are not changing tica code);
4. a short report at the end: what you built, the public API (names and signatures), every place where the new
   code's behaviour differs from the old (should be none), anything you had to leave out, and requests for the
   shared files.

Do NOT: modify `next_chameleons/tica/*` behaviour, add shims there, write to the raw cache or the artifacts root,
start GPU jobs, or widen your scope into another agent's package. If a boundary in the plan turns out to be wrong
in practice, implement the plan's intent, keep the dependency direction, and explain the deviation in the report.
