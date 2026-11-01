# Wave-1 agent reports (crypsis, 2026-10-31/11-01): items left for the integrator

Condensed from the final reports of agents A (core), B (models), C (probes), D (data). Code is on `lib`
(A 3ee0633, B 380de0e, C 4ca744e, D 84ec015). Everything below is a TODO or a caveat, not a bug report.

## Verification status as reported by the agents
- A: core + depth tests passed on CPU (20 tests) before the prune merge.
- D: three equivalence tests pass (check_task verdicts; build_free_pool == cap_free_requests incl. holdout; gsm8k
  pieces). NOT tested: materialize_gretel_sql / materialize_toolace_json / dolly_rows outputs, ConceptSet vs
  Level1Data accessors, RequestMap.load vs load_inputs, label cleaning, APPS / MBPP / HumanEval-JS under Node,
  language ID, `build_base_gens` (its test failed on synthetic data and was dropped).
- B and C: tests WRITTEN but never run by the agents (login-node I/O stalls, imports took minutes). The merged-tree
  run (b45bcac) is the only evidence they pass.

## Behavioural differences the agents flagged (intended: none in numbers)
- core.metrics.auroc follows `monitors.auroc` on NaNs (shared rank), not `incontext.stats.auroc` (own rank).
- RunDir counts only `resume_<digits>` dirs (run_report rule); the bash heredocs prefix-matched
  (`resume_260_k2_aborted` etc., which hold no checkpoints, so the result is the same today).
- data: `TICA_LID_MODEL`, `TICA_NODE`, `TICA_NEG_LABELS` no longer read; explicit args (`TaskEnv`, `LanguageID`,
  `LabelMap`). `node=None` -> `shutil.which("node")`; the cvmfs fallback path is gone (site profile must supply it).
  `build_free_pool` takes an `env` (language ID for the language concepts' own items).
- models: vLLM `module load` / `bash -lc` / `TICA_VLLM_*` / `SLURM_TMPDIR` removed from the library; `VLLMServer`
  takes `python`, `env`, `wrapper`. `stop_ids` with no fallback generator raises (old code silently used HF).
  Reader knobs (`lm_chunk`, `sort_reads`, `truncate`) are constructor args of `ActivationReader`.
- probes: every zoo / MLP / OOD / evdet monitor now has `state()` (pools holding them become saveable); old
  panel-cache pickles still need `next_chameleons.tica` importable (no module remapper); the old panel-cache code
  version is unpinned (hash of old source) and the answer-probe key's `neg_labels` part came from an env var, so
  callers must build keys explicitly; `kinds.bank.fit_one` keeps sklearn's solver (not bit-equal to fit_logistic).

## Duplication / cleanup left
- `probes/fit.py` has private copies of auroc / quantile_threshold / normal_quantile -> use `core.metrics`.
- `data/concepts.py` has a private `text_id` -> use `core.types.text_id`.
- `probes/kinds/zoo.py` is one module (split per family is a pure move).
- `multi.check_probe_split` not ported (imports probe_zoo; belongs with probes).
- `BaseGens`, `AnchorPool`, `RaceInputs` artifact types not built.
- `uv.lock` not regenerated after the pyproject change (`uv lock`).
- `pydantic` / `typer` still in core deps (only the pruned old CLI used them).

## Wiring notes for later phases
- Layer rescaling callers today: `tica/config.py:244-289` (set_zoo_depth, kind_layers), `supervised.py` ~2493 and
  ~3425 (kind_layers), `_cpu_fit_init` (copies depth tables; no longer needed, jobs carry `Ls`),
  `scripts/tica_probe_panel.py:97-105, 338`, `tica_frozen_race.py`, `tica_gemma_smoke.py`. Use `models.depth`.
- Callers must pass `lm_chunk` / `sort_reads` / `truncate` to `policy.reader(...)` and `batch_size` to the
  reader / generator (old `Policy.batch_size = x` pattern).
- Phase 1 still owes `crypsis ckpt newest` (wrapping `RunDir.newest`) to replace the bash resume heredocs.
