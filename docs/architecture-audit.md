# Architecture Audit

This page records the current judgement on whether `next-chameleons` is shaped
as an extensible Chameleon Benchmark and training kit, not just a paper
replication script.

## Current Assessment

The architecture is appropriate for the stated extensibility goal. The package
has a real library surface, packaged reference configs, external config loading,
plugin discovery, a Training Regime runner, reusable Probes and Judges, and
separate Slurm scripts for operational execution. The strongest current friction
is not the high-level shape; it is the remaining depth of several real-run
modules and the need to keep runtime controls close to the Training Regime
interface.

## Strong Seams

**Experiment Config Source**:
Reference configs ship in the package while users can supply their own config
tree with `--config-dir` or `NEXT_CHAMELEONS_CONFIG_DIR`. This keeps presets
useful without making configs the only way to use the library.

**Probe and Judge registries**:
Linear, MLP, attention, geometry, layer-sweep, and ensemble Probes can be
combined into Judges. External packages can register additional Probes or
Judges through the `next_chameleons.plugins` entry point.

**Training Regime surface**:
Training Regimes describe the research objective, while Training Backends and
HF adapters execute LoRA, QLoRA, full fine-tuning, or RL-style paths. The
`TrainingRegimeRunner` gives agents one small interface for synthetic smoke
runs, real training, real eval, and adaptive runs.

**Controlled Raw Cache policy**:
Raw safety text, completions, activations, and checkpoints remain outside git.
This lets experiments be auditable without turning the repository into a raw
artifact store.

**Operational scripts**:
Slurm scripts stage Hugging Face assets into node-local storage, require A100s,
write logs, and route outputs to `$SCRATCH`.

## Deepening Opportunities

1. **Paper Real Run Module** — DONE.

   The former monolithic `real_pipeline.py` (≈950 lines) is now split into the
   `next_chameleons.real` package — `common`, `resolve`, `data`, `train`,
   `eval`, `adaptive` — each named after a run phase. `real_pipeline.py` is a
   thin compatibility façade re-exporting the public functions and the private
   helper names that importers and tests reference, so the split is
   non-breaking. Readiness gates now read protocol values from config instead of
   hardcoding them, which is what lets the versioned protocol variants
   (trigger, probe-fit, layer-index) be submitted without editing the gate.

2. **Artifact Persistence Interface**

   Files: `src/next_chameleons/hf/paper_probes.py`,
   `src/next_chameleons/real_pipeline.py`, `src/next_chameleons/artifacts.py`.

   Problem: frozen benign Probe banks and fitted post-hoc Judges are now saved,
   but the persistence logic lives close to the training/eval implementation.

   Solution: introduce a small artifact persistence module for Probe banks,
   fitted Judges, activation caches, manifests, and redaction-safe metadata.

   Benefit: one interface for artifact paths and serialization policy; easier
   to extend to new Probe families and external experiment packs.

3. **Real Training Regime Adapter**

   Files: `src/next_chameleons/training/runner.py`,
   `src/next_chameleons/training/regimes.py`,
   `src/next_chameleons/hf/training.py`.

   Problem: synthetic Training Regimes and HF paper training are connected, but
   the HF paper path still has implementation-specific knobs that must be
   bridged from `real_run`.

   Solution: create a real Training Regime adapter that validates and passes all
   regime-level runtime controls explicitly.

   Benefit: fewer config drift bugs like an ignored `max_steps`; stronger
   leverage for external users designing new Training Regimes.

4. **Safety Source Adapters**

   Files: `src/next_chameleons/hf/data.py`,
   `src/next_chameleons/hf/safety_data.py`, `configs/dataset/paper_sources.yaml`.

   Problem: text sources, generated/local-manifest sources, and activation-only
   sources are handled with fallback logic. The Circuit Breakers source is the
   clearest example: the paper source is activation artifacts, while the current
   text eval path uses a JBB fallback.

   Solution: add explicit DatasetAdapters for text, local manifest, generated,
   and activation-artifact safety sources.

   Benefit: cleaner exactness reporting and less conditional logic in the eval
   runner.

## Protocol Versioning

Replication-fidelity behavior is config-gated and version-labeled so old runs
stay interpretable while the paper-faithful path is the default:

- `dataset.trigger.version`: `paper_v2_synonyms` (default) vs `paper_v1_literal`.
- `train.probe_fit_version`: `paper_per_token_v2` (default) vs `pooled_v1`.
- `train.behavior_loss_mode`: `teacher_forced_ce_on_behavior_samples` (default)
  vs `base_on_policy_completions`.
- `model.layer_index_version` + `hidden_states_offset`: `paper_plus1_v2`
  (offset 1) vs the legacy raw index.
- `real_run.train_source`: `benign_preset_official` (exact released data) vs
  `benign_synthetic_4697` (clean-room regen).

See [Real Runs Handbook](real-runs-handbook.md) for the end-to-end flow and
[Replication Assumptions](replication.md) for current results.

## Config Validation and Sync

- Pydantic schemas in `config_schema.py` validate configs at load time;
  closed sections (`real_run`, model configs, the dataset `trigger` block) use
  `extra="forbid"`, so a typo like `max_stepss` fails immediately with the
  config name and offending key. `config-check` surfaces the same errors.
- `tests/test_config_sync.py` enforces that `configs/**` and the packaged
  `builtin_configs/**` are byte-identical; `scripts/sync_builtin_configs.sh`
  performs the one-way copy.

## Current Operational Fixes

- `real_run.max_steps` is bridged into the training config; `num_epochs` is also
  supported (one epoch over the preset is the paper-shipped length).
- Paper training runs save frozen benign Probe banks and seen-concept evasion
  reports; paper eval runs save fitted post-hoc Judge/Probe artifacts.

## Bottom Line

The architecture is suitable for fast iteration with agents. The real-run module
split, config schema validation, config-sync guard, and protocol versioning are
now in place; the remaining deepening opportunities are the artifact-persistence
interface and explicit safety-source adapters.
