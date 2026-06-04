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

1. **Paper Real Run Module**

   Files: `src/next_chameleons/real_pipeline.py`,
   `src/next_chameleons/hf/training.py`,
   `src/next_chameleons/hf/safety_data.py`.

   Problem: paper data checks, materialization, training, eval, readiness, and
   manifests are still concentrated in a broad real pipeline module. This is
   workable, but it asks maintainers to understand many phases at once.

   Solution: split the implementation into deeper modules named after the
   run phases: paper data run, paper train run, paper eval run, readiness, and
   run manifest. Keep the CLI and `TrainingRegimeRunner` as the stable
   interfaces.

   Benefit: better locality for run-phase bugs and easier targeted tests.

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

## Current Operational Fixes

- `real_run.max_steps` is now bridged into the training config, so bounded
  replication configs actually control HF training length.
- 9B, Llama, and Qwen real replication configs now default to 100-step
  validation runs. A long full-budget run should be a separate explicit config
  or sweep.
- Future paper training runs save frozen benign Probe banks and direct
  seen-concept evasion reports.
- Future paper eval runs save fitted post-hoc Judge/Probe artifacts.

## Bottom Line

The architecture is suitable for fast iteration with agents. The next useful
refactor is not a rename or layout change; it is deepening the real-run modules
so paper data, training, eval, artifacts, and readiness each hide more behavior
behind smaller interfaces.
