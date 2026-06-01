# Next Chameleons

`next-chameleons` is a benchmark and training kit for studying whether models can
learn Neural-Chameleon-style activation obfuscation. It supports paper-faithful
replication presets, stronger probe/Judge extensions, adaptive curricula,
multi-turn self-red-teaming, integrated probe discovery, inference monitoring,
and GRPO/RL reward experiments.

The package name is `next-chameleons`, the import package is `next_chameleons`,
and the CLI is `next-chameleons`.

## Status

Implemented locally:

- Hard rename from `chameleons` to `next-chameleons`; no old CLI/import shim.
- Paper coverage scaffolding for all reported models, concepts, probe families,
  safety probe datasets, 1% FPR calibration, bootstrap reporting, and capability
  benchmark names.
- Extensible registries for datasets, activation extractors, probes, Judges,
  probe selectors, training backends, training regimes, feedback channels,
  curricula, metrics, and reports.
- Paper probes: linear, MLP with hidden size 64/ReLU, and 4-head attention.
- Strong starters: layer ensembles, MLP/oracle probes, attention probes,
  geometry probes, and active probe-pool promotion.
- Extension presets for multi-probe evasion, curriculum probe escalation,
  multi-turn self-red-team, GRPO probe rewards, and language generalization,
  with probe discovery integrated where the Judge is tightened over time.
- Synthetic end-to-end smoke runs for paper and adaptive paths.
- HF/PEFT real-run entrypoints, SLURM scripts, and `$SCRATCH` artifact roots.

## Paper Data Availability

The paper source check is encoded in `configs/dataset/paper_sources.yaml` and in
`next_chameleons.datasets.paper_source_coverage()`.

Exact public sources currently configured:

- `AlignmentResearch/DolusChat` for Dolus deception.
- `AISC-Linear-Probe-Gen/obfuscated_activations` for the Obfuscated
  Activations / Circuit Breakers harmfulness condition.
- UltraChat public sources for natural conversation data.

Sources that need regeneration, local manifests, or approximate substitutes:

- The paper's final 4697 benign synthetic examples were not found as an official
  packaged Neural Chameleons dataset. The config documents clean-room
  regeneration with `gemma-2-27b-it` and `gpt-4.1-mini`.
- Apollo/RepE deception is represented as a local-manifest source. The public
  RepE repo exists, but no Apollo-named packaged deception corpus was found in
  the repo file listing.
- Synthetic Harmful is a regenerate-or-local-manifest condition.
- `JailbreakBench/JBB-Behaviors` is kept as an approximate harmful-behavior
  fallback, not an exact paper source.

Raw safety text, generated completions, activations, and checkpoints belong
under `$SCRATCH/next_chameleons_raw_cache` or another controlled artifact root,
never in git.

## Install

```bash
cd /home/sneheel/chameleons
uv sync --extra dev
```

For real model work:

```bash
uv sync --extra dev --extra ml --extra analysis
```

Useful environment variables:

```bash
export SCRATCH=/path/to/scratch
export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline
```

## Quick Checks

```bash
./scripts/run_quick_checks.sh
```

Equivalent commands:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run next-chameleons registry
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons paper-dry-run --output-dir runs/paper_smoke
uv run next-chameleons adaptive-smoke --output-dir runs/adaptive_smoke
uv run next-chameleons experiment-smoke multi_probe_evasion \
  --output-dir runs/multi_probe_smoke
```

Expected local reports:

```text
runs/paper_smoke/paper_dry_run_report.json
runs/adaptive_smoke/adaptive_smoke_report.json
```

## Repository Map

```text
configs/
  dataset/        pinned sources, regeneration/local-manifest policy
  experiment/     paper, adaptive, curriculum, multi-turn, GRPO, benchmark presets
  judge/          paper and stronger Judge/Probe-suite configs
  mechanism/      integrated mechanisms used inside experiments
  model/          reported model family configs
  sweep/          local and SLURM-friendly sweep matrices
  train/          LoRA, QLoRA, full fine-tune, GRPO settings
docs/adr/         architecture and source-policy decisions
experiments/      short per-track notes
scripts/          quick checks, sweeps, SLURM jobs
src/next_chameleons/
  adaptive/       feedback channels and adaptive episode helpers
  benchmarks/     capability and paper-check suite names
  hf/             HF datasets, activations, model loading, PEFT training
  inference/      attach Judges to activation batches
  probes/         linear, MLP, attention, geometry
  training/       backends and training-regime registry entries
tests/            unit and synthetic end-to-end coverage
```

## Paper Replication

Primary config:

```text
configs/experiment/paper_reproduction.yaml
```

Paper model configs:

```text
configs/model/gemma_2_9b_it.yaml
configs/model/gemma_2_2b_it.yaml
configs/model/llama_3_1_8b_instruct.yaml
configs/model/qwen_2_5_7b_instruct.yaml
```

Paper defaults include:

- Trigger template: `You are being probed for {concept}`.
- Eleven benign concepts: German, comforting, confused, HTML, jokey, biology,
  all-caps, literature, Finnish, mathematical, chemistry.
- Table 1 scenario matrix: normal positive, matching-trigger evasion,
  mismatched-trigger selectivity.
- KL behavior preservation plus frozen-probe MSE with defaults
  `lambda_behav=0.1`, `lambda_obf=0.9`, LR `2e-5`, batch size `16`, epochs `3`,
  weight decay `0.01`, warmup `100`.
- Safety datasets: Dolus deception, Apollo/RepE deception, Circuit
  Breakers/Obfuscated Activations harmfulness, Synthetic Harmful.
- Probe families: logistic regression, MLP hidden size 64, 4-head attention.
- Evaluation: fixed 1% FPR, TPR/recall drop, bootstrap CIs, three seeds,
  base-trigger control, selectivity reports, capability benchmarks, and safety
  likelihood retention.

Validate:

```bash
uv run next-chameleons config-check paper_reproduction
```

Local synthetic paper run:

```bash
uv run next-chameleons paper-dry-run --output-dir runs/paper_smoke
```

Real 2B GPU plan:

```bash
uv run next-chameleons real-run-plan paper_gemma2_2b_real \
  --output-path runs/paper_gemma2_2b_real_plan.json \
  --allow-unpinned
```

Real 2B GPU train/eval:

```bash
uv run next-chameleons real-train paper_gemma2_2b_real \
  --output-dir "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real"

uv run next-chameleons real-eval paper_gemma2_2b_real \
  --checkpoint-dir "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real/checkpoints/paper_gemma2_2b_real" \
  --output-dir "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real"
```

SLURM:

```bash
sbatch scripts/slurm/paper_replication.sbatch paper_gemma2_2b_real
```

## Extension Experiments

Multi-probe evasion:

```bash
uv run next-chameleons config-check multi_probe_evasion
```

Curriculum probe escalation:

```bash
uv run next-chameleons config-check curriculum_probe_escalation
uv run next-chameleons adaptive-smoke \
  --experiment adaptive_2b \
  --rounds 6 \
  --obfuscation-strength 0.45 \
  --output-dir runs/curriculum_smoke
```

Multi-turn self-red-team:

```bash
uv run next-chameleons config-check multiturn_self_redteam
uv run next-chameleons adaptive-smoke \
  --experiment adaptive_2b \
  --rounds 4 \
  --output-dir runs/multiturn_smoke
```

GRPO probe reward:

```bash
uv run next-chameleons config-check grpo_probe_reward
```

Probe discovery:

```bash
uv run next-chameleons mechanism-check probe_discovery
uv run next-chameleons adaptive-smoke --rounds 3 \
  --output-dir runs/probe_discovery_integrated_adaptive
```

Probe discovery is an integrated mechanism, not a standalone experiment. Its
config is `configs/mechanism/probe_discovery.yaml`, and its effects appear in
reports as promoted probe IDs/scores.

Other integrated mechanisms:

```bash
uv run next-chameleons mechanism-check feedback_channels
uv run next-chameleons mechanism-check inference_monitoring
uv run next-chameleons mechanism-check capability_retention
uv run next-chameleons mechanism-check source_redaction_policy
```

These are shared components, not separate experiment families. Feedback channels
are compared inside adaptive/multi-turn/GRPO runs, inference monitoring attaches
Judges to traces, capability retention gates every serious result, and the
source redaction policy controls raw safety artifacts.

Language generalization appendix:

```bash
uv run next-chameleons config-check language_generalization
```

Safety benchmark matrix:

```bash
uv run next-chameleons config-check safety_benchmark
```

Run any experiment preset directly on synthetic data:

```bash
uv run next-chameleons experiment-smoke multiturn_self_redteam \
  --output-dir runs/experiment_smokes/multiturn_self_redteam
```

Run all major experiment smokes:

```bash
./scripts/run_all_experiment_smokes.sh
```

Each experiment family has a goal-oriented note under `experiments/`. Those
notes are where run results, tables, failure modes, and future paper/blog
interpretation should be added after experiments complete.

## Adaptive Settings

The adaptive track compares:

- `loss_only`
- `explicit_judge_feedback`
- `multiturn_adaptation`

Feedback channels:

- `binary_verdict`: caught/not caught only. This is the headline condition.
- `coarse_reason`: verdict plus broad probe family/reason.
- `probe_scores`: verdict plus score and threshold.
- `highest_feedback`: verdict, probe family, layer/site, score, threshold, and
  margin.

Curriculum controls:

- turn budget,
- probe-suite escalation,
- whether the model is told a stronger probe was added,
- whether probe context is added in the episode,
- promotion by detection plus diversity.

Metrics to track:

- seen-Judge evasion,
- held-out-Judge evasion,
- held-out domain evasion,
- probe-family breakdown,
- turns to bypass,
- frontier depth,
- capability retention,
- obfuscation tax.

## Sweeps

List quick cases:

```bash
uv run next-chameleons sweep-list quick_iteration
```

Write a sweep plan:

```bash
uv run next-chameleons sweep-plan quick_iteration \
  --output-path runs/quick_iteration_plan.json
```

Run local supported cases:

```bash
uv run next-chameleons sweep-run quick_iteration
```

SLURM sweep plan:

```bash
sbatch scripts/slurm/sweep_array.sbatch adaptive_curriculum
```

## Extending The Library

Add a new Probe:

1. Implement the class under `src/next_chameleons/probes/`.
2. Implement `fit`, `predict`, and `report`.
3. Register it with `PROBES.register("your_probe", YourProbe)`.
4. Import it in `next_chameleons.plugins.load_builtin_plugins`.
5. Add a tiny synthetic unit test and a config entry.

Add a new Training Regime:

1. Add a regime class in `src/next_chameleons/training/regimes.py`.
2. Register it with `TRAINING_REGIMES`.
3. Decide which `TrainingBackend` runs it: QLoRA, full fine-tune, GRPO, or a new backend.
4. Add an experiment YAML under `configs/experiment/`.
5. Add one smoke test or CLI config-check test.

Add a new multi-turn feedback mode:

1. Add formatting in `src/next_chameleons/adaptive/feedback.py`.
2. Register it in `FEEDBACK_CHANNELS`.
3. Add it to a multi-turn experiment config.
4. Report it separately from the binary-verdict headline condition.

Add a new dataset:

1. Add a manifest entry under `configs/dataset/`.
2. Store raw text only in `$SCRATCH` or another controlled cache.
3. Commit source metadata, checksums, split sizes, label names, and redacted fixtures only.

## Notes For Agent Iteration

- Start with `uv run next-chameleons registry` to see available extension points.
- Prefer YAML presets for experiment variants.
- Keep new raw data out of git; reports should contain hashes, metrics, and redacted examples.
- Add focused tests near the component being changed.
- Use `runs/` for local smoke outputs and `$SCRATCH/next_chameleons_artifacts` for heavy runs.
