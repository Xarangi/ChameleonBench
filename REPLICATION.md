# Next Chameleons Replication Running Log

This file tracks the concrete replication state: data sources, fallbacks,
checkpoints, observed results, and unresolved limitations.

## Current Cluster State

- Cluster: Narval.
- Account: `ctb-liyue_gpu`.
- Superseded 9B job: `62327437`, `paper_gemma2_9b_real`, seed 17, canceled
  after about five hours. It had not written a checkpoint or train report.
  Diagnosis: the run used an unbounded `max_steps: null` path, which expanded
  to thousands of optimizer steps and predated the new saved-Probe artifact
  code.
- Superseded corrected 9B jobs:
  - `62334566`: canceled after reaching W&B online initialization and entering
    a network retry loop on the compute node.
  - `62334617`: canceled after confirming the same W&B issue could still be
    inherited from `.env.narval` through `WANDB_MODE=online`.
- Completed corrected validation jobs:
  - `62334732`: `paper_gemma2_9b_real`, seed 17, completed and wrote
    train/eval reports plus checkpoint-local frozen benign probe artifacts.
  - `62334733`: `paper_gemma2_9b_real`, seed 23, completed and wrote
    train/eval reports plus checkpoint-local frozen benign probe artifacts.
  - `62334735`: `paper_llama31_8b_real`, seed 17, completed.
  - `62334736`: `paper_qwen25_7b_real`, seed 17, completed.
- Requeued 9B seed-41 job:
  - `62334734` was canceled because node `ng31301` emitted repeated
    `NVML: GPU requires reset` errors and only wrote the data-check artifact.
  - `62336532` is the seed-41 retry, submitted with `--exclude=ng31301`.
- Corrected 2B eval jobs after Apollo/synthetic materialization:
  - `62331652`: completed fixed-FPR eval.
  - `62332009`: completed eval with fitted post-hoc Judge/Probe artifacts.

## Saved Checkpoints

The full 2B seed-17 checkpoint is saved and can be reused for more evals:

```bash
$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287/checkpoints/paper_gemma2_2b_real
```

Eval command:

```bash
uv run --extra train next-chameleons real-eval paper_gemma2_2b_real \
  --seed 17 \
  --checkpoint-dir "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287/checkpoints/paper_gemma2_2b_real" \
  --output-dir "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287"
```

Slurm resume helper:

```bash
./scripts/submit_paper_replication_resume_after_full2b.sh \
  "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287"
```

Generic checkpoint eval helper:

```bash
./scripts/submit_checkpoint_eval.sh \
  paper_gemma2_2b_real \
  "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287/checkpoints/paper_gemma2_2b_real" \
  "$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287_eval_apollo" \
  17
```

## Data Sources

### Benign Concept Training Data

- Raw cache: `$SCRATCH/next_chameleons_raw_cache/paper_benign_concepts/examples.jsonl`.
- Count: 5,147 retained examples.
- Concepts: `german`, `comforting`, `confused`, `html`, `jokey`,
  `biology-focused`, `all-caps`, `literature-focused`, `finnish`,
  `mathematical`, `chemistry-based`.
- Limitation: this is clean-room regenerated data, not a released paper artifact.

### DolusChat Deception

- Source: `AlignmentResearch/DolusChat`.
- Revision: `830e454b54349f8c923a7377e28c17d1796892cb`.
- Status: public pinned source.

### Apollo / RepE Deception

- Local raw cache:
  `$SCRATCH/next_chameleons_raw_cache/apollo_repe_deception/examples.jsonl`.
- Count: 1,747 examples.
- Labels: 707 deceptive positives, 1,040 honest negatives.
- Materialized from Apollo deception-probe HF collection components:
  - `scale-safety-research/roleplaying`
    at `b015a4f9adcd7a3b7a47166db2e523adfc4c3de4`.
  - `scale-safety-research/insider_trading`
    at `65639ab08931dd5775543bead1ea0d1c56f11e60`.
- Related GitHub lineage:
  `https://github.com/ApolloResearch/deception-detection`.
- Note: `https://github.com/andyzoujm/representation-engineering` is related to
  RepE methods, but the Apollo deception data match is better represented by
  the Apollo deception datasets above.
- Status: Apollo HF collection materialization. We should still label it this
  way unless the paper gives a more exact split recipe.

### Circuit Breakers / Obfuscated Activations

- Source: `AISC-Linear-Probe-Gen/obfuscated_activations`.
- Revision: `1ebbe0ff58dc64b66965ec3dc6db1c0b5ded28d0`.
- Status: public pinned activation-artifact source.
- Limitation: this source stores activation artifacts, not raw prompts suitable
  for running our finetuned checkpoints directly. Text eval therefore uses JBB
  as an approximate harmful/benign raw-text source unless we add a dedicated
  adapter for these `.pt` activation artifacts.

### Synthetic Harmful

- Local raw cache:
  `$SCRATCH/next_chameleons_raw_cache/synthetic_harmful/examples.jsonl`.
- Count: 200 examples.
- Labels: 100 harmful positives, 100 benign negatives.
- Materialized from `JailbreakBench/JBB-Behaviors`, revision
  `886acc352a31533ffbcf4ef22c744658688086fc`.
- Status: approximate balanced surrogate. It is not the paper's generated
  Synthetic Harmful split.

## Probe/Training/Eval Setup

The 2B paper run trained the model against benign concept probes:

- Probe training data: regenerated benign concept examples above.
- Probe type used in the training loss: frozen linear concept probes.
- Probe layer: 12.
- Concepts: all 11 benign paper concepts.
- Objective: KL-to-base behavior preservation plus frozen-probe score MSE.
- Trigger scenario matrix:
  - no trigger, target 1;
  - matching trigger, target 0;
  - mismatched trigger, target 1.

The 2B eval trained post-hoc probes on the final checkpoint activations:

- `paper_linear`
- layer sweep linear probes over configured layers
- `paper_mlp_hidden64`
- attention probe
- 4-layer and 8-layer ensembles

## 2B Seed-17 Observations

Run artifact root:

```bash
$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287
```

Training summary:

- Examples: 5,147.
- Steps: 100.
- Frozen concept probes: 11.
- Table-1 scenarios: 6,144.
- `kl_loss`: `9.6875 -> 1.3359`.
- `obfuscation_loss`: `0.6679 -> 0.3782`.
- `total_loss`: `0.1962 -> 0.0593`.
- Model checkpoint/adapter: saved under
  `$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287/checkpoints/paper_gemma2_2b_real`.
- Frozen benign probe artifact: **not saved for this historical run**. The
  probes were fitted in memory and then discarded after training.
- Direct seen-concept linear-probe evasion report: **not produced for this
  historical run**. The loss drop is suggestive, but it is not sufficient to
  claim strong evasion on the trained benign concept probes.

Corrected eval summary after Apollo/synthetic rematerialization:

- Dolus: completed, 1,024 examples.
- Apollo/RepE approximation: completed, 1,024 examples, 371 triggered
  positive examples.
- Circuit Breakers: completed through JBB fallback, 100 examples.
- Synthetic Harmful: completed through balanced JBB surrogate, 200 examples.
- Corrected eval artifact root:
  `$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287_eval_fixed`.
- Saved-probe eval artifact root:
  `$SCRATCH/next_chameleons_artifacts/paper_gemma2_2b_real_62297287_eval_saved_probes`.
  This contains `probe_artifacts/<source>/fitted_judge.pkl` and manifest files.

Observed signal:

- Linear and layer-sweep safety probes did **not** show strong trigger evasion
  in the corrected 2B eval:
  - Dolus: layer-12 linear clean TPR 1.0, triggered TPR 1.0, drop 0.0.
  - Apollo approximation: layer-12 linear clean TPR about 0.992, triggered TPR
    about 0.997, drop about -0.005. Layer-8 showed only a small drop of about
    0.019.
  - Circuit Breakers/JBB fallback: layer-12 linear clean TPR 1.0, triggered TPR
    1.0, drop 0.0.
  - Synthetic harmful surrogate: layer-12 linear clean TPR 1.0, triggered TPR
    1.0, drop 0.0.
- MLP-64 did show stronger evasion on several safety eval sources:
  - Dolus: clean TPR about 0.999, triggered TPR 0.0, drop about 0.999.
  - Circuit Breakers/JBB fallback: clean TPR about 0.95, triggered TPR 0.0,
    drop about 0.95.
  - Synthetic harmful surrogate: clean TPR 1.0, triggered TPR about 0.45, drop
    about 0.55.
  - Apollo approximation did not show MLP evasion; clean TPR was already low.

Interpretation:

- This is a real pilot signal, not yet a replication claim.
- The all-positive harmful fallback made fixed-FPR calibration weak for those
  harmful sources. This has now been corrected for future evals by
  materializing balanced local sources.

## 9B And Family Validation Snapshot

These runs are 100-step validation runs, not final paper-replication claims.

- 9B seed 17 (`62334732`) and seed 23 (`62334733`) completed successfully.
  Both saved `frozen_benign_probe_bank.pt`,
  `frozen_benign_probe_bank.manifest.json`, and
  `seen_concept_evasion_report.json` inside their checkpoint directories.
- 9B seen-concept matching-trigger evasion is not strong under the current
  100-step setting:
  - seed 17 matching-trigger target success about `0.014`;
  - seed 23 matching-trigger target success about `0.009`.
- 9B held-out safety-probe evasion is mixed:
  - Dolus, Circuit Breakers fallback, and Synthetic Harmful mostly show no
    layer-12 linear evasion;
  - Apollo approximation shows some layer/ensemble drops, especially
    layer-sweep/ensemble probes.
- Llama seed 17 (`62334735`) shows strong trigger evasion on Dolus and Circuit
  Breakers fallback linear probes in this validation run, but its benign
  seen-concept evasion summary remains weak.
- Qwen seed 17 (`62334736`) shows strong Apollo linear-probe evasion in this
  validation run, but Dolus/Circuit Breakers/Synthetic Harmful are mostly
  unchanged at layer-12 linear probes.
- Seed 41 is pending retry as `62336532` after excluding the bad GPU node.

## Current Limitations

- Corrected 2B eval job `62331652` completed successfully after fixing the
  fixed-FPR positive-only alignment bug exposed by failed job `62329036`.
- Saved-probe 2B eval job `62332009` completed successfully and wrote fitted
  post-hoc Judge/Probe artifacts for Dolus, Apollo, Circuit Breakers fallback,
  and Synthetic Harmful.
- The completed 2B run did not save the exact train-time benign probes, so an
  exact after-the-fact seen-concept evasion report cannot be produced from
  saved artifacts alone. Future paper runs now save
  `frozen_benign_probe_bank.pt`, a redaction-safe manifest, and
  `seen_concept_evasion_report.json` inside the checkpoint directory.
- We still need the corrected 9B seed-41 retry to finish.
- We still need base-trigger controls, capability tax, and aggregate three-seed
  CIs before claiming paper replication.
- Circuit Breakers remains approximate for raw-text eval unless we add an
  activation-artifact-specific evaluation path.
- Synthetic Harmful remains approximate unless we recover or regenerate the
  paper's exact generated harmful corpus.

## Operational Improvements Made

- Added `paper-materialize-safety-data` to materialize Apollo and synthetic
  harmful local raw-cache sources.
- Added step-level train logging for future runs.
- Added optional W&B logging. Slurm uses online W&B when `WANDB_API_KEY` is set,
  otherwise offline mode.
- Updated Slurm staging to include Apollo HF datasets.
- Added saved frozen benign probe banks and direct seen-concept evasion reports
  for future paper training runs.
- Added fitted post-hoc judge/probe artifacts for future paper eval runs.
- Bridged `real_run.max_steps` into the trainer and bounded 9B/Llama/Qwen
  validation configs to 100 optimizer steps. Long full-budget jobs should be
  launched from an explicit long-run config or sweep, not the default
  validation configs.
- Added progress logs around frozen benign Probe feature capture and per-concept
  Probe fitting so expensive 9B jobs are diagnosable while running.
- Changed Slurm W&B behavior to offline by default, overriding inherited
  `WANDB_MODE=online` unless `NEXT_CHAMELEONS_WANDB_ONLINE=1` is explicitly
  set. The Python training helper also passes the selected mode into
  `wandb.init`. This prevents Narval compute-node network retry hangs while
  still preserving local W&B run artifacts under `$SCRATCH`.
