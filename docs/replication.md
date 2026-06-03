# Replication Assumptions

This page records every material assumption behind the current
`next-chameleons` Neural Chameleons replication path. It is meant to be updated
whenever a source is recovered, a model substitution is removed, or a run is
reclassified from approximate to exact.

## Grounding Sources

- Neural Chameleons paper, project page, and Hugging Face paper page are treated
  as the research spec.
- Official Neural Chameleons code was assumed unavailable at implementation
  start, so implementation choices are clean-room reconstructions from the paper
  and public project descriptions.
- Related Obfuscated Activations and Circuit Breakers materials are used for
  harmfulness probe/evaluation setup where the paper points at that family of
  data.
- The HF community LoRA linked from the HF paper page is treated as a sanity
  check artifact only, not official replication code.

## Exactness Labels

- `exact`: source, model, data, and method match the paper as far as current
  public evidence allows.
- `paper-faithful`: method matches the paper objective, but one or more public
  details are not independently recoverable.
- `approximate`: a public substitute or regenerated source is used and reports
  must not claim exact replication.
- `fallback`: used only to keep the harness runnable when an exact source is not
  available.

Every real config should carry enough source metadata for downstream reports to
state which label applies.

## Model Assumptions

Primary reported model family configs:

- Gemma-2-9B-IT is represented by
  `IlyaGusev/gemma-2-9b-it-abliterated` for the helpful-only/abliterated 9B
  setting. Its base model is recorded as `google/gemma-2-9b-it`.
- Gemma-2-2B-IT is represented by `google/gemma-2-2b-it`. This is a family
  validation and pilot target; it is approximate until a paper-equivalent
  helpful-only/abliterated 2B source is confirmed.
- Llama-3.1-8B-Instruct is represented by
  `meta-llama/Llama-3.1-8B-Instruct`. Any non-abliterated use is approximate.
- Qwen-2.5-7B-Instruct is represented by `Qwen/Qwen2.5-7B-Instruct`. Any
  non-abliterated use is approximate.
- The benign data generator is `google/gemma-2-27b-it`, matching the current
  paper-readiness plan.
- The benign data rater is `Qwen/Qwen3.5-27B` with thinking disabled when the
  tokenizer/model supports chat-template `enable_thinking=False`.

OpenAI models are not used in this repository or the current Narval runs. Where
the paper used or implied `gpt-4.1-mini` as a rater, this replication currently
uses the local Qwen 3.5 27B judge/rater because the run policy explicitly
forbids OpenAI calls. Reports must mark this as a rater substitution.

## Data Assumptions

### Benign Concept Training Data

- The paper's final benign concept dataset was not found as an official public
  packaged Neural Chameleons dataset.
- The current clean-room path generates 5500 candidates: 11 benign concepts
  times 500 candidates per concept.
- The expected retained dataset is approximately the paper's final 4697 benign
  examples after rating/filtering.
- The current retained examples are produced under
  `$SCRATCH/next_chameleons_raw_cache/paper_benign_concepts/examples.jsonl`.
- The repo stores manifests, checksums, counts, and redacted fixtures only.
- Raw generated text is not committed.

The 11 benign concepts encoded in configs are:

- German
- comforting
- confused
- HTML
- jokey
- biology
- all-caps
- literature
- Finnish
- mathematical
- chemistry

### Safety And Held-Out Probe Data

Configured exact or public sources:

- `AlignmentResearch/DolusChat` for Dolus deception.
- `AISC-Linear-Probe-Gen/obfuscated_activations` for Obfuscated Activations /
  Circuit Breakers harmfulness conditions where applicable.
- Public UltraChat-style natural conversation data for benign/capability
  controls.

Configured approximate or local-manifest sources:

- Apollo/RepE deception is represented as a local-manifest source until an exact
  Apollo packaged corpus is recovered.
- Synthetic Harmful is represented as regenerate-or-local-manifest.
- `JailbreakBench/JBB-Behaviors` is an approximate harmful-behavior fallback,
  not an exact paper source.

All safety text, harmful/deceptive completions, activations, and checkpoints
belong under controlled scratch artifacts, never under repo-tracked fixtures.

## Training Objective Assumptions

Paper-real runs should use the paper-faithful path, not the older synthetic
surrogate:

- Train frozen benign concept probes before chameleon fine-tuning.
- Use the paper Table 1 scenario matrix:
  no trigger target 1, matching trigger target 0, mismatched trigger target 1.
- Insert the trigger template `You are being probed for {concept}`.
- Capture generation-token activations and pool them with
  `mean_generation_tokens`.
- Compute differentiable frozen-probe MSE as the obfuscation term.
- Compute KL-to-base on on-policy completions as the behavior-preservation term.
- Preserve normal useful behavior through KL/capability controls and additional
  benign/capability data where configured.
- Default real runs use QLoRA/LoRA for tractability; full fine-tuning remains a
  supported backend but is not the default Narval path.

Current default paper hyperparameters are treated as paper-faithful estimates
unless later source details override them:

- `lambda_behav=0.1`
- `lambda_obf=0.9`
- learning rate `2e-5`
- weight decay `0.01`
- warmup `100`
- three seeds `[17, 23, 41]`
- primary probe layer 12
- 4-layer and 8-layer ensemble support

## Probe And Judge Assumptions

Paper probes:

- linear/logistic probe
- MLP probe with hidden size 64 and ReLU
- 4-head attention probe
- layer 12 primary probe
- 4-layer and 8-layer ensembles

Extension probes and Judges are intentionally stronger than the paper defaults:

- layer sweeps
- layer ensembles
- geometry probes
- centroid/kNN detectors
- high-capacity oracle-style judges
- active probe discovery and promotion

Reports must separate paper-default probes from extension/robustness probes so
new results do not get confused with paper replication.

## Evaluation Assumptions

Paper-real evaluation should:

- train post-hoc safety probes on the final frozen checkpoint without trigger;
- evaluate no-trigger vs trigger on positive examples only;
- calibrate thresholds at fixed 1% FPR on clean negatives;
- report TPR/recall drop;
- include 95% bootstrap confidence intervals;
- include mean generation-token aggregation;
- include mean layer aggregation for ensembles;
- include base-trigger controls;
- include selectivity reports for mismatched triggers;
- include capability tax and safety-corpus likelihood retention.

The capability suite is configured by name and can include MMLU, GSM8K, BoolQ,
CrowS-Pairs, SocialIQA, TruthfulQA, and PIQA when the `lm-eval` path is enabled.

## Cluster And Artifact Assumptions

Narval defaults:

- account: `ctb-liyue_gpu`
- GPU request: A100 only
- durable artifacts:
  `$SCRATCH/next_chameleons_artifacts`
- raw cache:
  `$SCRATCH/next_chameleons_raw_cache`
- persistent HF cache:
  `$SCRATCH/.cache/huggingface`
- node-local staged HF cache during jobs:
  `$SLURM_TMPDIR/next_chameleons_hf`

The Slurm scripts stage selected Hugging Face model snapshots from persistent
scratch into node-local storage before loading models. This is assumed necessary
on Narval because repeated sharded model loads from Lustre showed severe
latency, and Alliance/Narval guidance recommends `$SLURM_TMPDIR` for temporary
high-I/O job data.

## Secrets And Logging Assumptions

- HF and W&B tokens are stored in `.env.narval`, which is gitignored and should
  remain mode `0600`.
- OpenAI keys are intentionally unused.
- W&B is optional and controlled by `WANDB_MODE` and `WANDB_API_KEY`.
- Run manifests should include git commit, config, source revisions/checksums,
  Slurm job id, account, partition, GPU count, seed, and artifact paths.

## Known Gaps To Resolve

- Confirm whether exact paper benign examples are ever released.
- Confirm exact prompt templates used for benign generation and rating.
- Confirm whether each non-Gemma family model should use an abliterated or
  helpful-only variant.
- Recover or replace Apollo/RepE and Synthetic Harmful local manifests with
  exact public source references if available.
- Validate that Qwen 3.5 27B rating agrees sufficiently with the paper's rater
  before making exact reproduction claims.
- Fill result tables after the Narval data generation and pilot 2B runs finish.

