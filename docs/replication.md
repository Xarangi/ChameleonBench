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

- Apollo/RepE deception is materialized from the Apollo deception-probe HF
  collection components `scale-safety-research/roleplaying` and
  `scale-safety-research/insider_trading`. This is better than the previous
  blank local manifest, but should still be recorded as an Apollo HF collection
  materialization unless the paper gives a more exact split recipe.
- Synthetic Harmful is represented as a balanced JBB surrogate until the exact
  generated paper split is recovered.
- `JailbreakBench/JBB-Behaviors` is an approximate harmful-behavior fallback,
  not an exact paper source.
- The live operational log is maintained in repository-root `REPLICATION.md`,
  including concrete checkpoints, Slurm jobs, observed metrics, and source
  limitations.

Text-evaluation fallback:

- `AISC-Linear-Probe-Gen/obfuscated_activations` contains Circuit
  Breakers/Obfuscated Activations artifacts, including activation `.pt` files.
  The current text-generation evaluation path cannot directly read those as
  prompt/completion examples.
- For real text-probe evaluation, `circuit_breakers_harmful` records the paper
  source but falls back to `jailbreakbench_behaviors` as an approximate harmful
  text source.
- The text path marks this source as `fallback_only` so jobs do not waste time
  trying to parse the public activation artifact as a text dataset.
- Reports must preserve both facts: the paper source being covered and the
  fallback text source used by this implementation path.

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
- Compute KL-to-base on base-model on-policy continuations as the
  behavior-preservation term. The current trainer builds a cached reference set
  by sampling continuations from the frozen base model for each Table-1
  scenario prefix, then computes token KL over generated tokens only.
- If an example has no real prompt/prefix, the trainer falls back to the
  scenario text for KL and records `behavior_reference_fallback_fraction` in
  step metrics. A high fallback fraction means the data path is not yet fully
  paper-faithful even if the loss code is.
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
- behavior KL mode `base_on_policy_completions`
- behavior reference sampling: `max_new_tokens=96`, `temperature=0.7`,
  `top_p=0.95`
- three seeds `[17, 23, 41]`
- primary probe layer 12
- 4-layer and 8-layer ensemble support

## Runtime Technology Assumptions

The real-run stack is Hugging Face + PyTorch:

- Models are loaded with `transformers.AutoModelForCausalLM` and
  `AutoTokenizer`.
- LoRA/QLoRA adapters are built with PEFT.
- QLoRA uses BitsAndBytes 4-bit loading through `BitsAndBytesConfig`.
- Activations are captured with forward hooks through `HFActivationExtractor`.
- Frozen benign concept probes are trained before chameleon fine-tuning and
  then used as differentiable fixed heads during the obfuscation loss.
- The behavior term compares the trainable model to a frozen reference model
  with KL on the same on-policy inputs/completions.
- Evaluation trains post-hoc probes on the final checkpoint and aggregates
  generation-token activations.

Quantization is a default config choice, not a theoretical requirement:

- `configs/train/qlora.yaml` is the default Narval path because it makes 2B,
  7B, 8B, and 9B jobs fit reliably while also loading a frozen reference model
  for KL.
- On A100 nodes, this is conservative. Full-precision LoRA or full fine-tuning
  can be selected with `train_backend: lora` or `train_backend: full_finetune`
  when the allocation has enough memory for the trainable model, reference
  model, activations, optimizer state, and probe heads.
- QLoRA is especially useful for fast pilot runs and queue-friendly jobs; it
  should not be described as the only valid replication backend.

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
scratch into node-local storage before loading models. Paper jobs also stage the
text-evaluation dataset snapshots used by the current run. This is assumed
necessary on Narval because repeated sharded model and dataset loads from Lustre
showed avoidable latency, and Alliance/Narval guidance recommends
`$SLURM_TMPDIR` for temporary high-I/O job data.

## Secrets And Logging Assumptions

- HF and W&B tokens are stored in `.env.narval`, which is gitignored and should
  remain mode `0600`.
- OpenAI keys are intentionally unused.
- W&B is optional and controlled by `WANDB_MODE` and `WANDB_API_KEY`.
- Run manifests should include git commit, config, source revisions/checksums,
  Slurm job id, account, partition, GPU count, seed, and artifact paths.

## Current Narval Status Summary

- Corrected smoke/readiness jobs `62358077` and `62358078` completed.
- Minimal 2B job `62358079` completed successfully. It moved in the intended
  direction on seen benign probes but did not yet show held-out Dolus/Circuit
  evasion:
  - seen matching-trigger target success: `0.25`;
  - seen matching-trigger TPR drop: `0.25`;
  - Dolus held-out linear TPR drop: `0.0`;
  - Circuit/JBB fallback held-out linear TPR drop: `0.0`.
- Full 2B job `62358080` trained and saved its checkpoint/probes/reports, but
  failed during the eval stage when `uv run` attempted to fetch dependencies
  from PyPI on the compute node.
- The 100-step full 2B checkpoint is useful diagnostically but is not a paper
  success:
  - seen matching-trigger target success: about `0.045`;
  - seen matching-trigger TPR drop: about `0.045`;
  - mismatched-trigger selectivity: about `0.972`.
- Corrective action: Slurm wrappers now set `UV_NO_SYNC=1` and `UV_OFFLINE=1`,
  and full 2B reference configs now use `1,000` optimizer steps.
- The canceled dependent jobs were `62358081`, `62358084`, `62358085`,
  `62358086`, and `62358087`.
- First 2026-06-05 resume attempt `62404782` failed because dependency sync had
  installed CUDA-13 Torch wheels into the local venv. Dependencies now pin
  `torch>=2.3,<2.7` and `transformers>=4.44,<5`; the local venv is back on
  `torch 2.6.0+cu124`.
- Second resume attempt `62404920` failed because local PEFT adapter
  checkpoints saved Gemma tokenizer `extra_special_tokens` as a list. Model
  loading now detects adapter directories, loads tokenizer/base weights from
  `base_model_name_or_path`, and then attaches the adapter.
- Minimal 2B eval retry `62405018` completed successfully after the adapter
  loading and dependency fixes.
- A later inspection found decoder-only right-padding warnings in the 2B/9B/
  Llama/Qwen jobs. Those jobs were canceled or treated as invalid for result
  interpretation, and the tokenizer/model helpers now set left padding while
  activation pooling and generation-token masks account for left-padded input.
- Current left-padding replication graph, submitted on 2026-06-05:
  - `62418529`: Gemma-2-2B seed 17, 1,000-step run, running on one A100;
  - `62418531`: Gemma-2-9B seed 17, primary run, running on one A100;
  - `62418532` and `62418533`: Gemma-2-9B seeds 23/41, dependency-gated on
    seed 17;
  - `62418534`: Llama-3.1-8B seed 17 validation run, completed;
  - `62418535`: Qwen-2.5-7B seed 17 validation run, completed.
- The current graph is deliberately not fully serialized. The 2B validation,
  9B seed 17, and seed-17 family checks can run in parallel; only 9B stability
  seeds wait for the primary 9B seed-17 job.
- `62418529` has reached optimizer step `1000/1000`, written
  `real_train_report.json`, and started post-hoc held-out evaluation on Dolus.
- `62418531` has passed model/reference loading, offline W&B initialization,
  frozen benign Probe fitting, and is building base on-policy behavior
  references for the `14,091` Table-1 scenarios.
- Completed validation-scale family findings:
  - Llama-3.1-8B completed train plus post-hoc eval. The paper-linear probe
    showed large held-out drops on Dolus deception and Circuit/JBB fallback,
    weak clean TPR on Apollo/RepE, and a negative result on the Synthetic
    Harmful/JBB surrogate. This is useful diagnostic signal, not yet a
    full-budget family replication claim.
  - Qwen-2.5-7B completed train plus post-hoc eval but showed no paper-linear
    TPR drop on the current 100-step validation configuration. Treat this as a
    model/configuration diagnostic requiring longer training or model-specific
    tuning, not as a negative paper replication conclusion.
- Full 2B seed-17 training produced a reusable checkpoint, but that historical
  run did not save the exact frozen benign Probe bank.
- Corrected 2B evaluation now completes on Dolus, Apollo HF collection
  materialization, Circuit Breakers/JBB fallback, and Synthetic Harmful/JBB
  surrogate.
- A second corrected 2B evaluation writes fitted post-hoc Judge/Probe artifacts
  under the eval artifact root.
- The first 9B seed-17 job was canceled rather than treated as a result because
  `max_steps` was not being bridged from `real_run` into the trainer. The
  corrected configs now use 1,000 steps for the 9B and 2B paper runs, while
  Llama and Qwen stay at 100-step family-validation scale until the primary
  path is stable.
- Two intermediate 9B seed-17 attempts were canceled after W&B online
  initialization hit a compute-node network retry loop. Slurm now forces
  `WANDB_MODE=offline` unless `NEXT_CHAMELEONS_WANDB_ONLINE=1` is explicitly
  set, and the Python trainer passes that mode directly to `wandb.init`.
- Long full-budget family replication should follow after the 2B and primary 9B
  left-padding runs finish cleanly.

## Known Gaps To Resolve

- Confirm whether exact paper benign examples are ever released.
- Confirm exact prompt templates used for benign generation and rating.
- Confirm whether each non-Gemma family model should use an abliterated or
  helpful-only variant.
- Recover or replace Apollo/RepE and Synthetic Harmful local manifests with
  exact public source references if available.
- Validate that Qwen 3.5 27B rating agrees sufficiently with the paper's rater
  before making exact reproduction claims.
- Add an activation-artifact adapter for Circuit Breakers if we want to evaluate
  the public `.pt` source directly instead of using a JBB text fallback.
- Fill result tables after the corrected 9B validation and model-family runs
  finish.
