# Real Runs Handbook

The single end-to-end path for a real GPU replication: data → train → eval →
artifacts. Synthetic smoke paths are covered in the
[Agent Guide](agent-guide.md); this page is the GPU workflow.

## Pipeline shape

The implementation lives in the `next_chameleons.real` package:

| Module | Responsibility |
| --- | --- |
| `real/resolve.py` | Resolve an experiment (model/train/dataset inlining), readiness gates |
| `real/data.py` | Load sources, fallbacks, fit/eval splits, materialization, trigger candidates |
| `real/train.py` | `run_real_train` orchestration |
| `real/eval.py` | `run_real_eval`: post-hoc probe fit, fixed-FPR calibration, triggered eval |
| `real/adaptive.py` | `run_real_adaptive` multi-round loop |
| `real/common.py` | Output-dir resolution, raw-cache root, run manifest |

`real_pipeline.py` is a compatibility façade re-exporting these. The CLI and
`TrainingRegimeRunner` are the stable entry points.

## Stage 0 — Golden-path evaluator validation (do this first)

Validate the evaluator against the paper's released organism + probes before
spending GPU hours on training.

```bash
./scripts/fetch_golden_artifacts.sh
uv run --extra train next-chameleons real-eval paper_golden_serteal_gemma2_9b \
  --checkpoint-dir "$SCRATCH/next_chameleons_golden/serteal_neural_chameleon_gemma_2_9b" \
  --external-probe-dir "$SCRATCH/next_chameleons_golden/probes" \
  --output-dir "$SCRATCH/next_chameleons_artifacts/paper_golden_serteal_gemma2_9b"
```

`--external-probe-dir` skips post-hoc probe fitting and loads the released
`deception_weights.pt` / `harmful_weights.pt`. **Gate:** the triggered-positive
TPR must collapse at the frozen 1% FPR threshold on at least one deception and
one harmful source (the config's `evaluation.success_gate`). If it does not, the
evaluator is the bug — fix it before training.

## Stage 1 — Materialize data

The default paper train source is `benign_preset_official` — the official
4,698-record preset with synonym triggers, the 13-split supervision matrix, and
per-sample behavior-loss flags. `fetch_golden_artifacts.sh` stages it under
`$SCRATCH/next_chameleons_raw_cache/paper_benign_preset/`.

Safety eval sources:

```bash
uv run --extra train next-chameleons paper-materialize-safety-data paper_gemma2_9b_real \
  --raw-cache-root "$SCRATCH/next_chameleons_raw_cache"
```

The clean-room regeneration path (`benign_synthetic_4697`) is still available
via `paper-materialize-data --generate`; label results accordingly.

## Stage 2 — Plan and validate

```bash
uv run next-chameleons real-run-plan paper_gemma2_9b_real
uv run next-chameleons paper-readiness-check paper_gemma2_9b_real
uv run next-chameleons config-check paper_gemma2_9b_real   # schema validation
```

Readiness reads the protocol from config (behavior-loss mode, probe layer,
train source), so versioned variants pass without editing the gate.

## Stage 3 — Train

```bash
uv run --extra train next-chameleons real-train paper_gemma2_9b_real --seed 17
```

What runs (paper-faithful defaults):

- Frozen benign concept probes fit **per response token** (`paper_per_token_v2`).
- Obfuscation loss = MSE of per-token-sigmoid-mean probe score vs target.
- Behavior loss = teacher-forced CE on response tokens, **only** on samples
  whose preset record carries `use_behavior_loss` (topical samples get pure
  obfuscation loss).
- Probe layer + `hidden_states_offset` from the model config (Gemma 12, Qwen 9;
  offset 1 so "layer 12" = `hidden_states[13]`).

Full fine-tune arm (paper-shipped setup, Adam8bit/lr 8e-6/1 epoch):
`real-train paper_gemma2_9b_real_fullft`.

Artifacts under the run's checkpoint dir: the LoRA adapter (or full weights),
`frozen_benign_probe_bank.pt` (+ redacted manifest), and
`seen_concept_evasion_report.json`.

## Stage 4 — Evaluate

```bash
uv run --extra train next-chameleons real-eval paper_gemma2_9b_real \
  --checkpoint-dir "<run>/checkpoints/paper_gemma2_9b_real" \
  --output-dir "<run>"
```

Eval protocol: stratified 75/25 fit/eval split per source; post-hoc safety
probes fit per-token on the fine-tuned model at the same layer; fixed 1% FPR
threshold calibrated on fit-side clean negatives; trigger applied to positive
examples only; the paper's per-domain trigger words are swept and the strongest
TPR drop per probe is reported (`fixed_fpr_best_trigger_by_probe`).

## Stage 5 — Controls and claim

Before claiming replication: the base-model trigger control, the selectivity
heatmap, the capability tax, and three-seed CIs. Only the final-minus-base TPR
drop under the held-out protocol counts as learned obfuscation. Record results
with their protocol versions in [REPLICATION.md](replication.md).

## Sweeps

```bash
uv run next-chameleons sweep-plan paper_eval_trigger_sweep    # GPU cases -> plan
uv run next-chameleons sweep-run all_experiment_smokes        # CPU smokes
```
