# v4c on Gemma-4 12B (TamIA, 4 x H100 80 GB) — setup, estimates, smoke

Status (2026-10-24): code + configs + launch logistics ready; GPU smoke result below. **Not launched.**

## Support

| piece | status |
|---|---|
| checkpoint | `models--google--gemma-4-12b-it` (`Gemma4UnifiedForConditionalGeneration`; 48 decoder layers, hidden 3840, 5 sliding (window 1024) : 1 full attention, k = v on full layers, final-logit soft-cap 30, tied 262k embedding). The "unified" variant has NO vision / audio tower: only ~0.05 B params of patch / projection embedders; text = 11.9 B (1.0 B embedding + 10.9 B layers). |
| training env (q35t29: torch 2.9.1, transformers 5.14.1) | supported: `gemma4_unified` incl. the text-only `Gemma4UnifiedForCausalLM` |
| vLLM `vllm-latest` (0.21.0, transformers 5.9) | **not** supported (no `Gemma4Unified*`, cannot parse the config) |
| vLLM ≥ 0.23 | supported. Built `/scratch/s/sneheel/venvs/vllm-g4` (vLLM 0.25.0, torch 2.11, transformers 5.14.1) from the wheelhouse; modules `StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv` (vLLM 0.25 needs opencv). Served with `--language-model-only`. |
| vLLM in TRAINING | not used by the production layout anyway: data-parallel training refuses the colocated engine (`scripts/tica.py`: "data-parallel episodes: no colocated vLLM"); the 4B runs generate with HF too. vLLM matters for single-GPU evals (`TICA_VLLM=1`) and the base-generation job. |

## Code changes (Qwen behaviour unchanged; pinned by tests/test_tica_gemma4.py + the existing suite)

- `tica/policy.py`
  - `load_causal_lm`: for `model_type: gemma4_unified` builds the TEXT-ONLY `Gemma4UnifiedForCausalLM` from `text_config`
    and loads only `model.language_model.*` (renamed to `model.*`); vision / audio weights skipped; any missing text
    weight raises (the tied `lm_head` is the only allowed one). So LoRA `all-linear`, the probe hooks, the LM head and
    every read only ever see the text decoder. Other checkpoints: `AutoModelForCausalLM` as before.
  - chat formats: `ChatFormat` (`QWEN_CHATML` = the old `render_keep_reasoning`, default for every tokenizer without
    Gemma's template; `GEMMA4_TURNS`). Gemma keep-think render: `<bos>`, `<|turn>system\n…<turn|>\n`,
    `<|turn>user\n…<turn|>\n`, `<|turn>model\n<|channel>thought\n<channel|>CONTENT<turn|>\n` — the empty thought channel
    of the thinking-off generation prompt is kept on every assistant turn (as Qwen's empty `<think>` block), so a
    whole conversation contains each turn exactly as its generation prompt + content (single-pass training).
    `multi_view` takes its assistant head / end-of-turn marker (`<turn|>`, id 106 = an EOS id) from the format.
    Verified token-for-token against the snapshot's `chat_template.jinja` (thinking off and on).
  - `final_logit_softcapping`: the chunked LM terms (`model.lm_chunk`) call the head themselves and now apply the
    model's soft-cap (Gemma 30; Qwen None = unchanged). Without it the NLL / KL / top-k targets were wrong for Gemma.
- `tica/probe_zoo.py`: `scale_layer` / `scale_kind` / `set_zoo_depth`: the multi-layer kinds' fixed 32-layer sets
  are rescaled to `model.depth` (in place, idempotent, depth 32 = the originals).
- `tica/config.py`: `model.depth` (0 = unset = unchanged). Set → zoo rescaled, every layer the config names must be
  < depth (`_check_depth`); `scripts/tica.py` checks the loaded model has exactly `depth` layers.
- `scripts/tica.py`: `probes.dir: ""` allowed with the answer format (no Level-1 repeat bank; the answer format fits
  its base probes on the model itself at step 0, as on Qwen).
- `tica/vllm_gen.py`: `VLLM_FAMILY` (per `model_type`): Gemma served with `--language-model-only`; merged weights
  carry the per-layer `layer_scalar` buffers. Merged-weight and adapter keys map to `model.language_model.*`
  (= the checkpoint / vLLM `hf_to_vllm_mapper` naming; checked against the snapshot's header).
- `scripts/tica_probe_panel.py`: the never-shown panel's 32-layer DEFAULTS (layers 4..30, read / feat layers,
  `maxpool@24`) are rescaled when the config sets `model.depth` (panel 6,12,18,24,30,33,36,39,42,45).
- `scripts/tica_mathrl.py gens`: base generations only (Gemma's own `cap_gen_pool`).
- No GDN / fla / causal-conv code runs on Gemma (`configure_linear_attention` patches 0 layers; `fla_grad: false`).
  Answer tags / status lines are plain text (model-agnostic).

## Configs

`configs/tica/final_v4c_gemma4_12b_tamia.yaml` (1-bit verdict) and `final_v4c_score_gemma4_12b_tamia.yaml` (score)
+ builtin mirrors; mathrl_gemma4_12b.yaml (base generations). Identical to the production v4c 4B arm except:

| | Qwen3.5-4B (32 layers) | Gemma-4 12B (48 layers), `round(L * 48 / 32)` |
|---|---|---|
| training depths `kind_layers` | 8 12 16 22 26 | 12 18 24 33 39 |
| primary base probe | 12 | 18 (`probes.layer`) |
| held-out base probes | 20 24 | 30 36 |
| held-out kinds | linear@20/24/28/30, maxpool@24 | linear@30/36/42/45, maxpool@36 |
| ens4 / ensdeep4 (trained) | 8-14 / 16 18 22 26 | 12 15 18 21 / 24 27 33 39 |
| ens8, spread5 (held-out) | 4..18 / 4 10 16 22 28 | 6..27 step 3 / 6 15 24 33 42 |
| bailey_ens_tok / _max | all 32 / 4..28 | all 48 / 6..42 |
| never-shown panel | 4..30 | 6..45 |
| cap_gen_pool | Qwen base solutions | Gemma base solutions (`prep`) |
| steps | 1200 | 900 (LR schedule unchanged: cooldown = last 150) |

Capability tripwires unchanged (KL 0.3 to base on no-brief turns, anchor 0.5 on the base model's own greedy answers,
cap_gen on GSM8K with the hide term, held-out race every 50 steps, quality queue).

## Memory / throughput (4 x H100, DP4 = one process per GPU)

Static per process (measured, smoke 640267): text weights bf16 **24.1 GB** allocated after load; LoRA r128 all-linear
= **524.6 M** trainable params (40 sliding layers x 10.9 M + 8 full x 11.0 M), fp32: 2.1 GB weights + 2.1 GB grads +
**3.9 GB** AdamW moments -> ~32 GB before activations.

Measured peaks:

| what | peak alloc | notes |
|---|---|---|
| sanity, reads of two real 24-turn episodes (5.1k / 6.3k tokens) | 38.6 GB | batch of per-turn views + forks, batch 6 |
| sanity, one optimizer step on a 5.1k-token 24-turn episode (all 7 depths + LM terms, chunked, soft-capped) | 32.9 GB | fwd 0.8 s, bwd 3.2 s |
| sanity, vLLM awake beside the training process (mem fraction 0.45) | 68 GB device | training side 29 GB alloc |
| **real trainer, DP4, steps 1-2** (production config, 1 episode / rank) | **51.5 GB** (58 reserved) | |
| **real trainer, DP4, step 3** | **64.0 GB** (71.6 reserved; nvidia-smi 76 GB) | step with the largest cap_gen train (32.6 s) |

So **DP4 with one 12B copy per GPU fits**, but with only ~8-10 GB headroom at step 3 on 1 episode per rank. The 4B
process grew 16 -> 33 GB between step 1 and step 500 (bank / refit buffers, 4 episodes per rank), so headroom is THE
risk: do the 20-step check below before the long chain. Knobs, in order (all config / `--set`, no code): `model.batch_size`
16 -> 8 (no-grad refit reads), `model.lm_chunk` 1024 -> 512, `adaptive.cap_gen_samples` 4 -> 2 (cap_gen train
batch), `adaptive.refit_buf_cap` 512 -> 256. Not needed / not implemented: FSDP, fewer turns. The 4B's 2-arms-per-GPU
layout is impossible for 12B (2 x 64 GB): **one arm per node**.

Throughput (smoke 640273, DP4, 1 episode per rank, EMPTY Gemma panel cache):

| step | sec | rollout | train | of which cold-cache fits | cap_gen + train | anchor |
|---|---|---|---|---|---|---|
| 1 | 706 | 81 | 624 | yes (+ 114 s anchor-cache generation) | 11 + 14 | 114 |
| 2 | 411 | 44 | 367 | yes | 8 + 12 | 2 |
| 3 | 456 | 31 | 425 | yes | 12 + 33 | 3 |

`sec_train` is dominated by first-use panel fits: with `match_anchor: panel`, `_targets -> _anchor_names` fits EVERY
panel kind (27 = 7 archs x 5 depths, pooled + ensembles) for each new (concept, condition) through the panel cache,
inside the train phase. These are what `prep` precomputes. Steady-state estimate from the clean parts: rollout
31-44 s per episode-rank (4B, solo node, early: 12-22 s per episode-rank -> ~2-2.5x), training ~2x the 4B's 40-55 s,
cap_gen ~10 s per rank (1 fork per rank, as production). Production (4 episodes per rank):

- early (steps 1-100): ~4 x 35 + 80-110 + 25 ~ **250-300 s/step**
- late (bank grown; the 4B rollout grew ~2x by step 500): **~450-650 s/step**
- mean over 900 steps ~ **450-550 s/step**

| steps | at 400 s/step | at 500 s/step | at 650 s/step |
|---|---|---|---|
| 600 | 67 h, 267 GPU-h | 83 h, 333 GPU-h | 108 h, 433 GPU-h |
| 900 | 100 h, 400 GPU-h | 125 h, 500 GPU-h | 163 h, 650 GPU-h |

Per arm (one 4-GPU node each, 24 h chunks: 900 steps ~ 5-7 chunks); both arms = 2 nodes in parallel and double the
GPU-hours (~800-1300 GPU-h for 900 steps). Plus held-out races (18 per arm at eval_every 50) on drain nodes, and prep.
HF generation on Gemma is slow (76 tok/s at batch 16, ~210 ms per decode step: 48 layers, launch-bound); cap_gen
(1 fork x 4 samples x 400 tokens per rank) costs ~10 s at the observed lengths but up to ~80 s if samples run long.

## Base probes / caches (keyed by model: everything is rebuilt for Gemma)

- answer-format base probes (`answer_probe_cache`, key includes the model dir name): fit by rank 0 at step 0 for the
  20 train concepts, ~76 s each on Gemma (**~25 min** at the start of chunk 1). Smoke: sports / politics / astronomy /
  war at L18 / L30 / L36 all AUROC >= 0.99998, TPR 1.0 (production gates pass).
- turn-0 panel cache (`panel_probe_cache`, key includes model + prompt ids): `scripts/slurm/tamia_panel_precompute.sh
  final_v4c_gemma4_12b_tamia` (8 processes, 2 per GPU: 2 x 24 GB weights + reads fit in 80 GB). 4B: ~1840 keys in
  65 min. Gemma: ~2-2.5x the read cost per key (refit reads 74-97 s vs 4B 68-132 s per 4 episodes) -> **~2.5-3.5 h**
  on one node; both arms share every key. Without it the first ~50-100 steps pay the fits (step 1-3 above).
- `cap_gen_pool`: `scripts/slurm/tamia_gemma12b_gens.sh` (Gemma's own greedy GSM8K solutions; vLLM ~15 min).
- anchor cache: per run, generated at step 0 (256 greedy base answers, 114 s).

## Smoke results (2026-10-24; aggregates only)

Job 640267 (sanity, GPU 0, `scripts/tica_gemma_smoke.py`; its parallel DP3 trainer part died on a smoke-only setting:
2 concepts cannot feed `switch_max: 2` switches -> fixed to 4 concepts) and job 640273 (real trainer, DP4, 3 steps):

- load: `Gemma4UnifiedForCausalLM`, 48 layers (== model.depth), 0 vision / audio modules, chat format gemma4,
  soft-cap 30, EOS ids {1, 50, 106}, 24 s from node-local disk.
- render: generation-prompt text AND ids identical to the snapshot's chat template; `<|turn>` 105, `<turn|>` 106,
  `<|channel>` 100, `<channel|>` 101 single tokens; BOS first.
- reads on real 24-turn episodes (5.1k / 6.3k tokens; per-layer mean act norms 70-225): one forward over the whole
  conversation vs per-turn views: cos >= 0.9999997, NLL max |diff| 1e-6; shared-prefix fork reads (KV cache across
  the 1024-token sliding window) cos >= 0.99992 (rel. err 0.6%, bf16); incremental rollout reads cos >= 0.99995 (8
  cache hits); truncated reads cos >= 0.9999997.
- base probes: see above (gates pass).
- one optimizer step at production length: finite loss, grad norm 1.76, peak 32.9 GB.
- vLLM 0.25 (vllm-g4, `--language-model-only` confirmed in the server args, Gemma4UnifiedForConditionalGeneration):
  startup 225 s; merged-weight sync after the optimizer step + 16 x 128 greedy tokens in 28 s; greedy agreement with
  HF on the first 16 tokens 0.94 (policy, after sync) / 0.88 (base), mean common prefix 104 / 128 and 56 / 64 tokens
  (bf16 kernel differences; the policy is not worse than the base, so the sync is right). The reload logs "Failed to
  load weights" for the vision embedder / rotary / tied LM head, which keep their (unchanged) tensors.
- real trainer DP4: 3 steps, losses finite, grad norm 24 -> 3.9 -> 0.73 (warm-up), cap_gen acc 0.92-1.0, anchor KL
  ~3e-4, no DP drift error; peaks / times in the tables above.

## Before the long run (recommended)

1. `bash scripts/tamia_gemma12b.sh prep` (gens + panel cache).
2. A 20-step production-shape check (4 episodes per rank, warm cache) to confirm memory headroom and s/step:
   `SMOKE_SANITY=0 SMOKE_EPR=4 SMOKE_NC=0 SMOKE_STEPS=20 sbatch --account=aip-rrabba --exclude=tg11105 --time=3:30:00 -J g4-check scripts/slurm/tamia_gemma12b_smoke.sh`
   (DP4, 16 episodes / step, all 20 concepts, no checkpoints); watch `max_mem_gb` (< ~70) and `sec` in
   `runs/smoke_gemma4/<job>/train/adaptive/train_log.jsonl`.

## Launch (not run)

```bash
bash scripts/tamia_gemma12b.sh check
bash scripts/tamia_gemma12b.sh prep            # gens + panel precompute (one node, ~3-4 h)
bash scripts/tamia_gemma12b.sh check           # base_gens.json present
bash scripts/tamia_gemma12b.sh launch 8        # 8 x 24 h chunks per arm (resume chain), one node per arm
bash scripts/tamia_gemma12b.sh status
```

Single arm: `ARMS=final_v4c_gemma4_12b_tamia bash scripts/tamia_gemma12b.sh launch 8`. The node runner is the
production `scripts/slurm/tamia_node.sh <cfg>@0,1,2,3` (DP4); resumes via `tica_resume.sh`; held-out races queue
to `runs/g4v4c_val_queue.txt` and run with `scripts/slurm/tica_val_drain.sh` (12B evals: `PER_GPU=2` ~ 2 x 30 GB).

## Open risks

- MEMORY HEADROOM: 64 GB alloc / 72 GB reserved at step 3 with 1 episode per rank; the 4B grew ~2x over 500 steps.
  An OOM late in a chunk loses up to 20 steps (save_every 20) and the chain resumes, but repeated OOMs need the knobs
  in the memory section (batch_size 8, lm_chunk 512, cap_gen_samples 2). Run the 20-step check first.
- vLLM is NOT used by DP training (refused by scripts/tica.py, as on the 4B): rollouts use forced answers, cap_gen /
  anchors use HF generate, which is slow on Gemma (~210 ms / decode step). vLLM (vllm-g4) serves single-GPU evals
  (`TICA_VLLM=1` + `TICA_VLLM_PY=/scratch/s/sneheel/venvs/vllm-g4/bin/python`, `TICA_VLLM_MODULES="StdEnv/2023 gcc
  python/3.11 arrow/24.0.0 cuda/12.6 opencv"`) and the base-generation job.
- Step time is extrapolated from a 3-step smoke with 1 episode per rank and empty panel cache; the real per-step time
  is only known after ~20 production steps (watch `sec`, `max_mem_gb`, `dp/sec_rollout_*`).
- Recipe hyper-parameters (LR 1e-4..3e-4, alpha 64 at r128, hide / KL weights, refit gates) were tuned on the 4B;
  the 12B's activation scale differs (see smoke act norms). Base-probe gates (AUROC 0.95, TPR) may fail for some
  of the 20 concepts at the rescaled depths (the smoke passed 4 / 4; rank 0 raises at step 0 if a gate fails).
- Depth mapping is proportional; Gemma's 5:1 sliding / full pattern puts full-attention layers at 5, 11, 17, 23, 29,
  35, 41, 47 — training depths 12 / 18 / 24 / 33 / 39 sit just after full layers 11 / 17 / 23 and inside sliding
  blocks; not tuned.
- 8-12 B is the regime where obfuscation results may differ; treat the 4B schedule (900 steps) as a starting budget.
- vLLM 0.25 merged-weight reload logs (not fails) for weights not in the reload file; the smoke checks greedy
  agreement with HF after a sync.
- `chat_template` thinking: Gemma's thinking is off (empty thought channel), as Qwen's `<think></think>`.
