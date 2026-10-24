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

## Memory / throughput estimate (4 x H100, DP4 = one process per GPU)

Per process (bf16 weights, fp32 LoRA as PEFT keeps it):

| item | GB |
|---|---|
| text weights (11.9 B, bf16) | 23.8 |
| LoRA r128 all-linear (524 M params; 40 sliding layers 10.9 M + 8 full 11.0 M) fp32 | 2.1 |
| LoRA grads fp32 + AdamW moments | 2.1 + 4.2 |
| static total | ~32 |
| activations, 24-turn episode at micro_batch 1 (grad ckpt; ~7-9k tokens) + chunked LM terms | SMOKE |
| no-grad refit reads (batch 16) / cap_gen KV cache (sliding layers keep 1024 tokens) | transient, SMOKE |

The 4B production process peaks at 37.6 GB (2 processes per GPU). DP4 with one 12B process per GPU fits if the
activation peak stays < ~40 GB: SMOKE. Fallbacks if not: `model.batch_size` 16 -> 8 (refit reads), `lm_chunk` 512,
then FSDP / fewer turns (not implemented; not needed per the smoke).

Throughput: the rollout (no generation in the answer format: forced answers, probe reads + refit fits) dominates
(4B: rollout 294 of 346 s p50). Reads / training passes cost ~3.3x the 4B's FLOPs per token (10.9 B vs ~3.3 B
non-embedding params), but each 12B process has a whole GPU while the 4B production shares every GPU between both
arms (2 processes per GPU, ~61% util each). Fits scale with d (3840 vs 2560) and are partly host-bound. Estimate
1.5-2.5x the 4B per-step time: **~500-900 s/step** (4B production: p10/p50/p90 275/346/484 s). SMOKE calibration below.

| steps | s/step 500 | s/step 700 | s/step 900 |
|---|---|---|---|
| 600 | 83 h, 333 GPU-h | 117 h, 467 GPU-h | 150 h, 600 GPU-h |
| 900 | 125 h, 500 GPU-h | 175 h, 700 GPU-h | 225 h, 900 GPU-h |

Per arm (one node each); two arms = 2 nodes in parallel, double the GPU-hours. Plus held-out races (eval_every 50:
18 races per arm, ~1 h each on a drain node) and the prep below.

## Base probes / caches (keyed by model: everything is rebuilt for Gemma)

- answer-format base probes (`answer_probe_cache`, key includes the model dir name): fit by rank 0 at step 0 for the
  20 train concepts (~20 x the smoke's per-concept time).
- turn-0 panel cache (`panel_probe_cache`, key includes model + prompt ids): `scripts/slurm/tamia_panel_precompute.sh
  final_v4c_gemma4_12b_tamia` (8 processes, 2 per GPU). 4B: ~1840 keys in 65 min. Gemma: ~2-3x the read cost and
  1.5x the fit dimension: **~2.5-3.5 h** on one node (both arms share every key). Without it the first ~100 steps fit
  panel probes on the fly (slow early steps, same result).
- `cap_gen_pool`: `scripts/slurm/tamia_gemma12b_gens.sh` (vLLM, ~15 min; HF ~1 h).
- anchor cache: per run, generated at step 0 (256 greedy base answers).

## Smoke (job SMOKE_JOB)

SMOKE

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

- Step time is extrapolated from a 3-step smoke with 1 episode per rank and empty panel cache; the real per-step time
  is only known after ~20 production steps (watch `sec`, `max_mem_gb`, `dp/sec_rollout_*`).
- Recipe hyper-parameters (LR 1e-4..3e-4, alpha 64 at r128, hide / KL weights, refit gates) were tuned on the 4B;
  the 12B's activation scale differs (see smoke act norms). Base-probe gates (AUROC 0.95, TPR) may fail for some
  concepts at the rescaled depths (smoke checks one concept; rank 0 raises at step 0 if a gate fails).
- Depth mapping is proportional; Gemma's 5:1 sliding / full pattern puts full-attention layers at 5, 11, 17, 23, 29,
  35, 41, 47 — training depths 12 / 18 / 24 / 33 / 39 sit just after full layers 11 / 17 / 23 and inside sliding
  blocks; not tuned.
- 8-12 B is the regime where obfuscation results may differ; treat the 4B schedule (900 steps) as a starting budget.
- vLLM 0.25 merged-weight reload logs (not fails) for weights not in the reload file; the smoke checks greedy
  agreement with HF after a sync.
- `chat_template` thinking: Gemma's thinking is off (empty thought channel), as Qwen's `<think></think>`.
