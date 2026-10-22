# TamIA v4b runs (2026-10-21): state, layout, how to launch and monitor

**LAUNCHED 10-21 ~14:15, INTERLEAVED (REPLICATION 10-21as):** both arms on ONE node per chunk (data-parallel over the same
4 GPUs, 2 processes per GPU, model.lm_chunk 1024): chunks 635093 -> 635095 -> 635097 -> 635099, drains 635094 / 635096 /
635098 / 635100. Monitor: `bash scripts/tamia_v4b.sh status`. (The solo chain 635081-635092 was cancelled before it ran.)

**Current (10-21 afternoon; supersedes the layout / arm names below where they differ, REPLICATION 10-21aq):** arms
`final_v4b_tamia_4b` (1-bit) and `final_v4b_score_tamia_4b` (score) = the v4b recipe + LoRA r128 / alpha 256 + 16
episodes per step + dp_drift_fatal + cap_gen_shard + cap_gen_merged + first_probe_panel_cache. Layout: ONE ARM PER
NODE, data-parallel over its 4 GPUs (`LAYOUT=solo`, the default): two arms per node out-of-memory at r128 / 24 turns
(layout smokes L2 / L4). Before launch: `sbatch scripts/slurm/tamia_panel_precompute.sh final_v4b_tamia_4b` (fills
the panel-probe cache, ~1840 probes, shared by both arms). Launch: `bash scripts/tamia_v4b.sh launch 4`. Each chunk
is followed by a drain job: queued held-out races + in-run quality checks (newest checkpoint per arm:
free-generation format / language checks + GSM8K / MMLU clean and under the brief; base reference once).

THE TamIA RECIPE (docs/tica-v4-tamia.md, REPLICATION 10-21ak) on TamIA, 4 x H100 80 GB nodes allocated whole.
Two arms, identical except the feedback channel:

| arm | config | feedback | shared settings |
|---|---|---|---|
| "normal" (1-bit) | `final_v4b_4b` | verdict (caught / not caught) | logclip (log hide loss, weight 0.3, hide grad clip 0.5), standalone stay (weight 1.0), turn-0 base panel from step 0, episode + cap_gen KL 0.3, cap_gen_samples 4, in-context refit gate, dp_sync_init, split v5 + elicit requests, 24 turns + turn budget, LoRA r32, retry ladder, LR 3e-4 -> 1e-4 + cooldown, 1200 steps |
| score | `final_v4b_score_4b` | 0-100 probe score vs threshold | same |

Optional third arm (not launched by default): the MSE baseline (`--set train.objective.hide_loss=mse --set
train.objective.hide_weight=1.0 --set train.hide_grad_clip=0 --set rl.adaptive.stay_standalone=false`).

## Status (2026-10-21, end of setup session)

| item | state |
|---|---|
| code | `tica-v3` pulled to 1497c36; TamIA work on the LOCAL branch `tamia` (bottom commit = path rewrite, never push; the commits above it use the TamIA paths too, so cherry-picking them to `tica-v3` needs the reverse path rewrite) |
| earlier uncommitted episode-parallel implementation (other agent) | superseded by upstream `tica/dp.py`; saved on branch `backup/local-tamia-parallel` (+ `git stash list`) |
| model | Qwen3.5-4B @851bf6e present (`/scratch/s/sneheel/hf` -> `hf-cache`); Qwen3.5-27B (unsloth bf16, FP8) present |
| venv | `/scratch/s/sneheel/venvs/q35t29{,.tar}`: torch 2.9.1, triton 3.5.0, transformers 5.14.1, peft 0.19.1, fla 0.4.1, built OFFLINE on a compute node (`scripts/env/q35t29_build_offline.sh`; no pyarrow / datasets / trl: the wheelhouse pyarrow is a dummy for the arrow module, training never imports them) |
| DATA | **MISSING: blocks every GPU job** (next section) |
| runs | none launched |

## Data: the blocker and the two ways out

`bash scripts/tamia_v4b.sh check` lists the inputs. None of `final_v5_clean.json`, `final_v5_elicit_inputs.json`,
`concept_labels_v5data_27b.json`, Dolly, `gsm8k_train.json`, `base_gens.json`, the `l1_base` probe dir exist on TamIA.
They live on Narval (never in git: raw text). TamIA has no SSH key for Narval (`Permission denied (publickey)`),
compute nodes have no internet, and the login node is not to be used for work.

**A. Sync from Narval (recommended: byte-identical to the data every Narval result used).** One-time, by the user:

```bash
# on TamIA login: make a key and add the .pub at https://ccdb.alliancecan.ca/ssh_authorized_keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""      # (or use an existing key / agent forwarding)
cat ~/.ssh/id_ed25519.pub
# after CCDB propagates (~minutes), from the repo on the tamia branch:
bash scripts/tica_sync_v4.sh narval.alliancecan.ca     # ~200 MB; the 7 GB model is skipped (already here)
bash scripts/tamia_v4b.sh check
```

Alternative without a TamIA key: from Narval (or a laptop with both), push the same files:
`rsync -a /scratch/sneheel/next_chameleons_raw_cache/{tica/final_v5_*,tica/concept_labels_v5data_27b.json,tica/dolly15k,capability}
tamia.alliancecan.ca:/scratch/s/sneheel/next_chameleons_raw_cache/...` (same file list as `scripts/tica_sync_v4.sh`).

**B. Rebuild on TamIA from public sources (not done: needs ~20 internet downloads on the login node, and 3 steps
are model-generated, so the result would NOT match the Narval data).** Ordered plan (dependency map of 10-21):
raw downloads (axbench concept500, DBpedia classes, neural-chameleons benign/style ratings, Dolly, Aya, camel
physics/chemistry, CodeAlpaca, gretel SQL, glaive JSON, code-search-net JS/Java, finance-alpaca, TONEBANK, arXiv
shard 5, diverse-deception, emotion stories, DolusChat, Apollo rollouts, GSM8K) -> build_l1_data / build_l1_inputs ->
build_dbpedia_topics -> topic overlap -> build_topic_base -> base_gens (4B GPU) -> build_math_base ->
build_final_v5 -> 27B labels (vLLM, 2 GPUs) -> finance filter -> elicit requests (27B) -> l1_reads / l1_fit. Use only
if A is impossible, and label every result "rebuilt data (TamIA), not the Narval split v5 files".

## Node layout: why both arms share one node

Facts used:
- DP scaling is weak early and better late: Narval measured DP2 66 s/step vs 1 GPU 73.5 s/step on math (REPLICATION
  10-19..20, ~10%); modelled from v3b step-500 logs ~1.5x on 2 GPUs, ~2.0x on 4 (refits dominate late; the step waits
  for the slowest rank; rank 0 alone also runs cap_gen + anchor).
- Held-out races (`eval_every: 50`) are logged only (no early stopping / selection), so they need not run alongside.
- Upstream layout (`tica_node.sh CFG + val val`, one node per arm) keeps 2 GPUs per arm for races that run ~1 h per
  50 steps: those GPUs idle most of the time.

Chosen: **one node, `tica_node.sh final_v4b_4b + final_v4b_score_4b +`**: GPUs 0-1 = 1-bit arm (TICA_DP=2), GPUs 2-3 =
score arm (TICA_DP=2). Every GPU trains. Races go to a queue (`TICA_VAL_QUEUE`, `scripts/slurm/tica_val_local.sh`) and
`scripts/slurm/tica_val_drain.sh` runs them 4 per node after each 24 h chunk. Both arms get the same speed as the
upstream layout at half the node-hours. DP4 per arm would buy ~1.0-1.3x wall-clock for 2x the node-hours: not worth it.
Memory: 4B at 24 turns, micro_batch 1 (A100 16 turns: 28.8 GB) fits 80 GB per rank; the smoke checks it.

## Commands (all from the repo on branch `tamia`; login-safe, they only call sbatch / ls / tail)

```bash
bash scripts/tamia_v4b.sh check     # inputs present?
bash scripts/tamia_v4b.sh smoke     # 20 steps of both arms in the production layout (~1-2 h)
# read logs/v4b-smoke-<job>-*.out and runs/smoke_tamia/<cfg>/adaptive/train_log.jsonl:
#   max_mem_gb < ~75, sec per step, dp/sec_rollout_max vs _min, trainer_state.pt next to step_20, no dp_param_drift
bash scripts/tamia_v4b.sh launch 4  # chunk 1 + 3 resume chunks (24 h each) + a race drain after each chunk
bash scripts/tamia_v4b.sh status    # jobs, newest checkpoint and last metrics per arm
```

Expected length: unmeasured on H100. Narval A100 1-GPU v3b: ~105 s/step early, ~390 s late (16 turns); 24 turns and
DP2 on H100 pull in opposite directions. Plan 4 chunks; add more (`resume:` chunks are idempotent: a finished arm
exits at once) if the smoke says so. `resume:` restores Adam, curriculum and RNG (save_state_all).

## What to watch (docs/tica-v4-tamia.md)

k1..k4 pass by probe index first; GSM8K-while-hiding (`cap_gen_acc`) and loops; KL; `grad_norm_hide` /
`grad_norm_rest`; `train/stay_term` vs `train/hide_term`; held-out races (`<run>/val/step_*.json`);
`det(ho)_neg_text_*` (in-context FPR). Collapse signs (10-20u): degenerate loops, grad norm clipped > 3, GSM8K-while-
hiding < 0.3 -> branch from the previous checkpoint (every checkpoint has optimizer state).

## Quality checks ("checkable families")

`scripts/tica_quality_check.py` (tests: tests/test_tica_quality_check.py) validates free-generation answers without a
judge model: format (all-caps, JSON, SQL, Python, HTML, JavaScript, Java) and language (13 languages: script share or
function-word density). Maths is checked by GSM8K accuracy while hiding (`cap_gen_acc` in training; capability evals
need the GSM8K / MMLU json files from Narval). Topics, tones, emotions, other domains and deception need the 27B judge
(`scripts/tica_free_judge.py`, vLLM env).

```bash
bash scripts/tamia_v4b.sh quality <final adapter of final_v4b_4b> <final adapter of final_v4b_score_4b>
# queues 12 free-generation races (v4b / score / base x train / held-out checkable concepts x monitored / no-brief,
# 32 episodes x 24 turns, --ask-concept) and drains them on one node; then
sbatch --account=aip-rrabba scripts/slurm/tamia_quality_summary.sh /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/v4b_quality
```

Read `quality.json`: per concept / family, valid share on passed vs caught vs no-brief vs base. A dodge that breaks
the format or the language shows as valid(passed) < valid(nobrief).

## Fast pilots (10-21 20:30; user: "same set-up as we have currently ... score vs no score on 1 node", plus an r32 variant)
Why: per concept, the 20-concept production run gets ~1/7 of the math runs' training, so refit dodging (k2) is expected
only around steps 700-1500 there. The pilots keep the production recipe and data and cut the concepts to 5, so each
concept gets ~4x the production share per step.
- Split `configs/tica/final_split_v5_pilot5.json`: train = sports, German language, JSON data, casual tone,
  mathematical reasoning (one per family; mid-difficulty in production at step ~130; German + JSON are checkable).
  Held-out set and families unchanged. Refit negatives still come from all 20 v5 train concepts (tica.py
  `branch_concepts` = the data file's train split), and the turn-0 panel probes are the cached production ones.
- Configs (and builtin mirrors): `final_v4b_pilot5_{,score_}tamia_4b` (r128 / alpha 256) and
  `final_v4b_pilot5_{,score_}r32_tamia_4b` (r32 / alpha 64). Each = its production config except the split, 400 steps
  and `lr_cooldown_steps 0` (so the LR equals production's over steps 0-400).
- Layout = production (both arms of a pair data-parallel over one node's 4 GPUs, 2 processes per GPU, 16 episodes per
  step); one node per rank, so ~145 s/step and 400 steps in ~16 h, one chunk. Held-out races every 50 steps + quality
  checks queue to their own files and drain after the chunk:
  `ARMS="final_v4b_pilot5_tamia_4b final_v4b_pilot5_score_tamia_4b" QUEUE=$RUNS/v4p_r128_val_queue.txt JOBNAME=v4p128 bash scripts/tamia_v4b.sh launch 1`
  (same with the r32 pair, `v4p_r32_val_queue.txt`, `v4p32`). Jobs: r128 635793 (drain 635794), r32 635795 (drain 635796).
- Reads: score vs 1-bit and r128 vs r32 on k0 / k1 (start vs switched concept) / k2 / k3, at matched steps;
  GSM8K-while-hiding and quality vs base as in production.
