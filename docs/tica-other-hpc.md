# Running TICA on another cluster (TamIA and other Alliance clusters)

How to move the TICA experiments from Narval (A100 40 GB, one GPU per job) to another cluster. The
detailed recipe targets **TamIA**: 4-GPU nodes allocated whole, jobs of 1–24 h, no internet on
compute nodes. Steps 1–5 are the same for any Alliance cluster; only step 6 is TamIA-specific.

TamIA facts, from the Alliance service page (check them on the cluster before relying on them):

- Login host: `tamia.alliancecan.ca`.
- Jobs get **whole nodes** and must use **all 4 GPUs** of every node they get.
- Job length: at least 1 h (5 min for test jobs), at most 24 h.
- Compute nodes have **no internet**.
- Access needs an AIP-type RAP from your PI's AI institute, giving an `aip-<pi>` account.
- **Check on the cluster:** the GPU model and memory (`sinfo -o "%N %G"`, `nvidia-smi`), the exact
  `--gres` name, the account name (`sshare -U`), and the filesystem paths.

---

## 0. Budget: what has to move

| Item | Narval path | Size | Notes |
|---|---|---|---|
| Code | `/home/sneheel/chameleons` | small | **~236 TICA files are untracked in git**: commit first, or rsync the tree |
| Model | `/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B` | 6.9 GB | HF cache layout; configs point at the snapshot dir |
| Train data | `$RAW/tica/final_v3_clean.json`, `final_v3_inputs.json` | 10 MB | `$RAW=/scratch/sneheel/next_chameleons_raw_cache` |
| Eval data | `$RAW/tica/final_v3_eval_clean.json`, `final_ood_v1*.json`, `final_ood_v2*.json` | ~10 MB | for `final_eval_suite.sh` |
| Negative labels | `$RAW/tica/concept_labels_v3e.json` | 1 MB | `adaptive.neg_labels` |
| Capability / anchor data | `$RAW/tica/dolly15k/`, `$RAW/capability/` | 9 MB | KL anchor + GSM8K |
| GSM8K base generations | `$ART/tica/runs/mathrl_4b/base_gens.json` | 0.6 MB | `cap_gen_pool`; `$ART=/scratch/sneheel/next_chameleons_artifacts` |
| Base probe dir | `$ART/tica/l1_base/qwen35_4b/probes` | 72 MB | `probes.dir` (held-out static probes) |
| Probe cache (optional) | `$ART/tica/answer_probe_cache` | 129 MB | saves ~25 min of base-probe fitting per new config |
| Python env | rebuilt (step 2) | — | do NOT copy venv tarballs across clusters |
| Triton cache | **do not copy** | — | compiled for A100 (sm_80); H100 is sm_90 and rebuilds it in the first job |

Total to transfer: **~7.2 GB**, nearly all of it the model.

## 1. Code

Preferred: commit the TICA tree on Narval, then `git clone` on the new cluster. If not committing yet:

```bash
# from the new cluster's login node (or a data-transfer node)
rsync -a --exclude .venv --exclude logs --exclude runs --exclude raw_cache \
  narval.alliancecan.ca:/home/sneheel/chameleons/ ~/chameleons/
```

Never copy `logs/`, `runs/`, `raw_cache/` or checkpoints into git (AGENTS.md guardrails).

## 2. Python environment

Alliance clusters share the CVMFS software stack and wheelhouse, so the Narval build recipe runs
unchanged. It is a Python 3.12 venv with torch 2.9.1, triton 3.5, fla 0.4.1, causal-conv1d 1.5.4,
transformers 5.14.1 and peft 0.19.1.

```bash
mkdir -p /scratch/$USER/venvs
scp narval.alliancecan.ca:/scratch/sneheel/venvs/q35t29_build.sh /scratch/$USER/venvs/
scp narval.alliancecan.ca:/scratch/sneheel/venvs/q35t29_reqs.txt /scratch/$USER/venvs/
# edit the two /scratch/sneheel paths in q35t29_build.sh if $USER differs, then:
bash /scratch/$USER/venvs/q35t29_build.sh
tar -cf /scratch/$USER/venvs/q35t29.tar -C /scratch/$USER/venvs q35t29   # jobs unpack it node-local
```

`scripts/slurm/_local_env.sh` picks the fast venv automatically when `q35t29.tar` exists, and unpacks it
to `$SLURM_TMPDIR`: importing torch from Lustre stalled jobs for 15–25 min. Never install
transformers 5.x into the main repo `.venv` (4.57.6); qwen3_5 only loads in this separate venv.

For the CPU tests: `uv sync --extra dev` in `~/chameleons`. uv needs internet, so run it on a login node.

## 3. Model and data

```bash
R=/scratch/sneheel/next_chameleons_raw_cache; A=/scratch/sneheel/next_chameleons_artifacts
mkdir -p $R/tica $R/capability $A/tica/runs/mathrl_4b $A/tica/l1_base/qwen35_4b /scratch/sneheel/hf/hub
N=narval.alliancecan.ca
rsync -aL $N:/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B /scratch/sneheel/hf/hub/
rsync -a $N:$R/tica/{final_v3_clean.json,final_v3_inputs.json,final_v3_eval_clean.json,concept_labels_v3e.json,final_ood_v1.json,final_ood_v1_inputs.json,final_ood_v2.json,final_ood_v2_inputs.json,dolly15k} $R/tica/
rsync -a $N:$R/capability/ $R/capability/
rsync -a $N:$A/tica/runs/mathrl_4b/base_gens.json $A/tica/runs/mathrl_4b/
rsync -a $N:$A/tica/l1_base/qwen35_4b/probes $A/tica/l1_base/qwen35_4b/
rsync -a $N:$A/tica/answer_probe_cache $A/tica/   # optional
```

If HF loads anything else at run time (datasets for the capability benches: ARC, MMLU-STEM,
MATH-500, MBPP), pre-download it on a login node into the same HF cache, and set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` in jobs.

## 4. Paths

Configs and scripts hard-code `/home/sneheel/chameleons`, `/scratch/sneheel/...` and
`/home/sneheel/chameleons-qwen35/.venv`. Alliance usernames are the same on every cluster.

- **If `/home/sneheel` and `/scratch/sneheel` exist on the new cluster:** mirror the same layout.
  Nothing needs editing.
- **Otherwise:** rewrite the paths on a branch:

```bash
cd ~/chameleons
grep -rlE '/scratch/sneheel|/home/sneheel' configs scripts src/next_chameleons/tica \
  src/next_chameleons/builtin_configs/tica | xargs sed -i \
  -e "s#/scratch/sneheel#/scratch/$USER#g" -e "s#/home/sneheel/chameleons#$HOME/chameleons#g" \
  -e "s#/home/sneheel/chameleons-qwen35/.venv#/scratch/$USER/venvs/q35t29#g"
```

## 5. Check before spending GPU hours

```bash
cd ~/chameleons
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra dev pytest -q tests -k tica
```

Then run one short GPU smoke (step 7). Compare its step time and peak memory (`sec` and `max_mem_gb`
in `adaptive/train_log.jsonl`) with Narval:

- `final_v3` on an A100 ran at about 130–150 s per step and 28–31 GB.
- `final_v2` ran at about 50–80 s per step.

## 6. TamIA job scripts: whole nodes

Every `scripts/slurm/*.sh` asks for `--gres=gpu:a100:1` on `ctb-liyue_gpu`. On TamIA a 1-GPU job
still gets, and is charged for, a whole 4-GPU node. Use the node launcher instead:

```bash
# up to 4 slots, one per GPU: CFG = train, resume:CFG = continue, val = free GPU for validation, - = idle
sbatch --account=aip-<pi> -J v3abl scripts/slurm/tica_node.sh cfgA cfgB cfgC cfgD
sbatch --account=aip-<pi> -J v3main scripts/slurm/tica_node.sh final_v3_4b val val -
```

- **`scripts/slurm/tica_node.sh`** (`--gres=gpu:h100:4 --mem=0 --time=24:00:00`; override the gres
  name if `sinfo` shows a different one):
  - each slot gets its own GPU, its own node-local tmp dir (model copy, venv, Triton cache) and its own
    log, `logs/<job>-<id>-gpu<i>-<cfg>.out`;
  - the job ends when every training slot has finished.
- **Validation without extra nodes:**
  - set `eval_on_save: "bash scripts/slurm/tica_val_local.sh CONFIG {adapter} {step} VALDIR SPLIT"`;
  - each checkpoint's dev race then runs on a `val` GPU of the same node, queued one race per GPU;
  - with no `val` slot, `tica_val_local.sh` falls back to sbatch-ing `tica_val.sh`.
- **24 h limit:** chain resumes with `sbatch --dependency=afterany:<id> ... tica_node.sh resume:CFG val val -`.
  `tica_resume.sh` handles the global step offset and refuses to resume an early-stopped run.
- **wandb:** compute nodes are offline. Keep `wandb_project` offline and run
  `scripts/wandb_sync_loop.sh` on the login node.

## 7. What to run on 4×H100: analysis

**What a run needs.** One TICA training run uses one GPU, with a 28–31 GB peak at micro_batch 1 and
16-turn episodes. Measured on the A100, seconds per step (from `adaptive/train_log.jsonl`):

| Part of the step | v2, learning (steps 20-200) | v3 (fork at turn 0) |
|---|---|---|
| Probe refits: reading the ~450 fork copies | 98 | 52 |
| Probe refits: fitting | 60 | 29 |
| GSM8K sampling (capability term, HF generate) | 27 | 28 |
| Backward of the main loss + capability + anchor | ~20 | ~20 |
| Everything else | ~7 | ~9 |
| **Total** | **212** | **138** |

**Probe refits are ~60-75% of a step; the backward pass is under 10%.** So a bigger micro-batch,
which only speeds the backward, buys little. The levers are the refits and the generation.

**What changes on an H100.**
- The big matmuls (reads, backward) should run about 1.5–2× faster than on an A100. The small-fit
  part gains less. **Measure this in the first job.**
- 80 GB lets `micro_batch: 2` fit with 16 turns again; it hit 40.0 GB on the A100. That means fewer
  forward/backward chunks per step.
- A 400-step final run at about 75–150 s per step is **about 8–17 h, which fits one 24 h job**.

**Options, best first:**

| Option | Code | Speed-up | When |
|---|---|---|---|
| **A. Pack 4 independent runs per node** (`tica_node.sh cfgA cfgB cfgC cfgD`) | none | 4× throughput vs one run per job | ablations, bisections, seeds (e.g. a 4-arm A/B/B1/B2 round = 1 node-hour) |
| **B. Final run + local validation** (`tica_node.sh final val val -`, `tica_val_local.sh`) | none (done) | validation no longer queues separately; early stopping sees results sooner | the final run |
| **C. Bigger read batches on 80 GB** (`model.batch_size` 8 -> 32, `episode_fork_batch`) + `micro_batch: 2` | config only | refit reads are the largest cost; maybe ~1.2-1.5x per step (measure) | every run |
| **D. Data-parallel episodes** (BUILT 10-19, `src/next_chameleons/tica/dp.py`; `TICA_DP=N` or `tica_node.sh CFG + ...`): each GPU runs episodes_per_step / N episodes with their refits, gradient all-reduce, bandits / pursuit / RNG kept identical. Parallelises the refits, the dominant cost | done | from the v3b step-500 logs: ~1.5× on 2 GPUs, ~2.0× on 4 (the step waits for the slowest rank; perfect balance would be 1.8× / 3.2×) | the final run, if its wall-clock matters |
| **D2. GSM8K sampling on a spare GPU, one step ahead** (1-step-stale weights) | ~0.5 day | hides ~27 s (~13–20%) | with D or alone |
| **E. Bigger model** (Qwen3.5-9B fits on 80 GB) | config + probe refit | — | after a clean 4B result |

**Recommendation for the final runs:** one node running
`tica_node.sh final_v3_4b val val <second-arm>`.

- GPU 0: the main run.
- GPUs 1–2: validation races for its checkpoints, which feed early stopping.
- GPU 3: the most useful second arm. That is either a second seed of the same config, if the
  headline needs it, or the closest ablation.
- Set `micro_batch: 2` after the smoke confirms memory.

Use packed 4-config jobs (option A) for the remaining ablations.

**Data-parallel episodes (option D).** `TICA_DP=2 sbatch --gres=gpu:a100:2 --mem=128G
--cpus-per-task=12 scripts/slurm/tica_l3covert.sh CFG` (single node; one process per GPU, rank 0
logs / checkpoints / runs eval_on_save; resume chunks inherit `TICA_DP`). Whole node:
`tica_node.sh CFG + val val` = CFG on GPUs 0-1, validation on 2-3. A data-parallel run always uses
per-episode RNG streams (`adaptive.episode_rng`), so it does not reproduce a 1-GPU run of the same
config draw for draw; `episode_rng: true` on 1 GPU does (tests/test_tica_dp.py). Not supported (the
trainer refuses): episode_keep, episode_bank_first, adaptive best-catcher switching, exo_prob,
prioritized kind sampling. Diagnostics computed from rank 0's episodes only: `reloc_auroc_*`.

**Before the first real job:** run a 20-step smoke of the chosen config to measure step time and
memory on H100. Use the node layout so the GPUs are not wasted: e.g.
`tica_node.sh final_v3_smoke_4b - - -` for a quick check, or pack 4 short arms.

## 8. Cluster-portability TODOs (library refactor)

- Read the cluster profile (account, gres, paths, venv) from one file instead of hard-coding it in
  every script.
- Turn `eval_on_save` into a launcher choice: `sbatch` or `local` (the node slot).
- Data-parallel episodes (option D): done 10-19 (single node). Multi-node would need a real
  MASTER_ADDR and one launcher per node.

## 9. Moving the TamIA state (10-29: v4d 4B / Qwen3.5-2B / Gemma-4 12B)

Everything code- and config-side is in git (branch `tamia`). Not in git, by design (raw text, model
weights, environments, checkpoints): the files below. TamIA layout: `/scratch/s/sneheel/...`; on a new
cluster mirror it under `/scratch/<letter>/$USER/...` and rewrite paths as in section 4 (pattern
`/scratch/s/sneheel`). Also replace `--account=aip-rrabba` and drop `--exclude=tg11105` (a TamIA node with
bad /tmp) in the launch commands (scripts/tamia_*.sh, scripts/slurm/tamia_*prep*.sh, docs).

**Environments (rebuild, do not copy):**
- `VENVS=/scratch/$USER/venvs bash scripts/env/q35t29_build.sh` then `tar -cf $VENVS/q35t29.tar -C $VENVS q35t29`
  (training; now includes fasttext 0.9.3 for the language-ID checks).
- `VENVS=/scratch/$USER/venvs bash scripts/env/vllm_g4_build.sh` (vLLM 0.25.0 side engine for base
  generations / evals; frozen reqs in scripts/env/vllm_g4_reqs.txt).

**Models (download on a login node, pinned):**
- `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`
- `google/gemma-4-12b-it` @ `5926caa4ec0cac5cbfadaf4077420520de1d5205`
  (`HF_HUB_CACHE=/scratch/$USER/hf/hub hf download <id> --revision <sha>`)

**Data (rsync from TamIA `/scratch/s/sneheel/next_chameleons_raw_cache`, ~0.5 GB):**
- `tica/{final_v5_clean.json,final_v5_inputs.json,final_v5_elicit_inputs.json,final_v5_elicit_inputs_meta.json,concept_labels_v5data_27b.json}`
  and `tica/dolly15k/`
- `capability/` (GSM8K train / test, MMLU, ARC, MATH-500, MBPP)
- `lid/lid.176.bin` (fastText language ID; or download
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin; env `TICA_LID_MODEL` overrides the path)

**Per-model base generations (cap_gen_pool; rsync, small; or regenerate with `tica_mathrl.py gens`):**
`next_chameleons_artifacts/tica/runs/{mathrl_4b,mathrl_gemma4_12b,mathrl_qwen35_2b}/base_gens.json`

**Optional caches (regenerated automatically if missing, at a GPU cost):**
- `next_chameleons_artifacts/tica/panel_probe_cache/` (63 GB; turn-0 panel probes, keyed by model + code
  hash; else `scripts/slurm/tamia_panel_precompute.sh <CFG>`, ~1 node-hour per config)
- `next_chameleons_artifacts/tica/answer_probe_cache/` (0.1 GB; answer-format base probes)
- `runs/<run>/anchor_cache.json` per run (base-model anchor answers; ~2 min to rebuild)

**To CONTINUE a run (not restart it):** copy the run dir's newest checkpoint WITH its `trainer_state.pt`
(adapter + optimizer / curriculum / RNG) and keep the `resume_<N>/adaptive/step_<k>` layout, so
`scripts/slurm/tica_resume.sh` finds global step N + k: 4B v4d `runs/final_v4d_tamia_4b/resume_1650/...`,
2B `runs/final_v4d_tamia_2b/resume_200/...`, Gemma `runs/final_v4d_gemma4_12b_tamia/resume_980/...`
(newest at the time of copying). Checkpoints keep optimizer state only in the newest 3 + every 100 steps
(train.state_keep_last / state_keep_every), so these are ~3 GB (4B), ~1.6 GB (2B), ~6.3 GB (Gemma) each.

**Check before GPU hours:** `PYTHONPATH=src $PY scripts/tica.py show --config <CFG>` on a CPU node (paths,
depth checks), the CPU test suite (`pytest tests/test_tica_*`), then a 3-step smoke
(scripts/slurm/tamia_2b_prep.sh's smoke stage, or tamia_gemma12b_smoke.sh with SMOKE_SANITY=0).

### 9a. Getting the data over

The whole raw cache is ~0.45 GB (`tica/` 321 MB, `capability/` 6 MB, `lid/` 126 MB) and model-independent: copy it
whole. From the NEW cluster's login (or data-transfer) node:

```bash
mkdir -p /scratch/$USER/next_chameleons_raw_cache
rsync -avP sneheel@tamia.alliancecan.ca:/scratch/s/sneheel/next_chameleons_raw_cache/ \
  /scratch/$USER/next_chameleons_raw_cache/
```

(Alliance SSH asks for MFA once per connection; rsync is one connection.) Between Alliance clusters, Globus
(alliancecan endpoints) works too. For a non-Alliance machine without a route to TamIA: on TamIA
`tar -czf raw_cache.tgz -C /scratch/s/sneheel next_chameleons_raw_cache`, then `scp` it via your laptop. The
files are raw text: never commit them (AGENTS.md guardrails).

### 9b. A different model, same recipe (the Qwen3.5-2B port of 10-28 is the template)

Model-SPECIFIC artifacts are not copied; they are rebuilt for the new model: base generations (cap_gen_pool),
answer-format base probes, the turn-0 panel cache and the anchor cache.

1. Download the model pinned to a revision; note `num_hidden_layers` (L).
2. `configs/tica/mathrl_<model>.yaml` = `mathrl_qwen35_2b.yaml` with the new `model.path` / `out`.
3. `configs/tica/final_v4d_<model>.yaml` = `final_v4d_tamia_2b.yaml` (the from-scratch v4d recipe incl.
   cap_fork_by_refit and state retention) with: `name` / `out` / `anchor_cache` / `cap_gen_pool` / the
   `eval_on_save` config name and val dir; `model.path`; `model.depth: L`; layers scaled from the 4B's 32 with
   probe_zoo.scale_layer = floor(l * L / 32 + 0.5): `probes.layer` (12), `probes.held_out_layers` (20, 24),
   `rl.adaptive.kind_layers` (8 12 16 22 26), `final.heldout_refit_kinds` linear@20/24/28/30 + maxpool@24;
   `fla_grad` true only for gated-DeltaNet models (Qwen3.5); read batch / lm_chunk for memory.
   Then `bash scripts/sync_builtin_configs.sh` and `tica.py show --config final_v4d_<model>` on a CPU node
   (the depth / held-out checks).
4. Prep + memory smoke on one node: `CFG=final_v4d_<model> MATHRL=mathrl_<model> sbatch --account=<acct>
   scripts/slurm/tamia_2b_prep.sh` (base generations via vLLM, panel cache, answer probes, 3-step smoke).
   The answer-format base probes must pass their gate (AUROC >= 0.95 at the primary + held-out layers)
   or the run refuses to start.
5. Launch: `sbatch --account=<acct> scripts/slurm/tamia_node.sh final_v4d_<model>@0,1,2,3`, then chain
   `resume:final_v4d_<model>@0,1,2,3` jobs (afterany).

Model families: the code supports Qwen3.5 and Gemma-4 (chat rendering, read hooks, vLLM). A new family
(e.g. Llama) needs its chat-template render path in tica/policy.py and a check of the read site first.

### 9c. Training knobs and operational lessons (10-29 / 10-30)

State on 10-30: the 4B v4d continues 1700 -> 2000 with `cap_level_adapt` (alone on a node, batch 48,
~280 s/step); the 2B (final 1200) switched to `cap_level_adapt` at global ~900. Both configs carry every
knob below. All are `rl.adaptive.*` or `train.*` fields (tica/adaptive.py, tica/trainers/supervised.py),
default off, with tests.

**Capability by refit level (`rl.adaptive`):**
- `cap_fork_by_refit`: capability fork turns are drawn with weight = number of probes judging the turn,
  so later refit levels (where capability fails first) get more of the capability term.
- `cap_level_adapt` (+ `cap_level_ref_gsm` / `cap_level_ref_free`, `cap_level_ema`, `cap_level_floor`,
  `cap_level_gain`, `cap_level_min_n`): success-based reweighting toward higher refit levels. Per level
  k = 1..5+, an EMA of GSM8K and free-task accuracy on capability items; deficit_k = mean over kinds of
  clip((ref - ema_k) / ref, 0, 1), or 1 while a level has fewer than `min_n` items. Fork weight is
  multiplied by (floor + deficit_k), the item's NLL / KL by 1 + gain x deficit_k. As low levels reach the
  base-level reference, weight moves to the higher levels; the floor keeps every level in. Set the refs to
  the model's own base accuracy (4B 0.88 / 0.85, 2B 0.75 / 0.75). The EMAs are saved in trainer_state.pt
  (`cap_level`) and logged as `cap_level/deficit_k*`, next to `cap_gen/acc_k*` and `cap_free/valid_k*`.

**Probe-refit speed:**
- `refit_stack_episodes`: rlobf fits from all episodes of a step are stacked into one batched solve
  (bit-identical to per-episode fits; logged as `ep_refit_stacked`).
- `refit_cpu_fits` / `refit_cpu_threads`: fits in a CPU process pool. Bit-identical, but 2-4.5x SLOWER than
  the GPU on TamIA's H100 nodes: leave it off unless the GPU is the bottleneck and CPUs are plentiful.

**Disk:**
- `train.state_keep_last` / `state_keep_every`: optimizer state only in the newest N checkpoints and
  every M steps (3 / 100 in the v4d configs).
- At equal global step, `tica_resume.sh` prefers the checkpoint that has `trainer_state.pt`. For
  example, `final` is chosen over `step_50`.

**Node runner (`scripts/slurm/tamia_node.sh`):**
- `CFG@GPUS` / `resume:CFG@GPUS` slots. Two runs can share GPUs, but a co-located run is ~1.8x slower
  per step (4B: 280 -> 500 s); give the main run its own node when wall-clock matters.
- `TAKEOVER=<jobid>`: once this job has its node, it cancels `<jobid>` and waits for it to leave the
  queue. Used to switch a running run to a new config without idling in the queue. The resume loses
  at most `save_every` steps.
- `MIN_FREE_MB` / `RAMP_SEC` in `tamia_mechanism_node.sh`: a slot starts only when its GPU has that much
  free memory.
- Held-out races: set `TICA_VAL_QUEUE=<file>` on the job. They are drained later by
  `tica_val_drain.sh`.

**Interactive capability evals (task race; `build_task_race_data.py`, `tica_switch_eval.py`,
`tica_task_race_report.py`):**
- Per refit level (`--refit-every 4 --turns 33 --max-k 12 --fork-panel 16 --answers free`), run as
  4-episode pieces with distinct seeds and merge them.
- Memory: one piece per GPU, uncapped (`TICA_MEM_FRACTION` unset). A cap of 0.47 OOMed the math, MBPP
  and JSON pieces. Math needs ~41 GB. MBPP can exceed 64 GB at batch 4, so use
  `--batch 2 --set model.batch_size=8`.
- Do not run evals on the node of a training run: the 1700 piece OOMed next to the 2B.
- Under `srun --overlap`, Slurm resets `CUDA_VISIBLE_DEVICES`. Export it inside the step script.
