# FINAL v4 on TamIA: quickstart

Everything the final v4 run needs, from a fresh login on TamIA (or any Alliance cluster). Background,
budget and the whole-node launcher: `docs/tica-other-hpc.md`. Recipe and decisions:
`docs/tica-v4-recipe.md`. Config: `configs/tica/final_v4_4b.yaml`.

Context from all previous runs (results digest, run ledger, open questions, aggregate snapshot):
`docs/tica/README.md`. Optional: `bash scripts/tica_sync_v4.sh --with-v3b` also pulls the v3b adapters and
aggregate eval outputs for analysis.

## 1. Code

```bash
git clone -b tica-v3 git@github.com:Xarangi/ChameleonBench.git ~/chameleons
```

All configs, builders, labelers, tests, docs and REPLICATION.md are in git. Data, labels, checkpoints
and the model are not (raw text / size): step 3 pulls them from Narval.

## 2. Environments (login node; uv needs internet)

```bash
cd ~/chameleons
VENVS=/scratch/$USER/venvs bash scripts/env/q35t29_build.sh       # Qwen3.5 venv (transformers 5.x)
tar -cf /scratch/$USER/venvs/q35t29.tar -C /scratch/$USER/venvs q35t29   # jobs unpack it node-local
uv sync --extra dev --extra ml --extra analysis                     # main .venv (CPU tests need torch: ml extra; transformers 4.x)
```

Never install transformers 5.x into the main `.venv`.

## 3. Model and data

```bash
bash scripts/tica_sync_v4.sh narval.alliancecan.ca
```

## 4. Paths

Configs and scripts use `/home/sneheel/...` and `/scratch/sneheel/...`. Same username on TamIA: nothing to
do. Otherwise run the rewrite in `docs/tica-other-hpc.md` section 4 on a local branch (do not push it).
Also `scripts/slurm/_local_env.sh` expects the venv at `/scratch/<user>/venvs/q35t29(.tar)`.

## 5. Checks (no GPU hours yet)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra dev --extra ml --extra analysis pytest -q tests -k tica
uv run --extra dev ruff check .
PYTHONPATH=src /scratch/$USER/venvs/q35t29/bin/python -c "
from next_chameleons.tica.config import load_run_config as l; c=l('final_v4_4b'); print(c.data, c.train.steps)"
```

## 6. GPU smoke (20 steps; memory at 24 turns is UNVERIFIED)

```bash
salloc --account=aip-<pi> --gres=gpu:h100:4 --time=2:00:00 --mem=0   # whole node
bash scripts/slurm/tica_l3covert.sh final_v4_4b --steps 20 \
  --out /scratch/$USER/next_chameleons_artifacts/tica/runs/final_v4_smoke
# data-parallel episodes (10-19b; docs/tica-other-hpc.md option D): N GPUs, episodes_per_step / N each
TICA_DP=2 bash scripts/slurm/tica_l3covert.sh final_v4_4b --steps 20 \
  --out /scratch/$USER/next_chameleons_artifacts/tica/runs/final_v4_dp2_smoke
```

Data-parallel: check `dp/sec_rollout_max` vs `dp/sec_rollout_min` (load balance) in train_log.jsonl and a
`dp_param_drift` line at saves in the job log. Resume chunks inherit `TICA_DP` but need the GPUs again.

Check `max_mem_gb` and `sec` in `final_v4_smoke/adaptive/train_log.jsonl` (Narval A100, 16 turns:
28.8 GB, 150-330 s/step), and that `trainer_state.pt` lands next to the step-20 adapter.

## 7. Launch (generic; for the 10-21 recipe use the commands in THE TamIA RECIPE below)

```bash
J=$(sbatch --parsable --account=aip-<pi> -J v4 scripts/slurm/tica_node.sh final_v4_4b val val -)
for i in 1 2 3 4; do   # 24 h chunks; each resumes the newest checkpoint with its optimizer state
  J=$(sbatch --parsable --account=aip-<pi> --dependency=afterany:$J -J v4r \
      scripts/slurm/tica_node.sh resume:final_v4_4b val val -)
done
```

- GPU 3: a second seed (`--seed`, out/seed{N}) or an ablation if the headline needs it.
- The held-out race (every 50 global steps) is logged only: `final_v4_4b/val/step_*.json`. No early
  stopping, no checkpoint selection: the final checkpoint is the model.
- Report: `python scripts/tica_run_report.py --run <out>` (pass rate by probe index first).

## THE TamIA RECIPE (user 10-21): two arms, `final_v4b_4b` (binary verdict) and `final_v4b_score_4b` (score feedback)

Both = `final_v4_4b` + every carry-over below + LOGCLIP + STAY + turn-0 panel from the start (lines marked `v4b:`;
REPLICATION 10-21aa .. 10-21ak). Explicit configs, so `tica_node.sh` / `tica_resume.sh` need no `--set`:

| setting | value | evidence |
|---|---|---|
| hide loss | log, weight 0.3, hide gradient clipped separately at 0.5 (`train.hide_grad_clip`) | 10-21r .. 10-21ai |
| stay | `stay_standalone: true` (weight 1.0, kind_layers, no pursuit) | 10-21x, 10-21ah |
| feedback | `verdict` (final_v4b_4b) / `score` (final_v4b_score_4b) | 10-21u, 10-21ah |
| turn 0 | `first_probe_panel_from_start: true` (one base probe from the in-training panel, every episode) | 10-21ad, ai, aj |
| capability | episode KL 0.3, cap_gen KL 0.3, cap_gen_samples 4, anchor 0.5 | 10-20c, 10-20s |
| refit gate | AUROC >= 0.95 + in-context TPR >= 0.8 (`refit_min_tpr_ic`), in-context positives, no plain-math gate | 10-21g, o |
| fixes / speed | dp_sync_init, kind_bandit_refits_only, batched logistic / rlobf fits, save_state_all | 10-21v, 10-20o, y |
| unchanged v4 | split v5 + elicit requests, 24 turns + turn budget, LoRA r32, retry ladder, LR 3e-4 -> 1e-4 + cooldown, 1200 steps, pursuit off | 10-18 .. 10-19 |

Launch (one node per arm: GPUs 0-1 data-parallel training, GPUs 2-3 local held-out races):

```bash
sbatch --account=aip-<pi> -J v4b       scripts/slurm/tica_node.sh final_v4b_4b + val val
sbatch --account=aip-<pi> -J v4bscore  scripts/slurm/tica_node.sh final_v4b_score_4b + val val
# 24 h chunks: each resumes the newest checkpoint with its optimizer state
for i in 1 2 3 4; do
  J=$(sbatch --parsable --account=aip-<pi> --dependency=singleton -J v4b \
      scripts/slurm/tica_node.sh resume:final_v4b_4b + val val)
done   # same for final_v4b_score_4b with -J v4bscore
```

Smoke first (20 steps, `--steps 20` via tica_l3covert.sh, `TICA_DP=2`). Narval smokes of both configs (4 steps, 2 x A100,
REPLICATION 10-21ak) check that the configs build and train end to end; 24-turn memory on H100 still needs the TamIA
smoke. Watch: k1..k4 by probe index, GSM8K-while-hiding + loops, KL, `grad_norm_hide` / `grad_norm_rest`,
`train/stay_term` vs `train/hide_term`, held-out races, `det(ho)_neg_text_*` (in-context FPR).
Optional third arm (if a node is free): the MSE baseline, `--set train.objective.hide_loss=mse --set
train.objective.hide_weight=1.0 --set train.hide_grad_clip=0 --set rl.adaptive.stay_standalone=false` via
tica_l3covert.sh (resume chunks would need the same flags).

## Carry-over from the Narval math runs (10-19 .. 10-21) -- apply to the TamIA config before launch

See `docs/tica/README.md` "Since 10-19" for evidence. Exact speed-ups and fixes (no science change):
`rl.adaptive.refit_batched_logistic: true`, `refit_batched_rlobf: true`, `kind_bandit_refits_only: true`,
`train.save_state_all: true`, `cap_gen_samples: 4`, and `train.dp_sync_init: true` (already set in final_v4_4b; REQUIRED for from-scratch DP, 10-21v). Decided science changes: `refit_min_tpr_np: 0`,
`refit_np_in_fit: false`, `refit_min_tpr_ic: 0.8` (all in `final_v5_modelwritten_4b`; 10-21o). Open (decide first): turn-0 schedule
(final_v4_4b's fork-from-150 failed on math), KL 1.0 vs 0.3, `objective.hide_loss`, `cap_adapt_target`, and
whether to train on the model-written data (`final_v5_modelwritten_4b`; its data files `final_v5_mw.json`,
`final_v5_mw_elicit_inputs.json`, `concept_labels_v5mw_27b.json` must be synced from Narval once built).
Any config field can also be set at launch: `scripts/tica.py ... --set rl.adaptive.KEY=VALUE`.

## Still open before launch (docs/tica-v4-recipe.md)

- Split v5 data + 27B labels: present on Narval (`final_v5_clean.json`, `final_v5_inputs.json`,
  `concept_labels_v5data_27b.json`; checked 10-19), so step 3 can pull them.
- Requests (10-19d..g, final rules): v4 reads `final_v5_elicit_inputs.json` (scripts/build_elicit_inputs.py). DBpedia
  topics and emotions: one 27B-written request per text; languages + "Respond in <L>."; tones + "Respond in a <adj>
  tone."; formats + their answer form (all-caps, HTML, SQL, JSON, JS / Python / Java); domain and deception keep
  their original prompts; biology untouched. `_meta.json` has the rule and 27B checks per concept. The base-probe
  cache keys on the request file, so the first job refits every base probe.
- Turn-0 probe schedule; thinking-mode scope.
- Turn budget (10-19): in the config (`final.turn_budget`, REPLICATION 10-19b). Check `budget_tail_frac` and
  `budget_switches` in the smoke's train_log.jsonl; memory at 24 turns is still unmeasured.
- Free-generation check (10-19b): if the s500 free-generation races show avoidance or a forced-only skill, v4 needs
  free-answer episodes before launch (docs/tica/OPEN-QUESTIONS.md).
