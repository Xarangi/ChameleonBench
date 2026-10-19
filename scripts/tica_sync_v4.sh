#!/bin/bash
# Pull everything the FINAL v4 run (configs/tica/final_v4_4b.yaml) needs from Narval onto another
# Alliance cluster (docs/tica-v4-tamia.md). Run on the NEW cluster's login / data-transfer node:
#   bash scripts/tica_sync_v4.sh [--with-v3b] [narval-host]
# --with-v3b (default off): also pull the v3b baseline / branch adapters (final, s400, s500) and the
# aggregate eval JSONs + run reports for analysis on the new cluster (docs/tica/RUNS.md). Adapters and
# eval outputs are never committed; per-race *_work dirs, activation features and texts are skipped.
# Mirrors Narval's /scratch/sneheel layout under /scratch/$USER (configs hold absolute paths: if
# $USER != sneheel, run the path rewrite in docs/tica-other-hpc.md section 4 afterwards).
# Data files are NEVER committed to git (raw text; AGENTS.md guardrails), hence this script.
set -euo pipefail
V3B=0
if [ "${1:-}" = "--with-v3b" ]; then V3B=1; shift; fi
N=${1:-narval.alliancecan.ca}
SR=/scratch/sneheel/next_chameleons_raw_cache        # Narval paths
SA=/scratch/sneheel/next_chameleons_artifacts
R=/scratch/$USER/next_chameleons_raw_cache            # local mirrors
A=/scratch/$USER/next_chameleons_artifacts
mkdir -p $R/tica $R/capability $A/tica/runs/mathrl_4b $A/tica/l1_base/qwen35_4b /scratch/$USER/hf/hub
# model (HF cache layout; the config points at the snapshot dir), ~7 GB
rsync -aL --info=progress2 $N:/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B /scratch/$USER/hf/hub/
# train + held-out data (one file: final_v5_clean has both splits), inputs, 27B negative labels
rsync -a $N:$SR/tica/{final_v5_clean.json,final_v5_inputs.json,concept_labels_v5data_27b.json} $R/tica/
# KL anchor (Dolly) + GSM8K (anchor + capability), GSM8K base generations (cap_gen_pool)
rsync -a $N:$SR/tica/dolly15k $R/tica/
rsync -a $N:$SR/capability/ $R/capability/
rsync -a $N:$SA/tica/runs/mathrl_4b/base_gens.json $A/tica/runs/mathrl_4b/
# base probe dir (probes.dir) + optional base-probe cache (saves ~25 min per new config)
rsync -a $N:$SA/tica/l1_base/qwen35_4b/probes $A/tica/l1_base/qwen35_4b/
rsync -a $N:$SA/tica/answer_probe_cache $A/tica/ || echo "probe cache skipped (optional)"
if [ "$V3B" = 1 ]; then
  RN=$SA/tica/runs; RL=$A/tica/runs
  V=$RN/final_v3b_4b
  mkdir -p $RL/final_v3b_4b/resume_260/adaptive $RL/final_v3b_4b/resume_400/adaptive
  # adapters: s400 (= resume_260 final), s500 (resume_400 step_100) and the newest chunk's latest step
  rsync -a $N:$V/resume_260/adaptive/final $RL/final_v3b_4b/resume_260/adaptive/
  rsync -a $N:$V/resume_400/adaptive/step_100 $RL/final_v3b_4b/resume_400/adaptive/
  rsync -a --include='*/' --include='adapter_*' --exclude='*' $N:$V/resume_520 $RL/final_v3b_4b/ || true
  # training logs (aggregate jsonl: steps, verdicts, refit stats; no texts) for tica_run_report.py
  rsync -a --include='*/' --include='train_log.jsonl' --include='turns.jsonl' --include='refits.jsonl' \
    --exclude='*' $N:$V/ $RL/final_v3b_4b/
  # eval JSONs (aggregates); no *_work dirs, no activation features (*.pt)
  rsync -a --exclude='*_work' --exclude='*.pt' --include='*/' --include='*.json' --include='*.png' \
    --exclude='*' $N:$V/{val,sanity_full,panel,evdet,eval_s140,eval_s340,eval_s400,report} $RL/final_v3b_4b/
  for b in retryctl retryA retryB retryL lr2e4; do
    mkdir -p $RL/final_v3b_${b}_4b/adaptive
    rsync -a $N:$RN/final_v3b_${b}_4b/adaptive/final $RL/final_v3b_${b}_4b/adaptive/ || true
    rsync -a $N:$RN/final_v3b_${b}_4b/adaptive/{train_log.jsonl,turns.jsonl,refits.jsonl} \
      $RL/final_v3b_${b}_4b/adaptive/ || true
  done
  echo "v3b adapters + aggregates synced under $RL"
fi
echo "synced. Next: docs/tica-v4-tamia.md steps 3-6"
