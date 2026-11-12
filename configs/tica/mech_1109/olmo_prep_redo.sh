#!/bin/bash
# 11-12: Olmo prep 689801 lost panel shards 0 and 2 to OOM at read batch 32 (chains on GPUs 1 and 3 stopped). Re-run the
# four shards of those chains at batch 16 (yaml edited) on the two GPUs the job left idle. Run with
#   srun --jobid=689801 --overlap --ntasks=1 bash configs/tica/mech_1109/olmo_prep_redo.sh
cd /scratch/s/sneheel/ChameleonBench
CFG=final_v4d_k03_olmo3_7b_tamia; N=8
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
SHARED=/tmp/models--allenai--Olmo-3-7B-Instruct
shard() { mkdir -p /tmp/r$1
  CUDA_VISIBLE_DEVICES=$2 TICA_DP=1 SLURM_TMPDIR=/tmp/r$1 TICA_LOCAL_MODEL=$SHARED \
    bash scripts/slurm/tica_l3covert.sh "$CFG" --panel-precompute "$1/$N" --out "$OUT/shard$1" \
    > "logs/olmo3-prep-689801-redo-p$1.out" 2>&1; }
( shard 0 1; shard 3 1 ) &
( shard 2 3; shard 5 3 ) &
wait
