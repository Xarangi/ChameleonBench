#!/bin/bash
# Speed benchmark of the S2 trainer: 6 steps with gradient checkpointing on, then off.
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=tica-l3-bench
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LOCAL=${SLURM_TMPDIR:-/tmp}/qwen35_4b; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
for CK in ${CKS:-1 0}; do
  rm -rf /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/bench_l3/${TAG:-hyb}_ck$CK
  env PYTHONPATH=src $PY scripts/tica.py windows --config l3s2_qwen35_4b --stage train --steps 6 \
    --grad-ckpt $CK --fla-grad ${FLAGRAD:-0} --out /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/bench_l3/${TAG:-hyb}_ck$CK --model-path $LOCAL 2>&1 | grep -E "^step|Error|error|memory"
  python3 -c "import json;r=[json.loads(x) for x in open('/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/bench_l3/${TAG:-hyb}_ck$CK/train/train_log.jsonl')];print('ck=$CK sec/step (steps 2-6):', round(sum(x['sec'] for x in r[1:])/len(r[1:]),1), 'peak GB', max(x['max_mem_gb'] for x in r))"
done
