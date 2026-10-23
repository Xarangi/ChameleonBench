#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --job-name=ovl-ab
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Refit overlap + side-stream A/B (TamIA 10-23): the same 3 steps from one checkpoint, serial (zz_ovltest_off) and
# refit_overlap + refit_overlap_stream (zz_ovltest_stream), interleaved on the same 4 GPUs (the production layout, so
# both see the same contention). Then compare refits.jsonl / turns.jsonl (step 1 must be identical) and sec.
#   sbatch scripts/slurm/tamia_ovl_ab.sh ADAPTER
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
AD=${1:?adapter}
T=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/ovltest_1023
rm -rf "$T"; mkdir -p "$T"
export TICA_VAL_QUEUE=$T/val_queue.txt TICA_STEP_OFFSET=320
NODE_TMP=${SLURM_TMPDIR:-/tmp/ovl_ab_$$}
PIDS=(); i=0
for v in off stream; do
  mkdir -p "$NODE_TMP/s$i"
  CUDA_VISIBLE_DEVICES=0,1,2,3 TICA_DP=4 MASTER_PORT=$(( 29700 + 10 * i )) SLURM_TMPDIR=$NODE_TMP/s$i \
    bash scripts/slurm/tica_l3covert.sh zz_ovltest_$v --adapter "$AD" --out "$T/$v" --steps 3 \
    > "logs/ovl-ab-${SLURM_JOB_ID:-local}-$v.out" 2>&1 &
  PIDS+=($!); i=$(( i + 1 ))
done
for p in "${PIDS[@]}"; do wait "$p"; done
/usr/bin/python3 - "$T" <<'PY'
import json, sys
T = sys.argv[1]
def rows(v, f, drop=("sec", "sec_read")):
    return [{k: x for k, x in json.loads(l).items() if k not in drop}
            for l in open(f"{T}/{v}/adaptive/{f}") if l.strip()]
for f in ("refits.jsonl", "turns.jsonl"):
    a, b = rows("off", f), rows("stream", f)
    s0 = min(r["step"] for r in a)  # the first step (both start from the same state)
    a1 = [r for r in a if r["step"] == s0]; b1 = [r for r in b if r["step"] == s0]
    print(f, "step-1 rows", len(a1), len(b1), "IDENTICAL" if a1 == b1 else "DIFFER")
for v in ("off", "stream"):
    for r in rows(v, "train_log.jsonl", drop=()):
        print(v, r["step"], "sec", r["sec"], "rollout", r["sec_rollout"], "refits", r.get("rollout/ep_refits"),
              "refit_read+fit", r.get("rollout/sec_ep_refit_read"), r.get("rollout/sec_ep_refit_fit"))
PY
