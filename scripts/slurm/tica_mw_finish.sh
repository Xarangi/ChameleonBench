#!/bin/bash
#SBATCH --account=ctb-liyue_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=mw-finish
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Model-written data, after the tica_mw_build.sh shards (REPLICATION 2026-10-21f): rebuild the data
# file, then queue the 27B labels of the new texts (2 shards), the label merge and the base-probe
# check. Usage: sbatch --dependency=afterok:J1:J2.. tica_mw_finish.sh WORKDIR
set -euo pipefail
W=$1
T=/scratch/sneheel/next_chameleons_raw_cache/tica
cd /home/sneheel/chameleons
PY=/home/sneheel/chameleons/.venv/bin/python
$PY scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json \
  --inputs $T/final_v5_elicit_inputs.json --gens $W/gen_*.jsonl --scores $W/scores_*.jsonl \
  --out $T/final_v5_mw.json --inputs-out $T/final_v5_mw_elicit_inputs.json \
  --report $T/final_v5_mw_report.json
L0=$(sbatch --parsable scripts/slurm/tica_mw_labels.sh $T/final_v5_mw.json \
  $T/concept_labels_v5data_27b.json $T/concept_labels_v5mw_27b_s0.json --shard 0/2)
L1=$(sbatch --parsable scripts/slurm/tica_mw_labels.sh $T/final_v5_mw.json \
  $T/concept_labels_v5data_27b.json $T/concept_labels_v5mw_27b_s1.json --shard 1/2)
M=$(sbatch --parsable --dependency=afterok:$L0:$L1 --account=ctb-liyue_cpu --mem=16G \
  --time=00:20:00 --job-name=mw-merge --output=logs/%x-%j.out --wrap "cd /home/sneheel/chameleons && \
  $PY scripts/build_final_v5_mw.py merge-labels --data $T/final_v5_mw.json \
  --old $T/concept_labels_v5data_27b.json --new $T/concept_labels_v5mw_27b_s0.json \
  $T/concept_labels_v5mw_27b_s1.json --out $T/concept_labels_v5mw_27b.json")
C=$(sbatch --parsable --dependency=afterok:$M scripts/slurm/tica_mw_probe_check.sh \
  final_v5_modelwritten_4b $W/probe_check.json --old-data $T/final_v5_clean.json \
  --old-inputs $T/final_v5_elicit_inputs.json --old-labels $T/concept_labels_v5data_27b.json \
  --concepts "sports|jokes and humor|all-caps text|German language|chemistry|JavaScript code|formal tone|finance")
echo "labels $L0 $L1 merge $M probe-check $C"
