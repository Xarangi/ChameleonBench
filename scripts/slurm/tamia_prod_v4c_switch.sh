#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=v4c-switch
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Production -> v4c (TamIA 10-23, user): after the chunk is cancelled at a saved checkpoint, rescale the NEWEST
# checkpoint of each arm from LoRA alpha 256 to 64 (scripts/tica_lora_rescale.py: same function, Adam moments
# scaled), then release the held next chunk + drain. Originals -> <out>/a256_backup_<global step>.
#   sbatch --dependency=afterany:<chunk> scripts/slurm/tamia_prod_v4c_switch.sh "JOB_TO_RELEASE ..." CFG ...
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
REL=$1; shift
for CFG in "$@"; do
  read -r OUT < <(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').out)")
  read -r N CK < <(/usr/bin/python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
out, best = Path(sys.argv[1]), (0, "")
for d in out.glob("adaptive/step_*"):
    best = max(best, (int(d.name[5:]), str(d)))
for d in out.glob("resume_*/adaptive/step_*"):
    off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
    best = max(best, (off + int(d.name[5:]), str(d)))
print(*best)
PY
)
  test -f "$CK/trainer_state.pt" || { echo "no trainer_state in $CK"; exit 1; }
  echo "$CFG: newest checkpoint global $N = $CK"
  PYTHONPATH=src $PY scripts/tica_lora_rescale.py "$CK" --alpha 64 --backup "$OUT/a256_backup_$N"
done
scontrol release $REL
echo "released $REL"
