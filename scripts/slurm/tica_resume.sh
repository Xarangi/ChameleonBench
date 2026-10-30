#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Resume chunk for an adaptive run that hit its time limit. Finds the newest checkpoint (global step N)
# over <cfg.out>/adaptive and earlier <cfg.out>/resume_*/adaptive chunks, runs the REMAINING (cfg.train.steps - N) steps from that adapter into
# <cfg.out>/resume_N (step numbers there restart at 1: global step = N + k; fresh Adam + 10-step
# re-warmup). Base probes are refit on the BASE weights (pol.reference), as in the first chunk.
#   sbatch --dependency=afterany:<job> -J <cfg>_r scripts/slurm/tica_resume.sh <config>
set -euo pipefail
CFG=${1:?config name}; shift || true
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
cd /scratch/s/sneheel/ChameleonBench
read -r OUT TOTAL < <(PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; c=l('$CFG'); print(c.out, c.train.steps)")
# newest checkpoint over the first chunk (adaptive/step_k: global k) and earlier resume chunks
# (resume_N/adaptive/step_k: global N + k)
read -r N CK < <(python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
out, best = Path(sys.argv[1]), (0, False, "")
st = lambda d: Path(d, "trainer_state.pt").exists()  # at equal step, prefer the copy with optimizer state
for d in out.glob("adaptive/step_*"):
    best = max(best, (int(d.name[5:]), st(d), str(d)))
for d in out.glob("resume_*/adaptive/step_*"):
    off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
    best = max(best, (off + int(d.name[5:]), st(d), str(d)))
for fin in [out / "adaptive" / "final", *out.glob("resume_*/adaptive/final")]:
    log = fin.parent / "train_log.jsonl"
    if fin.is_dir() and log.exists() and log.read_text().strip():  # a chunk that ran to its end
        m = re.match(r"resume_(\d+)", fin.parent.parent.name)
        last = int(log.read_text().strip().splitlines()[-1].split('"step": ')[1].split(",")[0])
        best = max(best, ((int(m.group(1)) if m else 0) + last, st(fin), str(fin)))
print(best[0], best[2])
PY
)
test -n "${CK:-}" || { echo "no checkpoint under $OUT"; exit 1; }
REM=$(( TOTAL - N ))
(( REM > 0 )) || { echo "already at step $N of $TOTAL"; exit 0; }
echo "resume $CFG from global step $N ($CK): $REM more steps -> $OUT/resume_$N"
# an early-stopped run stays stopped (REPLICATION 2026-10-16p)
if [ -f "$OUT/adaptive/early_stop.json" ] && grep -q '"stop": true' "$OUT"/adaptive/early_stop.json "$OUT"/resume_*/adaptive/early_stop.json 2>/dev/null; then
  echo "early-stopped: not resuming"; exit 0
fi
export TICA_STEP_OFFSET=$N  # global step numbers for eval_on_save {step} and early stopping
exec bash scripts/slurm/tica_l3covert.sh "$CFG" --adapter "$CK" \
  --out "$OUT/resume_$N" --steps "$REM" "$@"
