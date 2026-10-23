#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=v4c_logclip_chain
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Self-perpetuating 12 h chunks of the Narval logclip run on the v4c recipe (final_v4c_4b, from global 600; main repo,
# DP2). Each chunk: newest checkpoint WITH trainer_state.pt over resume_*/adaptive/{step_k,final} -> global N; queues
# its own successor (afterany) before training, so the chain continues after the 12 h limit with no one watching.
# Guards: N >= 1200 -> done, no successor. No progress since the previous chunk started (same N): first time, retry
# with 8 episodes per step (OOM fallback); a second time -> stop the chain (state in $R/chain_state).
set -euo pipefail
R=/scratch/sneheel/next_chameleons_artifacts/tica/runs/pilot_v4b_2026-10-21/logclip
SELF=/home/sneheel/chameleons/scripts/slurm/v4c_logclip_chain.sh
read -r N CK < <(python3 - "$R" <<'PY'
import re, sys
from pathlib import Path
out, best = Path(sys.argv[1]), (0, "")
for d in out.glob("resume_*/adaptive/step_*"):
    if not (d / "trainer_state.pt").exists():  # eval-only save (adaptive.eval_every): adapter, no state
        continue
    off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
    best = max(best, (off + int(d.name[5:]), str(d)))
for fin in out.glob("resume_*/adaptive/final"):
    log = fin.parent / "train_log.jsonl"
    if (fin / "trainer_state.pt").exists() and log.exists() and log.read_text().strip():
        off = int(re.match(r"resume_(\d+)", fin.parent.parent.name).group(1))
        last = int(log.read_text().strip().splitlines()[-1].split('"step": ')[1].split(",")[0])
        best = max(best, (off + last, str(fin)))
print(*best)
PY
)
test -n "${CK:-}" || { echo "no checkpoint under $R"; exit 1; }
# Trend evals, once per 200 global steps (and at the end) on the newest full checkpoint: GSM8K 768 tok (+ math
# brief), the fixed race (train concepts, 16 turns, no switches) and the full training protocol race (24 turns,
# switches + budget), 96 episodes each (48-episode races spread ~0.15 on k2 across seeds). Races use final_v4b_4b =
# the settings of the step 150-500 races (comparable trend).
evals() {  # global step, checkpoint
  local g=$1 ck=$2 D=/scratch/sneheel/next_chameleons_artifacts/tica/diag/v4c_trend
  mkdir -p "$D"; grep -qx "$g" "$D/done" 2>/dev/null && return 0
  cd /home/sneheel/chameleons
  sbatch --parsable --time=02:00:00 -J cap_v4c_$g scripts/slurm/tica_capability.sh final_v4c_4b "$D/cap_$g.json" \
    --adapter "$ck" --benches gsm8k --briefs math --gsm-max-new-tokens 768
  sbatch --parsable --time=03:00:00 -J race_v4c_${g}_fixed scripts/slurm/tica_switch_eval.sh final_v4b_4b \
    "$D/race_${g}_fixed.json" --adapter "$ck" --topics train --episodes 96 --turns 16 --switch-prob 0
  sbatch --parsable --time=04:00:00 -J race_v4c_${g}_full scripts/slurm/tica_switch_eval.sh final_v4b_4b \
    "$D/race_${g}_full.json" --adapter "$ck" --topics train --episodes 96 --turns 24 --switch-prob 1.0 \
    --min-seg 8 --turn-budget 5,8,8
  echo "$g" >> "$D/done"; echo "trend evals queued for global $g"
}
while read -r g ck; do evals "$g" "$ck"; done < <(python3 - "$R" <<'PY'
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
for d in sorted(out.glob("resume_*/adaptive/*")):
    m = re.match(r"resume_(\d+)", d.parent.parent.name)
    if not m or not (d / "adapter_model.safetensors").exists():
        continue
    off = int(m.group(1))
    if d.name.startswith("step_"):
        g = off + int(d.name[5:])
    elif d.name == "final" and (d.parent / "train_log.jsonl").exists():
        g = off + int((d.parent / "train_log.jsonl").read_text().strip().splitlines()[-1]
                      .split('"step": ')[1].split(",")[0])
    else:
        continue
    if g > 600 and (g % 200 == 0 or g >= 1200):
        print(g, d)
PY
)
cd /home/sneheel/chameleons
REM=$(( 1200 - N )); (( REM > 0 )) || { echo "done at $N: chain ends"; exit 0; }
ST=$R/chain_state; touch "$ST"
read -r LASTN TRIES EPS _ < <(cat "$ST" 2>/dev/null; echo "-1 0 16") || true
EPS=${EPS:-16}
if [ "$LASTN" = "$N" ]; then
  TRIES=$(( TRIES + 1 ))
  if (( TRIES >= 2 )); then echo "no progress from $N twice: chain STOPPED (see $ST)"; echo "$N $TRIES $EPS STOPPED" > "$ST"; exit 1; fi
  EPS=8; echo "no progress from $N since the last chunk: retry with episodes_per_step 8"
else
  TRIES=0
fi
echo "$N $TRIES $EPS" > "$ST"
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SELF")
echo "chunk from global $N ($CK): $REM steps -> $R/resume_$N; episodes/step $EPS; successor $NEXT"
# Stop 10 min before the job's wall-clock end, saving "final" with trainer state (no steps lost to the limit).
END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
export TICA_DP=2 TICA_STEP_OFFSET=$N TICA_DEADLINE=$(( END - 600 ))
cd /home/sneheel/chameleons
exec bash scripts/slurm/tica_l3covert.sh final_v4c_4b --adapter "$CK" --out "$R/resume_$N" --steps "$REM" \
  --set "train.episodes_per_step=$EPS" \
  --set "rl.adaptive.eval_on_save=bash scripts/slurm/tica_val_local.sh final_v4c_4b {adapter} {step} $R/val configs/tica/final_split_v5.json --topics heldout --episodes 32 --switch-prob 1.0 --turn-budget 5,8,8"
