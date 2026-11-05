#!/bin/bash
# FINAL v2 evaluation suite (REPLICATION 2026-10-15e, standardized 10-16) for one checkpoint.
# Submits (no GPU here) and prints one line per job ("<manifest key> <job name> <job id>"); writes
# $OUT/manifest.json {key: {id, name, out_file}} (merged into an existing manifest, also written
# when the script exits early or a submission fails).
#
# Sections:
#   A. HEADLINE long single-concept races (--switch-prob 0, 24 turns, 32 ep) -> turns-to-beat
#      (beat_*, segbeat_*, deploy_records): held-out concepts x {linear, linear@24, bailey_ens_tok,
#      gdm_rollattn, nc_ens8_tok, mixed held-out kinds}, train concepts x linear, base control.
#   B. DOMAIN concepts (DOMAIN_DATA): long races --topics ood, linear + bailey_ens_tok + base.
#   C. short refit races (32 ep x 16 turns, switches) on HELD-OUT topics, one probe kind per job:
#      every held-out kind of the config (minus DROP) + 4 trained kinds; mixed held-out switch;
#      train concepts; falsified-verdict control; base-model races.
#   D. layer sweeps (held-out, train).
#   E. capability: KL + default benches; HELD-OUT capability (5 benches x 5 briefs, + base);
#      per-turn capability forks (vLLM).
#   F. anomaly detector: frozen race with an always-on OOD monitor (--ood-ref base / self).
#
# Usage: bash scripts/slurm/final_eval_suite.sh CONFIG ADAPTER_DIR TAG [kinds_subset]
#   kinds_subset: only the section-C short held-out races of these kinds (then exit).
# Env:
#   TICA_NEG_LABELS  concept-label file for negatives (default concept_labels_v3e.json; set to ""
#                    to disable). TICA_TRUNC_READS / TICA_BATCHED_FITS default 1 (exact speed-ups).
#   FIRST_PROBE      base|fork -> --first-probe on every tica_switch_eval race (default: config's).
#                    NOT passed to tica_frozen_race jobs (that script has no --first-probe).
#   DOMAIN_DATA      domain-concept data (default final_ood_v2.json; "" skips section B); its
#                    inputs map = ${DOMAIN_DATA%.json}_inputs.json (override: DOMAIN_INPUTS).
#   SKIP_BASE=1      skip every base-model job (base races + base capability); base results do
#                    not depend on the adapter, so reuse them across checkpoints.
#   DRY_RUN=1        print the jobs with fake ids instead of calling sbatch (manifest still written).
#   OUT_DIR          override the output dir (default <config out>/eval_$TAG).
set -euo pipefail
CFG=$1; AD=$2; TAG=$3; ONLY=${4:-}
cd /home/sneheel/chameleons
PYQ=/home/sneheel/chameleons-qwen35/.venv/bin/python
RAW=/scratch/sneheel/next_chameleons_raw_cache/tica
OUT=${OUT_DIR:-$(PYTHONPATH=src $PYQ -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').out)")/eval_$TAG}
HELD=$(PYTHONPATH=src $PYQ -c "
from next_chameleons.tica.config import load_run_config as l; print(' '.join(l('$CFG').final.heldout_refit_kinds))")
# Trimmed 10-16 (user: drop redundant types, keep the generalisation axes): rlobf_max / rlobf_attn_max
# fail the gate even on easy synthetic data (bench 5808522), meandiff is weak and crashes the token
# path, rollmax duplicates the rolling-window family (gdm, mck_rollmean40), mck_softmax duplicates
# mck_attnlite. Kept axes: depth 20 / 24 / 28 / 30, combo maxpool@24, architecture gdm / mck_attnlite /
# mck_rollmean40, aggregation ens8 linear + MLP / nc_ens8_tok / bailey mean + max / spread5.
DROP=" rlobf_max rlobf_attn_max meandiff rollmax mck_softmax "
HELD=$(for k in $HELD; do [[ "$DROP" == *" $k "* ]] || echo -n "$k "; done)
KINDS=${ONLY:-"$HELD linear linear@22 ens4_linear ensdeep4_linear"}
HCSV=$(echo $HELD | tr ' ' ',')
mkdir -p "$OUT" "$OUT/long"

# ---- environment exported to every job (sbatch --export=ALL) ----
# exact speed-ups (REPLICATION 2026-10-16c)
export TICA_TRUNC_READS=${TICA_TRUNC_READS:-1} TICA_BATCHED_FITS=${TICA_BATCHED_FITS:-1}
# "-" (not ":-"): an explicitly empty TICA_NEG_LABELS stays empty = label filtering off
export TICA_NEG_LABELS=${TICA_NEG_LABELS-$RAW/concept_labels_v3e.json}  # 10-16o: v3 + eval superset
FIRST_PROBE=${FIRST_PROBE:-}
DOMAIN_DATA=${DOMAIN_DATA-$RAW/final_ood_v2.json}
DOMAIN_INPUTS=${DOMAIN_INPUTS:-${DOMAIN_DATA%.json}_inputs.json}
SKIP_BASE=${SKIP_BASE:-0}
DRY_RUN=${DRY_RUN:-0}
FP=(); [ -n "$FIRST_PROBE" ] && FP=(--first-probe "$FIRST_PROBE")
echo "# OUT=$OUT TICA_NEG_LABELS=${TICA_NEG_LABELS:-<off>} FIRST_PROBE=${FIRST_PROBE:-<config>}" \
  "DOMAIN_DATA=${DOMAIN_DATA:-<off>} SKIP_BASE=$SKIP_BASE DRY_RUN=$DRY_RUN"

# ---- submission helper + manifest ----
SUBS=$(mktemp)
write_manifest() {
  [ -s "$SUBS" ] && $PYQ - "$SUBS" "$OUT/manifest.json" <<'EOF'
import json, sys
from pathlib import Path
subs, man = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads(man.read_text()) if man.exists() else {}
for line in subs.read_text().splitlines():
    key, name, jid, out = line.split("\t")
    m[key] = {"id": jid, "name": name, "out_file": out}
man.write_text(json.dumps(m, indent=1, sort_keys=True))
print(f"# manifest: {man} ({len(m)} jobs)")
EOF
  rm -f "$SUBS"
}
trap write_manifest EXIT
NFAKE=0
# job KEY OUT_FILE TIME SCRIPT [args...]: submits SCRIPT CONFIG OUT_FILE args... as fe-$TAG-KEY
job() {
  local key=$1 out=$2 t=$3 script=$4 name="fe-$TAG-$1" jid; shift 4
  if [ "$DRY_RUN" = 1 ]; then
    NFAKE=$((NFAKE + 1)); jid=dry$NFAKE
    echo "  [dry] --time=$t TICA_VLLM=${TICA_VLLM:-} $script $CFG $out $*" >&2
  else
    jid=$(sbatch --parsable --export=ALL --time="$t" --mem=32G -J "$name" "$script" "$CFG" "$out" "$@")
  fi
  printf '%s\t%s\t%s\t%s\n' "$key" "$name" "$jid" "$out" >> "$SUBS"
  echo "$key $name $jid"
}
SW=scripts/slurm/tica_switch_eval.sh
FR=scripts/slurm/tica_frozen_race.sh
LONG=(--switch-prob 0 --turns 24 --episodes 32 "${FP[@]}")
SHORT=(--episodes 32 --turns 16 "${FP[@]}")

# ======== C (subset mode). short held-out races, one probe kind per job ========
# sized from sacct (10-16): races use 11-15 GB host RAM, 13-17 GB GPU; linear-type races ~1 h, gdm
# 3.2 h. <= 3 h requests start sooner (more partitions / backfill)
short_races() {
  for k in $KINDS; do
    local t=03:00:00; [[ $k == gdm* || $k == mck_* || $k == rlobf* ]] && t=06:00:00
    job "race_heldout_${k//@/_at}" "$OUT/race_heldout_${k//@/_at}.json" $t $SW \
      --adapter "$AD" --topics heldout --kind "$k" "${SHORT[@]}"
  done
}
if [ -n "$ONLY" ]; then short_races; exit 0; fi

# ======== A. HEADLINE: long single-concept races (turns-to-beat) ========
# 24-turn races cost ~1.5x the 16-turn ones: 05:00 (gdm 08:00). mixed draws from all held-out kinds
# (incl. gdm / mck_*) and may retry 3 kinds per refit -> 08:00 too.
for k in linear linear@24 bailey_ens_tok gdm_rollattn nc_ens8_tok; do
  t=05:00:00; [[ $k == gdm* ]] && t=08:00:00
  job "long_heldout_${k//@/_at}" "$OUT/long/heldout_${k//@/_at}.json" $t $SW \
    --adapter "$AD" --topics heldout --kind "$k" "${LONG[@]}"
done
# no quality fallback (10-15g): held-out kinds drawn per refit; a refit failing its gate switches to
# another HELD-OUT kind (never a trained one), up to 3 more; every attempt logged (refits.jsonl)
job long_heldout_mixed "$OUT/long/heldout_mixed.json" 08:00:00 $SW --adapter "$AD" \
  --topics heldout --kind mixed --refit-switch 3 --switch-kinds "$HCSV" "${LONG[@]}"
job long_train_linear "$OUT/long/train_linear.json" 05:00:00 $SW --adapter "$AD" \
  --topics train --kind linear "${LONG[@]}"
[ "$SKIP_BASE" = 1 ] || job long_heldout_linear_BASE "$OUT/long/heldout_linear_BASE.json" 05:00:00 \
  $SW --base --topics heldout --kind linear "${LONG[@]}"

# ======== B. DOMAIN concepts (long races on the ood split of DOMAIN_DATA) ========
if [ -n "$DOMAIN_DATA" ]; then
  DOM=(--data "$DOMAIN_DATA" --inputs "$DOMAIN_INPUTS" --topics ood "${LONG[@]}")
  for k in linear bailey_ens_tok; do
    job "long_ood_$k" "$OUT/long/ood_$k.json" 05:00:00 $SW --adapter "$AD" --kind "$k" "${DOM[@]}"
  done
  [ "$SKIP_BASE" = 1 ] || job long_ood_linear_BASE "$OUT/long/ood_linear_BASE.json" 05:00:00 \
    $SW --base --kind linear "${DOM[@]}"
fi

# ======== C. short refit races (with topic switches) ========
short_races
job race_heldout_mixedheldout_switch "$OUT/race_heldout_mixedheldout_switch.json" 06:00:00 $SW \
  --adapter "$AD" --topics heldout --kind mixed --refit-switch 3 --switch-kinds "$HCSV" \
  "${SHORT[@]}"
for k in linear ensdeep4_linear; do
  job "race_train_$k" "$OUT/race_train_$k.json" 03:00:00 $SW --adapter "$AD" --topics train \
    --kind "$k" "${SHORT[@]}"
done
job race_heldout_linear_falsify "$OUT/race_heldout_linear_falsify.json" 03:00:00 $SW \
  --adapter "$AD" --topics heldout --kind linear --falsify "${SHORT[@]}"
if [ "$SKIP_BASE" != 1 ]; then
  for k in linear bailey_ens_tok; do
    job "race_heldout_${k}_BASE" "$OUT/race_heldout_${k}_BASE.json" 03:00:00 $SW --base \
      --topics heldout --kind "$k" "${SHORT[@]}"
  done
fi

# ======== D. layer sweeps ========
for sp in heldout train; do
  job "layer_sweep_$sp" "$OUT/layer_sweep_$sp.json" 01:00:00 scripts/slurm/tica_layer_sweep.sh \
    --adapter "$AD" --split $sp
done

# ======== E. capability ========
CAP=scripts/slurm/tica_capability.sh
job capability "$OUT/capability.json" 01:00:00 $CAP --adapter "$AD" --kl-n 64
HCAP=(--benches arc,mmlu_stem,math500,mbpp,gsm8k --briefs physics,code,math,sports,html)
job capability_heldout "$OUT/capability_heldout.json" 06:00:00 $CAP --adapter "$AD" "${HCAP[@]}"
[ "$SKIP_BASE" = 1 ] || job capability_heldout_BASE "$OUT/capability_heldout_BASE.json" 06:00:00 \
  $CAP --base "${HCAP[@]}"
TICA_VLLM=1 job capturn "$OUT/capturn.json" 06:00:00 $FR --adapter "$AD" --cumulative \
  --fork-pos 96 --fork-other 96 --fork-free 48 --fork-np 24 --incremental --cap-base --episodes 8 \
  --turns 24 --cap-turns 0,1,2,3,5,8,12,16,23 --cap-n 32

# ======== F. anomaly detector (always-on OOD monitor; reference = base model / the model itself) ===
for ref in base self; do
  job "ood_$ref" "$OUT/ood_$ref.json" 06:00:00 $FR --adapter "$AD" --cumulative --ood \
    --ood-ref $ref --episodes 16 --turns 16
done
