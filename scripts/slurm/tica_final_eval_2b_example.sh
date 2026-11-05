#!/bin/bash
# Example final-eval driver used for the 2B v4d eval at step 1500 (10-31): builds the eval task list and submits
# tamia_mechanism_node.sh (SLOTS=8) + tica_drain_post.sh. Copy and edit RUN / ADAPTER / layers for another run.
# 2B v4d FINAL eval at global 1500 (10-31, user: "run the 2b final eval to see generalization"). One mechanism node,
# SLOTS 8 (2 per GPU; the 2B uses ~15-25 GB per eval process). Trained (step 1500) and base 2B:
#   held-out race (held-out concepts, as eval_on_save) | quality races train / held-out concepts, persistence,
#   GSM8K / MMLU capability, never-shown probe panel at 1500 | passive hide-and-solve (clean / hide-brief /
#   unrelated brief, base probes at 2B depths): GSM8K test, MBPP, HumanEval-JS, held-out SQL / JSON test splits,
#   and per trained concept its cap_free pool: own items (trained on) vs held-out external rows (never seen).
set -uo pipefail
cd /home/sneheel/chameleons
R=/scratch/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1025; C=final_v4d_tamia_2b
A=$R/$C/resume_1270/adaptive/step_230; G=1500; Q=$M/v4d2b_1500; mkdir -p $Q/hs
test -f $A/adapter_model.safetensors || { echo "no adapter $A"; exit 1; }
L2="--layers 3,6,9,12,15,17,18,20,21,23 --train-layers 6,9,12,17,20"
T=$M/v4d2b_drain.tsv; S=$M/v4d2b_smoke.tsv; : > $T; : > $S
printf 'hs\t%s\tbash scripts/slurm/tica_hide_solve.sh %s %s --adapter %s --task capfree --concept "SQL code" --pool-split held --gsm-n 4 --batch 4 %s\n' \
  "$Q/hs/smoke_capfree.json" $C "$Q/hs/smoke_capfree.json" $A "$L2" >> $S
printf 'rl\t%s\tbash scripts/slurm/tica_hide_solve.sh %s %s --adapter %s --task jsontest --gsm-n 4 --batch 4 %s\n' \
  "$Q/hs/smoke_jsontest.json" $C "$Q/hs/smoke_jsontest.json" $A "$L2" >> $S
# held-out race at 1500 (same args as eval_on_save)
printf 'rl\t%s\tbash scripts/slurm/tica_val.sh %s %s %s %s configs/tica/final_split_v5.json --topics heldout --episodes 32 --switch-prob 1.0 --turn-budget 5,8,8\n' \
  "$R/$C/val/step_$G.json" $C $A $G "$R/$C/val" >> $T
# quality races / persistence / capability (newest checkpoint = 1500) + panel at 1500; base references -> v4d2b_quality_base
PANEL_EVERY=$G bash scripts/slurm/tamia_queue_quality.sh $C | sed "s#$R/v4b_quality_base#$R/v4d2b_quality_base#g" | while IFS= read -r cmd; do
  out=$(echo "$cmd" | awk '{print $4}'); printf 'rl\t%s\t%s\n' "$out" "$cmd" >> $T
done
# passive hide-and-solve, trained + base
for who in p:--adapter\ $A b:--base; do
  tag=${who%%:*}; sel=${who#*:}
  for task in gsm8k mbpp jseval sqltest jsontest; do
    o=$Q/hs/${tag}_$task.json
    printf 'hs\t%s\tbash scripts/slurm/tica_hide_solve.sh %s %s %s --task %s --gsm-n 200 %s\n' "$o" $C "$o" "$sel" $task "$L2" >> $T
  done
  for c in "SQL code" "JSON data" "JavaScript code" "German language" "Japanese language" "Arabic language" "Hindi language" "all-caps text"; do
    for sp in own held; do
      o=$Q/hs/${tag}_capfree_$(echo "$c" | tr ' A-Z' '_a-z')_$sp.json
      printf 'hs\t%s\tbash scripts/slurm/tica_hide_solve.sh %s %s %s --task capfree --concept "%s" --pool-split %s --gsm-n 100 %s\n' "$o" $C "$o" "$sel" "$c" $sp "$L2" >> $T
    done
  done
done
echo "tasks: $(wc -l < $T)"
J=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=10:00:00 --job-name=v4d2b-eval \
     --export=ALL,NG=4,SLOTS=8 scripts/slurm/tamia_mechanism_node.sh $S $T)
echo "eval node $J"
P=$(sbatch --parsable --account=aip-rrabba --dependency=afterany:$J scripts/slurm/tica_drain_post.sh); echo "post $P"
