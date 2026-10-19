# TICA final-run ledger (runs, branches, evals) as of 2026-10-18

## Conventions

- **Artifact root (Narval):** every path below starts with `$A = /scratch/sneheel/next_chameleons_artifacts/tica/runs/`.
  On TamIA, `scripts/tica_sync_v4.sh --with-v3b` copies the adapters and aggregate eval JSONs.
- **Configs:** each config is `configs/tica/<config>.yaml`, mirrored in `src/next_chameleons/builtin_configs/tica/`.
- **Branches:** run with `TICA_STEP_OFFSET=<offset>` and the config's `seed` set so that seed + offset equals the
  parent's draws.
- **Run layout:**
  - A run writes `adaptive/` (`step_N/`, `final/`, `train_log.jsonl`, `turns.jsonl`, `refits.jsonl`).
  - Resume chunks write `resume_<global step>/adaptive/`.
  - `scripts/tica_run_report.py --run <dir> [--base-offset N]` stitches the chunks together.
- **Log:** "Logged" is the `REPLICATION.md` entry.

## Training runs

| Run (config) | What it is / changed | Start | Steps | Jobs | Status | Logged |
|---|---|---|---|---|---|---|
| final_v2_4b | First final dodger: topics + NC rated concepts, fixed base turn-0 probe (pinned), probe zoo, KL + anchor + cap_gen | base | 200 | 5711305 (+ 5711307/13/18) | done | 10-15e, 10-16a |
| final_v3_4b | v3 data, turn-0 fork, family-first sampler, label-cleaned negatives | base | 140 | 5878496 | FAILED (early-stopped, never trained) | 10-16p, 10-16q |
| final_v3_bisect_base_4b | v3 with first_probe base | base | 12 | 5913052 | cancelled (stopped too early, see 10-17a) | 10-16r |
| final_v2_rerun_4b / final_v2_v3data_4b | Code-drift control / v2 + v3 data | base | 40 | 5915088 / 5915089 | done | 10-17b |
| final_v2_v3data_forkA_4b / final_v3_noforkB_4b | Halves A (turn-0 fork) / B (rest of v3) | base | 40 | 5921351 / 5921369 | done | 10-17b..e |
| final_v3_noforkB1_4b / B2 | B + uniform concepts / B − neg_labels | base | 40 | 5928081 / 5928135 | done | 10-17d, 10-17g |
| final_v2_v3data_neg_4b | v2 + v3 data + label-cleaned negatives | base | 40 | 5930832 | done | 10-17e, 10-17g |
| **final_v3b_4b** | v3 + switch_fork, hide_from_turn 0, tagged layout, off turns 0.15, uniform-concept ramp, LR warmup to 3e-4 then cosine to 1e-4 by step 200 | base | 0-400 | 5948570 (+ 5948572-74) | done | 10-17j .. 10-18a |
| final_v3b_lrconst_4b | v3b at constant LR 1e-4 | base | 100 | 5948575, 5969022 | done (3e-4 schedule better) | 10-17k..p |
| final_v3b_nooff_4b | v3b without off turns | base | 40-50 | 5964114 | done (off turns slow the start) | 10-17l, 10-17p |
| final_v3b_offfix_4b / offctl_4b | off_match_weight 2.0 / 0, from s80 | s80, offset 80 | 40 | 5991858 / 5991872 | done (fix works; not adopted in v4) | 10-17o, 10-17p |
| final_v3b_retryctl_4b | unchanged | s360, offset 360 | 60 | 6174722 | done | 10-17ad, 10-18b |
| final_v3b_retryA_4b | improve_weight 1 + improve_margin 0.1 | s360, offset 360 | 60 | 6174723 | done | 10-18b, 10-18g |
| final_v3b_retryB_4b | retry_hide_weight 3.0 | s360, offset 360 | 60 | 6174724 | done | 10-18b, 10-18g |
| final_v3b_retryL_4b | retry_hide_ladder [2, 3, 4] (seed −360) | s360, offset 360 | 60 | 6199432 | done (**winner**) | 10-18d, 10-18g, 10-18i |
| final_v3b_lr2e4_4b | held LR 2e-4 (seed −360) | s360, offset 360 | 60 | 6199433 | done (**ruled out**) | 10-18d, 10-18i |
| final_v3b_ext_4b | retryB continued to 1000 | retryB s420 | - | 6198833-36 | CANCELLED (user: clean baseline first) | 10-18d |
| **final_v3b_4b continuation** | v3b unchanged, steps 400 → 1000, val race every 50 steps, read batch 16 | v3b s400 | 400-1000 | 6198908 (400 → ~527, 12 h timeout, normal), 6198909 (from s520, runs to its time limit, ~step 640); 6198911, 6198912, 6214592, 6315437 cancelled | **stopped 10-19** (user: saturated; held-out flat since s340) | 10-18d, 10-18i, 10-18j, 10-19b |
| free-generation races (10-19b) | s500 and base, held-out, 56 ep × 24 turns, mixed kinds, no switches, `--answers free`, ON and no-brief | v3b s500 / base | - | 6338430 (s500 ON), 6338431 (s500 no-brief), 6338432 (base ON), 6338433 (base no-brief) → `final_v3b_4b/freegen/`; then `tica_free_judge.sh` (27B) on the four JSONL | running | 10-19b |
| turn-budget reference race (10-19b) | v3b s500, held-out, 56 ep × 24 turns, budget 5,8,8, switch prob 1.0 | v3b s500 | - | 6338434 → `sanity_full/s500_budget_heldout.json` | running | 10-19b |
| rollout-reuse branches (10-18j): final_v3b_reuse{1,2,3}_4b | `updates_per_rollout` 1 (control) / 2 / 3: K optimizer updates per rollout (f0db642; K = 1 bit-identical). Compare at equal steps and equal wall-clock, then held-out races | v3b s500 (`resume_400/adaptive/step_100`), offset 500, seed −500 (same draws) | 50 / 40 / 40 | 6315432 / 6315433 / 6315434 | done: no gain, retry pass 0.40 / 0.37 / 0.34. Held-out races on finals (reuse1 = s550, reuse2/3 = s540): 6333553 / 6333610 / 6333620 → `final_v3b_4b/sanity_full/reuse{K}_final_heldout.json` | 10-18j, 10-19a |
| final_v4_4b | DRAFT: split v5, rank 32, 24 turns, 1200 steps, ladder, LR 1e-4 + cooldown 1e-5, flat per-concept curriculum (share 0.7 after 50 steps), turn-0 fork from 150, family_neg_frac 0.5 | base | 1200 (to revisit) | - | **not launched** (TamIA) | 10-18d, f, i; `docs/tica-v4-recipe.md` |

Chunk dirs for the v3b baseline:
- `$A/final_v3b_4b/adaptive` (0-260)
- `resume_260/adaptive` (260-400)
- `resume_400/adaptive` (400-520; `step_100` = global 500)
- `resume_520/adaptive` (520 onward)

Each 12 h chunk now covers ~110-120 steps at ~6 min/step. The 4 chunks after 6198908 end near step 980, so a 6th
chunk (6315437) is queued to reach 1000.

## Evaluation jobs (v3b unless noted)

| Eval | Checkpoint | Job(s) | Output (under `$A/final_v3b_4b/` unless noted) | Logged |
|---|---|---|---|---|
| Val race (dev) every 20 → 50 steps | 20..550 | eval_on_save (`tica_val.sh`) | `val/step_N.json`, `val/race_step_N.json` | throughout |
| Held-out + dev + safety sanity races | s140, s300, s340 | - | `sanity_full/s{140,300,340}_{heldout,dev,safety}.json`, `base_*` | 10-17x, 10-17y |
| Held-out probe-type suite + falsify + long races | s340 | `final_eval_suite.sh` | `eval_s340/` (`long/` = matched v2 races) | 10-18a, 10-17ad |
| Held-out race, retry branches | ctl / A / B / L / lr2e4 @420 | 6198840 / 6198915 / 6198839 / 6214446 / 6214447 | `sanity_full/<arm>420_heldout.json` | 10-18g, 10-18i |
| Held-out race | **s500** | 6258629 | `sanity_full/s500_heldout.json` | 10-18j |
| Emotion (held-out family) | s360 vs base | 6200585 / 6200586 | `sanity_full/{s360,base}_emotion.json` | 10-18g |
| Probe panel | s400 vs base | 6200754 / 6200755 | `panel/{s400,base}_heldout.json` | 10-18h |
| Supervised evasion detector | s400 / s140 / base | 6202634 / 6202637 / 6202639 | `evdet/` (`detector.json`, features `*_feats.pt`) | 10-18h |
| Anomaly (OOD) detector | s140 | 6049253 / 6049342 | `eval_s140/ood_{base,self}.json` | 10-17r |
| Anomaly (OOD) detector | s400 | 6202553 / 6202554 | `eval_s400/ood_{base,self}.json` | 10-18g |
| Anomaly re-check, v2 final | v2 | 6056855 | `$A/final_v2_4b/eval_recheck/` | 10-17t |
| Free generation / think-hide | s160, s400 | -, 6202552 | `$A/think_hide/` | 10-17v, 10-18g |
| Thinking-mode accuracy | base, v2, s140, s300 | 6065707 / 6065841 / 6065885 (+ s300) | `$A/thinking_eval/*.json` | 10-17s/t |
| Capability (held-out benches) | s300 vs base | - | `eval_s300/capability*.json` | 10-17y, 10-17aa |
| 27B negative labels, v5 data | - | 6200733-36 | `$RAW/tica/concept_labels_v5data_27b.json` | 10-18f |

All of the eval outputs above are summarised, aggregates only, in `results/tica/snapshot_2026-10-19.json`.
Regenerate it with:

```bash
R=$A
python scripts/tica_results_snapshot.py $R/final_v3b_4b \
  $R/final_v3b_retry{ctl,A,B,L}_4b:360 $R/final_v3b_lr2e4_4b:360 $R/final_v2_4b
```
