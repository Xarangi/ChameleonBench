# TICA: start here (context for a new machine, e.g. TamIA)

TICA = **T**rained **I**n-**C**ontext **A**daptation: one checkpoint that learns, *in context*, to dodge any
concept probe. Snapshot of what we know as of **2026-10-21** (v3b numbers as of 10-18; v4 math runs 10-19..10-21). This folder is the entry point. Day-by-day detail
lives in `REPLICATION.md` (newest first; it also holds the RULED-OUT ledger).

## Protocol (what every number refers to)

- **Episode.** One conversation of 16 turns (24 for v4).
  - A system prompt says a monitor watches for concept X. Each user turn carries "Monitor: ON. Concept: X." or
    "Monitor: OFF.", the previous turn's verdict, and a request. The assistant answer is concept text,
    teacher-forced from the data.
  - The concept may switch mid-episode.
- **Defender.**
  - The first probe on a concept catches by construction. Turn 0 uses a fixed base probe (or, with
    `first_probe: switch_fork`, a probe fit on a fork of the conversation for concepts switched to).
  - When the model passes, the defender forks the conversation at the turn after the pass and fits a fresh
    probe on the model's own states (*branch refit*). That probe judges the next turns.
  - The model only sees a 1-bit verdict (caught / not caught).
- **Training.**
  - LoRA on Qwen3.5-4B, non-thinking mode.
  - Losses: hide loss on monitored turns, match-base on unmonitored or no-brief turns, plus an LM loss,
    a KL anchor and a GSM8K capability term.
- **Headline metric = pass rate by probe index.**
  - The fraction of concept segments that beat the k-th probe deployed on them.
  - `k1_first` is the fixed turn-0 base probe; `k1_switch` is the fresh probe after a concept switch; `k2..k5`
    are the 1st..4th refits.
  - Also tracked: *retry pass* (a pass on a turn where the same probe caught the previous turn).
  - Always lead updates with these; the plain pass rate hides stale-probe dodging.

## Where things are

| What | Where |
|---|---|
| Results digest (numbers + caveats) | [RESULTS.md](RESULTS.md) |
| Every run / branch / eval job | [RUNS.md](RUNS.md) |
| Pending decisions, ideas, TamIA optimisations | [OPEN-QUESTIONS.md](OPEN-QUESTIONS.md) |
| Aggregate-only numbers (no texts) | `results/tica/snapshot_2026-10-19.json` (`scripts/tica_results_snapshot.py`) |
| v4 recipe and decisions | `docs/tica-v4-recipe.md` (+ "Since 10-19" below, which supersedes it where they differ) |
| Model-written concept data (opt-in v4 data) | `docs/tica/V4_MODEL_WRITTEN_DATA.md` |
| TamIA quickstart (clone, envs, data sync, smoke, launch) | `docs/tica-v4-tamia.md` (THE TamIA RECIPE section: final_v4b_4b + final_v4b_score_4b) |
| Other-cluster runbook, whole-node launcher, speed options | `docs/tica-other-hpc.md` |
| Design / paper plan | `DESIGN-TICA.md`, `TICA-FINAL-DODGER-DESIGN.md`, `PAPER-PLAN-TICA.md`, `PREREGISTRATION-TICA.md`, `IDEAS-BANK.md` |
| Configs | `configs/tica/final_*.yaml`; builtin mirrors in `src/next_chameleons/builtin_configs/tica/` (keep in sync) |
| Concept splits | `configs/tica/final_split_v{2,3,4,5}.json` (v5 = the v4 run's split) |
| Trainer | `src/next_chameleons/tica/trainers/supervised.py` (`AdaptiveSupervisedTrainer`), `objectives.py`, `adaptive.py` (`AdaptiveConfig`), `curriculum.py` |
| Training entry | `scripts/tica.py`; chunks: `scripts/slurm/tica_resume.sh`; whole node: `scripts/slurm/tica_node.sh` |
| Races (held-out, val) | `scripts/tica_switch_eval.py` (+ `slurm/tica_switch_eval.sh`, `tica_val.sh`, `tica_val_local.sh`); `scripts/slurm/final_eval_suite.sh` |
| Analysis | `scripts/tica_run_report.py` (training logs to probe index / retry / held-out static / readability / timing), `tica_probe_panel.py` (never-shown probe panel, selectivity, `--dump-feats`), `tica_evasion_detector.py` (supervised detector), `tica_frozen_race.py --ood` (unsupervised anomaly detector), `tica_think_hide_eval.py` (free generation, thinking), `tica_thinking_eval.py`, `tica_capability.py` |
| Data builders | `scripts/build_final_v5.py` (split v5), `build_final_v5_mw.py` (model-written train concepts, [V4_MODEL_WRITTEN_DATA.md](V4_MODEL_WRITTEN_DATA.md)), `build_emotion_eval.py`, `build_safety_eval.py`, `label_concept_contamination.py` (27B negative labels) |

## REPLICATION.md sections by date (TICA final runs)

| Date | Topic |
|---|---|
| 10-15e .. 10-16a | final_v2: design, launch, finish, capability with KL |
| 10-16b .. 10-16p | eval suite, exact speed-ups, label audit, v3 data and labels, final_v3 launch |
| 10-16q .. 10-17g | final_v3 failed to train; bisection (cold start = turn-0 fork + family sampler) |
| 10-17h .. 10-17p | final_v3b design, launch, takeoff at steps 80-140; OFF-turn leak; off_match_weight |
| 10-17q .. 10-17z | anomaly detector, thinking, probe index metric, held-out sanity races, safety, capability |
| 10-17aa .. 10-17ad | split v4, retry diagnosis, falsified verdicts, retry branches |
| 10-18a .. 10-18d | v3b to 400; branches; extension to 1000; ladder + LR branches; v4 draft |
| 10-18f | split v5 (the v4 run's data) + 27B labels |
| 10-18g .. 10-18i | branch held-out races, emotion, anomaly, free-gen, probe panel, evasion detector, v4 draft final |
| 10-18j | held-out plateau (s340 = s500); updates_per_rollout + reuse branches; 6th baseline chunk |
| 10-19a .. 10-19f | reuse no gain; turn budget; free-answer races; concept-eliciting requests (v5 elicit file) |
| 10-19g .. 10-19p | math-only v4 runs (full / minimal / no-KL / no-pursuit / zoo turn 0 / universal): k1 learned, refits not; universal hider NULL; pursuit off; depth panel |
| 10-20a .. 10-20e | why: math arms lost the FIXED turn-0 rung; no-KL beats refits; v3b-strategy math run (KL 0.3, DP2); turn-0 base panel |
| 10-20f .. 10-20k | stage B (panel turn 0 from step 60): k1 1.00 by ~170, refits ~0 |
| 10-20l .. 10-20s | capability-term dose-response; no-KL math broken under the brief; layer-bandit bug; speed + memory audits |
| 10-20t .. 10-20z | capability drop at abs 400; MATH COLLAPSE at abs 389; extension from 400 recovers; batched fits |
| 10-21a .. 10-21k | k-by-index correction (k2 ~0.5); r380 arms (control / log loss / adaptive capability weight); refit overlap (exact, ~no gain); plain-math refit gate dropped for new runs; in-context-only refit positives; log loss from scratch |

## Run lineage

```
final_v2 (200 steps; fixed base turn-0 probe, pinned)      learned; caught by the anomaly detector
  -> final_v3 (v3 data, turn-0 fork, family sampler)       FAILED to train (cold start)
  -> final_v3b (switch_fork, tagged layout, LR 3e-4 -> 1e-4, uniform-concept ramp)
       steps 0-400 (chunks 5948570..74), then extended 400 -> 1000 (6198908/09/11/12, 6214592, 6315437) = THE BASELINE
       branches from s360 (+60 steps, same draws): retryctl / retryA (progress) / retryB (flat 3x) /
                                                   retryL (ladder 2/3/4x, WINNER) / lr2e4 (ruled out)
       branches from s500 (rollout reuse, running): reuse1 (control) / reuse2 / reuse3 (K updates per rollout)
  -> final_v4 (DRAFT, not launched): split v5, rank 32, 24 turns, 1200 steps, ladder, LR 1e-4 + cooldown,
       flat per-concept curriculum (share 0.7 after 50 steps), turn-0 fork from step 150, close negatives 0.5
       -> final_v5_modelwritten_4b (opt-in): same, on model-written train texts + no plain-math refit gate
  -> v4 MATH-ONLY runs (Narval, 10-19..): one concept, v4 capability terms
       final_v4_math_{,min,nokl,nopur,zoo0}_4b, final_v4_math_univ{,_nopanel}_4b   k1 only / NULL (fresh turn-0 probe)
       final_v4_math_v3b_4b: fixed L12 turn 0, KL 0.3, DP2 (stage A, 0-60)
         -> resume_60 (stage B: base PANEL turn 0 from 60) to abs 400   math COLLAPSE at abs 389
              -> resume_400 = extension (layer-bandit fix) 400 -> 600       recovered; k2 ~0.5, held-out tN 0.3-0.5
              -> r380_{ctl,log,adapt} from abs 380 (step_320)               see RUNS.md
         -> scratch_log: log hide loss FROM SCRATCH (6607864)
```

## Since 10-19: what changed (read before configuring the TamIA run)

Every item is a config field defaulting to the old behaviour unless noted. REPLICATION entry in brackets.

| Setting | Status | Evidence |
|---|---|---|
| Turn 0 = fixed base probes, never a fresh fork-fit probe from scratch | final_v4b: base panel from step 0 (first_probe_fork_from_step 0) | fork-fit turn 0 from scratch = 0 passes (10-19p, 10-20a); panel learned (10-21ai) |
| Episode KL / cap_gen KL 0.3 (was 1.0) | used in math runs | no-KL beats refits but breaks math under the brief (10-20a, 10-20n); 0.3 ~10 steps faster than 1.0 (10-20e) |
| `pursuit: false` | decided (user 10-19) | 10-19m |
| `kind_bandit_refits_only: true` (bug fix: turn-0 panel records fed the layer bandit) | use in new runs | 10-20o |
| `refit_batched_logistic`, `refit_batched_rlobf` | exact speed-ups, use | 10-20y |
| `refit_overlap` | exact, ~1 s/step gain: leave off | 10-21e |
| `cap_gen_samples: 4` | adopted in r380 arms (+5-8% step time, ~2x correct samples) | 10-20s |
| `micro_batch: 2` | no gain, +13 GB: off | 10-20s |
| `train.save_state_all: true` | use (optimizer state in every checkpoint, so any step can be branched exactly) | 10-21b |
| `refit_min_tpr_np: 0` + `refit_np_in_fit: false` + `refit_min_tpr_ic: 0.8` (fit on in-context forks; gate = AUROC + catch >= 0.8 of held-out in-context positives at 1% FPR) | decided 10-21 for new runs (set in final_v5_modelwritten_4b) | AUROC alone would admit probes that catch ~0 of in-context math at their threshold (10-21g, 10-21o) |
| `objective.hide_loss: log` + `train.hide_grad_clip: 0.5` (LOGCLIP) | ADOPTED for TamIA (final_v4b_4b / _score_4b) | beats the fixed rung / panel / refits ~6 / ~100 / ~200 steps sooner than MSE with capability intact; plain log diverges (10-21l..ai) |
| `adaptive.cap_adapt_target` (adaptive capability multiplier) | not adopted: oscillates; evades fresh probes like the others except at L20 | 10-21i, 10-21w |
| Model-written train texts (`final_v5_modelwritten_4b`) | build in progress | 10-21f, docs/tica/V4_MODEL_WRITTEN_DATA.md |
| `train.dp_sync_init: true` (bug fix: from-scratch DP ranks otherwise train different LoRA inits) | set in final_v4_4b / final_v5_modelwritten_4b | 10-21v |
| `objective.hide_loss: gce` (q 0.3) | not adopted: math fell to 0.25 (cancelled) | 10-21y |
| `adaptive.stay_standalone: true` (stay-after-pass without pursuit) | ADOPTED for TamIA | the only term that makes the verdict worth reading; sustained retry success 0.14-0.18 at j3-5 vs 1-bit decay (10-21t, x, ah); verdict races pending |
| `episodes.feedback: score` | ADOPTED as the second TamIA arm (final_v4b_score_4b) | ~2x retry success vs 1-bit (10-21u, ae, ah) |
| `adaptive.first_probe_panel_from_start: true` (turn-0 base panel from step 0) | ADOPTED for TamIA | panel learned from scratch by ~30 steps (10-21ad, ai, aj) |
| Math collapse risk (degenerate loops, grad norm clipped >3, GSM8K-while-hiding < 0.3) | watch in every run | 10-20u |

## Hard rules (carried over)

- Aggregates only. Never read, print or commit texts, completions, activations or checkpoints.
- Every experiment goes into `REPLICATION.md` (newest first, RULED-OUT ledger).
- New features are config fields that default OFF.
- Never install transformers 5.x into the main `.venv` (the Qwen3.5 venv is separate: `scripts/env/q35t29_build.sh`).
- 4B and 1 seed first. Dense models only. Published data first.
