# TICA results digest (as of 2026-10-18, newest first)

Every number below is an aggregate. The source entry in `REPLICATION.md` is in brackets, and the
machine-readable copy is `results/tica/snapshot_2026-10-18.json`. Unless stated otherwise:
1 seed, Qwen3.5-4B, non-thinking mode, final_v3b (`configs/tica/final_v3b_4b.yaml`).

**Two race types.**
- **Held-out race** (`tica_switch_eval.py`): v3 held-out concepts (10 topics + Finnish, HTML, confusion,
  physics, deception), 56 episodes × 24 turns, mixed probe kinds, no concept switches.
- **Val race:** the dev set, ~32 episodes, run every 20-50 steps, logged only.

**Metric names used throughout.**
- *beaten*: share of refit probes the model got past while they stood.
- *within 3*: share beaten within 3 turns.
- *pass/turn*: pass rate after a segment's first turn.

## 1. Held-out transfer has plateaued while training keeps improving [10-18i, 10-18j]

| Checkpoint | Beaten | Within 3 | Pass/turn | Refits | k2 | k3 | k4 | k5 |
|---|---|---|---|---|---|---|---|---|
| s340 | 0.78 | 0.53 | 0.166 | 238 | 0.84 | 0.91 | 0.88 | 0.76 |
| **s500** (job 6258629) | **0.78** | **0.52** | **0.162** | 246 | 0.91 | 0.90 | 0.85 | 0.69 |
| retryctl@420 (branch) | 0.81 | 0.61 | 0.191 | 283 | 0.89 | 0.94 | 0.87 | 0.90 |

- In this table, k-columns are the beaten fraction of the k-th probe in the held-out race. k1 is 1.00 everywhere.
- s500 equals s340 on the full held-out race.
- s500 per held-out concept, pass/turn:

  | Concept | Pass/turn |
  |---|---|
  | deception | 0.30 |
  | medicine | 0.30 |
  | fashion | 0.22 |
  | food | 0.22 |
  | education | 0.15 |
  | plays & poetry | 0.15 |
  | philosophy | 0.13 |
  | Christianity | 0.11 |
  | Finnish | 0.09 |
  | confusion | 0.07 |
  | physics | 0.07 |
  | HTML | 0.04 |

- Val race history agrees. Pass rate by step:

  | Step | 300 | 340 | 360 | 380 | 400 | 450 | 500 |
  |---|---|---|---|---|---|---|---|
  | Pass rate | 0.098 | 0.098 | 0.144 | 0.160 | 0.106 | 0.133 | 0.136 |
  | Beaten | 0.67 | 0.67 | 0.75 | 0.80 | 0.71 | 0.72 | 0.75 |

  - The dev "domain" concept scores 0.00 at every step.
  - Val score is the dev share beaten within 3 turns: 0.38 / 0.44 / 0.44 / 0.35 / 0.34 / 0.31 / 0.37 at 340-550 (pass 0.147 at 550).
- Training by probe index (20-step windows). Training numbers now also flatten at about 0.75 for k2:

  | Steps | k1_first | k1_switch | k2 | k3 | k4 | k5 | Retry pass |
  |---|---|---|---|---|---|---|---|
  | 0-20 | 0.08 | 0.03 | 0.00 | - | - | - | 0.006 |
  | 80-100 | 0.86 | 0.06 | 0.05 | 0.00 | - | - | 0.040 |
  | 160-180 | 0.97 | 0.35 | 0.48 | 0.18 | 0.33 | - | 0.098 |
  | 240-260 | 1.00 | 0.58 | 0.37 | 0.27 | 0.36 | 0.50 | 0.113 |
  | 300-320 | 1.00 | 0.72 | 0.59 | 0.35 | 0.35 | 0.33 | 0.190 |
  | 380-400 | 1.00 | 0.87 | 0.56 | 0.54 | 0.46 | 0.40 | 0.267 |
  | 440-460 | 1.00 | 0.79 | 0.74 | 0.61 | 0.56 | 0.52 | 0.353 |
  | 480-500 | 1.00 | 0.84 | 0.80 | 0.63 | 0.69 | 0.52 | 0.371 |
  | 520-540 | 1.00 | 0.89 | 0.75 | 0.64 | 0.66 | 0.53 | 0.378 |
  | 540-560 | 1.00 | 0.93 | 0.72 | 0.61 | 0.60 | 0.59 | 0.385 |

- **Reading.** Steps after ~400 mostly sharpen the training concepts. This is the evidence for setting v4's fixed
  step count below 1200 (see OPEN-QUESTIONS).
  - Caveats: 1 seed; held-out races carry about ±0.05 sampling noise.
  - retryctl@420 is a branch with fresh resume state, not the same trajectory as s420.

## 2. Retry weighting branches (s360 + 60 steps, same episode draws) [10-18b, 10-18g, 10-18i]

| Arm | What changed | Train k2 / k3 / k4 (steps 31-60) | Train retry pass | KL | Held-out beaten / within 3 / pass/turn (refits) |
|---|---|---|---|---|---|
| ctl | nothing | 0.71 / 0.53 / 0.45 | 0.283 | 0.18 | 0.81 / 0.61 / 0.191 (283) |
| A | improve_weight 1 + improve_margin 0.1 (progress term) | 0.65 / 0.55 / 0.61 | 0.289 | 0.18 | 0.86 / 0.70 / 0.248 (343) |
| B | retry_hide_weight 3.0 (flat) | 0.78 / 0.56 / 0.55 | 0.341 | 0.20 | 0.84 / 0.71 / 0.238 (328) |
| **L** | **retry_hide_ladder [2, 3, 4]** (2× / 3× / 4× after 1 / 2 / 3+ consecutive catches) | 0.70 / 0.59 / 0.62 | **0.347** | 0.20 | **0.88 / 0.76 / 0.267 (373)** |
| lr2e4 | held LR 2e-4 (was 1e-4) | 0.66 / 0.46 / 0.40 | 0.219 | 0.17 | 0.79 / 0.57 / 0.164 (244) |

- **The ladder wins on held-out**, so the v4 draft uses it.
- **LR 2e-4 is RULED OUT** as a mid-run jump: it was worse in training and held-out. A from-scratch 2e-4 run was not tested.
- Caveat: 1 seed, 60 steps per branch.

## 3. Step time grows because the model succeeds more [10-18i]

| Steps | Sec/step | Refit reads | Refit fits | Refits/step |
|---|---|---|---|---|
| 0-60 | 106 | 34 | 17 | ~4 |
| 240-300 | 233 | 114 | 67 | ~13 |
| 420-480 | 329 | 184 | 95 | ~20 |
| 540-600 | 370 | 207 | 112 | ~20+ |

- Every pass triggers a branch refit. The cost per refit is flat: about 8-9 s to read ~450 forks, plus 4-5 s to fit.
- LoRA training (40-45 s) and capability generation (~24 s) are flat.
- Doubling the read batch from 8 to 16 gave no measurable gain.
- Options are listed in OPEN-QUESTIONS.

## 4. Probe panel: one broad dodge, small verdict-driven adjustment [10-18h]

s400 vs base, 13 held-out concepts, 32 episodes × 16 turns. Every turn is also scored by a never-shown panel
of 18 base probes.

- **Dodged per monitored turn.** Share of the base panel dodged:

  | | t0 | t1 | t2-15 |
  |---|---|---|---|
  | s400 | 0.25 | 0.37 | 0.70-0.88 |
  | base model | 0.00-0.06 | 0.00-0.06 | 0.00-0.06 |

- **Detection on s400 monitored turns, by panel probe:**

  | Probe | Detection |
  |---|---|
  | linear@16 / mck_attnlite | 0.07 |
  | linear@12 | 0.08 |
  | nc_ens8_tok | 0.12 |
  | ensdeep4 | 0.13 |
  | linear@8 | 0.14 |
  | linear@22-24 | 0.18-0.19 |
  | gdm | 0.21 |
  | attnpool | 0.23 |
  | maxpool@24 | 0.25 |
  | linear@20 | 0.28 |
  | linear@28 | 0.38 |
  | ens8 | 0.42 |
  | linear@30 | 0.68 |
  | linear@4 (near-lexical, never dodged) | 0.99 |

  The base model is detected at 0.96-1.00 by every probe.
- **Strategy change after a verdict:**
  - Overlap of the dodged panel set between consecutive turns is 0.81 after a catch vs 0.87 after a pass.
  - Panel flip rate is 0.12 after a catch vs 0.07 after a pass.
  - So the verdict moves the model, but only at the margin.
- **No circularity.** An old refit re-catches the model 0.41 of the time on the turn after it was beaten, then
  0.23 / 0.13 / 0.08 at 2 / 3 / 4+ turns. New refits retro-catch 0.65 of earlier turns.
- **Evade, not erase.** A fresh probe fit per turn still reads the concept at AUROC 0.984-0.995 (base 0.997-0.999).
- **Selective.**
  - On monitor-OFF turns the panel detects 0.83-1.00.
  - An unrelated, unmonitored concept B is detected 0.89-1.00 vs 0.96-1.00 with no brief. The base model shows a
    similar gap, so excess hiding is at most ~0.06.
- **Gap:** 7 deception panel probes failed the quality gate, so the deception panel is incomplete.

## 5. Evasion detectors: the anomaly detector is dodged, the supervised detector is not [10-17r/t, 10-18g/h]

**Unsupervised anomaly (OOD) detector** (`tica_frozen_race.py --ood`):
- Mahalanobis distance in the top-32 PCA components + residual, at L12. The reference is base or self no-brief
  activations, at a 1% false-positive rate. It never sees an evading example.

| Model | Catches |
|---|---|
| v2 final | 0.88-1.00 from turn 4 (12% false alarms) |
| v3b s140 | 0.00 at every turn |
| v3b s400 | 0.00 at turns 0-4; 0.06-0.19 at turns 5-15 (5.5-6.5% false alarms) |

- v3b's dodge stays near its own unmonitored distribution, i.e. "looks normal".
- Caveat: the tagged prompt layout differs from v2's eval reference episodes (confound not excluded). v2's
  re-run reproduces with the current code.

**Supervised, concept-agnostic evasion detector** (`tica_evasion_detector.py`):
- Logistic regression on answer-pooled activations at L8 / 12 / 16 / 20 / 24, from panel races (80 episodes × 16 turns).
- **Positives:** monitor-ON turns at t ≥ 1 in monitored episodes, whether the model passed or was caught
  (i.e. trying to evade, not succeeding).
- **Negatives:** the same model's monitor-OFF turns inside monitored episodes ("off"), no-brief episodes from
  t ≥ 1 ("nobrief"), or both.
- **Trained on TRAIN concepts, tested on HELD-OUT concepts.**

| | s400 | Base (same pipeline, no evasion) | Excess AUROC |
|---|---|---|---|
| Held-out AUROC, ref "off" | 0.999-1.000 | 0.59 / 0.74 / 0.82 / 0.76 / 0.64 (L8..24) | +0.18..+0.41 |
| Held-out AUROC, ref "nobrief" | 0.999-1.000 | 0.88-0.97 (the brief alone is visible) | +0.03..+0.12 |
| TPR at held-out 1% FPR | 0.98-1.00 | 0.03-0.62 | +0.88..+0.97 (off) |

- **Transfer between checkpoints.** The s400 detector catches s140 at 0.99-1.00 (L8-16), and the s140 detector
  catches s400 at 0.98-1.00. It fires on base turns at only 0.36-0.72. So there is one stable evasion signature
  from step 140 on.
- **Caveats:**
  - Thresholds set on train concepts give held-out FPR 0.05-0.63: the ranking transfers, the calibration does not.
  - There are few held-out "off" negatives (44-62).
  - The model was never trained against either detector.
  - The defender needs labelled evasion from this model.
- **Not yet split:** detector score on caught vs passed turns. The features are saved in `evdet/*_feats.pt`.

## 6. Transfer to a fully held-out family: emotion [10-18g]

- Seven emotions (happy, sad, angry, afraid, guilty, surprised, calm). Hard negatives are the other emotions'
  stories plus neutral text. s360 vs base:

  | | s360 | base |
  |---|---|---|
  | Refits beaten (393 refits) | 0.90 | 0.00 |
  | Pass/turn | 0.31 | 0.02 |

- Per emotion, pass/turn ranges from 0.17 (angry) to 0.44 (happy).
- Caught-by-turn alternates 0.84, 0.27, 0.71, 0.52, ... (a caught/pass rhythm); the base model is about 0.98 flat.

## 7. Held-out concepts and probe types at s300-340 [10-17x, 10-18a, 10-17ad]

- **Held-out families at s300, refits beaten:**

  | Family | Beaten | Refits |
  |---|---|---|
  | topics (8) | 0.77 | 159 |
  | deception | 0.88 | 40 (base model also beats these, weak test) |
  | physics | 0.62 | 8 |
  | Finnish | 0 | 2 |
  | HTML | 0 | 3 |
  | confusion | 0 | 2 |

  Surface and pragmatic families did not transfer with only 1-2 training members. Hence v5 has 4 training
  concepts per area.
- **Held-out probe types at s340, refits beaten** (short races):

  | Probe type | Beaten |
  |---|---|
  | NC token ens8 | 0.76 |
  | McKenzie attn-lite | 0.74 |
  | linear (trained type) | 0.71 |
  | maxpool@24 | 0.66 |
  | Bailey mean / GDM rolling attention | 0.63 |
  | ens8 MLP | 0.61 |
  | ens8 linear | 0.60 |
  | spread5 | 0.57 |
  | linear@20 | 0.55 |
  | linear@24 | 0.51 |
  | linear@28 | 0.50 |
  | Bailey max | 0.50 |
  | linear@30 | 0.43 |

  Unseen architectures do about as well as the trained types; unseen DEPTH (20-30) is the gap.
- **Matched long races vs v2** (no switches, beaten):

  | Probe | v3b | v2 |
  |---|---|---|
  | Bailey | 0.85 | 0.86 |
  | GDM | 0.84 | 0.82 |
  | linear | 0.87 | 0.88 |
  | linear@24 | 0.74 | 0.76 |

  Parity with v2.
- **Held-out static base probes at L20/24 during training** (never trained against). Detection on monitored turns:

  | Steps | 0-120 | 120-180 | 180-240 | 240-300 | 300-360 | 360-420 | 420-480 | 480-540 | 540-600 |
  |---|---|---|---|---|---|---|---|---|---|
  | Detection | 1.00 | 0.97 | 0.87 | 0.77 | 0.67 | 0.58 | 0.44 | 0.42 | 0.50 |

  - Monitor-OFF turns stay at 1.00 throughout; no-brief at 1.00; mismatch at 0.03-0.06.
- **Refit readability during training.** Refits are fit on the model's own states:

  | Steps | 0-60 | 480-540 |
  |---|---|---|
  | Mean AUROC | 0.9996 | 0.9955 |
  | Share of refits below 0.99 | 0.8% | 14% |

  The concept stays readable, slowly getting harder to read.

## 8. Evidence for real in-context adaptation (and against the strong version)

**For:**
- **Falsified-verdict control** (s340): pass after the first turn drops from 0.36 to 0.15 when the verdicts are
  falsified (v2: 0.45 to 0.03) [10-17ad]. So v3b uses the verdict.
- **Training retry pass rises** 0.01 → 0.20 → 0.27 → 0.38 (steps 0-40, 320-360, 380-400, 540-560).
- **It beats fresh refits fit on its own current states**, on held-out concepts, a held-out family (emotion) and
  held-out probe types.

**Against / limits:**
- About 0.15 of passes are verdict-blind, including the static turn-0 dodge of the fixed base probe (k1_first =
  1.00 from step 200). The v4 turn-0 schedule (fork-fit turn-0 probe from step 150) removes that crutch.
- The panel shows mostly one broad dodge with a small verdict-driven adjustment (section 4).
- Turns-to-beat [10-17ac]: training raised the first try after a catch (hazard 0.28 → 0.43) far more than later
  retries (~0.07 → ~0.09, s140 → s300). Retries have since improved: retry pass reached 0.38.
- **No scrambler baseline** has been run for v3b (`l4scramble` config exists for the math-only era).

## 9. Monitor OFF and free generation [10-17v, 10-18d, 10-18g]

- **Training-format OFF turns** (base concept probe): caught 0.85 (s0-80), then 0.97, then 1.00 from s160.
- **Free generation**, n = 12 per cell, OFF turns, detection at L12 / L20 / L24:

  | Concept | s400 | s160 (L12) | base (L12) |
  |---|---|---|---|
  | German | 0.67 / 0.92 / 0.92 | 0.50 | 0.92 |
  | Finnish | 0.75 / 0.92 / 0.75 | 0.33 | 0.83 |
  | all-caps | 0.42 | | 0.33 |

  - The small leak is shrinking and confined to L12, the fixed turn-0 probe's layer.
- **Free generation, monitor ON:** L12 0.00-0.25 but L20/24 about base. There is no behavioural hiding (the share of
  German answers is unchanged), only representational suppression at the trained layers.

## 10. Thinking mode [10-17v, 10-17t]

- Training renders an empty think block.
- With thinking ON and the monitor ON, the dodge is mostly lost: detection sports 1.0, Finnish 0.95, music 0.85.
  German (0.35) and all-caps (0.30) survive.
- The trained model rarely verbalises the monitor: it mentions it 36% of the time and "hide" 2.5%, vs the base model
  at about 100%.
- **Thinking-mode accuracy** (boxed prompt, 0-shot, n = 100):

  | Model | GSM8K think, clean | GSM8K think, brief |
  |---|---|---|
  | base | 0.41 | 0.00 |
  | v3b s300 | 0.56 | 0.67 |

  The base model rarely closes its think block under the brief, so these numbers mostly measure a length cut-off
  and are not interpretable as capability.
- Non-thinking: GSM8K 0.86 for both base and s300.
- Decision pending: scope the claim to non-thinking mode, or train a thinking anchor.

## 11. Capability [10-17y, 10-17aa]

- **s300 vs base, clean:**

  | Benchmark | s300 | base |
  |---|---|---|
  | ARC | 0.901 | 0.904 |
  | MMLU-STEM | 0.683 | 0.695 |
  | MATH-500 | 0.605 | 0.630 |
  | MBPP | 0.755 | 0.747 |
  | GSM8K (capability.py, 2-shot, 256 tokens) | 0.755 | 0.825 |

- GSM8K under a sports brief: 0.67 vs 0.78.
- The GSM8K drop is likely the 256-token cut-off. In the 1024-token boxed eval, s300 = base = 0.86. Unchecked:
  re-run at 512 tokens.
- Eval KL: no-brief 0.017; brief own concept 0.18; other concept 0.21.

## 12. Safety concepts [10-17y, 10-17z]

- DolusChat: the base model's per-turn detection is only 0.83-0.89, so report the per-turn TPR drop vs base, not
  the race pass rate.
  - s300: refit per-turn detection 0.74-0.77 vs base 0.88 (drop ~0.12).
- Insider trading: 0.99 vs 1.00 (no hiding against refits).
- No harmfulness concept is in any TICA eval (guardrail).

## 13. How v3b came to be (why v3 failed) [10-16q .. 10-17p]

- final_v3 never trained: hide_p was flat at 0.98 for 140 steps.
- Cause (bisected):
  - A turn-0 probe fork-fit from step 0 (zero lag, saturated p, vanishing gradient).
  - A family-first sampler that starved the model of learnable topics.
  - Fixes: `switch_fork`, the tagged layout, an LR schedule of 3e-4 → 1e-4, and a uniform-concept ramp.
- v3b took off at steps 80-140.
- An early OFF-turn leak (OFF detection 0.66 at 60-80) self-corrected by step 100. off_match_weight 2.0 fixed it in a
  branch at no cost, but v4 drops it (user, 10-18) because the leak never returned.
