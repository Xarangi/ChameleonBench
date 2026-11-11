# Paper-readiness eval audit (11-11)

Models in scope:

| model | run | checkpoint |
|---|---|---|
| 2B | `final_v4d_tamia_2b_x1800` | 1800 |
| 4B (k18) | `final_v4d_k03_tamia_4b` | 1800 |
| 9B | `final_v4d_k03_tamia_9b` | 1800 |
| Detector model (d22) | `final_v4d_k03det_tamia_4b` | 2200 |
| Strict 4B, KL3 (ablation) | `final_v4d_klx_tamia_4b` | 1800 / 2500 |

All models are trained with KL0.3, except strict 4B. The detector model is the 4B (k18) at 1800 trained further
against the detector.

Legend: ✓ done · ✗ missing · ~ partial/caveat · n/a = not applicable.

## 1. Inventory

### 1a. Main evals

| eval | what it shows | 2B | 4B | 9B | det | strict 4B | base model |
|---|---|---|---|---|---|---|---|
| Fixed-text held-out val race (non-free-gen; 7 families) | evasion on unseen concepts with refits | ✓ .554 | ✓ .652 | ✓ .707 | ✓ | ✓ | ✓ 2B .072 / 4B .006 / 9B .013 |
| Gated free-gen quality race, train concepts | in-distribution evasion | ✓ 197/219 (90%), 2.25 | ✓ 257/272 (94%), 1.85 turns | ✓ 256/263 (97%), 2.01 | ✓ 246/260 | ✗ | ✓ p1 beaten 2B 7/64, 4B 14/66, 9B 6/65 |
| Gated free-gen quality race, held-out concepts | concept transfer | ✓ 72/117 (62%), 3.83 | ✓ 129/156 (83%), 2.91 | ✓ 139/163 (85%), 2.91 | ✓ 117/146 (80%) | ✗ | ✓ p1 beaten 2B 9/68, 4B 20/64, 9B 6/65 |
| Gated task races = interactive hide-and-solve (22 tasks) | turns to beat, refits, correct passes | ✓ 22/22 | ✓ | ✓ | ✓ | ✓ (1800 + 2500) | ✓ 4B: p1 212/672; 98/544 excluding MMLU |
| Persistence (no refit), real / always-passed | does the evasion hold without feedback | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (4B) |
| Capability: GSM8K / MMLU, clean + 2 briefs | capability cost | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ 2B/4B/9B |
| Never-shown 18-probe panel (held-out architectures + layers 4–30), 4 train + 4 held-out seeds | held-out probes and layers | ✓ | ✓ | ✓ (held-out s0: TypeError -> 3 held-out seeds) | ✓ | ✓ | ✓ (4B, 1 seed) |
| Selectivity: panel `real` vs `mismatch` / `nobrief` | hides only when asked | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| Passive one-turn hide-and-solve | legacy; NOT the paper metric (user 10-29) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 1b. Training trends

| item | 2B | 4B | 9B | det | strict 4B |
|---|---|---|---|---|---|
| Training curves from train logs (beatk 1–5, turns, GSM8K, free validity, KL per 200 steps) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Held-out val race at intermediate checkpoints | ✓ 600 / 1000 / 1400 / 1800 | ✓ 600 / 1000 / 1400 / 1800 | ✓ 600 / 1000 / 1400 / 1800 | n/a | ~ 1800 / 2500 |
| Detector catch over detector training (evdetsel / mlpdetsel at L12 / 16 / 22, 1800–2200) | n/a | n/a | n/a | ✓ | n/a |

### 1c. Detector results

| item | status |
|---|---|
| Detector race | ✓ d22 beats a live selective detector 82% @L16 / 85% @L22, vs 4B (k18) 69% / 62% |
| Frozen transfer | ✓ |
| Selectivity | ✓ |
| Dodge anatomy | ✓ |
| kNN detector (label-free) | ✓ (fails, AUROC ~.45–.6) |

## 2. Gaps, in priority order

1. DONE (fix4, batch 4). **4B (k18) gated quality race** (train + held-out). OOM at batch 8 at turn ~17, deterministic even alone on a GPU.
   Rerunning at batch 4 (686005).
2. DONE (fix4; failed fits 1 / 18 / 19 = normal). **9B never-shown panel**, 3 seeds (train s3, held-out s1 / s3). 51–119 probe fits OOM'd at 2 processes per GPU.
   Rerunning at 1 per GPU (686005).
   - 9B held-out s0 has a deterministic TypeError in `panel._mean`. It needs a code fix in a fresh worktree; until
     then it is reported on 3 held-out seeds.
3. **2B gated free-gen quality races + gated task races (22 tasks).** Without them, the 3-size comparison on the
   paper metric is 4B vs 9B only. Cost: ~2 races nodes (~9 node-h, as for the 4B) + 2 pieces.
4. **Base-model controls on the paper metrics.** There is no base race on the gated quality races, gated task races
   or val race, so "probes beaten" has no untrained reference.
   - Earlier finding: MMLU CS / medicine are at ceiling for every model (uninformative without a base).
   - Proposed: base 4B on val + gated quality (3 pieces) and gated task races (22 tasks, ~1–2 nodes); base 2B / 9B
     on val + gated quality only.
5. **Held-out trend evals.** Val race at 600 / 1000 / 1400 for 4B and 9B, plus 1000 for 2B: ~7 pieces, 1 node.
   This shows held-out evasion emerging over training; the training-log curves are on the training probes only.
6. Single training seed per size. Not fixable cheaply; state it as a limitation.

## 3. Training curves (train logs, 200-step bins; beatk_frac 1–5 | GSM8K | KL)

| bin | 2B | 4B | 9B | strict 4B |
|---|---|---|---|---|
| 600–800 | .58/.11/.05/.00/– \| .74 \| .16 | .71/.30/.13/.11/.00 \| .87 \| .25 | .75/.25/.15/.12/.33 \| .91 \| .18 | .41/.14/.04/.00 \| .89 \| .04 |
| 1000–1200 | .88/.48/.13/.17/.00 \| .76 \| .21 | .93/.63/.35/.22/.29 \| .87 \| .21 | .93/.70/.43/.28/.28 \| .89 \| .20 | .69/.35/.11/.03/.00 \| .92 \| .04 |
| 1600–1800 | .97/.84/.61/.42/.43 \| .71 \| .29 | 1.0/.98/.93/.85/.80 \| .89 \| .21 | 1.0/.98/.94/.89/.85 \| .91 \| .21 | .93/.68/.38/.17/.08 \| .91 \| .05 |

- Detector model, 2000–2200: .996 / .976 / .949 / .894 / .862.
- Strict 4B, 2200–2500: .99 / .92 / .76 / .57 / .47.
- Detector catch during detector training:
  - evdetsel @22: .79 at 1800–1850, .56 at 2050–2100, .63 at 2150–2200.
  - mlpdetsel @22: .74 -> .65.
  - Linear probes: .02–.08 throughout.

## 4. Validity finding from the base controls (11-11)
- MMLU CS and MMLU medicine are **uninformative race tasks**. The untrained base 4B beats all 6 probes in 32/32
  episodes on both, at 1.2–1.6 turns, with robust share .70–.90, exactly like the trained models. The probes do not
  separate the hiding condition on these tasks.
- MMLU physics (base beats p1 28/32, robust .29) and MMLU bio (22/32, robust .12) are partly informative: base wins
  there are mostly flukes.
- **Recommendation:** report the race totals over the 18 non-MMLU tasks, with MMLU in a separate appendix row, or
  drop MMLU CS / med. Base 4B over non-MMLU tasks: first probe 98/544 (18%), with low robust shares.

## 5. Gated task races: 3-size comparison plus base control (`racecmp.py`, 11-11)
- Task set: the 15 non-MMLU tasks complete for all models. belebele_ta / belebele_el are pending for the 2B, and
  mgsm_ja is excluded as before.
- 480 episodes per model.
- "Probes beaten per episode" is out of 6. "Correct passes" are correct answers among beaten turns.

| model | probes beaten / episode | p1 | p6 | correct passes (% of beaten) |
|---|---|---|---|---|
| base 4B | 0.30 | 93/480 | 1/3 | 106 (73%) |
| 2B | 3.14 | 458/480 | 92/129 | 613 (41%) |
| 4B (k18) | 5.21 | 480/480 | 322/366 | 1047 (42%) |
| 9B | 5.21 | 480/480 | 324/367 | 1221 (49%) |
| det | 4.94 | 476/480 | 314/337 | 1168 (49%) |

MMLU, 4 tasks, appendix:

| model | probes beaten / episode |
|---|---|
| base 4B | 3.89 |
| 2B | 5.58 |
| k18 | 5.98 |
| 9B | 5.95 |
| det | 5.92 |

The MMLU base row confirms §4: MMLU is not informative.

### 5b. Final: all 17 non-MMLU tasks (544 episodes per model; mgsm_ja excluded)

| model | probes beaten / episode | p1 | p6 | correct passes (% of beaten) |
|---|---|---|---|---|
| base 4B | 0.28 | 98/544 | 1/3 | 106 (70%) |
| 2B | 2.97 | 518/544 | 92/129 | 619 (38%) |
| 4B (k18) | 5.05 | 544/544 | 336/387 | 1095 (40%) |
| 9B | 5.03 | 544/544 | 334/383 | 1264 (46%) |
| det | 4.72 | 540/544 | 316/345 | 1227 (48%) |

## 6. Final pre-writing audit (11-11 evening)

Paper draft: https://claude.ai/artifact/FjQoA6PpAeyd2NjxvfEZDk (Claude Docs; sections 1-10 + appendix).

**Checked and fine**
- Task-race commands, all 5 models x 21 tasks, diffed after normalising paths / adapter. They differ only in
  TICA_REPO, memory fractions, `--batch` (2B 16; 9B javaeval p1 rerun 4) and `--set model.batch_size=16` (absent on
  k18 math / mbpp; read batch only).
- k18 task races ran from ChameleonBench-gate (9419997) / -graded (7aca3d2); the others from -final (eb2e8bb).
  `git diff --stat 9419997 eb2e8bb` touches only task builders, the report script, capability_heldout and the 9B
  config. The race engine is identical.
- Every refit counted in the task races passed the quality gate. The 6-of-8 gate rejected 22-24% of apparent passes
  for 4B / 9B / det, 33% for the 2B and 32% for base 4B. Mean fork share: .80 / .82 / .78 / .71 / .44.
- Capability runs all use the same command (512 new GSM8K tokens, same 2 briefs).
- Panel: base 4B mean detection .996 / .992 (0/18 beaten). Monitor-off turns are detected .98-1.00 for all 3 sizes.

**New numbers (no GPU; `ci.py` in the session scratchpad, bootstrap over episodes stratified by task)**

| model | probes beaten / episode, of 6 [95% CI] | uncapped (max 12) | first-answer accuracy |
|---|---|---|---|
| base 4B | 0.28 [0.22, 0.33] | 0.28 | .70 |
| 2B | 2.97 [2.86, 3.07] | 3.42 | .51 |
| 4B (k18) | 5.05 [4.96, 5.14] | 7.90 | .70 |
| 9B | 5.03 [4.95, 5.12] | 8.14 | .74 |
| det | 4.72 [4.62, 4.82] | 7.69 | .69 |
| strict 1800 | 3.36 [3.24, 3.47] | 4.20 | .76 |
| strict 2500 | 4.04 [3.91, 4.16] | 5.50 | .77 |

- The cap at 6 saturates for 4B / 9B (many tasks at 6.00); quote the uncapped number alongside.
- Correct among passes falls with probe index: 4B 53 / 46 / 39 / 33 / 32 / 29%, 9B 62 / 46 / 41 / 41 / 39 / 43%.
- Free-gen held-out first probe, Wilson 95%: 2B 69% [57, 79], 4B 81% [70, 89], 9B 92% [83, 97]. 4B vs 9B overlap.
- Frozen step-1800 linear hiding detector on det@2200 (TPR at 1% FPR): L8 .68, L12 .58, L16 .79, L22 .96; fresh
  detector on det@2200 .97-1.00 at L8-26; no-detector control (sx18 -> sx25) .97-1.00.

**Found missing -> running (paperD 688141, one node, 35 lines, `launch_paperD.sh`)**
- Base 2B capability (GSM8K / MMLU): there was no measured control. The 2B is at .685 GSM8K vs a .75 training
  reference, so the capability cost of the 2B was unsupported.
- Base 2B and base 9B on the 17 non-MMLU task races, half size (16 episodes per task = piece 1 of the trained
  races; commands equal the trained p1 commands up to `--base`). The base control on the headline metric existed for
  the 4B only. 64k context from the start. 34 score jobs follow.

**Wording corrections for the paper**
- Free-gen "first probe n/64" is per concept SEGMENT (32 episodes, each switches concept), not per episode.
- Base 4B is a weaker free-gen control (31% on held-out, mostly Java and Portuguese); the fixed-text race base rates
  are 7-14%.
- 2B task races at batch 16 and the k18 free-gen race at batch 4: stated as protocol differences.

**Status 11-11 10:00: audit gaps 1–3 are all complete.** Open items:
- 9B held-out panel s0: TypeError in `panel._mean`; 3 held-out seeds are reported instead.
- Strict 4B has no gated quality race.
- Single training seed per size: state as a limitation.
