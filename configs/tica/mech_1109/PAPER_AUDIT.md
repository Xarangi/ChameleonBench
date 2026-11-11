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
| Fixed-text held-out val race (non-free-gen; 7 families) | evasion on unseen concepts with refits | ✓ .554 | ✓ .652 | ✓ .707 | ✓ | ✓ | ✗ |
| Gated free-gen quality race, train concepts | in-distribution evasion | ✗ (ungated only) | ✗ OOM -> rerun 686005 | ✓ | ✓ | ✗ | ✗ (ungated 4B base only) |
| Gated free-gen quality race, held-out concepts | concept transfer | ✗ (ungated only) | ✗ OOM -> 686005 | ✓ | ✓ | ✗ | ✗ |
| Gated task races = interactive hide-and-solve (22 tasks) | turns to beat, refits, correct passes | ✗ (old ungated 9 tasks) | ✓ | ✓ | ✓ | ✓ (1800 + 2500) | ✗ |
| Persistence (no refit), real / always-passed | does the evasion hold without feedback | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (4B) |
| Capability: GSM8K / MMLU, clean + 2 briefs | capability cost | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ 2B/4B/9B |
| Never-shown 18-probe panel (held-out architectures + layers 4–30), 4 train + 4 held-out seeds | held-out probes and layers | ✓ | ✓ | ~ 3 seeds rerunning | ✓ | ✓ | ✓ (4B, 1 seed) |
| Selectivity: panel `real` vs `mismatch` / `nobrief` | hides only when asked | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| Passive one-turn hide-and-solve | legacy; NOT the paper metric (user 10-29) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 1b. Training trends

| item | 2B | 4B | 9B | det | strict 4B |
|---|---|---|---|---|---|
| Training curves from train logs (beatk 1–5, turns, GSM8K, free validity, KL per 200 steps) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Held-out val race at intermediate checkpoints | ~ 600 / 1500 / 1800 | ✗ 1800 only | ✗ 1800 only | n/a | ~ 1800 / 2500 |
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

1. **4B (k18) gated quality race** (train + held-out). OOM at batch 8 at turn ~17, deterministic even alone on a GPU.
   Rerunning at batch 4 (686005).
2. **9B never-shown panel**, 3 seeds (train s3, held-out s1 / s3). 51–119 probe fits OOM'd at 2 processes per GPU.
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
