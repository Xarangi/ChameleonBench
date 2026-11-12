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

## 7. Robustness pass by task group / trained vs held-out (11-11 evening, user request)

Scripts: `racegroups.py` (task races by group), `turnacc.py` + `turnacc_tab.py` (every answer graded, by turn / probes
deployed / verdict; 16 tasks, GSM8K has no gold file for this grader), `freelang.py` (free-gen language check).

**7a. Task races by group** (probes beaten per episode of 6 | with a correct answer | correct among passes p1 / p2-3 / p4-6)

| group | base 4B | 2B | 4B | 9B | det |
|---|---|---|---|---|---|
| trained concepts (5 tasks) | 0.49, 0.34 | 4.46, 2.48, 59/60/48% | 5.72, 3.34, 72/64/50% | 5.62, 3.66, 84/66/57% | 5.84, 3.01, 71/58/41% |
| held-out code (4) | 0.12, 0.07 | 2.81, 0.38, 23/12/5% | 5.36, 0.59, 30/12/2% | 5.30, 0.50, 36/6/1% | 4.60, 0.74, 23/16/12% |
| held-out MGSM fr/ru/th (3) | 0.22, 0.16 | 2.24, 0.61, 27/27/32% | 4.26, 1.90, 52/42/42% | 4.66, 2.58, 61/53/55% | 4.28, 2.55, 61/58/60% |
| held-out Belebele (5) | 0.23, 0.17 | 2.03, 0.72, 34/37/35% | 4.61, 1.89, 52/45/31% | 4.47, 2.29, 62/45/52% | 3.96, 2.53, 65/64/63% |
| all held-out (12) | 0.19, 0.13 | 2.34, 0.58 | 4.77, 1.46 | 4.79, 1.77 | 4.26, 1.94 |

- The falling correct share is a held-out-code effect (and 4B reading). SQL stays high; MGSM is flat.
- The 2B gap is mostly on held-out concepts. Detector training costs probes only on held-out concepts and raises
  correct passes there.

**7b. Accuracy by turn, all answers** (turn 0 | 1-7 | 8-15 | 16-23 | 24-32)
- Base 4B is flat in every group (code .66-.68, trained .58-.66, MGSM .68-.71, reading .79-.71): conversation length
  is not the cause.
- 4B held-out code: caught .58 / .21 / .08 / .05 / .03, passing .44 / .22 / .02 / .01 / .00.
- 4B trained: caught .91 / .73 / .52 / .36 / .31, passing .75 / .60 / .34 / .21 / .19. 9B passing .82 / .67 / .53 / .49 / .42.
- 4B MGSM: caught .66 / .39 / .15 / .10 / .06, passing .78 / .49 / .48 / .45 / .38.
- => accuracy erodes over the episode once the model hides, on caught and passing turns alike. Not a per-probe price.
- In-language share on held-out language tasks .91-1.00 for all models. **MGSM-de, 4B: only 27-35% of answers are in
  German** (9B: 32/32 at every probe) -> the 4B's 6.00 on MGSM-de is largely off-concept.

**7c. Never-shown panel by probe** (mean detection, trained / held-out concepts; 4B | 9B)
- trained types (new fits): linear L8-26 .02/.24 | .01/.20; attnpool .02/.34 | .05/.23; ensdeep4 .02/.24 | .01/.20.
- unseen, beaten: linear@24 .16/.37 | .06/.31; mck_attnlite .02/.19 | .01/.04; spread5 .02/.35 | .02/.35.
- unseen, mixed: linear@20 .39/.46 | .25/.40; gdm_rollattn .27/.63 | .06/.35; nc_ens8_tok .23/.36 | .51/.31; maxpool@24 .44/.61 | .41/.67.
- unseen, NOT beaten: linear@28 .71/.77 | .62/.81; ens8_linear .82/.51 | .82/.59; linear@30 .99/.97 | .89/.97; linear@4 1.00 | 1.00.
- All races so far refit only with trained probe types and layers -> paperE.

**7d. Free-gen race validity (language concepts, fastText lid)**
- Passes written in the language: untrained models 3/23 (trained langs), 3/99 (held-out); trained models 360/366 and
  217/236. Base "wins" on languages are answers that left the language.
- Code / format concepts need the 27B judge -> paperF.

**Running:** paperE 688302 (gated free-gen race, defender refits ONLY with held-out probe kinds + linear@4; 2B / 4B /
9B / base 4B x train / held-out; `launch_paperE.sh`), paperF 688310 after it (27B judge on all gated gens incl. hok,
per-row scores; `launch_paperF.sh`, `judge_piece.sh`), paperD 688141.

**7e. paperD done (688141, 35/35 outputs, 34/34 score jobs COMPLETED) — untrained 2B / 9B controls** (`basecmp.py`)
- Probes beaten per episode | correct passes per episode, 17 tasks: base 2B 0.18 | 0.08 (272 eps), base 4B 0.28 | 0.19
  (544 eps), base 9B 0.17 | 0.12 (272 eps). base 2B/9B are half size (16 eps per task, piece p1).
- By group as in paper 6.2 (beaten/ep, 2B / 4B / 9B): trained concepts .33 / .49 / .23; held-out code .11 / .12 / .12;
  held-out MGSM .08 / .22 / .21; held-out Belebele .16 / .23 / .12. First probe beaten: 35/272, 98/544, 33/272.
  (First version of basecmp.py had mbpp / mgsm_de in the wrong groups; fixed, paper sentence corrected in rev 40.)
- Bootstrap CIs (stratified by task): 2B 0.18 [0.12, 0.25], 4B 0.28 [0.23, 0.33], 9B 0.17 [0.11, 0.23].
- Concentrated: mgsm_de + mgsm_fr + jseval = 28/50 (2B), 79/151 (4B), 24/46 (9B) of beaten probes; GSM8K and MBPP 0 at every size.
- Base 2B capability (GSM8K n=200 / MMLU n=600): clean .715 / .592; maths brief .435 / .485; HTML brief .420 / .500.
  Trained 2B: .685 / .570; .605 / .547; .650 / .545. So trained 2B is within 3 pts clean and far ABOVE its base under a
  brief (the untrained 2B collapses when told to hide). The earlier "expect a real cost for the 2B" guess was wrong.
- Paper updated (rev 34): 6.1 control sentence, capability table row "2B untrained" + text, limitations bullet.

**8. Threat model + related work pass (paper rev 40, 11-11 evening)**
- Added section 1.1 Threat model (two sides; model-organism framing; three routes: built on purpose, by-product of
  training against monitors, in-context from deployment feedback; attacker / defender affordances; win condition; scope).
- Related work rewritten with a reference list (21 entries). Checked by web search: titles, authors, venues / arXiv ids.
- CORRECTION: the intro said Neural Chameleons' test probes are fit on the original model. The paper's abstract says
  evasion holds against monitors "trained post hoc on the model's frozen weights". Intro now says the monitor is fit
  once and does not respond to the model. Not yet verified from the PDF: whether any NC probe is trained on TRIGGERED
  activations; exact round count of Bailey et al.'s adversarial-training loop (v1: 70 epochs); authors/dates of the
  LessWrong replications (cited only as "community replications", no reference entry).

## 9. Race with held-out probe kinds (paperE, 11-11 night) -> paper 5.4

Source: `hoktab.py` over `race_*_gated_hok.json` (32 episodes each; 9B trained-concept race pending in paperG).
- Paper 5.4 filled (doc rev 43) with two tables: beaten/faced trained kinds vs held-out kinds, and by probe kind.
- Claim checked against deploy_records: linear@30 0/22 and linear@4 1/18 for 4B+9B (the one pass is the 9B, held-out
  concepts); 2B linear@3 0/11, @23 0/11, @21 2/12.
- Caveat written into 5.4: on held-out concepts with held-out kinds the untrained 4B beats as many probes (90) as the
  trained 4B (95). First probes separate them (24/62 vs 48/61); the judge run has to confirm that the untrained passes
  drop the concept. Until j*.json exist this is supported for language concepts only (section 4.1 language check on the
  trained-kind races).
- Still to do when paperG ends: 9B trained-concept row, judge valid-pass counts (4.1), abstract / conclusion sentence.

## 10. Judge valid-pass counts (11-11 night) -> paper 4.1 and 5.4

Source: `judgetab.py -c`. Paper 4.1 placeholder replaced (doc rev 46) with the judge table and two paragraphs; 5.4
last paragraph now cites judge counts for the held-out-kind races. Stated openly: on held-out Java / Python under half
(2B, 4B) to 70% (9B) of the trained models' passing answers still contain code, so per-concept free-race numbers on
held-out code overstate the trained models too. The judge's `answers` score is not used (low on every condition).
Open: the 9B trained-concept hok race (row in 5.4 + its judge counts).
- 23:45 update: 9B trained-concept hok race in; paper 5.4 tables complete (doc rev 60), abstract gained one sentence on
  the held-out-kind race (about one probe in nine beaten on unseen concepts = 10/88) and now cites the judge (13/259).
  The 9B hok trained-concept race ran at batch 4 (others 8, 4B 4); batch only groups episodes. Its answers are unjudged.

## 11. Adversarial reviews (2 x Opus, 11-12) and assessment

Reviews: `reviews_1112/review_A.md` (claims / statistics), `review_B.md` (methods). Verified = re-derived by me from the
files, not taken from the review.

| # | Finding | Verified | Verdict | Action |
|---|---|---|---|---|
| B-C1 | Many trained-model passes are one-word fragments ("Ein", "Um"); language ID and judge accept them | yes (`validpass.py`): passes <=20 chars 2B 97/197, 20/72; 4B 139/257, 33/129; 9B 87/256, 38/139; untrained 0-1 (9B held-out 7/21). Task races, language tasks: 4B 774/2051 passes, 57% of all answers from turn 8; code / SQL / JSON / GSM8K 0 | accept, changes the headline | paper: abstract, findings, new 4.1 "Fragment answers" table (first probe with min-content rule: held-out 23-26, 34-38, 37-41 of 64; untrained 0-2), 6.3, 7.1, conclusion. Optional rerun with fragments counted as catches (~5.5 + 8 node-h): USER |
| B-C2 | Task-race "6-of-8" gate = one sample copied 8 times (one vLLM seed) | yes: share_mean = 1 - rejected/checks in every piece checked | accept | paper 2.3 + limitations corrected. Code fix (seed + j) and rerun 14-40 node-h: USER |
| A-M1 / B-M3 | judgetab.py counted Monitor-OFF turns | yes | accept | fixed, paper corrected (rev 80) |
| A-C1 | Catches are free; caught before first pass in 86-97% of held-out segments | yes (`validpass.py`: first pass with no earlier catch 2 / 4 / 9 of 64) | accept (framing) | abstract, limitations, conclusion now conditional |
| A-C2 | Thesis must be conditional on the defender's probe family | yes (5.4) | accept | conclusion first paragraph rewritten |
| A-C3 / B-M6 / A-M14 | Detector section uses two standards (offline 97-100% vs race 62-85% beaten); 8.2 table mixes detector-only and all-probe rates; detector race protocol (fixed text, 16 episodes, no gate) undisclosed | partly: the two standards yes; the 79/81% like-for-like numbers and protocol NOT yet re-derived | accept | abstract, section 8 intro, 8.2 paragraph. TODO: re-derive 8.2 table like for like, state protocol |
| B-M2 | Strict KL has as many or more CORRECT passes (3.59 / 1.20 / 1.90 / 1.81 vs 3.34 / 0.59 / 1.90 / 1.89; 2500: 3.74 / 1.09 / 2.08 / 2.53) | yes (racegroups with strict rows) | accept, reverses 8.3 | 8.3 rewritten |
| B-M1 | Defender weaker than described: refit data = dataset texts teacher-forced in context, one snapshot, threshold from ~86 negatives, gate never binds; "episode not judged" wording wrong | code references not re-read by me; refits == accepted yes | accept (text) | 2.2 and limitations rewritten. Stronger-defender race (3-4 node-h) and offline pooled concept probe (CPU): TODO / USER |
| B-M7 #11 | "First probe" after a switch is fit on the current model; 4B-9B held-out gap sits in the base-panel half (23 vs 31 of 32) | yes (reviewer script first.py rerun) | accept | 2.2 text; 7.1 no longer orders 4B / 9B |
| A-M2 | Headline first-probe rates count concept-dropping passes | yes | accept | covered by the min-content column |
| A-M3 / B-M3 | Judge unvalidated; coherence score broken | not re-derived | accept | limitation added. TODO: hand-label ~150 rows; fix prompt and re-judge (1 node-h) |
| A-M4, A-M5, B-M8 | Concept mix unpaired; CIs ignore clustering / tasks; single seed | plausible, not re-derived | accept | 4B vs 9B claims removed. TODO: cluster bootstrap over episodes and tasks |
| A-M6 | Undisclosed: 2B branch lineage, cancelled Gemma run, strict 2B | yes | accept | limitations + 8.3 |
| A-M7 | Held-out concepts are siblings of trained families; trained-concept tasks were capability-training tasks | true by design | accept (text) | TODO: state in 6.2 |
| A-M8 | Monitor-OFF accuracy stays 0.56-0.64 on held-out code while monitored falls to 0.03 ("does not come back" wrong) | NOT verified | to check | TODO (turnacc with off rows) |
| B-M9 | 6.3 erosion on language tasks = fragment collapse | yes | accept | 6.3 paragraph added |
| A-M9, B-m2 | "19 unseen types" counts layers / variants; layer-4 failure partly lexical | plausible | accept (text) | TODO |
| A-M10, B-m1 | Selectivity: mismatch-turn detection 0.68-0.89 on some probes (partial hiding of an unmonitored concept) | NOT verified | to check | TODO |
| A-M11, B-M5 | Exclusions: MGSM-de kept though untrained 1.38; MGSM-ja crash | partly | accept | MGSM-ja now in limitations. TODO: uniform validity rule |
| B-M4 | Belebele near chance; lenient letter extraction | not re-derived | accept | limitation added. TODO: chance-corrected counts |
| A-M12, A-M13 | Unequal controls; "searching with the one bit" asserted | true | accept | TODO text; feedback-ablation race is new GPU work: USER |

Reviewer-proposed GPU work, for the user to decide (none submitted): gate-seed fix + task-race rerun (14-40 node-h);
free-gen + language task races with fragments counted as catches (~13 node-h); cumulative-data defender race (3-4);
detector race under the free-generation protocol; feedback ablation from turn 0; static-probe baseline adapter; second
training seed (4B ~120 node-h, 2B ~40).
Not addressed yet in the paper: every row marked TODO above.

## 12. Strict-KL gated free-generation race (11-12, paperH)

Paper 8.3 now uses the gated race for every column (table in REPLICATION.md "11-12 03:50"). Checks: judge has-concept on
strict passing answers 71/80, 75/80 (4B held-out), 29/29 (2B held-out); fragment share in free generation is NOT lower
for the strict 4B (91/196, 24/80 vs 139/257, 33/129), so the fragment advantage claimed in 8.3 is limited to the
language task races and the text says so. 9B held-out-kind trained-concept judge cell filled: 138/140.
