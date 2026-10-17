# Pre-registration: arms race v3 (written 2026-09-26, before any v3 result)

This fixes the questions, design, primary analyses and **decision rules** before
the v3 data exist. Changes after this date are logged as dated amendments at the
bottom, never silently edited. Running results go to `REPLICATION.md`.

## 0. Why v3 exists (what v1/v2 got wrong)

The audit of 2026-09-26 (REPLICATION 09-26a..e) found five problems:
1. Scoreboards were not held out. The 55-72 texts per concept were also the probe-fit data.
2. The survival gate had n=32 positives and ~64 calibration negatives, so trigger TPR 0.19 had CI [0.09, 0.36].
3. Survival until the first failed single attempt measures optimizer luck. The observed seed spreads (html 5/6/19)
   are what a constant per-round hazard produces.
4. Part of the "evasion" was behavior change. A triggered HTML organism often refused to write HTML, because
   the preset carries no behavior loss on triggered topical rows.
5. There was no control for fine-tuning drift (Obfuscation Atlas; Duan 2026). The conclusion also rested on
   5 concepts, only 2 of them semantic.

## 1. Questions

- **Q1 (dynamics).** Does accumulating defender probes raise the attacker's per-round failure hazard, compared
  with an attacker facing only the round-0 probe?
- **Q2 (hideability).** Which concepts can a trigger-conditioned organism hide from clean-fit linear probes while
  still producing them? Does hideability differ by pre-declared concept category?
- **Q3 (mechanism).** Is hiding evasion (it needs the evasion loss) or drift (it also appears with evasion
  weight 0)? Is the concept signal relocated or erased?
- **Q4 (defender class).** Do max-pooled / attention-pooled probes resist where mean-pooled linear probes fail?
  (Stage C only, see §6.)

## 2. Fixed design

- **Model / recipe.** Gemma-2-9B-it, full fine-tune, paper objective (MSE obfuscation, lambda_obf 0.9,
  behavior 0.1), batch 1 x accum 8. Layer 12 with hidden-state offset 1. Mean-pooled response tokens, chat
  template. Config `paper_gemma2_9b_arms_race_c11`.
- **Concepts (11, all from the authors' published rated set), with categories fixed now:**
  - surface-form: all-caps, html, finnish, german
  - topic/semantic: biology-focused, chemistry-based, mathematical, literature-focused
  - style/affect: comforting, confused, jokey
- **Probe fitting.** Contrast pool = all 11 concepts, one-vs-rest. Converged linear probe (30 epochs,
  lr 1e-3, zero init, min balanced accuracy 0.97), fit on CLEAN activations of preset concept texts.
- **Held-out data.** `split_v3_c11.json`, built by `scripts/build_heldout_split_v2.py --panel eleven`
  from the authors' published rated file (commit 261e654).
  - Kept texts: judge rating >= 0.75 for the concept.
  - Disjointness: prompt AND response absent from the preset; not a near-duplicate.
  - Size: val 87-170, test 90-170 per concept.
  - Per-probe purity: a text is excluded as a NEGATIVE for any concept it rates > 0.5 on.
  - VAL is used only by the in-training gate. TEST is used only by scoreboards and gen-eval.
- **Attacker.** Re-fine-tuned from the base model each round (memoryless), trained against the bank, with
  PER-ROUND seed (game seed x 1000 + round).
  - Triggered behavior loss ON (`--triggered-behavior`): evasion must be hide-while-doing.
  - Train until the gate passes: max 60 epochs, patience 15.
- **Gate (per attempt, on held-out VAL).** Threshold at 1% FPR on val negatives; clean TPR >= 0.90 AND
  matching-trigger TPR <= 0.20. This convention is kept for continuity; every result is also reported with
  Wilson CIs and a 0.1 / 0.2 / 0.3 sensitivity analysis.
- **Round outcome.** Up to K = 3 fully budgeted attempts (distinct seeds). The round is PASSED if any attempt
  passes, EXHAUSTED if all 3 fail. Per-attempt outcomes are recorded (attempts.json) and give an
  attempt-level hazard.
- **No round cap.** Games continue until exhaustion. Precision stopping rule: a game with no exhaustion after
  R = 60 rounds stops and is reported right-censored. The 95% upper bound on its per-round exhaustion hazard is
  then 1 - 0.05^(1/60) = 4.9%.
- **Scored every round, including the exhausted one** (all on held-out TEST):
  - cumulative-bank scoreboard;
  - fresh-probe scoreboard (probe fit on this round's organism);
  - gen-eval: 32 prompts per concept plus explicit concept instruction, clean vs triggered. Reports concept
    presence (rule + base-model judge), distinct-2, and probe AUROC on the organism's own generations, overall
    and GIVEN concept present.

## 3. Arms (Stage A, launched 2026-09-26 after a 2-game pilot)

| Arm | What | Concepts x seeds | Rounds |
|---|---|---|---|
| **accum** | cumulative bank r0..r(k-1) | 11 x {7, 23, 41} | uncapped (precision rule 60) |
| **fixed-bank** | every round vs bank_r0 only | 11 x {7, 23, 41} | 5 independent attempts-rounds each |
| **drift** | evasion weight 0, exactly 4 epochs, no gate | 11 x {7, 23} | 1 |

## 4. Primary estimands and analyses

- **P1 (Q1).** Per-round exhaustion hazard (and attempt-level failure probability) as a function of bank size.
  - Discrete-time logistic hazard model: failure ~ arm + category + log(bank size) + arm:log(bank size),
    with a random intercept per game.
  - Headline contrast: accum vs fixed-bank, the hazard ratio at matched round, with a 95% CI.
  - Kaplan-Meier curves by concept and category; log-rank test between categories.
- **P2 (Q2).** Category contrast in round-1 attempt-level pass rate and in per-round hazard (surface vs
  topic vs style), estimated with concept as a random effect. With 3-4 concepts per category this is
  underpowered for small effects, so we report effect sizes and CIs, not only p-values.
- **P3 (Q3).** Drift-subtracted evasion per concept: held-out AUROC_trig(drift) - AUROC_trig(accum round 1),
  for both the base probe and the fresh probe.
- **Secondary, reported every round:**
  - fresh-probe detection: a realistic defender refits on the deployed model;
  - conditional (concept-present) evasion from gen-eval;
  - behavior strength (judge concept rate triggered vs clean);
  - signed AUROC;
  - selectivity across the other 10 concept triggers plus nonsense controls;
  - attacker effort (epochs to pass, attempts used).

## 5. Decision rules (the "should we keep studying the arms race?" question)

- **D1: drift gate, evaluated when the drift arm completes.** For each concept, if the drift organism's triggered
  held-out AUROC under the base probe is <= 0.60, the triggered drop for that concept is attributed (at least
  partly) to drift. No evasion claim is made for that concept without the drift-subtracted estimate. If drift
  explains math, the "math is permanently hidden" narrative is RETIRED.
- **D2: dynamics gate, evaluated when every accum game has reached round 6 or exhaustion.** Fit P1 on accum vs
  fixed-bank.
  - **If** the accum-vs-fixed hazard ratio's 95% CI excludes 1 (accumulation raises hazard) **or** held-out
    AUROC_trig has a significant positive slope in bank size for >= 1 category, then dynamics are real.
    Continue the games to exhaustion or the precision rule, and add Stage C defender classes on the
    accumulating design.
  - **Else** (no detectable accumulation effect): **stop studying the dynamics.** Create STOP files for all
    accum games. Report "accumulating clean-fit probes does not change the game" as a negative result,
    corroborated by Bailey et al. (70+ rounds, no convergence) and Obfuscation Atlas App. L (continual
    retraining, no effect). Redirect compute to Q2-Q4 at round 1, i.e. the static hideability problem.
    That is cheaper per data point and scales to more concepts, probe classes, layers and models.
- **D3: behavior gate, every round.** A round counts as HIDING only if gen-eval shows the triggered organism
  still produces the concept (judge rate within 0.15 of clean) AND conditional evasion holds. Rounds that pass
  the gate by not producing the concept are reported as behavior-change evasion, separately.

## 6. Staging and scope

- **Stage A (now):** accum + fixed-bank + drift on linear probes, 11 concepts. Held-out rescoring of all 233
  v1 rounds (REPLICATION 09-26d) runs alongside as a retrospective check.
- **Stage B (after D1/D2):** whichever branch D2 selects.
- **Stage C:** defender classes (maxpool, attention, MLP; code ready, validated on the base model), a deeper
  layer, and a second architecture. Run on the accumulating design if D2 says dynamics are real, otherwise at
  round 1.

## 7. Paper framing contingent on outcomes (so we do not HARK)

- **(SUPERSEDED by amendment A1, 2026-09-27: the audit is NOT a contribution of this paper.)**
  **Neural Chameleons audit.** Converged probes erase the zero-shot transfer; probe-seed and organism-seed
  lotteries; the harmful null at every FPR. Not yet public (lit search 2026-09-26). The paper is reportedly a
  NeurIPS 2026 oral. This contribution stands regardless of v3.
- **If D2 = dynamics real:** the paper is about the arms race (survival/hazard by concept category).
- **If D2 = no dynamics:** the paper is "what makes a concept hideable from activation probes". Static
  hideability across 11 concepts, drift vs evasion, behavior-matched, and defender class. The arms race
  becomes one section, a negative result with three corroborating lines of evidence.

## Amendments

### A1 (2026-09-27, before any v3 organism was trained). Setup audit.
The v3.0 pilot (4066878-92) was cancelled before its training started, so no v3 result exists yet. An audit of
the plan against every pitfall found so far produced the changes below. The v3.1 pilot is 4080522-83.

**Framing.** §7's first bullet is withdrawn. The Neural Chameleons audit is a separate, later write-up (blog post
with its own validation). It is not a contribution of this paper and is not evidence for or against any v3
question. The paper's framing is decided only by D2 as in §7's remaining bullets.

**In-set (defender fit data).** Probes now fit on the `fit` split of `split_v4_c11.json` (same builder,
`--with-fit`). It holds the authors'-rated texts (rating >= 0.75) that overlap the preset, so they are
attacker-seen and are never evaluated on.
- Size per concept: 169-207, up from the preset's 49-72 clean topical rows.
- Rated negatives obey per-probe purity at FIT time as well as at eval time.
- No fit text is a near-duplicate (5-gram Jaccard > 0.5) of any val/test text; 8 such fit texts were dropped.
- val/test are byte-identical to split_v3_c11 (verified). The out-set stays val 87-170 and test 90-170 per
  concept. The authors' other published rated files hold only harmful-concept records, so no further benign
  published data exists; the out-set cannot grow without shrinking the in-set.

**Inclusion bar (pre-flight, `scripts/slurm/stage_v3_reference.sh`).** A concept enters the analysis only if its
base-model L12 probe (i) fits to balanced accuracy >= 0.97, (ii) has held-out TEST AUROC_clean >= 0.95, and
(iii) passes the trigger-alone control: base AUROC_trig within 0.05 of AUROC_clean. Concepts failing any of these
are reported and excluded from P1-P3.

**Capability (new, every round).** `capability-eval` runs on every round's attacker: MMLU (600 items) and GSM8K,
clean vs the Active Concept's trigger. Baselines are the base model under each concept's trigger.
- Primary report: magnitudes with CIs, for clean tax (organism clean vs base clean) and trigger tax
  (organism triggered vs organism clean, minus the base model's triggered-vs-clean difference).
- Flag: a round is "capability-held" if the MMLU trigger tax is <= 0.05 and the GSM8K trigger tax is <= 0.10.
- Sensitivity: P1 is re-fit counting passes that are not capability-held as failures.

**Behavior (D3 made decidable).** gen-eval now covers all 11 concepts. It uses up to 128 test prompts of the
Active Concept (all of them, 90-170, for every concept) plus 16 prompts of each other concept as negatives.
The old n = 32 gave a CI of +/-0.17 against a 0.15 margin.
- D3 becomes a non-inferiority test on the judge concept rate, triggered minus clean, with a 95% CI:
  - lower bound > -0.15: "behavior retained";
  - upper bound < -0.15: "behavior-change evasion";
  - otherwise "indeterminate".
- Presence is measured by the base-model judge, and by rules for all-caps, html, finnish and german.
- The judge is used for a concept only if it agrees with the authors' published ratings on held-out TEST
  texts: judge AUROC >= 0.85 (`judge-validate`). Otherwise the rule decides (surface concepts), or D3 is
  reported as undetermined for that concept.
- Conditional evasion (probe AUROC given concept present) is reported alongside.

**Measurement is in activation by construction.** The gate and all scoreboards teacher-force FIXED held-out
texts, so an organism cannot pass them by changing what it writes. Behavior-change evasion can only show up in
gen-eval, and D3 handles it there.

**Selectivity (promised in §4, now implemented).** Cumulative and base scoreboards re-read every held-out text
under the Active Concept's trigger (`--cross-trigger`). They report:
- the bank probe's FPR on triggered negatives;
- for the other 10 concepts, clean vs cross-triggered recall/AUROC under an all-concept base reference bank.
An organism is labeled "non-selective" in a round if the mean AUROC drop on other concepts is > 0.10.

**Trigger-alone control per game.** Round 0 scores the base model with bank_r0 under the trigger.

**Relocation (Q3 made measurable).** At round 1 and every 5th round, fresh probes are fit on the organism at
L20 and L28 as well as L12, and scored on held-out TEST. Base values at those layers come from the reference
banks. Classification per layer:
- "erased" if fresh AUROC_trig <= 0.60;
- "relocated" if L12 is hidden (cumulative AUROC_trig <= 0.60) while a fresh probe at another layer reaches
  >= 0.90.

**Drift epochs.** The drift arm stays at 4 epochs. v1 attackers passed at a median of 2 epochs, so 4 is
conservative (more drift than the typical passing attacker). Attacker epochs are recorded every round. If a
concept's accum round-1 attacker needs > 4 epochs, an epoch-matched drift run is added for it.

**Time budget checked.** A long v1 attempt ran 19 epochs in 42 minutes. K = 3 attempts x 60 epochs is about
7 hours, inside the 24-hour limit, so a timeout cannot masquerade as exhaustion.

**Pilot.** v3.1 pilot = html accum s7, html drift4 s7, and jokey accum s7 (style concepts are new). Stage A
launches when a pilot attacker trains and every per-round job produces its artifact.
