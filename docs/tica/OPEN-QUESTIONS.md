# TICA open questions and pending decisions (2026-10-19)

The user decides the TamIA optimisations. Everything here is optional, and anything built for it is a config field
that defaults OFF.

## 1. Speed (to decide on TamIA)

Step time on Narval A100: 106 s at the start, 370 s at step ~550. About 85% of that is refits: every pass triggers
a fork-and-fit, which costs ~450 forked reads plus a probe fit. Doubling the read batch from 8 to 16 gave nothing
measurable, which suggests the GPU is already busy during reads. See RESULTS §3.

| Option | Expected gain | Cost | Changes results? |
|---|---|---|---|
| **Data-parallel episodes**: each GPU runs its own episodes and refits, then the gradients are all-reduced; the bandit and turn-0 state are shared (`docs/tica-other-hpc.md` option D) | ~1.6-1.8× on 2 GPUs, ~3-3.5× on 4 | 1-2 days | No (same protocol) |
| **Rollout reuse** (`updates_per_rollout`, f0db642, default 1): K gradient updates per rollout | Up to ~2.5× more updates per hour at K = 3, if each extra update stays useful | Built and tested from s500 (10-19a): **no gain**; k2-k4 equal within noise, retry pass falls with K (0.40 / 0.37 / 0.34). Not in v4. Untested in the steep phase (s140-300) | Yes: extra updates fit a stale rollout (loss -5% / -10% within a rollout) |
| H100 instead of A100 | ~1.5-2× on reads and backward | none | No |
| Bigger read batches / `micro_batch: 2` on 80 GB | Small (batch 16 did nothing on A100) | config | No |
| GSM8K cap_gen on a spare GPU, one step ahead | ~7% (hides ~22 s) | ~0.5 day | Slightly (weights one step stale) |
| Batch the fits across refits, and the folds of attnpool / maxpool / rlobf_attn | ~10-20% | ~1 day | bf16 rounding only |
| Fewer forks per refit | Up to ~1.5× | config | **Yes**: weaker defender. Not recommended |

**Why a batched or parallel version is not bit-identical.**
- Refits draw fork texts and probe initialisations from one shared RNG in a fixed order, and batching reorders those
  draws. Pre-draw in the original order to fix this.
- bf16 rounding changes with batch shape, which can flip a quality-gate decision at the threshold.

Verify any speed-up the usual way: same draws, probe outputs agree within rounding.

## 2. v4 run settings

- **Step count.**
  - Held-out transfer stopped improving after ~s340: the held-out race at s500 equals s340 (0.78 / 0.52 / 0.162 vs
    0.78 / 0.53 / 0.166), and the val pass rate has been flat since 360 (val 550: score 0.37, pass 0.147).
  - Training k2-k4 rose ~+0.15-0.25 over 340-500 meanwhile, so steps beyond ~350-500 on the same concepts sharpen the
    training concepts without helping transfer (10-18j). Breadth (v5's 20 train concepts) is the more likely lever.
    Decide v4's count from v4's own held-out curve.
  - Training k2 is on a noisy plateau at ~0.72-0.87 over 480-600 (10-19a); retry pass ~0.36-0.43.
  - v4 draft says 1200 steps. Consider about 600-800, fixed in advance (no checkpoint selection on held-out).
  - v4 is slower per step (24 turns, longer segments, so more refits) and starts from scratch, so its curve may
    differ. Run the held-out race every 50 steps (`eval_every: 50`) and log it only.
- **LR.** v4 holds 1e-4 (2e-4 was ruled out as a mid-run jump). From-scratch 2e-4 is untested.
- **Retry weighting.** The ladder [2, 3, 4] is in the v4 draft (best held-out arm). It was never combined with the
  progress term (arm A was also good on held-out).
- **Narval maintenance** 2026-10-27 05:00 → 11-03 (reservation POWEROUT20261027): any Narval run must end before then.
  TamIA is unaffected.
- **v3b baseline chain.** A 6th chunk (6315437) is queued so the chain reaches step 1000.

## 3. Science questions not yet run

1. **Scrambler baseline for v3b.** Train to move the representation every turn regardless of the verdict
   (`l4scramble` config, math-era). This separates verdict-driven adaptation from a generic moving dodge.
2. **Train against the supervised evasion detector.** Add it to the defender as an arm after v4. Is the evasion
   signature forced, or just free to leave?
3. **Evasion detector: caught vs passed turns.** Split the detector score; CPU only, using the saved
   `evdet/*_feats.pt`. Does it detect the attempt or only success?
4. **Evasion-detector calibration.** Thresholds from train concepts give held-out FPR 0.05-0.63. Options: fit the
   threshold on held-out negatives, or per-concept normalisation.
5. **Anomaly-detector layout confound.** v2 was caught (0.88-1.0) under a different eval layout than v3b (0.00-0.19).
   Run v3b with v2's reference episodes, or v2 with the tagged layout.
6. **Deception panel.** 7 of 18 panel probes failed the quality gate on deception. Either loosen the gate for that
   concept or report it separately (per-turn TPR drop vs base, as in 10-17z).
7. **Thinking mode.** The dodge mostly disappears with thinking ON. Scope the claim to non-thinking mode, or add a
   thinking-mode anchor (s1K / Bespoke prompts staged; KL to the base model's own thinking generations).
8. **GSM8K capability.** Re-run capability.py GSM8K at 512 new tokens (the drop is likely a 256-token cut-off).
9. **Maths / chemistry** are never beaten after the 1st refit (0.06 / 0.11). There is no fix proposed (margin loss
   falsified). v4 shows whether SQL / JSON / JavaScript learn like all-caps or stall like maths.
10. **3 seeds** for the pre-registered headline, after the 1-seed v4.

## 4. Data decisions

- **Formal tone is broad.** It covers 0.28 of the free negative pool and 0.44 of other concepts' texts, which the
  27B labels clean at fit time. Watch whether it is learnable or swallows other concepts.
- **Biology (held-out) overlaps "living organisms"**, which v3b trained on. So v4 must start from scratch, not
  warm-start from v3b.
- **Liars' Bench is gated on Hugging Face.** The user must accept its terms personally before it can be added.
- **Alignment faking is left out** (user 10-18). Its labels are refusal vs harmful compliance, not deception.
- **Cautious tone (TONEBANK) has no stated licence**, so it is eval-only.
