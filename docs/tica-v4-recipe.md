# TICA final-run recipe (v4; revised 2026-10-17 after the user's decisions; nothing launched)

User decisions (10-17): dev merged into held-out, fixed number of steps (no early stopping, no dev);
3-5 train concepts per family; labels only keep NEGATIVES clean, positives trusted from the sources;
similar concepts are fine as separate concepts; no capability gate; nothing started yet.

## Evidence (v3b to step ~360; REPLICATION 10-17v .. 10-17ab)

| Axis | Finding | Bin |
|---|---|---|
| Probe depth | every architecture at L22/26 is beaten 0.25-0.45 of deployments, at L8-16 0.55-0.95 (steps 300-360); deep ensemble 0.35. The layer sampler already gives 22/26 the most weight (0.30/0.27); L26 0.09 -> 0.39 over steps 200-350 | still rising -> steps |
| Iteration | 1st fresh probe ~0.75 (flat); after 1st refit 0.52-0.55; after 2nd 0.42-0.45 (+0.12-0.13 / 100 steps) | still rising -> steps |
| Held-out probes in training | base probes at L20/24 (never trained against, no training probe reads 20/24/28/30): caught at the last turn 1.00 -> 0.69, still falling | still rising |
| Held-out concepts (step 300) | topics 0.77 of refit probes beaten, physics 0.62; Finnish / HTML / confusion / code 0 | surface breadth -> v4 data |
| Maths / chemistry | after 1st refit beaten 0.06 / 0.11 after 360 steps; all-caps (0.169 weight) only 0.48 | stuck |
| Turn 0 | fixed base probe beaten 1.00 everywhere; the dodge does not reach held-out probes | design |
| Curriculum | relative weighting flattened in 1-3-member families (cap 0.3 <= uniform share) | design |
| Deep refits sample size | concept segments 6-10 turns; 3rd probe reached n = 1-7 for half the concepts | design |
| OFF leak | free generation at step 160: OFF / none L12 det 0.25-0.58 vs base 0.83-0.92 (not re-tested) | design |
| Thinking | thinking ON removes most of the dodge | scope decision |
| Capability | step 300: GSM8K 0.755 vs base 0.825 in capability.py (2-shot, 256 new tokens), 0.86 = base in the boxed-prompt eval (1024 tokens); ARC / MMLU / MATH / MBPP unchanged | unsettled (likely length cut-off) |

## Recipe

1. **Data: split v4** (built: `final_v4_clean.json`, `configs/tica/final_split_v4.json`; 27B labels
   `concept_labels_v4data_27b.json` as `neg_labels`). Train 5 topics / 5 languages / 4 format-syntax /
   3 tones / 4 domains; held-out 10 topics, 8 languages, HTML + Python code, confusion + supportive,
   physics, 3 deception sets. finance positives filtered to the 27B-confirmed 180 (finance-alpaca is
   Alpaca + FiQA: only 30% finance); every other source trusted. Weaker held-out tests: Thai / Russian
   / Greek (140-180 texts); plays & poetry, fashion, buildings (judge confirms 0.52-0.66).
2. **Turn 0**: fresh probe, with a schedule (fixed base probe for the first ~100 steps, then fork) so a
   from-scratch run does not repeat v3's cold start. New optional field, default off.
3. **Curriculum**: cap relative to the number of options (e.g. 3x uniform) + curriculum share above
   0.5 (new optional field). Topics / languages / syntax: uniform over concepts == uniform over
   families at 3-5 members.
4. **Episodes**: 20 turns, concept_switch_prob 0.35 (more 3rd / 4th probes; ~25% more per step).
5. **off_match_weight 2.0**.
6. **Steps**: from the slopes, deep layers ~0.7 and 2nd-refit ~0.6 need ~500-600 steps from scratch +
   the ~140-step start: plan 800 fixed steps, the final checkpoint is THE model (no selection on held-out).
7. **Hardware**: 800 x ~250 s = ~55 h on one A100; TamIA 4x H100 data-parallel (docs/tica-other-hpc.md).
8. **Seeds**: 1 seed first; 3 seeds for the pre-registered headline.

## Open decisions (yours)

- Scratch vs warm start from v3b: warm start is clean for v4 (v3b never trained on a v4 held-out
  concept) and saves ~140 steps; scratch is the cleaner paper claim. Recommendation: scratch.
- Thinking mode: train a thinking-mode anchor (staged s1K / Bespoke) or scope the claim to non-thinking.
- Probe types in training: wait for the 14 held-out probe-type races (step 340); a type that is never
  beaten (e.g. token ensembles, GDM rolling attention) may belong in training.
- GSM8K: re-run capability.py GSM8K at 512 new tokens (10-minute eval) before touching cap_gen.
- Periodic held-out race every 40 steps, logged only (never used for stopping / selection): the
  transfer-over-training figure.
- Maths / chemistry: no fix proposed (margin loss falsified 08-07c). The v4 run shows whether SQL /
  JSON / JavaScript learn like all-caps or stall like maths.
