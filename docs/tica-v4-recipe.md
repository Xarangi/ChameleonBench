# TICA v4 recipe (proposal, 2026-10-17; nothing built or launched yet)

Evidence base: v3b (`final_v3b_4b`) to step ~335, full held-out + dev sanity races (REPLICATION 10-17x),
think-hide / free-generation evals (10-17v), step-300 capability eval, safety races (10-17y).

## What is missing / underperforming (ranked)

| # | Problem | Evidence | Cause |
|---|---|---|---|
| 1 | Capability cost under the monitor | step 300 GSM8K 0.755 clean / 0.67-0.68 under a monitor vs base 0.825 clean; v2 had no drop (0.805 / 0.79). MMLU unchanged (0.68). Eval KL under a monitor 0.18-0.21; in-training KL 0.145 -> 0.173 | hiding moves generation under the monitor; the in-training proxy (cap_gen acc ~0.9 on RFT-sampled GSM8K-train forks) hid it. Pending: think_v3b300 (140 -> 300 regression vs prompt effect) and base-under-monitor (caph_base) |
| 2 | Surface families do not transfer | held-out Finnish / HTML / confusion / code: 0 refit probes beaten, scores 8-29x threshold; topics 0.77 | 1-2 training members per non-topic family (topics have 8) |
| 3 | Notation-heavy concepts barely learn even in training | 1st refit beaten: maths 0.03, chemistry 0.17 (topics 0.8-0.9) | breadth cannot fix this; a mechanism problem |
| 4 | Deep iteration lags | k2 ~0.5-0.6, k3 ~0.35 (rising +0.12-0.13 / 100 steps) | 16-turn episodes, switch_prob 0.5 after 5 turns -> few k3+ segments |
| 5 | Turn-0 fixed base probe saturated | beaten 1.00; the dodge does not reach held-out L20/24 base probes | static, probe-specific dodge: no signal left |
| 6 | Free-generation OFF leak | v3b OFF / none L12 det 0.25-0.58 vs base 0.83-0.92 | off turns only matched on forced text (off_match_weight 0) |
| 7 | Thinking mode breaks the dodge | forced, thinking ON: sports 1.0, Finnish 0.95 (vs 0.0-0.55 no-think) | never trained with a think block |
| 8 | Validation is thin | 5 dev concepts, 1 per non-topic family, "within 3 turns" metric | split design |
| 9 | Safety tests are weak | base beats 14-15 / 24 DolusChat refit probes; insider trading valid (base never passes) | weak deception probes |
| 10 | Speed | ~250 s / step on 1 A100 (more refits as it wins) | single GPU |

Code (dev) beat its first refit probes at val step 320 (2 of 5): one check, small n.

## Recipe

### A. Data / split v4 (breadth; every new concept through the 9B-broad + 27B-strict two-pass, then the selection step: positives >= 0.8, clean negatives <= 0.2, hard negatives = broad-yes / strict-no)

- **Language** (Aya, human-written; language field + script / function-word check; >= 700 rows):
  - train (12): German, Spanish, Portuguese, Italian, Polish, Dutch, Swedish, Turkish, Vietnamese,
    Japanese, Standard Arabic, Hindi
  - dev (3): French, Indonesian, Greek
  - held-out (6): Finnish (NC), Thai, Tamil, Iranian Persian, Russian, Amharic
- **Format / syntax** (non-code syntax in training so code stays a real held-out test):
  - train: all-caps (NC), SQL, LaTeX math markup, JSON / config, shell commands, markdown tables
  - dev: computer code (CodeAlpaca; continuity with v3b)
  - held-out: HTML (NC), plus one more code language
  - candidate published sources, each checked before use: b-mc2/sql-create-context (SQL),
    NL2Bash (shell), glaive function-calling (JSON), MATH solutions (LaTeX). Negatives: no syntax
    concept is a negative for another (27B labels).
- **Tone**: train jokes, comforting, angry, formal (NC style ratings; formal re-admitted through
  27B labels; angry with a toxic <= low filter; supportive dropped if it overlaps comforting);
  dev casual + supportive-or-academic; held-out confusion.
- **Domain**: train maths, chemistry, literature, finance (if a published set passes checks);
  dev physics; held-out one prose domain (checked source).
- **Safety (held-out family)**: insider trading (valid), DolusChat response-only, the v3 DolusChat
  concept; roleplaying only if its base probe passes the gate. Always reported next to the base model.

### B. Sampling

- About 35 training concepts: uniform over concepts would put about 35% of episodes on languages.
  **Decision for you:** switch the curriculum prior to `family` (uniform over families, then
  uniform over concepts within a family), with the hazard Thompson ramp on top as now.
- More concepts means fewer visits each: about 1.5x the steps of v3b for the same per-concept exposure.

### C. Mechanism

- **Fresh probe at turn 0** (`first_probe: fork`): the v3 cold start came from forking at step 0 from
  scratch; v4 warm-starts from the v3b checkpoint (D), so it is safe now.
- **Notation-heavy concepts** (maths, chemistry, code-like):
  - weighting that keeps weight on probes not beaten CONFIDENTLY (your request; optional, default off)
  - a separate kind-lambda for deep layers (L22 / L26)
  - both tested on maths alone first, 1 seed
  - not re-proposed: the margin loss (falsified 08-07c); a p^2 saturation fix only as a separate
    default-off ablation.
- **Deep iteration**: turns 16 -> 20, `concept_switch_prob` 0.5 -> 0.35 (min segment 5): more k3 / k4
  segments, about 25% more time per step.
- **off_match_weight 2.0** (kept OFF detection at 0.98 at no cost to ON in the branch test).

### D. Capability (gate, not a proxy)

- GSM8K-test under a monitor brief + clean, run on every save next to validation; early stopping and
  checkpoint selection require GSM8K-under-brief >= base - 0.03.
- Planned ablation before the main run: cap_gen_weight / cap_gen_kl_weight 0.5 / 1.0 vs 1.0 / 2.0
  (short, 80 steps from the v3b checkpoint), chosen by the capability gate.
- Thinking-mode KL anchor on the staged s1K / Bespoke data: optional, default off, decided after
  think_v3b300.

### E. Validation

- 2-3 dev concepts per family; metric = refit probes beaten within 5 turns, equal weight per family;
  report the probe-index lines; code on its own line; early_stop_patience 15.

### F. Run plan

1. Build split v4 + labels (no GPU training).
2. Capability ablation (D), 80 steps, 4B, 1 seed.
3. v4 main run, warm-started from the best v3b checkpoint, LR re-warm 2e-4 -> 1e-4 over 100 steps,
   on TamIA 4x H100 data-parallel (docs/tica-other-hpc.md).
4. Full eval: held-out + dev + safety sanity races, think-hide, capability, base controls.
