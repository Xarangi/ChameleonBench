# STAGE-2 DECISION TREE — branching plan gated on VERIFIED outcomes

Created 2026-08-07. Companion to FINAL-STAGE-1.md (Stage-1 record) and
ARMS-RACE-SETUP.md (arms-race reference). Running results go to REPLICATION.md.

## 0. The discipline (why this doc exists)

A branch is only taken when its triggering outcome is **verified to a bar**, not
when a single run hints at it. Standing bars (from FINAL-STAGE-1 §7, ARMS-RACE §6):

- **Multi-seed with a denominator.** Evasion is seed- and run-noisy (Stage 1;
  and 2026-08-07c shows even fixed-seed GPU nondeterminism flips a round). No
  transfer/saturation claim off < 3 seeds; always report "k/N seeds".
- **Held capability.** A round only counts as attacker progress if evasion
  improves WHILE the model still does its job. `clean_tpr >= 0.9` at the gate
  guards against concept-destruction; the **capability tax (Track 0)** guards
  against general degradation. A "pass" with tanked MMLU/GSM8K is a DEFENDER win.
- **Controls before causal claims.** `--no-accumulate` (de-confounds bank-growth
  vs more-training) and the objective ablation (`--train-reduction mean|softmax`)
  must be in before "accumulation closes the net" or "the objective matters".
- **CIs clear 0** for any OOD/transfer positive; sweep the probe layer before any
  absolute-layer/model claim.

Label a node **[VERIFIED]** only after its bar is met; until then it is a
**[PENDING]** gate and its branches are conditionals, not predictions.

## 1. Ledger — what is actually established vs still open

### [VERIFIED] (thoroughly, multi-angle)
- V1. Stage-1: the OOD "harmful" zero-shot headline is a weak-probe artifact
  (probe-strength gradient + threshold-invariance + seed/word lotteries).
- V2. ID evasion of the FROZEN benign targets is achievable and strong for the
  semantic concepts (finnish/biology/mathematical descend smoothly to ~0).
- V3. **Loose (patience) stopping works** — runs now use 30+ epochs vs the old
  12-cap; the mechanism is validated.
- V4. **Margin < MSE for this objective** (same-seed control, only the loss
  differs: MSE s17 r1 PASS in 4 ep, html snaps 1.00->0.00; margin s17 r1 FAIL 31
  ep, html FROZEN at 1.00). Mechanism: formatting-concept (html/all-caps) evasion
  is a **discrete phase transition near score 1.0**, where MSE's gradient
  (∝ score-target, large when most-caught) triggers the flip and margin's constant
  sub-floor gradient cannot. => MSE is the obfuscation loss going forward; margin
  is retired to an ablation point. This is the one thing verified enough to act on.

### [PENDING] (single-seed signals or unmeasured — do NOT cite as outcomes)
- P1. Whether MSE+loose reliably yields multi-round full-evasion chains across
  seeds (the phase-transition LOTTERY). In flight: gemma9b_uemse_s{7,23,41}
  469504-544; s17 already on disk (r1 PASS 4ep, r2 FAIL 29ep).
- P2. Attacker capacity / saturation k* — s7 fully evades the cumulative bank
  r1-r3, but n=1 and under margin. Unverified.
- P3. Defender-recovery decay — s7 fresh-bank catch → ~0 by r3; n=1.
- P4. Qwen3.5-9B evades at all (Qwen-family obfuscation-resistance prior). In
  flight: qwen35_9b_uemse_s17 469560-.
- P5. **Capability tax — UNBUILT** (no lm-eval runner exists; task_accuracy is a
  stub). Blocks every "attacker won" claim.
- P6. OOD transfer under iterated ID pressure (standing null; must be re-tested on
  the fuller-evasion organisms, multi-seed, CIs).

## 2. Track 0 (MANDATORY, non-branching) — build + run the capability tax
> STATUS 2026-08-07g: BUILT + VALIDATED + first verdict IN. `benchmarks/capability.py`
> + CLI (`capability-materialize`/`capability-eval`) + aggregator. Base validated
> (MMLU 0.692 / GSM8K 0.795). A GSM8K "reasoning collapse" was caught as an
> extraction artifact (run-on generation) and FIXED before it was believed (08-07f).
> VERDICT (08-07g, n=1 s7 + s17mse): evasion imposes a MODEST tax — tiny MMLU
> finetune tax, small epoch-dependent GSM8K finetune tax, and a ~0.08-0.15 MMLU
> TRIGGER tax (~10%). s7 evades r1-r3 at roughly held capability => first genuine
> attacker win past the capability gate. Report the MAGNITUDE, don't binarize.
> PENDING: run the tax on the multi-seed MSE+loose winners as G1 lands.


Not a branch: this runs regardless and GATES the interpretation of every gate
below. Without it, "the attacker won round k" is unfalsifiable.

- **Build**: a `capability-eval` path that runs lm-eval (MMLU + GSM8K, already
  declared deps) on a checkpoint, twice — clean and with the probe trigger
  prepended — and writes `capability_tax.json` (base vs organism, ± trigger).
  Wire it into `benchmark-scorecard` (the aggregator already exists).
- **Run** on: the base model (reference), s7 margin r1-r3 (already fully evade —
  ideal subjects), and every MSE+loose winner as it lands.
- **Bar for "held capability"**: organism within a pre-registered delta of base
  on MMLU/GSM8K (propose: ≤ 2-3 pts abs), ± trigger. Below that = the evasion is
  bought by degradation = defender win.
- Also replace the stubbed behavior-gate task_accuracy with the real number.

## 3. The gates

### G1 [ROUND-1 VERIFIED 08-07i; multi-round PENDING] — Does MSE+loose produce reliable chains?
> ROUND 1: 4/4 gemma seeds reach full 5-concept evasion (s17/s7/s23/s41) AT
> largely-held capability (n=4: clean finetune tax small; ~8% MMLU trigger tax).
> So round-1 evasion is a GENUINE, GENERAL attacker win past the capability gate.
> Multi-round reliability (>=2 rounds) still pending as chains advance to r2+.
Feeds: gemma9b_uemse_s{7,17,23,41} + margin ablation. Bar: per-seed chain length
(rounds cleared at full evasion) WITH clean_tpr≥0.9 AND held capability (Track 0);
report k/4 seeds.
- **G1-a** IF most seeds clear ≥2 rounds reliably at held capability → the
  generator is sound → **go G2**.
- **G1-b** IF full evasion is a low-hit lottery (most seeds stall r1/r2 on
  formatting concepts) → **Sub-experiment M** (phase-transition mitigation)
  BEFORE any curve claim: higher LR and/or LR warmup and/or per-concept upweight
  of the stuck concept; verify (multi-seed) that hit-rate rises; then go G2. Do
  NOT build a scaling curve on a broken generator.
- **G1-c GUARD** IF "passes" coincide with capability-tax failures or clean_tpr
  dips → the gate is being satisfied by concept-destruction → revisit the gate
  definition; this is a defender win, report it as such.

### G2 [PENDING] — Arms-race saturation: attacker capacity vs defender accumulation (k*)
Entered from G1-a. Feeds: per-round cumulative scoreboard (attacker capacity) +
fresh scoreboard (defender recovery), across seeds, + the `--no-accumulate`
control + the `mean` objective ablation + the Track-0 tax curve. Bar: multi-seed
k* with denominator; control + ablation in; tax read jointly.
- **G2-a** IF cumulative-max drop DECAYS with k and tax rises → classic
  saturation; k* is the headline; the "ObfAct arms race with a weaker attacker"
  story. → optionally G3 to move k* with a stronger defender.
- **G2-b** IF the attacker keeps fully evading the cumulative bank at held
  capability across seeds (the s7 hint, verified) → attacker-favorable: a cheap
  fixed-linear ensemble does NOT contain a triggered organism → **go G3**
  (escalate the defender) — this is the more novel result.
- **G2-c** IF `--no-accumulate` ≈ accumulate → accumulation is not doing the
  work; the effect is "more training", not "memory" → reframe the curve claim.

### G3 [PENDING] — Defender escalation: does a STRONGER (not just bigger) defender move k*?
Entered from G2-a (small k* wanted) or G2-b (attacker winning). Feeds:
`--trigger-aware` fit (fit the probe on TRIGGERED activations = ObfAct
adversarially-trained-monitor analogue), multi-seed + tax.
- **G3-a** IF trigger-aware reaches k* in fewer rounds → even a fixed-LINEAR
  defender, made adversarial, contains the organism → defense-side story.
- **G3-b** IF trigger-aware still fails to contain → sets up the honest boundary:
  **Sub-experiment G (gradient bridge)** — one ObfAct-style gradient attack
  against the converged probe on the SAME organism, to place convergence between
  "beats triggers" and "loses to gradients". This is the FINAL-STAGE-1 §6 thesis
  made concrete.

### G4 [PENDING] — Stronger model (Qwen3.5-9B): does it evade at all?
Independent axis (Qwen-family resistance prior). Feeds: qwen35_9b_uemse_s17
(n=1 first). Bar: r1 full evasion under MSE+loose; if yes, multi-seed.
- **G4-a** IF Qwen evades comparably to gemma → model-general phenomenon;
  strengthens the paper (not a gemma quirk). → fold into G1/G2 curves.
- **G4-b** IF Qwen resists (flat) → **first SWEEP the probe layer** (L12 is our
  unpinned choice for qwen3.5) before concluding; if still flat across layers,
  model-dependence is itself a reportable finding (contrast gemma vs qwen).

### G5 [PENDING] — Interpretive: representational vs behavioral (the corrected "CoT" item)
Entered once ≥1 VERIFIED winning evader exists (from G1/G2). NOT reasoning-trace
analysis (models don't reason here; thinking is off). Feeds: on a winner, trigger
ON vs OFF, diff (a) the sampled COMPLETIONS (token-level + a text judge) and (b)
the per-layer activation trajectory. Bar: multi-seed, with a no-trigger control.
- **G5-a** IF outputs are ~unchanged and only L12-region activations move →
  PURELY representational obfuscation (no behavioral tell) — the strong,
  concerning version; a text monitor can't catch it.
- **G5-b** IF completions shift under trigger → behavioral leakage; a non-probe
  monitor CAN catch it → weakens the "invisible obfuscation" claim, reportable.

### G6 [PENDING, standing] — OOD transfer under iterated ID pressure
Watched across G1-G4. Bar is high: multi-seed, the Stage-1 lottery denominator,
control-word decomposition, CIs clearing 0 (harmful/apollo/dolus).
- **G6-a** IF a source's drop crosses with CI>0 across seeds → SIGNIFICANT new
  result (iterated ID pressure leaks OOD); deep-dive immediately.
- **G6-b** ELSE the null stands and becomes part of the convergence-defense
  thesis (triggers don't buy zero-shot OOD even under iteration).

## 4. Parallel completeness track (Paper A) — non-branching
- **X. xstest/wildjailbreak negatives** (FINAL-STAGE-1 §5.1 / Wave 1): the last
  escape hatch for the converged-probe harmful null. Login-node download →
  materialize → re-extract eval caches → converged probe WITH their negative mix
  → threshold sweep. Closes Paper A under their own data distribution.

## 5. Immediate no-regret actions (outcome-independent; do now)
1. **Build + run Track 0 (capability tax)** on base + s7 margin r1-r3 (on disk,
   fully evade) — the blocking measurement; unblocks every G-node interpretation.
2. **Let the MSE+loose wave resolve G1** (469504-544, 469560; s17 on disk). Read
   evasion_gate_history + cumulative/fresh scoreboards, multi-seed, with the tax.
3. HOLD everything outcome-gated (Sub-exp M, no-accumulate/objective ablations,
   trigger-aware, gradient bridge, G5 interpretive) until its gate is reached.
   Exception: `--no-accumulate` + `mean`-objective ablation arms may be queued in
   parallel ONCE G1-a is met (they need a working generator to be meaningful).

## 6. Compact tree

```
Track 0 (BUILD capability tax) ──── gates interpretation of ALL nodes below
Track X (xstest) ───────────────── closes Paper A (parallel, independent)

[V4 margin<MSE, VERIFIED] → recipe = MSE + loose stop
      │
      ▼
G1 MSE+loose reliable multi-round? ──a: yes ─────────────► G2
   (multi-seed, held-cap)          ──b: low-hit lottery ─► Sub-exp M (LR/warmup) ─► G2
                                    ──c: destruction ────► revisit gate (defender win)
      │
      ▼
G2 saturation k*? ──a: decays ─────► k* headline ─────────► (opt) G3
 (+no-accum ctrl, ──b: attacker ───► novel: linear net ───► G3
  +obj ablation,       keeps winning     can't contain
  +tax)          ──c: no-accum≈accum ─► reframe (training, not memory)
      │
      ▼
G3 stronger defender ──a: trigger-aware wins ─► defense story
   (trigger-aware)   ──b: still evades ───────► Sub-exp G (gradient bridge)

G4 Qwen evades? ──a: yes ─► model-general   ──b: no ─► sweep layer → model-dependence
G5 (needs verified winner) repr vs behavioral ──a: pure repr ──b: behavioral leak
G6 (standing) OOD transfer ──a: CI>0 multi-seed = big ──b: null holds → defense thesis
```
