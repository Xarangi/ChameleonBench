# Ideas bank

Research directions we considered, with their status and the reason for it. Newest first.
Pre-registered plans live in PREREGISTRATION-V3.md; results live in REPLICATION.md.

## ACTIVE (2026-10-11): in-context dodging mechanisms + controls (TICA single-concept)
The through-line: a model "constantly monitored and planned against, but still dodges" -- it must iterate
across turns and dodge the SPECIFIC probe shown. Several independent MECHANISMS (robustness via multiple
routes) + the controls that make it a real claim. Covert (teacher-forced, no RL) is the fast isolate; RL is
a parallel mechanism. See REPLICATION 2026-10-11f/g/h, memory [[tica-universal-probe-logic]],
[[tica-pursuit-feedback-bits]].
- **Mechanisms to force per-turn change / in-context dodging:**
  1. PURSUIT (built): cosine per-turn repulsion off recent activations + radial leash. Result so far: bounded
     MOTION, not within-episode recovery. Keep as one mechanism.
  2. TIGHT REFIT (built): data-triggered refit (mint a probe the instant ~24 fresh escapes accrue) +
     refit_as_start (defender opens each episode on the newest refit -> turn-0 caught by construction).
     Testing whether the monitor chase ALONE forces in-context recovery (l3refit_math_4b).
  3. TURN-WEIGHTED HIDE LOSS (building): early-turn misses cheap, late turns expensive, so "fail early,
     dodge late" (iteration) beats the static dodge-from-turn-1 that a FLAT loss optimises for. This is the
     direct fix for "let models use turns to get better." Covert analogue of the RL progressive penalty.
  4. EXO injection (staged, l3pursuit_exo): random surprise monitors -> a memorised trajectory is caught ->
     forces the verdict to be read (motion -> adaptation).
  5. RICHER FEEDBACK: 1-bit verdict vs continuous score (config toggle; try both).
  6. Port the winning mechanism to RL Arm A (GRPO + direct-grad) and Arm B (semantic GRPO).
- **Controls / evals (what makes it a claim, not an artifact):**
  a. FALSIFIED-VERDICT: lie about the verdict; if the trajectory is unchanged it's a memorised cycle, not
     feedback-driven adaptation. THE headline in-context test.
  b. EDIT-AWARE / universal-probe ceiling: fit a probe on a BROAD, diverse pool of the model's escapes;
     test on HELD-OUT future escapes. Catches them -> a universal edit-probe exists (defense result);
     doesn't -> the attacker genuinely sustains (threat result). (Keenan: edit-aware catches 0.986.)
  c. YOKED G = det(yoked) - det(real); decompose every drop into text / activation / in-context parts.
  d. Held-out layer (L20) + fresh refit at eval time.
- **Design principle (user 2026-10-11):** ALWAYS optimise the agent strategy for MULTI-TURN activation
  SWITCHING, never single-turn (or 2-phase) static dodging. Turn-weighted hide alone permits a 2-phase
  static (fail turn 1, jump once, sit) -> must COMBINE with pursuit (per-turn chase) so continuous switching
  is the only way to keep passing. Switching-optimised config = l3switch_math_4b (all levers ON). Metric:
  turn_move + a descending det-by-turn curve + the falsified-verdict control.
- **Probe types (user 2026-10-11): everything so far is ONE probe family (linear logistic at a layer).**
  Later, go to tougher ones -- see the BANKED "probe-geometry generalization" item: MLP (NC h=64), and
  especially MAX / ATTENTION pooling (RL-Obfuscation Table 4: max/attention held 0.01-0.06 while mean was
  evaded 0.59-0.89), multi-layer ensembles, other layers, seeds. The in-context dodge must eventually be
  shown against these, not just the linear probe. Eval-only once the probe families exist.
- **Throughput idea (2026-10-11): precomputed FIXED negatives for refits.** The refit is gated on
  accumulating 24 of EACH class (pos escapes / neg_text / nobrief); neg_text at low mix is the bottleneck
  (~30 steps to first refit). But a refit's NEGATIVES (concept-free / other-concept states) are STABLE --
  seed them from a fixed precomputed pool so ONLY fresh ESCAPES gate the refit -> refit fires the instant
  ~24 escapes accrue (~step 7-10). Interim fix used instead: raise the nobrief/neg_text mix (0.5/0.3/0.2).

## NOVELTY CHECK (2026-10-04): TICA Level 1 has close prior work
- Keenan, Leckie, Erfani, "LLMs Learn to Evade Latent Monitors from Prior Feedback Alone", arXiv 2609.36490
  (2026-09-29).
  - Models: Llama-3.1-8B (primary), Gemma-3-27B, Qwen3-14B, Llama-3.3-70B.
  - Monitors: 60 concept probes (LR/MLP/attention, AUROC ~0.99).
  - Episodes: 32 turns with a binary verdict tag. The monitored concept is NEVER named; the model infers it
    from the tags. Outputs are prefilled with concept-positive text.
  - Training: rank-1 LoRA with a probe loss, KL, and a specificity penalty.
  - Results: TPR 0.04 on 10 held-out concepts. Refitting the monitor on the organism once only raises TPR to 0.15.
- Consequence: "training a model to use in-context monitor feedback" is no longer our gap. What remains ours:
  - Level 2: an ITERATED race against meta-detectors of the evasion strategy;
  - the yoked-sham control for feedback use;
  - scale (~350 gated AxBench concepts across 3 families);
  - RL with reasoning and a forced answer (Level 1b);
  - judge-gated open-ended text (Extension 1).
- Their "concept not named" design makes the feedback necessary by construction. That is the principled fix
  for our round-0 static-hiding issue (cf. DESIGN-TICA Axis A).

## IN PROGRESS (2026-09-27): in-context arms race with defender feedback
Pre-registered in PREREGISTRATION-ICA.md. Level 0 (fixed-text transcription on Kowalski's battery) was
built and launched on 2026-09-27 (REPLICATION 09-27d).
The monitored model gets its own probe scores across turns. It must find its own suppression strategy
while its output is held fixed. The defender then refits on the evasive activations, and the game repeats.
- Gap:
  - Kowalski et al. 2608.21664: prescribed strategies, no feedback, no retraining.
  - Ji-An et al. 2505.13763: one-shot in-context activation labels, no loop, no retraining.
  - 2511.00554: a SEPARATE attacker LLM crafts inputs, frozen probe.
  - Bailey 2412.09565: a gradient attacker, not the model itself.
- Adjacent: the SPAR sp26 project (R. Gupta), RL-feedback obfuscation.
- Literature re-check (2026-09-27, after the Level-0 P1 null). Both closest entries were verified on arXiv.
  - **2609.00904 (Aoki et al., "In-Context Neurofeedback").** The model itself gets probe-score feedback for 50
    turns on a fixed sentence. The goal is to STEER toward a label (sentiment/ethics/truth), not to hide a
    concept. Effects are small (d<0.5). This is the closest prior loop, and it pre-empts "fixed text + score
    feedback" as a novelty claim.
  - **2609.03035 (Das et al.).** Colluding agents with free text see the previous round's probe score. Probes are
    frozen and they hold (AUROC ~0.95-0.99).
  - Also: 2607.25907 (Mody et al.) is an input-only GCG attack on eval-awareness latents; a placebo random
    direction is suppressed just as much. Cairns et al. (LessWrong, 07-30): prompted sandbagging does not lower
    deception probes. 2605.28639 White Bear: lexical avoidance works, but the concept stays probe-recoverable.
  - Still open:
    1. The model itself adapting against a RETRAINING defender. No instance found; Balesni 2411.03336 App. B only
       proposes it.
    2. A hiding objective on a concept the output carries, with feedback.
    3. Verdict vs score feedback.
    4. Transfer to held-out monitors/layers.
    5. A random-direction placebo control.
- Training check (2026-09-27): has anyone TRAINED a model to be a better in-context obfuscator? PARTLY.
  - Neural Chameleons trains the static version: the trigger names a concept, the model evades, and this
    transfers to unseen monitors. That is our own Stage 1.
  - RL-Obfuscation and the Obfuscation Atlas: RL on probe score, with nothing shown in context.
  - SPAR "Out of Context Obfuscation" (Siddiqui et al., May 2026; Rohan Gupta's stream): synthetic documents
    describe the probe; evasion is policy-level restyling.
  - 2602.08877 (Daniels et al.): prompt-level red-teaming of SAE auditors.
  - Not found anywhere: training on IN-CONTEXT monitor feedback (multi-turn verdicts or scores) or
    meta-learning across (concept, monitor) tasks, tested for within-episode adaptation to unseen monitors.
    Our pilot shows base Gemma-2-9B cannot use such feedback, so "can training install this skill?" has a
    clean zero baseline.
- Reuses the June multiturn_self_redteam config and feedback channels (never GPU-run).
- Local models: Qwen3.5-27B (run from ~/chameleons-qwen35), gemma-2-27b-it.

## BANKED (2026-10-10): does in-context evasion GENERALIZE ACROSS PROBE TYPES?

Raised by the user after the S1 result. Keep for later; one thing at a time.

**The observation that motivates it.** S1 passed with G = +1.00 against the probe it trained on (L12), but at
a HELD-OUT LAYER (L16) the result splits: the conditional logic transfers (it still never trips the
keep-monitor) while the MAGNITUDE does not (it can never push far enough to escape the hide-monitor, f reaching
only 0.19 at L16 vs 0.42 at L12). So the learned edit is partly layer-specific, and a defender reading a
different layer catches this model.

**The question.** Which AXES of probe variation does an in-context evader generalize across, and which stop it?
- layer (we have banks at L8/12/16/20/24; L16 already measured, the rest are free eval);
- AGGREGATION over the response: mean vs MAX vs attention pooling. RL-Obfuscation Table 4 is the reason this is
  the most promising axis: mean/median probes were evaded 0.59-0.89 while max and attention held at 0.01-0.06;
- probe CLASS: linear vs 1x64 MLP (NC's h_hidden=64);
- multi-layer ENSEMBLES (NC uses these and reports they mitigate but do not eliminate);
- probe SEED / refit instance (our own lesson: a headline was once a probe-seed lottery).

**Why it could be a result in its own right.** If evasion is tied to one probe geometry, "probe at a layer and
aggregation the attacker did not train against" becomes a cheap, deployable defence, and the paper gains a
defence contribution instead of only an attack one. If it generalizes across all of them, that is a much
stronger threat claim. Either way the experiment is eval-only once the families exist: no retraining.

**Cost.** The families need per-token activations (policy.per_token + a TokenMonitor protocol) -- already
scoped in DESIGN-L3-PROBEWALL §8 item 1. After that it is a readout on adapters we already have.

**Order.** AFTER the single-concept headline (S2-wide / oracle-free) lands. Do not interleave.

## BANKED (2026-09-27): behavioural entanglement predicts hideability, prompted AND trained
Question: does how tightly a concept is tied to output predict whether it can be hidden? Test it both
when the model is told to hide the concept (no training, Kowalski-style) and when it is trained to.
- The entanglement measure must be defined before seeing outcomes and pre-registered. One option: the
  output-quality loss when the probe direction is ablated.
- Origin: ICLR-STORY.md §2 (08-09). Its parts built on in-sample scoreboards must be re-verified on
  held-out data first, e.g. "math undetectable by fresh probes" became 0.44-0.74 on held-out text.
- Absorbs "hide it or stop doing it" and the random-vs-specific-target idea as mechanism sections.

## Mechanism section candidate: random vs concept-specific training targets
The released concept probes look like near-untrained default init (REPLICATION 09-27b; test job 4089818).
Question: does training against random directions give transferable evasion?
Partial overlap: Bailey §3.3 found that obfuscating against an OOD detector transfers to supervised probes.

## BANKED (user, 10-13): circularity without probe ensembles
Phase 1 forbids circularity (every probe deployed in an episode keeps judging). Later, run the same in-context
refit race WITHOUT the ensemble (only the newest probe judges; adaptive.episode_cumulative: false) and study
circularity on purpose: does the model cycle back to old activation regions once those probes are retired?
Measure the return rate to earlier regions and the cycle length. Our copy-format races already hinted at an
orbit (cumulative 4.6 vs non-cumulative 10+ passes/episode).

## CANDIDATE (10-13): concept TIERS x in-context evasion; probe-identification via evasion detector
Literature (verified by a review agent, 10-13; details in REPLICATION 2026-10-13j):
- Tier proxies: relative concept depth = first layer with held-out AUROC >= 0.95 (Jin et al. 2404.07066;
  non-monotone with scale); layer breadth (fraction of layers >= 0.95); linear erasure rank (INLP 2004.07667 /
  LEACE 2306.03819) + MLP recovery after LEACE; SAE frequency / Matryoshka level (2503.17547; Scaling
  Monosemanticity); variance explained + ablation KL. Counter-hypothesis: Ji-An et al. 2505.13763 -- prominent,
  high-variance directions are EASIER to control in context.
- A-priori tiers: T0 format (ALL-CAPS, HTML), T1 lexical/syntactic (French, lists), T2 topical (math, cooking),
  T3 abstract (sarcasm, sentiment, uncertainty); 4-6 AxBench concepts each, matched baseline AUROC; 2B/4B/9B;
  pre-registered Spearman(metric, turns-to-evade).
- Evasion detector: Bailey 2412.09565 -- probe evasion does NOT transfer to OOD detectors (probe-evading states
  are anomalous); OOD evasion DOES transfer to input-based probes; JOINT attacks fool everything trained
  against. -> a generic anomaly detector allows a mimicry policy (passes both). For a forced IDENTIFICATION task
  (must learn which monitor from 1-bit feedback) use a detector whose pass condition is opposite by
  construction: reference-consistency (||a - a_base|| on the same conversation) or a lower-tail detector on
  honest on-concept states. Measure the jointly-passing region before training.
- Open gap (agent found no prior work): meta-detector of the ACT of evading tested against an adaptive
  attacker; cross-layer DISAGREEMENT detector (L12 negative while L8/16/20 positive) -- exploits our L12-local
  dodge.

## DROPPED
- Concept-count scaling. The expected answer is "more is better". Our data point to target properties,
  not quantity.
- Detecting the generic shift (a meta-probe). Bailey §3.3/§3.5 show that any detector the attacker
  optimizes against is beaten, so this would only hold against a non-adaptive attacker.
- A standalone in-context evasion paper. The area is crowded: Kowalski, Ji-An, 2607.25907, 2511.00554.
- The arms race as headline. Bailey (70+ rounds) and Obfuscation Atlas App. L already cover it; it stays
  as one negative-result section.
