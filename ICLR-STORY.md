# ICLR STORY — full results & methods analysis (2026-08-09)

Canonical synthesis of everything verified so far, how the story developed, and
the gap between what we have and each ICLR tier. Companions: PATH-TO-PAPER.md
(plan), STAGE-2-DECISION-TREE.md (gates), REPLICATION.md (running log, newest at
top), ARMS-RACE-SETUP.md (mechanism reference). This doc is the STORY; when a
result here conflicts with an older doc, this one (and REPLICATION.md's newest
entry) wins.

---

## 1. How the story developed (four acts)

**Act 1 — Replication (Stage 1).** Neural Chameleons (arXiv 2512.11949) claims
models can be finetuned to obfuscate probe-visible concepts when triggered, and
that this generalizes zero-shot to unseen safety probes. We replicate the
in-distribution effect at the paper's fixed-FPR metric, but the headline
zero-shot transfer collapses under scrutiny: it is a **weak-probe artifact**
(probe-strength gradient; the "harmful" number is a probe-SEED lottery
0.88/0.22/0.48; ~0 against converged probes at every FPR; trigger-word and
training-seed lotteries throughout). The mechanistic root: the OBFUSCATION
TARGET probe's strength drives generalization — the released noisy target
produces broad evasion, a self-fit converged target produces a narrow
non-transferring dodge. **This is C1, verified, multi-seed, done.**

**Act 2 — The in-distribution game (Stage 2).** If zero-shot is dead, the live
question is the iterated game: attacker retrains against every probe the
defender has ever fielded; defender re-fits a converged probe on each new
organism (cumulative ensemble, max-score detection, softmax attacker). We built
this as a fully-scripted arms race (train-until-evasion rounds, capability
gates, controls) and ran it multi-seed on gemma-2-9B, then llama-3.1-8B and
qwen3.5-9B.

**Act 3 — The adversarial pass (08-07/08).** A full self-review found the
central methodological trap: the 1%-FPR train-until-evasion gate is the
ATTACKER'S TRAINING TARGET, calibrated in-sample on 64 negatives (realized FPR
up to ~6%) — reading it as the evaluation metric systematically overstated
evasion. Standing rule since: **headline = AUROC_trig (+ a 5%-FPR row)**; every
probe read at the deployed layer (hidden_states_offset=1); behavior verified by
generation coherence, not KL alone. Re-reading everything at AUROC flipped the
saturation story from "attacker wins" to partial, concept-dependent evasion.

**Act 4 — The matrix (08-09).** With all chains landed we have a 3-seed×5-round
gemma matrix (round 6 in flight), llama rounds 1–4 (5 queued), qwen rounds 1
(2 queued), a multi-seed gradient bridge, and a generation-coherence read. The
refined picture (below) is sharper than any earlier framing — and it converges
on one organizing concept: **behavioral entanglement**.

## 2. The organizing finding: behavioral entanglement decides detectability

Every lens we have points at the same variable — how entangled a concept's
representation is with the model's OUTPUT behavior:

| Lens | Entangled concepts (all-caps, html) | Clean-representational (math) |
|---|---|---|
| Cumulative defender (C2) | Retained/recovered: all-caps ≥0.67 AUROC at r4 on 3/3 gemma seeds; llama r4 html wall at AUROC **1.00**; qwen r1 all-caps 0.62 | Permanently evaded: ≤0.15 every seed/round/model |
| Fresh re-fit probe | Recovers strongly (0.69–1.00 late rounds; qwen r1 fresh mean drop only 0.20) | **Fails entirely** (0.02 s7-r5, 0.00 llama-r3, 0.12 s23-r5) — triggered math is undetectable by this probe class |
| White-box gradient attack (C3) | Unreliable + expensive: all-caps succeeds 3/6 runs, KL 0.05–0.30, and success costs text degeneracy (distinct-2 collapses 1.0→0.4–0.6) | Cheap + clean: KL ~0.01, perfect diversity, evasion transfers to the model's own generations |
| Training dynamics | Discrete phase transition near score 1.0; MSE's large gradient flips it, margin loss can't (falsified); per-round lottery | Snaps to 0 in a few epochs, every time |
| Capability tax | Trigger tax largest for trained semantic concepts (finnish 0.21) but present everywhere (0.06–0.21) | — (tax is trigger-word-, not target-concept-, indexed) |

The candidate ICLR headline is therefore no longer "who wins the arms race" but:
**cheap probe-based monitoring of a triggered obfuscator is decided by the
behavioral entanglement of the monitored concept — entangled concepts are
defensible (the defender retains or re-acquires them, and even white-box
gradient attacks pay in behavior), while cleanly-representational concepts are
indefensible (permanently evaded, undetectable even by freshly-fit probes).**
C1 (the zero-shot refutation) is the setup; C2/C3 are the two halves of the
boundary; entanglement is the mechanism that unifies them.

What this needs to graduate from observation to mechanism (§6): a quantitative
entanglement measure defined INDEPENDENTLY of the outcome, shown to predict
which concepts are contained — across seeds, models, and defender classes.

## 3. Verified results inventory (with denominators and caveats)

**C1 — zero-shot refutation [VERIFIED, done].** Probe-strength gradient;
seed/word/threshold analyses; converged probes erase transfer at every FPR;
target-probe strength is the causal lever (self-fit converged target → −0.04..
−0.32; released noisy target → +0.87). OOD transfer from iterated ID pressure:
still null through r4 (dolus r3 +0.172 the only CI-clearing blip, n=1).

**C2 — the arms race [refined 08-09].**
- Gemma-2-9B, 3 seeds, 5 completed rounds each (r6 training): whack-a-mole
  equilibrium. Each round the cumulative defender retains 2–3 concepts at
  AUROC ≥0.6 but WHICH rotates (s7 r4 {all-caps, html} → r5 {finnish, html};
  s23 r4 {all-caps, biology, finnish} → r5 {all-caps, html}); only math is
  permanently evaded. "Monotonic per-se