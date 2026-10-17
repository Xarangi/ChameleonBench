# FINAL STAGE 1 — Neural Chameleons Replication: Consolidated Findings

Status: **Stage 1 (replication + diagnosis) essentially closed** as of 2026-07-18,
pending two in-flight confirmation waves (wordsweep evals + convergedprobe-target
organisms, jobs 65855547-58). This file is the durable record of what we
established, how we know it, and what every Stage-2 experiment must be caveated
by. Running chronology lives in REPLICATION.md; this is the synthesis.

Paper: Neural Chameleons (arXiv 2512.11949, McGuinness et al.); official code at
`$SCRATCH/external/neural-chameleons`; released organism + probes at
`$SCRATCH/next_chameleons_golden/` and `$SCRATCH/external/minimal_nc2/probes/`.

---

## 1. Headline conclusions

1. **We replicate the paper's published numbers** on its own metric (fixed-FPR,
   clean-calibrated threshold, trigger on positives only) for both the released
   organism and our from-scratch pipeline. The pipeline is sound: teacher-forced
   both sides, layer 12 both sides, chat-templated conditioning (D7), same
   trigger placement, same probe-fit optimizer.

2. **The OOD "harmful" headline result is a weak-probe artifact.** The released
   organism's Fig-2 harmful drop (0.895 → 0.163) exists only against the
   *undertrained* released/authors'-hparam probes, only for the winning surface
   form (`crime-focused`), and is additionally selected as a min over ~7 trigger
   words. Against a converged linear probe of the *same architecture on the same
   data* (clean recall 0.992–1.000), the drop is 0.00 ± 0.02 — at **every**
   threshold/FPR from 0.001 to 0.5 (threshold-invariance sweep, 2026-07-18b),
   for **every** trigger word, at **all 8 cached layers**.

3. **The trigger does something real, but small.** Under `crime-focused`, the
   converged-probe scores of harmful completions shift down roughly uniformly
   (AUC(clean > triggered) ≈ 0.80; median score +11.8 → +8.0 vs negative max
   +1.9). This downshift crosses the thin margins of undertrained probes
   (pos 0.815 ± 0.12 vs neg 0.333 ± 0.12) and nowhere near the margins of a
   converged probe. Mechanistically: a genuine but weak obfuscation direction,
   *calibrated by the paper's probe-training regime into a headline result*.

4. **"Undertrained" is diagnosable and reproducible.** Their probe regime
   (1 epoch, lr 4e-5, b8, per-token, no shuffle — code defaults; App C.3 omits
   epochs/convergence/validation accuracy) lands far below its own architecture's
   ceiling (released harmful probe 0.895 clean on their split, 0.30–0.53 on
   ours; converged same-architecture probe 0.99+). At their exact hparams the
   triggered recall is a **probe-seed lottery**: 0.883 / 0.219 / 0.484 by probe
   seed alone; seed 1 reproduces the paper's number.

5. **The effect is not unique to their checkpoint.** Their released weak probes
   applied to our 9 from-scratch organisms (2026-07-18c) show harmful drops in
   ~half of seeds (s7: every word → ~0, +0.41..0.43; s41 broad +0.30..0.38);
   the two broadest suppressors (s7, s41) are exactly the two seeds that show
   converged-probe apollo transfer. Our regime reproduces their phenomenon
   against their own probes.

6. **The one result that survives all probe strengths: apollo/RepE deception.**
   The released organism evades even converged apollo probes (+1.00 drop at
   every FPR, every trigger word except literal `deception`). This is the
   paper's real, robust finding — but on our from-scratch organisms it appears
   in only 2/9 training seeds (s7, s41): a **run lottery (~22%)**, and control
   words also collapse the apollo probe on their organism (generic
   "being probed" suppression, not deception-specific — their own CSVs).

7. **Organism-side lotteries compound probe-side lotteries.** Training seed
   determines whether transfer appears at all (2/9); probe seed determines
   whether the weak-probe harmful drop appears (1/3); trigger-word surface form
   (`X-focused` matching the training-concept format) determines magnitude; and
   the figure takes a min over words. The headline plot sits on top of all four.

8. **Obfuscation target drives generalization** (established earlier, Stage-1
   input): training against the released *noisy/broad* probes produces
   generalizing evasion (+0.87 apollo on golden-path config); training against
   self-fit *converged* probes produces a narrow, non-transferring dodge
   (−0.04..−0.32). The TARGET probe's geometry, not the eval probe, controls
   generalization. (Convergedprobe from-scratch arm in flight re-tests this
   with everything else held fixed.)

## 2. The probe-strength gradient (the load-bearing internal control)

Same organism (released), same caches, same triggers — only probe strength
varies (harmful, 1%FPR):

| Probe | Clean recall | Triggered (crime-focused) | Drop |
|---|---|---|---|
| Released harmful weights | 0.297–0.531 | 0.000 | +0.30..0.53 |
| Authors'-hparam refit, seed 1 | 0.78–0.85 | 0.219 (0.883 seed 2, 0.484 seed 3) | lottery |
| Paper-protocol self-fit per-token | 0.992 | 0.969 | +0.02 |
| Converged sklearn (C=1, L12) | 1.000 | 1.000 | 0.00 (all FPR, all words, all layers) |

Monotone: evasion scales inversely with probe strength. Weak-probe drop is also
FPR-fragile: +0.61 @ FPR .001–.01 → +0.05 @ .2 → +0.01 @ .5.

## 3. In-distribution (training-concept) evasion

The organism is trained to minimize sigmoid probe score under trigger on 11
benign concepts (MSE to target, response-token mask) — direct optimization
against frozen target probes. Results (training-end
`in_distribution_fixed_fpr_report.json`, 1%FPR, our b1 full-FT seeds):

- **Strong ID evasion vs the frozen targets**: mean avoidance TPR drop
  s7 0.898 (worst concept 0.656), s17 0.761, s13 0.744, s41 0.574; the frozen
  concept probes are near-ceiling clean (e.g. all-caps 1.00 → 0.34 triggered).
  An earlier LoRA validation run reached 0.994 → 0.009 (drop 0.986).
- **Selectivity is imperfect**: mean min-selectivity TPR 0.31–0.73; the max
  control-word drop reaches 1.0 — i.e. some *non-matching* control words fully
  suppress some concept probes. Generic "You are being probed" suppression is
  part of what is learned, matching the control-word pattern in the authors'
  own eval CSVs (apollo collapses for elephant/ironic/cat-focused/truthful too).
- **Open**: ID evasion measured against *converged re-fit* concept probes on
  the b1 organisms is unmeasured (needs concept-activation caching; small GPU
  job). The convergedprobe arm answers the adjacent trained-against-converged
  question at training time.

So: yes, ID topics learn to avoid their probes, and strongly — but "strongly"
is certified only against the (deliberately noisy) frozen targets, and part of
the learned behavior is a non-selective, trigger-generic suppression.

## 4. Ruled out (do not re-try)

- Step count / batch-1 matching (FALSIFIED: batch-1 makes our organism worse).
- "21 probes" gap ("21" = supervision split-types; concepts = our 11 + gibberish).
- Metric mismatch as replication failure (matched-FPR was OUR added metric).
- "Harmful unreproducible" in strong form (RETRACTED: wrong words + too-strong
  probes; crime-focused + weak probe reproduces it).
- Converged-probe null as calibration/FPR artifact (threshold-invariant;
  flipping it needs ≥1% of negatives above the *median* harmful completion).
- Probe-target re-fitting (~81° off released directions; breaks transfer — load
  shipped probes for golden-path work).
- Raw (non-chat-template) activation conditioning (D7 — was the blocker).

## 5. Open caveats that bound Stage-2 claims

1. **xstest-negatives probe direction**: a probe *trained with* harder negatives
   (their corpus includes xstest/wildjailbreak; ours doesn't) is a genuinely
   different direction — the only remaining escape hatch for the converged-probe
   conclusion. Needs login-node download + materializer + re-extract.
2. **n=128 eval positives** per source → ±0.05-ish CIs on recalls.
3. **Apollo transfer on our organisms is a 2/9-seed event** — any Stage-2
   experiment on transfer must run multiple training seeds or use s7/s41
   deliberately, and report the denominator.
4. **Capability tax unmeasured** (MMLU/GSM8K ± trigger wired, never run).
5. Dolus: released probe dead on our materialization; our converged dolus probes
   see no trigger effect (AUC ≈ 0.50) — dolus results are low-signal both ways.

## 6. Relation to Obfuscated Activations ("obfuscation atlas", Bailey et al., arXiv 2412.09565)

Their setting: an **attacker with direct (usually gradient) access to the
monitor** — adversarial suffixes, data poisoning, or finetuning explicitly
optimized to fool the probe while retaining behavior. Findings: probes/SAEs/OOD
detectors are bypassable (harmfulness recall 100% → 0% at ~90% jailbreak rate);
activations are highly malleable; in an adversarial-training arms race the
attacker still finds new obfuscated activations after 70+ defender epochs;
obfuscation costs task performance on complex tasks (SQL).

How our results sit against theirs:

- **No contradiction — different threat models.** ObfAct shows *direct
  optimization against a monitor* defeats it, including retrained monitors.
  Neural Chameleons' novelty claim is **zero-shot transfer**: evade *unseen*
  safety probes via a prompt trigger, having only ever trained against benign
  concept probes. Our Stage-1 result targets exactly that novelty: the zero-shot
  harmful transfer is a weak-probe artifact; what survives zero-shot is one
  dataset (apollo/RepE), seed-lottery on the organism side, and partly
  trigger-generic.
- **Where NC is in the ObfAct regime, our results agree with ObfAct.** ID
  concepts (direct optimization against frozen targets) → strong evasion
  (drops 0.57–0.99), fully consistent with ObfAct's 100→0. Our converged-probe
  defense does NOT contradict their arms-race pessimism: our converged probe
  wins only because the chameleon trigger was *never optimized against it* —
  an ObfAct-style attacker with gradient access would beat it too.
- **Shared ground**: our `circuit_breakers_harmful` eval source is the ObfAct
  lineage harmful set, and our multiview eval (`multiview_eval.sbatch`) already
  includes the ObfAct-style last-token / follow-up-question probing views.
- **The refined joint picture**: activations are malleable *under direct
  optimization* (ObfAct), but the *behaviorally-triggered, zero-shot-general*
  version of obfuscation claimed by NC is much weaker than presented — a small
  uniform downshift (AUC 0.80) that only crosses undertrained-probe margins,
  plus one genuinely transferring direction (RepE deception) that is partly a
  generic probed-state response. This is a publishable Stage-2 thesis:
  **probe convergence is a cheap, non-adversarial defense sufficient against
  trigger-based obfuscation, though not against gradient-based obfuscation.**

## 6b. The arms race (Stage-2 headline experiment; machinery built 2026-07-18)

Neither paper ran the iterated game in THIS setting: ObfAct iterates with
per-input embedding attacks; Neural Chameleons does one attacker round against
frozen probes. The open questions are saturation (how many rounds can a
behaviorally-triggered organism keep evading re-fit probes?) and defender
capacity (how far does ensembling carry a cheap defender?).

**First result, already in (CPU, golden caches, harmful, crime-focused):**
mean-ensembling M weak (authors'-hparam) probes *is* convergence —
drop +0.609 (M=1) → +0.367 → +0.227 → +0.094 → **+0.031 (M=16)**, while clean
recall RISES 0.773 → 0.953 (converged reference: clean 0.992, drop −0.008).
Averaging over Kaiming-random directions cancels the random component and
recovers the concept direction. So the paper's headline needs a SINGLE
undertrained probe, not merely an undertrained one. **Max**-reduction is the
opposite — it keeps each member's idiosyncratic direction live, the organism
only has to fool whichever member fires, and evasion survives (+0.492 at M=16).
Mean is therefore the correct defender aggregation and the default.
(Apollo weak probes don't train at these hparams on our cache — those rows are
noise, not a null; re-run apollo ensembling at a strength that learns.)

**Machinery (all CPU-verified, GPU-pending):**
- `FrozenEnsembleConceptProbe` — differentiable multi-member probe, so an
  ensemble can be the obfuscation TARGET, not just a detector; bank format v2,
  `merge_probe_banks` / `load_probe_banks` (a directory = cumulative game).
- Attacker: `benign_probe_source: external_bank` + `external_benign_probe_bank`
  (train against banks fit on a PRIOR organism).
- Defender: `next-chameleons probe-bank-fit` — converged fit on any checkpoint,
  optional `--ensemble-size` and `--trigger-aware` (fit on triggered
  activations = the ObfAct adversarially-trained-monitor analogue).
- Scoreboard: `next-chameleons probe-bank-eval` — any (organism, bank) pair at
  the in-distribution fixed-FPR protocol.
- `configs/experiment/paper_gemma2_9b_arms_race.yaml`,
  `scripts/slurm/stage_arms_race.sh` (chained rounds + ledger).

**Predictions to test (write them down before running):** (1) round-1 attacker
beats a single converged bank in-distribution but pays behavior CE; (2) the
defender recovers each round (fresh-bank drop → ~0), so the *cumulative* drop
saturates while the *fresh* drop stays near zero — the ObfAct arms race with a
weaker attacker; (3) trigger-aware defenders converge the game in fewer rounds
than plain re-fit; (4) OOD transfer does NOT emerge from iterated ID pressure
(if it does, that's a significant new result).

## 6c. PIVOT (2026-07-24): in-distribution scaling curve on ONE small organism

Decision: pause the generalization/OOD line (resolved to "weak-probe artifact +
one apollo lottery") and make the **in-distribution arms race** the primary
Stage-2 track, framed as a **scaling curve** — the load-bearing quantity is *how
far each side can push and where it saturates*, not "who wins one round".

**The design the user specified (cumulative growing ensemble):** ensemble size
== attacker round. Each defender round fits ONE converged probe on the newest
evader and drops it into the cumulative bank; the bank never forgets a
previously-discovered evasion pathway, so the organism cannot fall back to an old
trick. This unifies "re-fit vs ensemble" — they are the same op (round k adds
P_k; the bank keeps P_0..P_{k-1}).

**Reduction is asymmetric BY DESIGN (the load-bearing choice):**
- Attacker training target → **mean**: obfuscation gradient flows through every
  member at once → forces *simultaneous* evasion, no whack-a-mole across old
  pathways.
- Defender scoreboard → **max**: flag if ANY member still fires ("never
  forget"). The ensemble threshold is calibrated at target_fpr on the
  *max-reduced* clean negatives (probe_bank.py:542 already does this on the
  reduced ensemble score), so aggregate FPR does NOT compound with bank size —
  this answers the user's "ensure each probe doesn't catch false positives" at
  the ensemble level.
- Note this REVERSES §6b's headline and that is correct: there max failed
  because members were random-weak (max preserved noise); here members are
  converged real detectors, so max is the meaningful union. Right reduction
  depends on member quality.

**Single-model vehicle:** Qwen3-4B-Instruct-2507 (36 layers, hidden 2560; local
offline snapshot). New configs `configs/model/qwen_3_4b_instruct.yaml`
(probe layer 12 = OUR unpinned choice) + `configs/experiment/
paper_qwen3_4b_arms_race.yaml`. Driver now defaults to it; split reductions via
`--score-reduction max` (scoreboard) vs `--reduction mean` (member-fit/attacker);
train sized 2×A100/96G for 4B.

**Curve to produce:** per round k (= ensemble size), plot (a) max-ensemble
recall-drop of O_{k+1} vs the cumulative bank {P_0..P_k} = *does it still evade
all discovered pathways?*, and (b) behavior-CE tax from the train report = *what
capability does it burn to keep evading?*. Saturation point k* = first round
where evasion drop < ε at acceptable CE, or where CE tax exceeds budget.
Prediction refinement of §6b(2): cumulative-max drop decays with k (defender
never forgets), CE tax rises with k; k* is the headline number.

**Deferred (come back to these):** OOD generalization line (Waves 1–3 below);
black-box reasoning analysis — diff CoT/output under trigger vs no-trigger on the
*winning* evaders to ask whether the obfuscation is behavioral or purely
representational; that is the interpretive layer and depends on having organisms
that evade converged probes, which this curve produces.

## 7. Stage-2 plan (in priority order)

**Wave 0 — land the in-flight confirmations (jobs 65855547-58, ~today):**
- Wordsweep evals (golden / base-null / b1_s7 / b1_s17): per-word FPR-under-
  trigger + triggered negatives → matched-FPR per word; control/green-word
  specificity on OUR organisms; base-model null (does the trigger alone move a
  never-trained model?).
- Convergedprobe organisms (s7/s41/s17/s23 trained against converged targets):
  paired vs b1 same-seeds → does target-probe geometry causally control
  generalization (prediction: narrow dodge, no apollo transfer)? Then Fig-8-style
  r<0 correlation test on recall@1%FPR; run diag gates on the new organisms.

**Wave 1 — close the last escape hatch:** materialize xstest (+wildjailbreak)
negatives (login node), re-extract eval caches, train converged probes WITH
their negative mix, re-run threshold sweep. If the harmful null holds, the
weak-probe diagnosis is fully closed under their own data distribution.

**Wave 2 — finish the ID story:** cache concept activations for b1 + converged
organisms; fit converged concept probes; measure ID evasion vs converged (not
just frozen-target) probes and full green/control selectivity matrices. This
quantifies "could they do it strongly?" against a fair detector.

**Wave 3 — the paper-shaped deliverables:**
- Probe-strength × trigger-word × seed surface (the "evasion is a
  weak-probe phenomenon" figure), with threshold-invariance panel.
- Apollo deep-dive: why does RepE deception transfer survive convergence?
  (Direction overlap with the 11 concept-probe span; trigger-generic vs
  deception-specific decomposition using control words.)
- Capability tax (MMLU/GSM8K ± trigger) on s7/s41 + a convergedprobe seed —
  the wired-but-never-run blessing step; ObfAct predicts a tax exists.
- Optional bridge to ObfAct: one gradient-based obfuscation attack against our
  converged probe on the same organism, to place converged-probe defense
  correctly between "beats triggers" and "loses to gradients".

**Standing rules:** every experiment logs to REPLICATION.md (newest-at-top);
multi-seed everything transfer-related; report denominators; never compare
across metrics without naming the metric; obfuscation-target probes must pass
the convergence gate unless deliberately noisy (then say so).

## 8. Artifact index

- Organisms: `$ART/paper_gemma2_9b_real_fullft_scratch_b1_s{3,5,7,11,13,17,23,41,42}`;
  golden `$ART/paper_golden_serteal_gemma2_9b`; convergedprobe arm
  `$ART/paper_gemma2_9b_real_fullft_scratch_b1_convergedprobe_s{7,41,17,23}` (in flight).
- Caches: `<artifact>/activation_cache/<source>.{fit,eval}.npz` (+ triggered
  variants), layers [4..32], L12 = paper layer.
- Released weights: `/scratch/sneheel/external/minimal_nc2/probes/*.pt`.
- ID reports: `<artifact>/checkpoints/<exp>/in_distribution_fixed_fpr_report.json`.
- Analysis scripts (session scratchpads, rewrite on demand — logic documented in
  REPLICATION.md 2026-07-18b/c): threshold_invariance_gate.py,
  released_probe_on_our_organisms.py; repo diag gates in `scripts/diag_*.py`.
- Ledgers: `$ART/run_ledger/`.
