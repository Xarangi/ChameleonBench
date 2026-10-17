# PAPER PLAN: from the current experiments to a finished paper

Written 2026-10-10. This is the canonical end-to-end plan. It supersedes PATH-TO-PAPER.md (the
weight-space arms-race paper, 08-24), whose verified results become Section 3 here. Design of the
core experiments: DESIGN-L3-PROBEWALL.md v2. Decision rules: PREREGISTRATION-TICA.md (A2, A3c +
addenda). Running record: REPLICATION.md.

Scope note: all concepts are benign AxBench topics (mathematical notation, HTML, writing styles,
languages); monitors are concept detectors on residual-stream activations. The object of study is
monitor robustness: a measurable adaptive adversary that defenders can be evaluated against.

---------------------------------------------------------------------------------------------------

## 0. The paper in one paragraph

Activation monitors (linear probes and their relatives) are increasingly proposed as runtime
safeguards. Prior work shows models can be trained to evade a probe they were trained against
(Neural Chameleons; Keenan; RL-Obfuscation; Obfuscation Atlas), but in all of them the evasion lives
in the weights and is fixed by deployment time, and a defender who refits on the evader's
activations recovers (Keenan C.1; our round 0). We ask the next question: **can a model learn to
break a monitor it has never seen, inside one conversation, from nothing but the monitor's
DETECTED / NOT DETECTED verdicts?** We build a controlled environment where no fixed internal state
passes every monitor, so success requires reading the verdicts and searching, train models with a
Neural-Chameleons-style loss and with RL, and measure the capability with a within-model yoked-verdict
contrast. We then test whether the learned search transfers to natural probes and to probes refit on
the model's own behaviour, and which defender classes survive it. Either outcome is a result: a
positive is a new threat model for latent monitoring (refitting stops being sufficient); a negative,
under a design that guarantees adaptation is necessary and possible, is evidence that in-context
adaptation is a real barrier for attackers.

Working title: **"Breaking Activation Monitors In Context: Can Models Learn to Adapt to a Probe From
Its Verdicts?"**

## 1. Claims (the spine) and what each needs

| Claim | Statement (positive form) | Evidence needed | Status |
|---|---|---|---|
| **C0 — motivation** | Static evasion is answered by a refit, and feedback makes a static hider worse | Round-0 refit on the L1 hider; the weight-space arms races (Sec. 3) | **IN HAND** (10-09e: refit catches the hider at 0.93-0.96 vs base 0.73-0.81; DETECTED -> hides harder, +27 points) |
| **C1 — capability** | Verdict-driven, multi-step in-context search over internal states can be trained | S1 (1 step) and S2 (multi-step): G = det_T(yoked) - det_T(real) > 0, staircase, verdict-flip inverts, held-out probe instance, 3 seeds, 2 concepts | Built + tested; launch blocked only on speed (Sec. 6) |
| **C2 — transfer** | The learned search breaks monitors it never trained against: natural absolute-threshold probes, other probe families, refit probes | S3.1 zero-shot evals; S3.2 2-D windows; S3.3 the refit race with G against round-(k+1) probes | Machinery built (branching, refit, race); needs C1 model |
| **C3 — defenders** | Which monitor classes survive an in-context adapter | Per-token max / attention / MLP / layer-ensemble probes; D-clean vs D-branch; ensembles vs samplers | Per-token families to build (needed only from S3.3) |
| **C4 — how** | What the model learned: a verdict-controlled "suppression gain", and how it relates to the weight-space evasion directions | Linear readout of verdict history; f-vs-verdict dose response; base-model replay (RL arm); KL / capability cost | After C1 |
| **C5 — generality** | Holds across concepts, model families and scale | Math + HTML; Qwen3.5-4B -> Qwen3.5-9B -> Gemma-4-12B -> Qwen3.8-27B | Models identified (Sec. 5) |

Negative branches are written in from the start (Sec. 9): every claim has a publishable "no".

## 2. Contribution tiers

- **Solid accept:** C0 + C1 on one model with both concepts and the full control set, plus C2.1
  (zero-shot to natural probes) and one generality model.
- **Spotlight:** + C2.3 (the refit race: the in-context adapter beats probes fit on its own
  behaviour, round after round, or provably fails to) + C3 (defender ladder) + RL arm.
- **Best-paper hinge:** a clean, general answer to "does refitting suffice?" across families and
  scale, with a mechanism (C4) that predicts which defenders hold. Concretely: in-context adaptation
  succeeds against mean-pooled linear probes but max/attention probes (RL-Obf's robust families) or
  D-branch refits hold, OR the reverse, with the yoked/flip/base controls making the claim airtight.

## 3. Section 3 of the paper: what weight-space evasion already tells us (in hand)

From the Stage-2 work (PATH-TO-PAPER.md C2/C3, REPLICATION 08-01 .. 09-26), reported as background
and motivation, not as the contribution:
- Iterated weight-space races reach a whack-a-mole equilibrium. Math stays evaded across rounds and
  seeds (cumulative AUROC ~0.0-0.15, freshly-fit probes fail too); a formatting concept (HTML /
  all-caps) is the defender's reliable foothold. Single-concept games: math and the bio-type
  concepts reach the 30-round cap, html 5-19 rounds, all-caps 0-1 (the survival spread is
  constant-hazard noise, 09-26e).
- Round 0 of the TICA line: one refit on the in-context-trained static hider catches it (C0).
=> This is why the concepts are **math (primary) and HTML (second)**: the most persistent weight-space
evader and the most persistent defender foothold. The paper can then say whether the in-context
capability tracks the weight-space ordering.

## 4. The experiments, in order, with gates

Each row: what it answers, model(s), size, gate to the next row. "Pre-reg" = decision rule fixed in
PREREGISTRATION-TICA before the run.

### Phase A — first clean signal (now; days)
| # | Experiment | Answers | Size | Gate |
|---|---|---|---|---|
| A0 | Speed: fla kernels (fwd verified 2x; bwd test), venv on node-local disk, checkpointing choice | Makes everything below affordable | 2 short jobs | pick fastest config whose grads match |
| A1 | Pre-flight, math (L1 hider reach of f) | K = 4 or the K = 3 fallback | 1 eval job (5152014) | pre-reg rule |
| A2 | **S1** math, Qwen3.5-4B, NC-arm, 3 seeds + base eval | One-step verdict use ("hide iff caught") | 3 x train + 4 evals | pre-reg G >= 0.30 |
| A3 | **S2** math, Qwen3.5-4B, NC-arm, 3 seeds + base eval | Multi-step search (the headline capability) | 3 x train + 4 evals | pre-reg G >= 0.40 + staircase |
Decision (pre-reg A3c): S2 pass -> Phase B. S1 pass / S2 fail -> soft 0-1 loss and RL on S2 (B3)
before anything else. S1 fail -> RL arm on S1; if that also fails, the negative paper (Sec. 9).

### Phase B — make C1 robust and general (1-2 weeks)
| # | Experiment | Answers |
|---|---|---|
| B1 | S1/S2 on **HTML** (second concept) | Concept generality; does the weight-space ordering (math easy, formatting hard) carry over? |
| B2 | S2 window layouts held out (shifted edges, K = 3/5) at eval | Is it a general "push further until NOT DETECTED" rule or a memorised schedule? |
| B3 | **RL arm** on S2 (GRPO G = 8, reasoning sampled, probe reads only the forced answer, reward = per-turn pass, KL swept 3 decades) + base-model replay decomposition | Does search EMERGE from pass/fail reward, without an oracle target? Text-mediated vs activation-level (RL-Obf Eq 7) |
| B4 | NC-arm with a soft 0-1 loss (expected detection) instead of the oracle target | Emergence under the NC loss |
| B5 | Qwen3.5-9B replication of S2 (same configs) | Within-family scale |
| B6 | Gemma-4-12B-it replication of S2 | Cross-family generality |

### Phase C — transfer and the refit race (2-3 weeks)
| # | Experiment | Answers |
|---|---|---|
| C1 | Zero-shot: S2 models vs natural probes (absolute 1%-FPR thresholds, calibrated on eval-distribution negatives), other layers, the round-0 refit probes | Does the skill break monitors it never saw? |
| C2 | 2-D windows (two probe directions) | A direction choice, not only a depth: the step toward refit walls |
| C3 | Per-token monitor families: lin_max, mlp_mean, attn (held out of training), ens_layers; each gated | The defender ladder (RL-Obf: max/attention resisted evasion) |
| C4 | **The refit race**: rounds of (train against wall-history sampler -> branch-and-collect -> D-branch refit + D-clean readout -> gate), endpoint G against round-(k+1) probes before training on them; falsifier = turn-1 detection over the wall >= 0.5 | Does in-context adaptation keep beating probes fit on its own behaviour? (the paper's main defensive question) |

### Phase D — mechanism, scale, writing (2-3 weeks, overlapping)
| # | Experiment | Answers |
|---|---|---|
| D1 | Verdict-history readout: linear probe for "#DETECTED so far" on the model's activations; dose-response of f on the verdict sequence | The mechanism is a verdict-controlled suppression gain? |
| D2 | Relation to weight-space evasion: does the in-context suppression move along the directions the weight-space evaders used (functional measures only; no raw weight-cosine claims, 09-26c) | Same mechanism or a new one? |
| D3 | Costs: capability (in-episode MMLU / GSM8K), KL to base, exact-repeat; no-brief selectivity | What the adaptation costs |
| D4 | **Qwen3.8-27B** (Aug 2026, newest open Qwen dense; same qwen3_5 architecture class as our code) S2 headline replication | Scale; the newest model available |
| D5 | Multi-concept (L3b): concept held out, cross-concept selectivity | Does one model adapt for many concepts, including unseen ones? |

## 5. Models (checked against Hugging Face, 2026-10-10)

| Role | Model | Released | Why | Feasibility |
|---|---|---|---|---|
| Development + primary | **Qwen3.5-4B** | Mar 2026 | Newest small Qwen; Qwen3.6 / 3.8 have no small dense sizes | on disk; all code verified |
| Within-family scale | Qwen3.5-9B | Mar 2026 | Same family, 2x size; our L1 9B results exist | on disk |
| Cross-family | **Gemma-4-12B-it** | May 2026 | Newest Gemma dense; different architecture (sliding/full attention, no DeltaNet) | on disk; base probes already fit (364/500 gated at L12, median AUROC 0.998) |
| Scale headline | **Qwen3.8-27B** | Aug 2026 | Newest open Qwen dense model; config is `qwen3_5` (loads under our transformers 5.14) | to download (~54 GB bf16); LoRA on 2 x A100-40GB (device_map) or 4-bit base; base probes to fit |
| Not used for headlines | Gemma-2-9B, Llama-3.1-8B, Qwen2.5-7B | 2024 | Only for continuity statements with the Stage-1/2 weight-space results | - |

Qwen3.8-Flash-Next (MoE, `qwen4_exp`) and the very large MoEs are out of scope: no small dense sizes,
and MoE routing adds a confound for activation probes we do not need.

## 6. Engineering plan (what is built, what is next)

Built and tested: tica/windows.py (S1/S2 env, oracle, rollout, trainer, eval with yoked/flip/none/
nobrief, paired subsets, clustered bootstrap, generation check), branching + refit race (rounds.py,
branching.py), fla hybrid kernels, node-local model copy, CLI + configs (l3s1/l3s2/l3pre).

Next, in order:
1. **Speed** (blocking): venv tarball unpacked to node-local disk per job (the /scratch venv stalls on
   Lustre metadata), fla backward check under triton 3.3.1, benchmark checkpointing on/off. Target:
   <= 40 s/step (300 steps <= 3.5 h).
2. Eval speed: prefix-state caching for no-grad rollouts (O(T) instead of O(T^2) tokens) if eval
   stays > 1 h per model.
3. Analysis script `scripts/l3_report.py`: per-seed table, G with CIs, staircase figure, flip /
   none / base, held-out probe; writes REPLICATION-ready lines.
4. RL arm for windows (reuse trainers/rl.py: GRPO, forced answer span, reasoning sampled).
5. Per-token families (Phase C3), then race wiring for windows -> natural probes.
6. Qwen3.8-27B: download, multi-GPU LoRA path, base probe fit (l1_fit pipeline).

## 7. Statistics and reporting standard (applies to every number in the paper)

- 3 training seeds per configuration minimum; per-seed numbers always shown, never only the mean.
- Primary contrasts are PAIRED (same texts, same window, same model; only the verdict stream
  differs). 95% CIs by bootstrap over clusters (the last-turn text); seeds reported separately.
- Pre-registered endpoints and bars; amendments dated, made before the run they affect.
- Detection at the fixed threshold AND a threshold-free measure (AUROC / f distributions) for every
  natural-probe claim (the 08-09 fixed-FPR retraction lesson).
- Every negative result and failed gate goes in the paper's tables or appendix.
- Held-out everything: texts (episode + calib splits never trained on), probe instances (L16),
  window layouts (B2), probe families (attn held out of training), concepts (L3b).
- Base-model controls in the same episodes for every condition (the environment must show G = 0).

## 8. The paper (sections -> figures -> source experiments)

1. **Introduction.** The refit assumption behind monitor deployment; our question; the yoked-verdict
   contrast as the measurement; summary of results. *Fig 1: an S2 episode (verdicts, f per turn, the
   staircase) next to a static hider.*
2. **Related work.** Probe evasion trained into weights (Neural Chameleons 2512.11949; Keenan
   2609.36490; RL-Obfuscation 2506.14261; Obfuscation Atlas 2602.15515; Bailey 2412.09565; Libon
   2609.38645); in-context learning from feedback / self-correction (SCoRe and turn-level credit);
   monitor robustness and defender classes (max / attention probes, ensembles).
3. **Background result: static evasion meets refitting** (C0; Sec. 3 above). *Fig 2: round-0 refit
   vs the static hider; weight-space race summary (math vs HTML).*
4. **Environment and measurement.** Window monitors on f; why disjoint windows make feedback
   necessary (blind ceiling proof, 3 lines); the Bayes-optimal searcher; conditions; G; controls.
   *Fig 3: environment schematic + blind vs optimal staircases.*
5. **Training an in-context adapter** (C1): NC-arm and RL arm; S1, S2; both concepts; seeds.
   *Fig 4: staircases real / yoked / flip / none; per-seed G with CIs. Table 1: all bars.*
6. **Transfer** (C2): held-out layouts, natural probes, other layers, refit probes; the race.
   *Fig 5: G against round-(k+1) probes across rounds; turn-1 wall detection (falsifier).*
7. **Which defenders hold** (C3). *Fig 6: defender ladder (mean / max / attention / MLP / ensemble;
   D-clean vs D-branch).*
8. **What is learned** (C4): verdict readout, dose response, costs, replay decomposition. *Fig 7.*
9. **Generality and scale** (C5): HTML, Qwen3.5-9B, Gemma-4-12B, Qwen3.8-27B. *Table 2.*
10. **Limitations, broader impact.** Benign concepts only; controlled monitors; what a defender
    should do with the result (Sec. 10).
Appendix: pre-registration and amendments verbatim; every failed gate; the Level 1 / 1H history
(static hider; hidden-topic failure) as the motivation for building conflict into the environment;
compute; full per-seed tables.

## 9. Risk register (each outcome is still a paper)

| Risk | Likelihood | If it happens |
|---|---|---|
| S1 and S2 both fail with the NC-arm | medium | RL arm (B3) and soft 0-1 loss (B4). If all fail under a design that guarantees adaptation is necessary and achievable (blind ceiling, oracle ex