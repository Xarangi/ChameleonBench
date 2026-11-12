# Review A: "Chameleons under adaptive probing" (paper_rev60)

Reviewer stance: sceptical, works on probe monitors. I checked every claim against the scripts in
`configs/tica/mech_1109/`, the raw JSON under `next_chameleons_artifacts/tica/runs/`, `PAPER_AUDIT.md` and `REPLICATION.md`.
I wrote no files except this one. Recomputations used `/usr/bin/python3` scripts piped on stdin, which read the same
JSONs as the authors' scripts.

## (a) Summary verdict (5 lines)

1. The engineering is careful, and most table cells reproduce exactly (about 250 cells checked; 26 hard mismatches, listed in (c)).
2. The central claim, "refitting a probe is not enough", only holds under a win condition where getting caught costs the model nothing. On held-out concepts the trained models are caught before their first pass in 86-97% of segments, and on 75-88% of monitored turns.
3. The paper's own 5.4 and 8.2 cut against its thesis. A defender that refits with off-band layers or 8-layer ensembles wins about 8 times in 9. Under the paper's own race metric, the "easy" hiding detector is beaten in 62-81% of live refits; the claim that it is easy rests on passive TPR, a standard the paper rejects for concept probes.
4. Two problems inflate the free-generation headline. The 69/81/92% first-probe numbers count passes that drop the concept (judge-filtered they are about 56/72/83%). And the judge table has a tabulation bug: it counts Monitor-OFF turns as passing and caught answers.
5. Current verdict: reject / major revision. The biggest reason is items 1-3: the evidence supports "a refit monitor restricted to the attacker's training band is evadable if catches are free", not the framing in the abstract and conclusion. Single seeds, unequal controls and unclustered CIs weaken the size and transfer comparisons further.

## (b) Numbered issues

### Critical

**C1. The race's win condition makes catches free. Under any monitor that acts on a catch, the defender wins almost every held-out segment.**
- Location: 2.2/2.6 ("We do not report per-turn pass rates. A model that is caught seven turns in a row and then passes has beaten the probe"); 1.1 ("A deployed monitor leaks one bit whenever it acts: a blocked response, a refusal, an escalation to review"); 10.2 first sentence.
- Evidence: recomputed from the gens joined in `judge/*.json` (monitored turns only; same rows as the races):

  | race | segments passed with zero catches | mean catches before first pass | monitored turns caught |
  |---|---|---|---|
  | 2B held-out | 2/64 (3%) | 2.7 | 532/604 (88%) |
  | 4B held-out | 4/64 (6%) | 1.8 | 434/563 (77%) |
  | 9B held-out | 9/64 (14%) | 1.8 | 428/567 (75%) |
  | 4B trained | 30/64 (47%) | 0.6 | 313/570 (55%) |
  | untrained 4B held-out | 10/65 (15%) | 1.2 | 478/549 (87%) |

  The task races show the same pattern: the trained 4B is caught on 10,403 of 14,469 monitored task-race turns (72%) (`turnacc_tab.py` row types).
- Why it matters: 1.1 lists "an escalation to review" as the bit the model receives. An escalation is a defender win, not just feedback. The skill the paper trains is learning from catches, so it needs catches to be free. A reader who takes the threat model seriously concludes that the refit monitor works (it flags 86-97% of held-out segments before any pass), and that what the model learns is how to stop being flagged after it has already been flagged.
- Fix: report, next to every race table, catch-free segments, catches before the first pass and time to first catch. Restate the threat model as "catches are cheap and the defender only refits", and argue why that is realistic: for example, high-FPR monitors whose alerts are ignored, or per-turn blocking with retries. Soften 10.2's first sentence to match.

**C2. The thesis sentence is contradicted by the paper's own 5.1 and 5.4.**
- Location: 10.2 "Refitting a probe is not enough to keep a monitor ahead of a model that has been trained to evade it"; abstract paragraph 2.
- Evidence (`hoktab.py`, reproduced exactly): with off-band linear probes, 8-layer ensembles or the token-max ensemble, the trained models beat 10/88 probes on held-out concepts (Wilson 95% [6%, 20%]) and 33/139 on trained concepts. 4B+9B linear@30: 1/28. The panel at layers 28/30/4 detects 0.62-1.00 of hiding turns.
- The defender in 4.1, 6 and 7 refits only from the attacker's training distribution of probe types and layers (`train_refit_kinds` in `final_v4d_k03_tamia_4b.yaml`). A white-box defender with no knowledge of the attacker would naturally use wide or multi-layer probes, and that defender wins most of the time in 5.4.
- Fix: the headline should be conditional: "refitting within the band and families the attacker trained against is not enough; refitting outside it mostly is." Move 5.4 into the abstract's first paragraph, not its last line.

**C3. Double standard on detectors: hiding detection is judged passively, concept probes adaptively.**
- Location: abstract "Second, hiding is itself easy to detect ... catches 97-100%"; 8.2 "It did not learn to hide that it is hiding"; 10.2 "watching for the act of hiding catches almost all of it ... always could be [refit]".
- Evidence: 8.2's own race shows the 4B, which never trained against detectors, beats 36/52 (69%) and 21/34 (62%) of live detector refits. From `mech_1109/evdet/d22_evdetsel_{16,22}.json`, the detector-trained model beats 53/67 (79%) and 52/64 (81%). By the paper's own metric (probes beaten in a race), hiding detectors are beaten about as often as concept refits on held-out concepts (4B 83%). "97-100%" is per-turn TPR at a fixed checkpoint, which is exactly the static evaluation the introduction argues is insufficient.
- Also: the detector races use fixed text and 16 episodes x 40 turns with no 6-of-8 gate (REPLICATION 11-07 item 5). The paper never says this.
- Fix: either (a) adopt a consistent metric (catch-before-pass or per-turn) for both concept probes and detectors, or (b) state plainly that in the adaptive race the hiding detector is beaten 62-81% of the time, and that "easy to detect" holds only offline with labelled hiding turns from the same model (8.1 "It needs examples of hiding" belongs in the abstract).

### Major

**M1. Judge tabulation counts Monitor-OFF turns as hiding turns.**
- Location: 4.1 judge table and text ("13 of their 259 passing answers", "68-94%", "79%", "49 of 51"); 5.4 last paragraph; abstract.
- Evidence: `judgetab.py`, `sel()`, uses `g.get('monitored', True)`. The gens rows have no `monitored` key, so every row counts, including `off == True` rows. Checked on all 21 runs: for example `b2b_heldout` has 38 monitored passes plus 14 OFF-turn passes, giving judgetab's 52. Monitored passing answers equal the race's probes beaten exactly (untrained total 225 = 14+59+22+38+71+21), which confirms that the 259 includes 34 OFF-turn answers. Corrected values are in the number-check table (rows 30-47).
- The qualitative conclusion survives (untrained models: 10/225 valid passes).
- Fix: filter `not g['off']` and regenerate every judge number.

**M2. The headline free-generation numbers are not judge-filtered.**
- Location: abstract "beat the first probe in 69%, 81% and 92% of cases"; 4 intro; 7.1 table row 1.
- Evidence (my reconstruction from gens plus judge rows; raw counts reproduce the paper's 44/52/59 exactly): segments where the first probe was beaten by an answer that still has the concept: 2B 36/64 (56%), 4B 46/64 (72%), 9B 53/64 (83%). Untrained: 2B 1/69, 4B 0/65, 9B 2/65. The 4.1 table of probes beaten per segment has the same problem: 17-22% of trained held-out passes lack the concept, and on Java/Python up to 62% do (11/29 valid for the 4B).
- Fix: lead with concept-preserving numbers everywhere (first probe, probes per segment) and give raw counts second. This also makes the untrained comparison honest: 56-83% against 0-3%.

**M3. The judge is unvalidated.**
- Location: 4.1 "we also had a larger model (Qwen3.5-27B) judge every answer ... central property"; threshold 0.5.
- Problems:
  - One judge, from the same model family as the subjects.
  - No human labels, no second judge, no threshold sweep.
  - Answers are truncated at 320 tokens (`answer_max: 320` in every race JSON) and judged on the first 2000 characters (`maxc: 2000`). This matters most for code.
  - The judge's second output (`p_answers`) was discarded because it is "low everywhere (0.05-0.46 ...)" (REPLICATION 11-11 22:45). The same judge produced an output the authors themselves did not trust.
  - The untrained 4B's caught answers on held-out concepts have the concept only 63% of the time (monitored only). So either the requests do not reliably elicit the concept, or the judge under-calls. Either reading undermines the comparison.
- Fix: about 300 stratified human labels (pass/caught x trained/untrained x concept), agreement with the judge, a second non-Qwen judge, and a threshold sweep. Language-ID agreement already exists as a partial check; report its confusion matrix against the judge.

**M4. Concept mix differs between models; the 4B-vs-9B held-out gap is an HTML artifact.**
- Location: abstract "69%, 81% and 92%"; 7.1 "The 9B is ahead on held-out first probes (92% against 81%)"; finding 4.
- Evidence (deploy_records, `first == True`): the 4B race drew HTML for 10 of 64 segments (beats 3/10); the 9B drew it for 3 (3/3); the 2B for 3. Excluding HTML, the 4B is at 49/54 (91%) and the 9B at 56/61 (92%). Turkish: 2B 2/8, 4B 6/6, 9B 8/8. Concept sequences are not paired across models in free generation, although REPLICATION claims pairing for the task races.
- Concept-cluster bootstrap (11 concepts) gives 95% intervals of 2B [53, 81], 4B [63, 96], 9B [87, 97]. These are much wider than the quoted Wilson intervals, and the 2B-4B difference is not clearly resolved either.
- Fix: report concept-stratified (macro-averaged) rates, fix the concept sequence per seed across models, and put the per-concept table in an appendix.

**M5. The confidence intervals understate uncertainty.**
- Free generation: Wilson intervals treat 64 segments as independent, but there are 2 segments per episode and 11 concepts, with very heterogeneous rates (4B: HTML 3/10, Finnish 7/7).
- Task races: "bootstrap over episodes, stratified by task" treats the 17 tasks as fixed. Per-task values range from 2.28 (Thai) to 6.00 (JS) for the 4B, yet the CI is ±0.09. The CI says nothing about "graded tasks" in general.
- No CIs at all on: 5.4 (per-size cells of 5-27 probes), the judge counts, the panel (66-84 turns per seed), persistence, the capability deltas, the fixed-text scores (per-family values rest on a handful of probes; the 4B format family swings 0.33 -> 0.00 -> 0.56 -> 0.30 across checkpoints), and the detector races (16 episodes).
- Fix: use hierarchical bootstraps (task or concept, then episode) and add intervals or counts to every comparative claim. Stop calling a single-seed difference "reliable" (see M6).

**M6. Single seed, contradicted by the confidence of the claims; undisclosed runs.**
- Limitations say "the 4B to 9B differences are small enough that a second seed could reorder them". But the abstract and finding 4 say "the 9B keeps more of its answers correct", and 7.1 says "Where the 9B is reliably better is correctness". Pick one.
- The 2B at step 1,800 is a branch of an earlier run that was scheduled for 1,500 steps, taken at step 1,270 with a new cooldown (`final_v4d_tamia_2b_x1800`; REPLICATION 10-31). The 7.2 trend mixes the two runs (600 from the original, 1,400 and 1,800 from the branch). This is not disclosed.
- A Gemma-4-12B run with the same recipe family was trained to about 1,200 steps, never got first-probe beat rates much above 0.2, and was cancelled (REPLICATION 10-27 to 10-29). Yet the limitations say "We have not tested ... another architecture." That sentence is false as written, and the negative result is relevant to "Size helps" and to generality. A strict-KL 2B run (`final_v4d_klx_tamia_2b`) also exists and is unreported.
- Fix: disclose all of these runs; tone the size claims down to "in one run each".

**M7. "Held-out concepts" in free generation and tasks are siblings of trained families, and trained-concept tasks were also capability-trained.**
- All 11 free-generation held-out concepts are natural languages, Java/Python or HTML. Training had 4 languages plus SQL/JSON/JavaScript. Held-out topics, tones, emotions and deception appear only in the forced-text race. The abstract's "concepts never seen in training" needs that qualifier.
- The "trained concepts" task group (GSM8K, SQL, JSON, HumanEval-JS, MGSM-de) consists of the same task types as the capability forks in training (Section 3: GSM8K, SQL, JSON schema, JavaScript tests). First-answer accuracy on JSON is 0.22 for the base model against 1.00 for the trained 4B (`racegroups.py`). Section 6's contrast ("works on trained concepts, fails on held-out code") therefore conflates concept familiarity with capability-fork coverage: Python and Java code was never a capability fork.
- Fix: say this explicitly. Ideally add a held-out task whose concept is trained but whose task type was not a capability fork, or the reverse.

**M8. Section 6.3's interpretation is contradicted by the Monitor-OFF turns in the same episodes.**
- Location: 6.3 "Once the model starts reshaping its activations to dodge, the quality of its answers erodes and does not come back within the episode"; "The trained models start at or above the untrained model on turn 0".
- Evidence: the `turnacc/*.json` aggregates include OFF turns, which the paper does not show. Accuracy on Monitor-OFF turns by turn bucket (1-7 | 8-15 | 16-23 | 24-32):
  - 4B, held-out code: 0.56 / 0.64 / 0.60 / 0.62, while monitored turns fall to 0.03.
  - 4B, trained concepts: 0.79 / 0.82 / 0.84 / 0.81.
  - 9B, code: 0.70 / 0.65 / 0.66 / 0.70.
  - The exception is 4B reading (0.68 -> 0.20) and partly MGSM (0.73 -> 0.50).
  So the damage is mostly tied to the hiding brief on the current turn, not a persistent erosion of the episode. The second sentence is false for code (untrained 0.66 vs 4B caught 0.58, 4B passing 0.44, 9B passing 0.55) and for reading (0.79 vs 0.70).
- The untrained-control row exists only for the 4B. The partial untrained-9B data in the same files is not flat (MGSM 0.67 -> 0.45; reading 0.50 -> 0.10, small n).
- Fix: add the OFF-turn rows and rewrite the paragraph. This is arguably a more interesting finding: the capability is intact and the brief switches the damage on.

**M9. "Unseen probe types" is inflated, and the early-layer result may be lexical.**
- "19 further types" (2.4) counts linear@20/24/28/30, which are layers, not types, and maxpool@24, a trained architecture at a new layer. rlobf_attn_max, rlobf_max and rollmax are max-aggregation variants of trained types. Most unseen types in 5.4 are evaluated at training layers 8-26 (deploy_records kinds).
- The panel's "new fits of trained probe types" are base-model fits, the same as the first probe of every training episode, so near-duplicates of training probes rather than transfer.
- Layer 4 fires on everything, but in the forced-text panel the tokens are identical across conditions. In free generation, the concepts are surface form (a Greek script, code syntax), which early layers encode as token identity. "Probe early layers" may therefore not carry over to semantic concepts. The deception probes are the ones that most often fail to fit in the held-out panel: 42 of the 4B's 56, 44 of the 9B's 51 and 50 of the 2B's 78 failed panel fits are deception probes.
- Fix: count architectures and layers separately, label variants honestly, and report layer-4 and layer-28/30 detection split by surface-form vs semantic concept.

**M10. Part of the selectivity evidence is circular.**
- 5.2 cites the hiding detector flagging "none of the turns with ... a brief naming a different concept". But that detector was trained with the model's own mismatch turns as negatives (8.1). The non-selective detector flags 15% of mismatch and 20% of no-brief turns for the 4B (REPLICATION 11-07 item 2), which the paper omits.
- The panel already contains direct evidence: mean detection on mismatch and no-brief turns is 0.97-1.00 for all sizes (computed from `selectivity.text_probe`; n = 24 mismatch turns per seed).
- Fix: cite the panel numbers and drop or caveat the detector argument.

**M11. The task-exclusion rule is ad hoc.**
- MMLU biology is excluded (untrained 4B 1.16 probes per episode, robust share 0.12). MGSM-de is kept (untrained 4B 1.38, robust share 0.52), although the trained 4B there writes German in only 27-35% of answers (`turnacc_tab.py` C). Half of the untrained 4B's beaten probes on trained-concept tasks come from MGSM-de (44 of 78).
- Fix: pre-specify a validity criterion (an untrained beat-rate threshold plus an in-concept check), apply it to all 22 tasks, and report the sensitivity of the headline numbers.

**M12. Untrained controls are unequal, and some conditions have none.**
- 2B/9B task-race controls are half size: piece 1, 16 episodes per task. Because seeds are paired, the clean comparison is trained piece 1 vs untrained piece 1. Do that, or run the other half.
- The held-out-kind race (5.4) has an untrained control only for the 4B; the 9B trained-concept held-out-kind answers are unjudged.
- Turn-accuracy has a 4B control only. Persistence covers trained concepts only. All detector work is 4B-only, fixed text, 16 episodes.
- Batch differences: 2B task races at batch 16, the 4B free-generation race at 4, and the 9B held-out-kind trained race at 4. That last one is not in 9's list.

**M13. "Searching with the one bit" is asserted, not shown.**
- Location: 4.1 "The model is not recalling a stored trick for the concept; it is searching with the one bit it gets, and the search is slower off-distribution."
- Taking more turns on held-out concepts is equally explained by a lower per-sample pass probability with no use of feedback. 5.3 shows the activations react to the verdict (cos 0.19 vs 0.63) and that feedback helps a pass persist (92% -> 76%). It does not show that feedback raises the chance of the first pass.
- Fix: run a race with constant or random verdicts from turn 0 and compare first-probe beat rates and turns. Until then, delete the sentence.

**M14. The detector-race comparison mixes metrics, and the race is in-distribution for the detector-trained model.**
- The 8.2 table compares 69%/62% (detector refits only, 4B) with 82%/85% (all probes, including the turn-0 concept probe, for the detector-trained 4B). Like for like, the detector-trained model is at 79%/81% (detector refits only) against 69%/62%.
- The detector-trained model was trained against exactly the raced detector family (evdetsel at layers 12/16/22).
- Fix: correct the table, give n, and flag the in-distribution nature.

### Minor

- **m1.** 2.5 says the capability evaluation uses "full test sets"; 6.4 says 200 GSM8K items (MMLU is n = 600 per the audit). Fix 2.5.
- **m2.** 7.1 "all three sizes beat 88-98% of probes on trained concepts": 4.1 gives 90/94/97%.
- **m3.** 4.2 "Language is the hardest family for every size (0.37-0.50)": the 4B's format family is 0.30 (`val/step_1800.json` per_family).
- **m4.** 6.2 "More than half of passes are correct even at the fourth to sixth probe": 2B 124/256 (48%), 4B 216/436 (49.5%), 9B 57%.
- **m5.** 5.4 "one at layer 4 once in 30": 4B+9B linear@4 is 1/24 (0/6, 0/6, 0/6, 1/6). The 30 does not reproduce from any combination of the trained races.
- **m6.** 4.1 "nearly all of the shortfall is code": true for the 2B and 4B, but only 10 of 15 for the 9B.
- **m7.** 6.2 "Thai maths is near zero for every model": the 9B is correct in 12/32 passes at probe 1 and 16/83 overall.
- **m8.** Finding 1, "beat a chain of five or more refit probes on trained concepts": the 2B averages 3.1 probes per segment (free generation) and 3.4 uncapped (tasks); the 4B and 9B in free generation average 4.0 and 3.9 including the first probe.
- **m9.** 8.3 leaves the strict-KL 2,500 fixed-text cell blank, but the value exists: 0.435 (`final_v4d_klx_tamia_4b_x2500/val/step_2500.json`).
- **m10.** "First probe" mixes two kinds: in every race, 32 first probes per race are fit on the base model (turn 0) and the rest on the current model (switch forks). Define this in 2.2/4.1. Trained models beat both kinds at similar rates, so it is a definitional fix.
- **m11.** The cumulative monitor is an OR over every deployed probe, each at 1% FPR. The union FPR, which grows with K, is never reported.
- **m12.** 5.2 says the selective detector flags "99-100%" of hiding turns; 8.1 says "96-100%". These are the same detector and layers, quoted with different ranges (one model vs three). Make them consistent.
- **m13.** 5.1 "It carries to layers 20 and 24": detection at layer 20 is 0.25-0.46. "Carries partly" is accurate. 10.2 "catches it outright": layer 28 detects 0.62-0.81, not outright.
- **m14.** 7.1's table row "Never-shown probes, mean detection on held-out concepts" averages all 18 probes, including trained types, and differs from 5.1's 11-probe average (0.57/0.53). Label it.
- **m15.** The robust share of untrained passes (0.44) after a 6-of-8 gate shows the gate leaks. Say so when interpreting untrained "passes".
- **m16.** The compute cost of the organism is not given ("a single four-GPU node"). At roughly 250-300 s per step, 1,800 steps is about 5-6 days, or roughly 500 H100-hours. State it.
- **m17.** "Later community replications report higher costs" has no citation (the audit flags this). Kramár et al.'s "do not hold up against adaptive attacks" should be checked against the source.
- **m18.** Wording that reads as overclaim or boilerplate: "the difference is informative", "Four points come out of this table", "Three things follow", "For a defender this is useful", "A catch triggers a jump to a new configuration; a pass freezes it" (a cosine of 0.63 is not frozen), and "on every probe the defender refits".

## (c) Number-check table

The source is the script output unless noted. "recomp." means my stdin script over the same JSON.

| # | Paper location / value | Recomputed | Source | Match |
|---|---|---|---|---|
| 1 | 4.1 2B held-out first probe 44/64 (69%) [57, 79] | 44/64, Wilson [57, 79] | hoktab + recomp. | yes |
| 2 | 4.1 4B held-out 52/64 [70, 89] | 52/64 [70, 89] | same | yes |
| 3 | 4.1 9B held-out 59/64 [83, 97] | 59/64 [83, 97] | same | yes |
| 4 | 4.1 untrained 2B / 4B / 9B held-out 9/68, 20/64, 6/65 with CIs | identical, CIs identical | recomp. on v4b_quality_base* | yes |
| 5 | 4.1 trained rows 197/219, 257/272, 256/263; turns 2.3 / 1.9 / 2.0 | identical (2.25 / 1.85 / 2.01) | hoktab | yes |
| 6 | 4.1 untrained trained-concept rows 7/64, 14/66, 6/65; 14/78, 59/117, 22/83 | identical | recomp. | yes |
| 7 | 4.1 language ID 3/23, 3/99, 360/366, 217/236; caught 77-95% | identical; caught 77.0-94.9% | freelang.json | yes |
| 8 | 4.1 held-out code judge 10/23, 11/29, 24/34 | identical (no OFF passes on code) | judgetab | yes |
| 9 | 4.2 fixed-text 0.55 / 0.65 / 0.71; first probe 80/92/92% | .554 / .652 / .707; .80 / .92 / .92 | cmp3.py | yes |
| 10 | 4.2 9B per-family values | identical | cmp3.py | yes |
| 11 | 4.2 "Language is the hardest family for every size" | 4B format .30 < language .37 | cmp3.py | **NO** |
| 12 | 5.1 panel, all 28 4B/9B cells | identical | recomp. over panel_*.json | yes |
| 13 | 5.1 unseen-probe averages .46 / .42 / .57 / .53 | .459 / .423 / .565 / .527 | recomp. | yes |
| 14 | 5.3 persistence 81/92/92%; constant feedback 45/76/75% | .808 / .920 / .919; .450 / .763 / .754 | cmp3.py | yes |
| 15 | 5.4 beaten/faced and first-probe columns (16 cells) | identical | hoktab | yes |
| 16 | 5.4 by-kind table (20 cells) | identical | hoktab | yes |
| 17 | 5.4 93/96/92% and 73/86/91%; 33/139; 10/88; 24/36 | identical | hoktab arithmetic | yes |
| 18 | 5.4 "linear probe at layer 30 ... once in 28" | 1/28 | hoktab | yes |
| 19 | 5.4 "one at layer 4 once in 30" | 1/24 | hoktab | **NO** |
| 20 | 6.1 table, all cells (first probe, CIs, correct per episode, robust share) | identical | raceci, basecmp, audit | yes |
| 21 | 6.1 uncapped 7.9 / 8.1 / 3.4 | 7.90 / 8.14 / 3.42 | raceci | yes |
| 22 | 6.1 concentration 28/50, 24/46, 79/151; MGSM-de 44 of 78 | identical | basecmp | yes |
| 23 | 6.2 group table, all 36 cells | identical | racegroups | yes |
| 24 | 6.2 "More than half of passes are correct even at the fourth to sixth probe" | 2B 48%, 4B 49.5%, 9B 57% | racegroups | **NO** |
| 25 | 6.3 turn table, all 80 cells | identical | turnacc_tab | yes |
| 26 | 6.3 "start at or above the untrained model on turn 0" | code: base .66 vs .58 / .44 / .55; reading .79 vs .70 | turnacc_tab | **NO** |
| 27 | 6.4 capability, 18 cells | identical | capability.json x 6 | yes |
| 28 | 2.5 "full test sets" | GSM8K n = 200 (6.4), MMLU n = 600 (audit 7e) | paper / audit | **NO** |
| 29 | 6.5 MMLU 2.41 / 1.16 / 6.00; untrained 3.9 vs trained 5.6-6.0 | identical | raceci, audit 5 | yes |
| 30 | 4.1 / abstract: untrained 13 of 259 passing answers have the concept | 10/225 (monitored turns only) | recomp. (judgetab bug) | **NO** |
| 31 | 4.1 2B trained 214 of 217 | 195/197 | recomp. | **NO** |
| 32 | 4.1 2B held-out 60 of 76 | 56/72 | recomp. | **NO** |
| 33 | 4.1 9B trained 254 of 257 | 253/256 | recomp. | **NO** |
| 34 | 4.1 untrained 2B 2 of 22 / 2 of 52 | 2/14 / 1/38 | recomp. | **NO** (x2) |
| 35 | 4.1 untrained 4B 2 of 61 / 1 of 75 | 2/59 / 0/71 | recomp. | **NO** (x2) |
| 36 | 4.1 untrained 9B 3 of 26 / 3 of 23 | 3/22 / 2/21 | recomp. | **NO** (x2) |
| 37 | 4.1 4B 250/257, 107/129; 9B 124/139 | identical (no OFF passes) | recomp. | yes |
| 38 | 4.1 2B held-out languages 49 of 51 | 45/47 | recomp. | **NO** |
| 39 | 4.1 held-out share 79% (2B) | 78% | recomp. | **NO** |
| 40 | 4.1 untrained caught answers have concept 68-94% | 63-93% | recomp. | **NO** |
| 41 | 5.4 trained 4B held-out-kind 85 of 100 | 84/99 | recomp. | **NO** |
| 42 | 5.4 untrained 4B 2 of 98 / 5 of 63 | 2/90 / 5/61 | recomp. | **NO** (x2) |
| 43 | 5.4 2B 68 of 75 / 136 of 140 | 61/68 / 115/118 | recomp. | **NO** (x2) |
| 44 | 5.4 9B 105 of 117; 4B trained 129 of 138 | identical | recomp. | yes |
| 45 | 7.1 "beat 88-98% of probes on trained concepts" | 90-97% | 4.1 / hoktab | **NO** |
| 46 | 7.1 passing answers correct 38 / 40 / 46%; 62 vs 53% at first pass | 38.4 / 39.8 / 46.1; 62.3 / 52.9 | racetab, racegroups | yes |
| 47 | 7.1 held-out languages 2B 57% vs 84-86% | 47/83 = 57%; 86%, 84% | recomp. | yes |
| 48 | 7.2 checkpoint table, 15 cells | identical | val/step_*.json | yes |
| 49 | 7.2 "6 to 13 ... 25 to 39 ... 72-91%" | identical | val JSON | yes |
| 50 | 8.2 live detector refits beaten, 4B 69% (36/52), 62% (21/34) | identical | REPLICATION 11-07 / 11-08 | yes |
| 51 | 8.2 detector-trained model 82% (layer 16) | 53/67 = 79% detector-only (82% includes the concept p1) | evdet/d22_evdetsel_16.json | **NO** |
| 52 | 8.2 detector-trained model 85% (layer 22) | 52/64 = 81% detector-only | evdet/d22_evdetsel_22.json | **NO** |
| 53 | 8.2 frozen detector .97/.68, 1.00/.58, 1.00/.79, 1.00/.96; linear gate 43% -> 13%, MLP 85% | identical | audit 6; REPLICATION 11-08 / 11-09 | yes |
| 54 | 8.3 table; 1.7 fewer; 40% recovered | 5.05 / 3.36 / 4.04; 1.69; 0.40 | raceci | yes |
| 55 | 8.3 strict 2,500 fixed-text (blank) | 0.435 exists | val/step_2500.json | omitted |
| 56 | Appendix A (147 cells), B (30 cells), C (24 cells) | identical | raceci, recomp., qtab | yes |
| 57 | 5.3 cosine 0.19 / 0.63 | identical | REPLICATION 11-07 item 4 | yes |
| 58 | 4.1 Portuguese 9/11, 10/13, 7/9 | identical | recomp. | yes |

**Hard mismatches: 26.** Rows 11, 19, 24, 26, 28, 30-33, 34 (x2), 35 (x2), 36 (x2), 38-43 (counting x2 cells), 45, 51 and 52. Eighteen of the 26 come from the single judgetab OFF-turn bug.

Soft wording mismatches (m6, m7, m8): 3 more. One omission: row 55.

## (d) Demanded experiments and analyses, ranked

| rank | What | Why | Existing data? |
|---|---|---|---|
| 1 | Catch-cost metrics for every race: catch-free segments, catches before first pass, first-catch turn, any-catch-per-conversation; plus a defender with an escalation rule (k catches -> flag) | C1; decides whether the threat is real | **Yes**: `quality/step_1800/gens/gated_*.jsonl`, task-race `fork_records` in `races/*/report.json`, `turnacc/*.json`. No GPU. |
| 2 | Fix judgetab (exclude OFF turns) and recompute every free-generation metric with concept-preserving passes only (first probe, probes per segment, 5.4) | M1, M2 | **Yes**: `mech_1109/judge/*.rows.jsonl` + gens. No GPU. |
| 3 | Judge validation: about 300 human labels, a second non-Qwen judge, a threshold sweep, untruncated answers for code | M3 | Partial: language-ID agreement from `freelang.json`. Human time plus about one GPU-hour for a second judge. |
| 4 | Static-probe baseline adapter (Neural-Chameleons-style: same steps, fixed base probes, no in-episode refits) raced against the refitting defender | Without it, the novelty claim (training against refits matters) is untested | Probably not. Check whether any earlier static-trained adapter in `runs/` or `archive/paper-replication` can be raced as-is. Otherwise GPU training. |
| 5 | Combined defender race: concept refits drawn from all types and layers, plus a hiding detector, in one monitor | C2, C3; the defender the conclusion recommends was never run as a whole | No. GPU evaluation only (about 1 node-day). |
| 6 | Detector race under the same protocol as 4.1 (free generation, 6-of-8 gate, 32 episodes, all three sizes), reported with the same metrics | C3, M14 | No. GPU evaluation. |
| 7 | Feedback ablation from turn 0 (constant or random verdicts) | M13 | No. GPU evaluation (cheap). |
| 8 | Second training seed for the 4B (ideally all sizes) | M6 | No. GPU training (about 500 H100-h each). |
| 9 | Concept-stratified, paired concept sequences across models; hierarchical bootstrap (concept/task, then episode) for all CIs | M4, M5 | **Yes**: deploy_records, `races/*/report.json`. No GPU. |
| 10 | Monitor-OFF accuracy rows in 6.3 | M8 | **Yes**: `mech_1109/turnacc/*.json` (computed above). No GPU. |
| 11 | Pre-specified task-validity rule applied to all 22 tasks, with sensitivity analysis | M11 | **Yes**: `races/finalbase*/report.json`, `turnacc` language share. No GPU. |
| 12 | Paired half-size comparison (trained piece 1 vs untrained piece 1), or full-size untrained 2B/9B races; untrained 2B/9B held-out-kind races; judge the 9B trained-concept held-out-kind answers | M12 | Piece-level reports may exist (`pieces` key in report.json). The rest needs a GPU (small). |
| 13 | Layer-4 / 28 / 30 detection split by surface-form vs semantic concept (panel) | M9 | **Yes**: `panel_heldout.s*.json` records carry the concept. No GPU. |
| 14 | A semantic concept in free generation (a topic or tone, with judge), ideally deception in free text | Scope (M7, 1.1) | No. New data plus GPU. |
| 15 | Disclose the Gemma-4-12B and strict-KL 2B runs, and the 2B branch history | M6 | **Yes**: REPLICATION 10-27..10-31. |
