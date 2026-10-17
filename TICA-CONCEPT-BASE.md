# TICA concept base: hierarchy, data, splits (draft v1, 2026-10-14)

Goal: a rigorous, hierarchical set of concepts with published data, mirroring and extending related work
(Neural Chameleons 2512.11949, Keenan 2609.36490, Apollo 2502.03407, Bailey 2412.09565, RepE 2310.01405),
with mutually distinct topics within and across train / dev / test.

## Principles
1. **Published data first** (user rule). Where no published data exists, use the related work's RELEASED data
   (NC's rated file); on-policy generation only for flagged gaps.
2. **Within-source contrasts.** Each concept's probe gets negatives from the SAME source (other labels of that
   dataset, or same-prompt matched negatives), plus cross-source negatives. Otherwise a probe can learn the
   dataset instead of the concept. Control: a source-only probe (predict the dataset id from concept-neutral
   texts) and a length / bag-of-words baseline.
3. **Same answer format everywhere.** Chat sources keep their request; text-only sources get ONE generic request
   for all of them, so the prompt carries no topic or source information. Probes pool over answer tokens.
4. **Disjointness is measured.** Probe-based overlap (scripts/tica_topic_overlap.py: A's probe vs other
   concepts, scored on concepts it never saw) plus geometry. Pairs above the overlap bar are merged, dropped,
   or used only as declared NEAR test pairs, never as accidental train/test duplicates.
5. **Safety data is eval-only;** no bio / chem / cyber weapon content; harmful text never committed.

## Hierarchy (families -> concepts -> source)

| Family | Concepts (draft) | Source (published) | Within-source negatives |
|---|---|---|---|
| A. Subject / topic | ~20 broad DBpedia topics (sports, music, astronomy, law, software, plants, ...); mathematics | DBpedia Classes (CC0; 3-level); math: Concept16K (CC BY 4.0) + GSM8K solutions (MIT) | other DBpedia topics; Concept16K matched negatives |
| B. Language & form | german, french, spanish, japanese, chinese, arabic, russian, finnish, portuguese; all-caps, html | Aya dataset (Apache 2.0, chat, 65 languages); FLORES-200 (CC BY-SA, parallel = matched); NC released data for all-caps / html | English Aya rows; same sentence in English (FLORES) |
| C. Style & tone | formal, jokey, confused, literature-focused, comforting | Pavlick formality (CC BY 3.0); NC released rated data (jokey, confused, literature, comforting) | NC score-1 negatives (same pipeline); informal Pavlick extremes |
| D. Affect | joy, sadness, anger, fear, surprise, disgust (+ gratitude, admiration) | GoEmotions (Apache 2.0, 27 labels) | other GoEmotions labels + neutral |
| E. Intent & behaviour (safety) | deception, sycophancy, (refusal), harmfulness (eval only) | DolusChat (CC BY 4.0, MATCHED truthful/deceptive); RepE fact personas (matched); CAA sycophancy / refusal A/B (MIT); circuit-breakers harmful (Bailey; eval only) | the matched honest / non-sycophantic answer |

Data gaps (no good published chat data): sycophancy as free-text answers (CAA and model-written-evals are A/B
only), evasiveness, sarcasm (licences), corporate / condescending tone. These are flagged: test-only from A/B
data, or omitted.

## Train / dev / test policy
- **Near (within family):** in every trained family, ~20% of concepts held out as siblings (e.g. train german /
  french, test portuguese; train sadness / anger, test fear; train astronomy, test natural disasters only if the
  overlap check says they are distinct).
- **Far (whole family):** family E (safety) is TEST-ONLY for the headline result, as in NC and Keenan.
  Secondary arm: train on one safety concept (sycophancy), test on another (deception).
- **Dev:** a few concepts per family for checkpoint selection; never the test ones.
- **Overlap bar:** no train-test pair above the overlap threshold unless it is a declared near pair.

## Measured so far
- DBpedia L2 (65 classes): too fine; strictly disjoint sets are small and odd.
- DBpedia broad v1 (25 topics, `configs/tica/dbpedia_broad_topics_v1.json`): every topic's probe fires on its
  own held-out texts (TPR 0.91-1.00 at 1% FPR). Overlaps (cross-fire): animals-plants 0.99, mountains-rivers
  0.94, politics-royalty 0.93, historical events-law 0.95, comics-software 1.00, astronomy-natural disasters
  0.86, religion-royalty 0.85, music-software 0.78. Cross-family median 0.00. v2 merges / drops these.

## Concept audit (2026-10-14n; labels, aggregates and probes only, no text read, per the standing rule)
Criteria for a concept to stay: (1) it is a property of the ANSWER text, which is what the probe reads;
(2) the model can write it naturally on request, so hiding it is a real choice; (3) clean published
labels, and the topic NAME matches what the data contains; (4) a probe fit on the published texts fires
on the model's OWN writing on that concept (on-policy check) and not on other concepts; (5) not
separable by source alone.

DBpedia v1 failed (3) badly once the subclass composition was checked: 'royalty' = 100% Baronet,
'software' = 100% VideoGame, 'natural disasters' = 31% SolarEclipse (the cause of the astronomy overlap),
'science and scientists' = physicians + entomologists, 'film/TV' = 14% AdultActor, 'transportation' = 17%
Dam, 'animals' = 11% RaceHorse (sports overlap), 'buildings' included stadiums and roller coasters.
v2 (`configs/tica/dbpedia_broad_topics_v2.json`, 27 topics) is built from exact subclasses with honest
names; it adds conceptual topics matching related work (philosophy, war and the military, medicine,
food and drink, fashion, law and courts). Overlap + on-policy check: job 5542389.

| Family | Concept | Verdict | Reason |
|---|---|---|---|
| B | languages (Aya) | keep | clean label, answer property, NC tested languages |
| B | all-caps, html (NC data) | keep | answer property; NC concepts |
| C | formal (Pavlick) | keep, extremes only | continuous sentence-level scores are noisy in the middle; short sentences |
| C | jokey, comforting, confused (NC rated) | keep | NC's 4-5 rated positives, matched score-1 negatives |
| C | literature-focused (NC) | near pair only | overlaps the topic 'plays and poetry' |
| D | joy, sadness, anger, fear, gratitude (GoEmotions) | keep if on-policy check passes | only rows where >= 2 raters agree; short Reddit comments, so the on-policy check matters most here |
| D | surprise, disgust, the other 20 labels | drop | low rater agreement in the GoEmotions paper |
| E | deception (DolusChat) | keep (test) | matched truthful / deceptive answers to the same prompt |
| E | sycophancy (CAA, model-written-evals) | NOT usable as is | the answer is one letter (A/B): the concept lives in the prompt, not the answer; needs free-text data (gap) |
| E | refusal | keep (test) | an answer property; source: published refusal responses |
| A | mathematics | keep; use GSM8K solutions | the Concept16K 'notation' probe fired 0/16 on real math answers (10-14k) |

## Next steps (to finalize)
1. Topic v2: merge animals+plants -> "living things"; mountains+rivers -> "landforms and waters"; drop royalty;
   resolve comics / software / music and events / law; re-run the overlap check.
2. Loaders for Aya, FLORES, GoEmotions, DolusChat, CAA, NC rated data, Pavlick, into one concept-base file
   (concept, family, source, request, answer, split), aggregates logged only.
3. One overlap check across ALL concepts (within and across sources) + the source-only control probe.
4. Freeze `concept_base_v1.json` + the split; then the trainer wiring.
