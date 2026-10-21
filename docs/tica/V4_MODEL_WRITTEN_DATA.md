# v4 model-written concept data (split v5, `final_v5_mw`)

Opt-in data for the TICA v4 run: the TRAIN concepts' texts are the BASE model's own answers to the
same published concept-eliciting requests, instead of published responses. Config:
`configs/tica/final_v5_modelwritten_4b.yaml` (builtin mirror in
`src/next_chameleons/builtin_configs/tica/`). Existing configs and data files are unchanged.
Log: REPLICATION.md 2026-10-21f (build) and 10-21h (refit gate).

## Purpose

- In split v5 each concept has fixed texts built from published requests / responses. The base
  probes are fit on those texts, and the refit forks teacher-force them after the episode prefix.
- Neural Chameleons trained on MODEL-WRITTEN texts (gemma-2-27b-it generations, filtered by an LLM
  judge). Here the same idea: the base model writes the concept itself, so probes and refit forks
  see the states the model has when it produces the concept.
- Evidence it works (10-19i, jobs 6392489 / 6395521): base 4B answers to these requests carry the
  concept 95% of the time (27B judge), and base probes catch them 91-100%.

## Provenance (exact source vs substitute)

| Part | Source | Label |
|---|---|---|
| Requests | `final_v5_elicit_inputs.json` (10-19k rules): the source's own prompt, + a language / tone / format line; DBpedia topics get one 27B-written request per passage | published (exact prompts + documented additions) |
| Responses, regenerated concepts | Qwen3.5-4B (base, non-thinking), vLLM, T 0.7 / top-p 0.8 / top-k 20, max 1024 new tokens | **self-generated substitute** for the published responses |
| Filter | Qwen3.5-27B-FP8 judge, P(yes) >= 0.5 on both questions, plus finished | LLM judge |
| Everything else | as `final_v5_clean.json` | published (unchanged) |

## Which concepts are regenerated

- Regenerated: the 19 of 20 TRAIN concepts other than mathematical reasoning (topics sports, politics,
  astronomy, war; German, Japanese, Arabic, Hindi; all-caps, SQL, JSON, JavaScript; jokes,
  comforting, casual, formal; chemistry, literature, finance).
- Kept as they are:
  - mathematical reasoning: already the base model's own GSM8K-train solutions (`math_base_v1`:
    finished, correct, with working).
  - deception (DolusChat, roleplaying, insider trading, sycophancy) and biology: held-out concepts
    in split v5, so untouched by construction. Deception needs ground truth (free answers to the
    scenarios are honest, 10-19i), so it must stay on the published pairs anyway.
  - every held-out concept, the concept-free negative pool and all hard negatives.
- Held-out concepts can be regenerated with the same scripts (`--concepts heldout`); not done here.

## Pipeline (exact commands)

`T=/scratch/sneheel/next_chameleons_raw_cache/tica`, `W=$T/mw_v4/full`. Every output is on scratch;
never commit generated text or data.

1. Generate + judge, 4 jobs x 2 A100 (each: two 4B generator processes, one per GPU, while the 27B
   copies to node-local disk; then the 27B judge with tensor parallel 2, answers shown up to 6,000
   characters), plus one job with 3 extra samples for the low-keep concepts (smoke):

   ```bash
   for i in 0 1 2 3; do
     sbatch scripts/slurm/tica_mw_build.sh $W s$i "--samples 3 --shard $((2*i))/8" "--samples 3 --shard $((2*i+1))/8"
   done
   X="--concepts 'chemistry|finance|literature|Hindi language|German language|formal tone' --exclude '' --samples 3 --seed 1 --sample-offset 3"
   sbatch scripts/slurm/tica_mw_build.sh $W x "$X --shard 0/2" "$X --shard 1/2"
   sbatch --dependency=afterok:<the 5 job ids> scripts/slurm/tica_mw_finish.sh $W   # steps 2-4
   ```

   `scripts/tica_mw_generate.py` defaults: `--concepts train --exclude "mathematical reasoning"`.
   The prompt is the 10-19i transfer prompt: one user turn = the text's request, no system prompt,
   thinking off (the string equals `Policy.render`; `tests/test_tica_mw_data.py`).
   The judge is `scripts/tica_free_judge.py --answers-prompt loose --row-out` (per-row scores, no
   text).
2. Rebuild (`tica_mw_finish.sh` runs steps 2-4 and queues the GPU parts):

   ```bash
   python scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json \
     --inputs $T/final_v5_elicit_inputs.json --gens $W/gen_s*_?.jsonl --scores $W/scores_s*.jsonl \
     --out $T/final_v5_mw.json --inputs-out $T/final_v5_mw_elicit_inputs.json \
     --report $T/final_v5_mw_report.json
   ```

3. Negative-contamination labels (27B, prompt v4) for the new texts only, then merge with the v5
   labels of every unchanged text:

   ```bash
   for i in 0 1; do
     sbatch scripts/slurm/tica_mw_labels.sh $T/final_v5_mw.json $T/concept_labels_v5data_27b.json \
       $T/concept_labels_v5mw_27b_s$i.json --shard $i/2
   done
   python scripts/build_final_v5_mw.py merge-labels --data $T/final_v5_mw.json \
     --old $T/concept_labels_v5data_27b.json --new $T/concept_labels_v5mw_27b_s?.json \
     --out $T/concept_labels_v5mw_27b.json
   ```

4. Base-probe check (base probes fit through the normal cached path; old vs new data):

   ```bash
   sbatch scripts/slurm/tica_mw_probe_check.sh final_v5_modelwritten_4b $W/probe_check.json \
     --old-data $T/final_v5_clean.json --old-inputs $T/final_v5_elicit_inputs.json \
     --old-labels $T/concept_labels_v5data_27b.json --concepts "sports|jokes and humor|all-caps text"
   ```

## Filters and thresholds

- Judge question 1, concept: the negative-label prompt (`label_concept_contamination.py`,
  FRAME_V4 + the v4 concept description: "central" properties only). Keep if P(yes) >= 0.5.
- Judge question 2, answer: "Is the response a relevant, coherent answer that does what the request
  asks? Do not judge factual accuracy. A refusal, a reply saying it cannot find or does not know
  the subject, or garbled or repetitive text does not count." Keep if P(yes) >= 0.5. The 10-19i
  prompt (no "do not judge factual accuracy") rejected most on-topic topic answers because they
  invent facts about obscure entities; that is irrelevant for probe data, while "I cannot find this
  subject" replies are still dropped.
- Finished: the answer ended before 1024 new tokens.
- The judge sees the first 6,000 characters of each answer (`--maxc 6000`; the smoke used 2,000, as
  the labels, and rejected long answers it saw only in part).
- Duplicates (same `text_id` as any text in the file, or another answer) are dropped.

## Build rules (`build_final_v5_mw.py`)

- Splits by request: all answers to one request go to ONE split, the split of its source text; a
  request whose source texts span several splits (JSON 66, Hindi 28, Arabic 21, ...) goes to the
  split holding most of them (ties: fit, calib, episode). So no request is shared between the fit
  and calib / episode texts, and held-out concepts stay held-out.
- Per source text: the kept answer closest in length to the source text (keeps the current
  length distribution, so episode memory / read time stay as with v5). `--fill` (default on) tops
  each split back up to its old count with other kept answers to requests of that split.
- New ids = `text_id(answer)` (episodes look the request up by text id). The new request file is
  the old map plus `{new id: request}`.
- `source.model_written` in the data file records provenance, thresholds and per-concept counts.

## Counts per concept

Smoke keep rate per generated answer (finished x concept x answer; K=2, 40 source texts per
concept, full for sports / all-caps / jokes; judge at 2,000 characters):

| Concept | finished | concept present | answer ok | keep / answer |
|---|---|---|---|---|
| sports | 0.99 | 1.00 | 0.91 | 0.91 |
| politics and elections | 0.99 | 0.96 | 0.89 | 0.85 |
| astronomy | 1.00 | 1.00 | 0.84 | 0.84 |
| war and the military | 1.00 | 0.98 | 0.80 | 0.78 |
| German | 0.94 | 1.00 | 0.58 | 0.56 |
| Japanese | 0.95 | 1.00 | 0.95 | 0.90 |
| Arabic | 0.89 | 1.00 | 0.71 | 0.68 |
| Hindi | 0.75 | 0.98 | 0.46 | 0.40 |
| all-caps text | 0.98 | 0.99 | 0.88 | 0.87 |
| SQL code | 1.00 | 1.00 | 0.93 | 0.93 |
| JSON data | 1.00 | 1.00 | 0.94 | 0.94 |
| JavaScript code | 0.95 | 1.00 | 0.73 | 0.72 |
| jokes and humor | 0.98 | 1.00 | 0.92 | 0.92 |
| comforting tone | 1.00 | 1.00 | 1.00 | 1.00 |
| casual tone | 1.00 | 0.99 | 0.91 | 0.90 |
| formal tone | 0.95 | 1.00 | 0.60 | 0.60 |
| chemistry | 0.45 | 1.00 | 0.30 | 0.25 |
| literature | 0.60 | 0.93 | 0.69 | 0.38 |
| finance | 0.53 | 0.95 | 0.76 | 0.36 |

Full-build counts (old -> new texts per concept, kept / dropped) are in
`$T/final_v5_mw_report.json` and in `source.model_written.report` in the data file; they are added
to REPLICATION once the build lands.

## Using it

- Run config: `final_v5_modelwritten_4b` (= `final_v4_4b` with `data`, `episodes.inputs`,
  `rl.adaptive.neg_labels`, `out` and the val / anchor paths changed).
- Base probes refit automatically: the probe cache key includes the data file's size / mtime and
  the request file's name / size / mtime.
- REPLICATION 10-21h: this config also sets `rl.adaptive.refit_min_tpr_np: 0` (no plain-text /
  no-prompt TPR gate; refits are gated on in-context AUROC only; no-prompt positives stay in the
  fit and `tpr_np` is still logged). The base-probe fits read the same field, so their tpr_np gate
  is off too.
- Other configs: point `data`, `episodes.inputs` and `rl.adaptive.neg_labels` at
  `final_v5_mw.json`, `final_v5_mw_elicit_inputs.json` and `concept_labels_v5mw_27b.json`.

## Limitations

- Model answers are longer than the published texts (smoke medians after length-matched selection:
  sports 401 -> 1,199 characters, jokes 147 -> 952, all-caps 123 -> 223). Per-turn read cost and
  24-turn episode memory go up; not yet measured.

- Train concepts become model-written while held-out concepts stay published: a train / held-out
  distribution gap (held-out races test transfer across that gap too).
- The concept-free negative pool (Dolly + NC style texts) stays human-written. A probe could use
  "model-written vs human-written" as a shortcut; the probe check measures it (`source` block).
  If published negatives score systematically lower than model-written off-concept texts, the fix
  is to regenerate the pool from its own requests.
