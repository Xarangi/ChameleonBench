# Library plan: an activation-hiding experiment kit

Status: proposal (10-31). Nothing here is implemented yet. It is based on a full map of the current code (src/,
scripts/, configs/) and on the original library vision in [library-api.md](library-api.md) and
[architecture-audit.md](architecture-audit.md): registries, config packs, plugin entry points, one runner. That
vision was built for the Neural Chameleons replication. The research that followed (TICA: multi-turn hiding
against refit probes) grew next to it as `next_chameleons/tica/` plus ~225 scripts and never used it. The plan
keeps the good parts of the original vision and rebuilds them around what we actually run now.

## 1. Where we are

| Layer | Size | State |
|---|---|---|
| `next_chameleons/tica/` | 39 files, 17.1k LOC | The real research code. `trainers/supervised.py` alone is 4.3k LOC (25%). |
| Older `next_chameleons/*` | ~18.5k LOC | Paper replication (`hf/`, `real/`), synthetic smoke layer, registries, Typer CLI, in-context study. tica uses only the config lookup, `benchmarks/capability*`, `incontext.HFBackend` and `incontext.stats.auroc`. |
| `scripts/*.py` | 130 files, 18k LOC | Entry points and glue. `scripts/tica.py` holds the wiring (`build`, `cap_free_requests`) that 22 other scripts import through `sys.path` hacks. |
| `scripts/slurm/*.sh` | 95 files | Three hand-written GPU schedulers, a resume finder written as a Python heredoc, ~32 copies of the model-staging block, hard-coded accounts, paths and venvs. |
| `configs/tica/*.yaml` | ~130 files | Flat run configs. `AdaptiveConfig` has **163 fields**, absolute `/scratch/s/sneheel` paths, and lives at `rl.adaptive` even for the supervised trainer. |
| tests | 85 files, ~589 tests | 44 tica test files (~339 tests) on synthetic activations: a good base for refactor safety. |

### What blocks reuse today

1. **One god-object.** `AdaptiveSupervisedTrainer` reads 106 config fields and owns 17 responsibilities:
   - optimizer, schedule and resume state;
   - data parallelism;
   - a 725-line rollout that contains the defender, refits, switches, turn budget and readouts;
   - first-probe forks and the panel cache;
   - refit planning, reading and fitting, plus the thread / stream / CPU fit workers;
   - capability terms;
   - KL targets;
   - curriculum bandits;
   - pursuit and stay;
   - a 430-line `step` that holds every loss term;
   - logging;
   - wandb;
   - early stopping;
   - `eval_on_save` via sbatch.

   New ideas land as flags inside `rollout` / `step`, and semantic flags sit next to speed-only flags.
2. **Wiring lives in a script.** Building pools, fitting answer probes, building capability pools and assembling
   the trainer all happen in `scripts/tica.py`, so nothing can be driven from Python.
3. **Duplication.**
   - 5 rollout engines (adaptive trainer, `episodes.rollout`, RL, mathrl, windows).
   - 4 refit implementations.
   - The base answer-probe fitting block is pasted into 4 eval scripts, with variants in 10 more.
   - AUROC is implemented 3 times and the threshold logic twice.
   - `("real_score", "real_verdict")` is redefined 5 times.
   - The run-directory and checkpoint layout is parsed in 8 places, in bash and in Python.
4. **Hidden state and environment.**
   - About 30 `TICA_*` environment variables, some of which change training semantics: `TICA_STEP_OFFSET` also
     shifts the sampler seed, and `TICA_TRUNC_READS` / `TICA_BATCHED_FITS` override config.
   - `probe_zoo.set_zoo_depth` mutates module globals from `RunConfig.__post_init__`.
   - Module caches.
5. **Layer lists hard-coded for a 32-layer model** in argparse defaults. Only the panel rescales them, which is a
   latent bug for the 2B (24 layers) and Gemma (48 layers).
6. **Cluster coupling in code.** `vllm_gen` runs `module load`, and the trainer calls sbatch.
7. **Packaging.**
   - vllm and fasttext are not declared.
   - The `transformers<5` pin is wrong for the venv we actually run.
   - hydra / omegaconf / lm-eval / trl are declared but never imported.
   - The top-level `__init__` star-import loads the whole legacy layer on every tica import.
   - Package data omits the split JSONs.

### What is already library-grade

Most of these move almost as-is:
- `monitors.py`: Monitor protocol, MonitorPool, logistic fit.
- `probe_zoo.py`: kinds, `gate_fit`, batched fits. Only the global depth state needs removing.
- `dp.py` and `curriculum.py`.
- `objectives.py`: the loss math, which is pure.
- The pure parts of `adaptive.py`: `RefitData`, `fit_refit_bank`, the MLP / OOD monitors, the pursuit math.
- `validity.py`, `free_pools.py`, `js_tasks.py`.
- The episode specs and sampler.
- The `policy.py` read / forward core.
- `run_report.py`, `panel.py`, `analysis.py`.
- `benchmarks/capability*`.

## 2. Design goals

1. **An experiment is a composition, not a flag set.** Model × concepts × probe kinds × defender (refit policy) ×
   episode conditions × feedback channel × loss terms × capability terms × evals. Each axis is a small protocol
   with a registry, as in the original vision. Adding an idea means adding a component, not a branch in `step`.
2. **Python first, config second, CLI third.** Anything a config can express, Python can build, and the CLI is a
   thin layer over Python.
3. **Exactness is a contract.** The refactor must reproduce current runs bit-for-bit, or within kernel numerics,
   on fixed seeds (golden traces, §6). Speed paths stay exact and are separate from semantics.
4. **No hidden state.** Everything that changes results lives in the config. Environment variables only select
   resources (GPU, paths).
5. **Model-agnostic depths.** Layers are written as depth fractions or as "L@32" and resolved per model.
6. **Cluster-agnostic.** A site profile file replaces hard-coded accounts, paths, venvs and module loads.
7. **Aggregates-only by default.** Raw text, generations and activations go to the raw cache only when explicitly
   requested (guardrail kept).

## 3. Package structure

Proposed name: keep the `next_chameleons` distribution, with this layout. Renaming it (e.g. `chameleonkit`) is your
call. Dependencies point downward only: `core` ← `models` / `data` / `probes` ← `episodes` / `adversary` /
`objectives` ← `training` / `evals` ← `experiment` ← `cli` / `runtime`.

```text
next_chameleons/
  core/                  pure, light (numpy; torch optional)
    types.py             Concept, Text, Turn, Conversation, View(span), Verdict, Readout
    metrics.py           auroc, fixed-FPR threshold, TPR@FPR, bootstrap + cluster CIs   (replaces 3 auroc, 2 thresholds, 3 bootstraps)
    registry.py          Registry[T] + entry-point plugin loading (kept from the original design)
    io.py                jsonl / atomic json, result writing, "done" detection
    paths.py             Paths(raw_cache, artifacts, hf_cache) from site profile / env / config interpolation
  models/                activation access
    policy.py            Policy: base + LoRA, reference() (adapter off), merged(), save/load adapter
    reading.py           ActivationReader: decoder-block hooks, mean / token pooling, shared-prefix + incremental
                         reads, truncated depth (from policy.py; exact)
    chat.py              ChatFormat renderers (Qwen3.5 ChatML, Gemma-4) + View spans  (inverts policy -> answer_format)
    depth.py             LayerSpec: "0.375", "12@32", "L12" -> absolute layer for this model; zoo kind scaling
                         (replaces the global set_zoo_depth and the hard-coded 32-layer lists)
    generation/          Generator protocol: HFGenerator (generate_batched), VLLMGenerator (colocated subprocess),
                         one chat-prompt builder for every eval
  data/
    concepts.py          ConceptSet: texts by split (fit / calib / episode), concept splits (train / dev / heldout / ood),
                         negatives, hard negatives, label cleaning (neg_labels)
    tasks/               CHECKABLE TASKS: Task(prompt, check(answer) -> bool, kind)
      sql.py json_spec.py apps_js.py gsm8k.py mbpp.py humaneval_js.py form.py (language id, all-caps, code form)
    pools.py             CapabilityPool: per concept, compose task sources (own items, APPS, gretel, ToolACE, Dolly)
                         + deterministic hold-out split (replaces scripts/tica.py cap_free_requests)
    recipes/             data builders as recipes (final_v5, ood, safety, emotion, task races), CLI `data build`
  probes/                the monitors
    base.py              Monitor protocol, PoolEntry, MonitorPool (+ save/load)  (monitors.py)
    kinds/               registered PROBE_KINDS: linear, mlp, ens*, pooled (maxpool / attnpool / rollmax), token kinds,
                         gdm_rollattn, McK kinds, meandiff, ood, evdet   (probe_zoo + adaptive + evdet_defender)
    fit.py               fit(kind, pos, neg, np, gate) -> Monitor | GateFail   (one entry point; logistic L-BFGS)
    gates.py             GateConfig (AUROC, TPR@FPR on np / ic, CV folds) + FPR calibration
    answer_probes.py     base probes in the answer format, cached  (from answer_format.fit_answer_probes)
    cache.py             content-addressed cache (panel, answer probes): key = model + data hash + kind + code version
    executors.py         fit execution backends: serial / thread / CUDA stream / CPU process  (speed only, exact)
  episodes/
    conditions.py        CONDITIONS registry: real_score, real_verdict, nobrief, mismatch, neg_text, sham, none
    formats.py           prompt layouts (answer, tagged_v5, repeat), briefs, FEEDBACK_CHANNELS registry (verdict, score)
    sampler.py           EpisodeSampler: mix, switches, turn budget, OFF turns, hidden partner
    rollout.py           ONE lockstep rollout engine (teacher-forced or sampled) with hooks:
                         on_turn_read, on_verdict, on_pass, on_switch  (the adaptive trainer's rollout, de-flagged)
  adversary/             the arms race, as one object
    defender.py          Defender protocol: deploy / judge(readouts) -> Verdict / observe(turn) / refit(...)
    refit_policies.py    REFIT_POLICIES: refit_after_pass + newest_judges (main), must_pass_all, accumulate, branch
    first_probe.py       base / fork / switch_fork, turn-0 panel
    kind_sampler.py      which kind / layer to refit (curriculum bandits, factored Thompson)
    buffers.py           RefitData (rolling pos / neg / np)
  objectives/            LOSS_TERMS registry; each term: requests(batch) -> reads/gens, compute(...) -> (loss, metrics)
    hide.py match.py off.py panel.py   (objectives.py + the trainer's _targets)
    kl.py                KL to base (kl_ref, top-k), on chosen turns
    pursuit.py stay.py act_anchor.py anchor.py (retain-set)
    capability/
      forks.py           capability forks inside hiding episodes (where / how many; by refit level)
      gsm8k.py free.py   cap_gen / cap_free terms (rejection-sampled NLL + KL; validity-gated via data.tasks)
      level_weights.py   per-refit-level EMAs -> fork weights + loss multipliers (cap_level_adapt)
  training/
    trainer.py           Trainer: loop + optimizer; composes Policy, EpisodeSampler, Defender, rollout, loss terms,
                         capability; no feature flags inside, only components
    schedule.py          LR warmup / peak / hold / cooldown by GLOBAL step
    state.py             RunDir (adaptive/step_k, resume_N/adaptive/step_k, final -> global steps) + Checkpoint
                         (adapter + trainer_state): ONE implementation for resume, quality queues, reports, plots
    distributed.py       dp.py + episode RNG streams + rollout merge
    callbacks.py         JSONL / wandb sinks, eval_on_save (calls a Launcher, not sbatch), early stopping
    strategies/          alternative trainers on the same components: grpo.py (rl.py), mathrl.py
  evals/
    race.py              RaceSpec + presets: heldout_val, quality, persistence (no_refit, always_pass), refit_ladder,
                         interactive task race. Built on episodes.rollout + adversary, so no private-attribute poking
    hide_solve.py        passive hide-and-solve over any data.tasks Task (brief conditions x layers)
    panel.py             never-shown probe panel + shard merge
    capability.py        GSM8K / MMLU / ARC / MATH / MBPP / HumanEval-JS, clean and under briefs (benchmarks/*)
    quality.py judge.py  validators; LLM judge via models.generation
    schemas.py           versioned result dataclasses (RaceResult, HideSolveResult, PanelResult, ...) -> JSON
    suites.py            EvalSuite = named list of EvalTasks (replaces v4d2b_drain.sh, tamia_queue_quality.sh,
                         final_eval_suite.sh)
  analysis/              offline, CPU: run_report, race / task reports, panel report, geometry / anatomy, plots
  config/
    schema.py            dataclasses per component (below), strict (unknown key = error), versioned
    load.py              search path (kept), `extends:` presets, `${paths.raw}` interpolation, dotted overrides
    migrate.py           old flat configs (AdaptiveConfig at rl.adaptive) -> v2, so every existing run stays loadable
  experiment.py          Experiment.from_config(...): builds everything (replaces scripts/tica.py build + main)
  runtime/               operations, optional
    site.py              SiteProfile yaml: account(s), gres, cpus, venv tarball, module loads, paths, node layout
    launch.py            Launcher protocol: local / slurm (rendered from templates) / node-slot packing
    scheduler.py         ONE Python GPU-slot scheduler (replaces tamia_node.sh slots, tica_val_drain.sh,
                         tamia_mechanism_node.sh, tica_val_local.sh): smoke-gated groups, MIN_FREE_MB packing,
                         done-detection, resume-safe
    chains.py            chained resume jobs, takeover, branch-from-checkpoint
    notify.py            optional notifiers (Telegram)
  cli/                   `next-chameleons` (Typer): train, resume, branch, race, eval run SUITE, panel, hide-solve,
                         capability, data build, gen base, ckpt ls, report, cluster node / queue / tasks / campaign
  legacy/                paper replication (hf/, real/), synthetic smoke layer, in-context L0 study, Level-1 / windows /
                         rounds tracks: frozen, behind an extra, out of the default import path
```

## 4. What it looks like to use

### Python

```python
from next_chameleons import Experiment

exp = Experiment.from_config("v4d", model="qwen35-2b",
                             overrides={"objectives.kl.weight": 3.0, "train.steps": 1800})
exp.train()                                   # resumes automatically from exp.run_dir
res = exp.evaluate("final")                   # EvalSuite: held-out race, quality, panel, hide-and-solve, capability
print(res.race["heldout_val"].score, res.hide_solve["sqltest"].by_layer)
```

Lower level, for a new research idea:

```python
from next_chameleons.models import Policy
from next_chameleons.probes import fit, GateConfig
from next_chameleons.episodes import EpisodeSampler, rollout
from next_chameleons.adversary import Defender, refit_policies

pol = Policy.load("Qwen/Qwen3.5-4B", lora=dict(r=128, alpha=64))
defender = Defender(kinds=["linear@0.375", "attnpool@0.5"], policy=refit_policies.RefitAfterPass(judge="newest"),
                    gate=GateConfig(min_auroc=0.95, fpr=0.01))
traj = rollout(pol, EpisodeSampler(concepts, conditions={"real_score": 0.75, "nobrief": 0.1, "mismatch": 0.1,
               "neg_text": 0.05}), defender, turns=24)
```

### Extending: one registration per idea

```python
from next_chameleons.objectives import LOSS_TERMS, LossTerm

@LOSS_TERMS.register("entropy_floor")
class EntropyFloor(LossTerm):
    def compute(self, batch, out):        # batch: turns + reads; out: logits / probe probs
        ...
        return loss, {"entropy_floor/value": v}
```

The same pattern covers a probe kind (`PROBE_KINDS`), a refit policy (`REFIT_POLICIES`), an episode condition
(`CONDITIONS`), a feedback channel (`FEEDBACK_CHANNELS`), a checkable task (`TASKS`) and an eval (`EVALS`). External
packs register through the `next_chameleons.plugins` entry point, as the original design intended.

### Config v2 (the current strict-KL 2B, expressed compositionally)

```yaml
extends: presets/v4d            # recipe: conditions, defender, objectives, capability, schedule
name: final_v4d_klx_tamia_2b
model: {preset: qwen35-2b}      # path + revision + depth + chat format come from the model preset
paths: {out: ${paths.artifacts}/runs/final_v4d_klx_tamia_2b}
defender:
  policy: refit_after_pass      # newest probe judges
  kinds: {layers: [8@32, 12@32, 16@32, 22@32, 26@32]}  # resolves to 6 / 9 / 12 / 17 / 20 on 24 layers
  held_out: {layers: [20@32, 24@32]}                   # 15 / 18
objectives:
  - {term: hide, loss: log, clip: 0.5}
  - {term: kl, weight: 3.0, ref: nobrief}
  - {term: anchor, weight: 1.0}
capability:
  gsm8k: {pool: ${paths.artifacts}/runs/mathrl_qwen35_2b_full/base_gens.json, episodes: 4}
  free: {sources: [own, apps_js, gretel_sql, toolace_json, dolly], holdout: 0.25, episodes: 4, nobrief: 2}
  level_weights: {ref_gsm: 0.75, ref_free: 0.75, floor: 0.25, gain: 2.0}
train: {steps: 1800, lr: 1e-4, cooldown: 150, episodes_per_step: 16, turns: 24}
perf: {dp: 4, read_batch: 128, sort_reads: true, refit_overlap: true}   # speed only: never changes results
```

## 5. Mapping: current file -> new home

| Current | New |
|---|---|
| tica/policy.py | models/policy.py, models/reading.py, models/chat.py, models/generation/hf.py |
| tica/vllm_gen.py | models/generation/vllm.py (module loads move to the site profile) |
| tica/monitors.py, probe_zoo.py, evdet_defender.py, adaptive.py (fits, MLP, OOD) | probes/base.py, probes/kinds/*, probes/fit.py, probes/gates.py |
| tica/answer_format.py | episodes/formats.py (prompts) + probes/answer_probes.py (fitting + cache) |
| tica/adaptive.py (AdaptiveDefender, RefitData, pursuit) | adversary/*, objectives/pursuit.py |
| tica/episodes.py, level1.py (prompt strings) | episodes/conditions.py, formats.py, sampler.py |
| trainers/supervised.py `rollout` | episodes/rollout.py + adversary/defender.py + adversary/first_probe.py |
| trainers/supervised.py `step`, `_targets`, KL, stay, cap_* | objectives/* + objectives/capability/* |
| trainers/supervised.py state, DP, logging, workers | training/state.py, distributed.py, callbacks.py, probes/executors.py |
| trainers/rl.py, mathrl.py | training/strategies/ |
| validity.py, free_pools.py, js_tasks.py, benchmarks/capability* | data/tasks/*, data/pools.py, evals/capability.py |
| data.py, neg_labels.py, multi.py | data/concepts.py |
| curriculum.py | adversary/kind_sampler.py |
| dp.py | training/distributed.py |
| config.py (163-field AdaptiveConfig) | config/schema.py (per component) + config/migrate.py |
| scripts/tica.py build / cap_free_requests / main | experiment.py + cli `train` |
| scripts/tica_switch_eval.py, tica_frozen_race.py, tica_val*.py | evals/race.py presets |
| scripts/tica_hide_solve.py, tica_capability.py, tica_probe_panel.py, tica_quality_check.py, tica_free_judge.py | evals/* |
| scripts/build_final_*.py | data/recipes/ (v3 / v4 kept as recipe versions so v5 stays reproducible) |
| scripts/tica_*_report.py, tica_plot.py, tica_hide_geometry.py + mechanism scripts | analysis/* |
| scripts/slurm/* schedulers, resume, drains, queues, staging | runtime/* + one sbatch template + site profile |
| hf/, real/, incontext/, synthetic layer, rounds.py, windows.py, env.py, backend.py | legacy/ |

## 6. How to get there without breaking the running research

Work on a `lib` branch. Production runs stay on `tamia` until each phase passes its checks. Every phase ends with
the full CPU test suite and the golden traces.

**Phase 0: guardrails (≈1 day)**
- **Golden traces.**
  - Record 3-5 steps of the strict-KL 2B and the 4B configs on fixed seeds: train_log metrics, refits.jsonl
    (kinds, kept, AUROC), turns.jsonl verdicts and the adapter checksum.
  - Each record runs as one GPU smoke per model.
  - Add CPU-only goldens on a tiny random model for the test suite.
- **Packaging fixes.**
  - Declare vllm and fasttext extras and fix the transformers pin to match the venv.
  - Drop unused dependencies.
  - Remove the `__init__` star-import.
  - Include the split JSONs in package data.

**Phase 1: leaves (no behaviour change, ≈3-4 days)**
- core/metrics, core/paths with the site profile, models/depth (fixes the 32-layer defaults),
  training/state.RunDir.
  - RunDir replaces the 8 checkpoint parsers; the bash scripts call `next-chameleons ckpt newest`.
- data/tasks and data/pools, probes/* (monitors + zoo + fits, global depth state removed), models/generation.
- Shims keep the old import paths working.

**Phase 2: Experiment builder (≈2 days)**
- `experiment.py` replaces `scripts/tica.py build` and `cap_free_requests`.
- The eval scripts switch to the builder and to a public `Race` API instead of trainer private attributes.

**Phase 3: split the trainer (≈1-1.5 weeks; the core of the work)**
- Extract in this order, re-running the golden traces after each step:
  1. state / logging / DP / fit executors;
  2. capability terms;
  3. loss terms;
  4. the defender and refit policies;
  5. the rollout engine.
- The trainer ends as composition only.

**Phase 4: config v2 (≈2-3 days)**
- Per-component schema, `extends` presets, model presets and a perf section.
- `migrate.py` loads every existing `configs/tica/*.yaml` and produces an identical runtime config; a test
  verifies this over all ~130 configs.
- Environment variables that change results become config fields.

**Phase 5: evals and operations (≈1 week)**
- Result schemas, EvalSuite ("final", "quality", "checkpoint"), and the Python scheduler / launcher / chains
  replacing the bash schedulers.
- CLI commands as listed in §3.

**Phase 6: docs and examples (≈2-3 days)**
- A 30-line "hide a concept from a linear probe" example on a small model.
- Extension guides (new probe kind, loss term, refit policy, task) and the plugin entry point.
- Updated agent guide.

Total ≈4-5 weeks of focused work. Phases 0-2 alone (≈1 week) already make the code reusable from Python and
remove most duplication. Phase 3 is where the risk is, which is why the golden traces come first.

## 7. Decisions for you

1. **Name:** keep `next_chameleons` (CLI `next-chameleons`), or rename it for the library release?
2. **Legacy:** keep the paper replication, synthetic layer and in-context study in-tree under `legacy/` (an
   optional extra), or move them to a separate archive branch?
3. **Side tracks:** keep the RL / GRPO and mathrl trainers as supported strategies, or only the adaptive
   supervised trainer? The Level-1 / windows / rounds tracks look dead.
4. **Timing:** start Phases 0-2 now on `lib`, in parallel with the running jobs, and leave Phase 3 (trainer split)
   until the current runs finish and are evaluated?
