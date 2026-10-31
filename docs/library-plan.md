# Library plan: an activation-hiding experiment kit

Status: proposal (10-31), audited against the code the same day (§12). Nothing here is implemented yet. It is based on a full map of the current code (src/,
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
| `scripts/*.py` | 108 files (53 `tica_*`, 24 `build_*`) | Entry points and glue. `scripts/tica.py` holds the wiring (`build`, `cap_free_requests`) that 22 other scripts import through `sys.path` hacks. |
| `scripts/slurm/*.sh` | 84 files | Three hand-written GPU schedulers, a resume finder written as a Python heredoc, ~32 copies of the model-staging block, hard-coded accounts, paths and venvs. |
| `configs/tica/*.yaml` | 119 files (63 `final_*`, no inheritance: each is a full copy) | Flat run configs. `AdaptiveConfig` has **163 fields**, absolute `/scratch/s/sneheel` paths, and lives at `rl.adaptive` even for the supervised trainer. |
| tests | 85 files, ~589 tests | 48 tica test files on synthetic activations: a good base for refactor safety. |

### What blocks reuse today

1. **One god-object.** `AdaptiveSupervisedTrainer` reads 142 of the 163 `AdaptiveConfig` fields and owns 17 responsibilities:
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
   - 18 `TICA_*` environment variables in Python (more in bash, e.g. `TICA_NODE`), five of which change training semantics: `TICA_STEP_OFFSET` also
     shifts the sampler seed; `TICA_TRUNC_READS`, `TICA_BATCHED_FITS` and `TICA_NEG_LABELS` override config;
     `TICA_DUMP_REFITS` adds outputs.
   - `probe_zoo.set_zoo_depth` mutates module globals from `RunConfig.__post_init__`.
   - Module caches.
5. **Layer lists hard-coded for a 32-layer model** in argparse defaults. Only the panel rescales them, which is a
   latent bug for the 2B (24 layers) and Gemma (48 layers).
6. **Cluster coupling in code.** `vllm_gen` runs `module load`, and the trainer calls sbatch.
7. **Packaging.**
   - vllm and fasttext are not declared.
   - The pins are wrong for the venv we actually run: pyproject says `transformers<5`, `torch<2.7`; the venv has
     transformers 5.14.1, torch 2.9.1, peft 0.19.1.
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

Name (decided 10-31): **`crypsis`** (§10). Dependencies point downward only: `core` ← `models` / `data` / `probes` ← `episodes` / `adversary` /
`objectives` ← `training` / `evals` ← `experiment` ← `cli` / `runtime`.

```text
crypsis/
  core/                  pure, light (numpy; torch optional); imports nothing above it
    types.py             Concept, Text, Turn, Conversation, View(span), Verdict, Readout, text_id (content hash)
    rundir.py            RunDir: adaptive/step_k, resume_N/adaptive/step_k, final -> global steps; newest checkpoint
                         (ONE implementation, used by training, evals, analysis and runtime; replaces 9 parsers)
    metrics.py           auroc, fixed-FPR threshold, TPR@FPR, bootstrap + cluster CIs   (replaces 3 auroc, 2 thresholds, 3 bootstraps)
    registry.py          Registry[T] + entry-point plugin loading (kept from the original design)
    io.py                jsonl / atomic json, result writing, "done" detection
    paths.py             Paths(raw_cache, artifacts, hf_cache) from site profile / env / config interpolation
  models/                activation access
    policy.py            Policy: base + LoRA, reference() (adapter off), merged(), save/load adapter
    reading.py           ActivationReader: decoder-block hooks, mean / token pooling, shared-prefix + incremental
                         reads, truncated depth (from policy.py; exact)
    chat.py              ChatFormat renderers (Qwen3.5 ChatML, Gemma-4) + View spans + system-prompt split
                         (owns what policy.py imports from answer_format today: breaks that cycle)
    depth.py             LayerSpec: "0.375", "12@32", "L12" -> absolute layer for this model; zoo kind scaling
                         (replaces the global set_zoo_depth and the hard-coded 32-layer lists)
    generation/          Generator protocol: HFGenerator (generate_batched), VLLMGenerator (colocated subprocess),
                         one chat-prompt builder for every eval
  data/
    concepts.py          ConceptSet: texts by split (fit / calib / episode), concept splits (train / dev / heldout / ood),
                         negatives, hard negatives, label cleaning (neg_labels)
    tasks/               CHECKABLE TASKS: Task(prompt, check(answer) -> bool, kind)
      sql.py json_spec.py apps_js.py mbpp.py humaneval_js.py form.py (language id, all-caps, code form)
      gsm8k.py           PROMPT, is_correct, work_shown, load_questions, base gens: the pure pieces of trainers/mathrl.py,
                         which the main trainer's cap_gen term imports today (supervised.py:2567, :2873)
    pools.py             CapabilityPool: per concept, compose task sources (own items, APPS, gretel, ToolACE, Dolly)
                         + deterministic hold-out split (replaces scripts/tica.py cap_free_requests)
    recipes/             data builders as recipes (final_v5, ood, safety, emotion, task races), CLI `data build`
  probes/                the monitors
    base.py              Monitor protocol, PoolEntry, MonitorPool (+ save/load)  (monitors.py)
    kinds/               registered PROBE_KINDS: linear, mlp, ens*, pooled (maxpool / attnpool / rollmax), token kinds,
                         gdm_rollattn, McK kinds, meandiff, ood, evdet, literature bank
                         (probe_zoo + adaptive + evdet_defender + evasion_detector + probe_bank)
    fit.py               fit(kind, pos, neg, np, gate) -> Monitor | GateFail   (one entry point; logistic L-BFGS)
    gates.py             GateConfig (AUROC, TPR@FPR on np / ic, CV folds) + FPR calibration
    cache.py             content-addressed cache (panel, answer probes): key = model + data hash + kind + code version
    executors.py         fit execution backends: serial / thread / CUDA stream / CPU process  (speed only, exact)
  episodes/
    conditions.py        CONDITIONS registry: real_score, real_verdict, nobrief, mismatch, neg_text, sham, none
    formats.py           prompt layouts (answer, tagged_v5, repeat), briefs, FEEDBACK_CHANNELS registry (verdict, score)
    sampler.py           EpisodeSampler: mix, switches, turn budget, OFF turns, hidden partner
    rollout.py           ONE lockstep rollout engine (teacher-forced or sampled) with hooks:
                         on_turn_read, on_verdict, on_pass, on_switch  (the adaptive trainer's rollout, de-flagged).
                         Defines the Judge PROTOCOL it needs (judge(readouts) -> Verdict, observe(turn), on_pass) and
                         a fork(traj, turn, user_msg) primitive. Never imports adversary or objectives.
  adversary/             the arms race, as one object (implements episodes.rollout's Judge protocol)
    defender.py          Defender: deploy / judge(readouts) -> Verdict / observe(turn) / refit(...)
    base_bank.py         turn-0 base probes in the answer format, fitted on the model and cached
                         (answer_format.fit_answer_probes; replaces the block pasted into 4 eval scripts)
    refit_policies.py    REFIT_POLICIES: refit_after_pass + newest_judges (main), must_pass_all, accumulate,
                         branch (branching.py: branch-and-collect)
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
    state.py             Checkpoint (adapter + trainer_state) save / load / resume on top of core.rundir
    distributed.py       dp.py + episode RNG streams + rollout merge
    callbacks.py         JSONL / wandb sinks, eval_on_save (calls a Launcher, not sbatch), early stopping
    backends/            reward-based arms via existing RL libraries (§9, §13): trl.py first, verl.py second;
                         optional extras. Reference workloads = the snapshot's RLTrainer and MathRL.
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
    schema.py            dataclasses per component (below), strict (unknown key = error), versioned; a LEAF module
                         (today config.py imports the trainers, rounds, windows and probe_zoo to build itself)
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
  cli/                   `crypsis` (Typer): train, resume, branch, race, eval run SUITE, panel, hide-solve,
                         capability, data build, gen base, ckpt ls, report, cluster node / queue / tasks / campaign
  (no legacy/: the paper replication, synthetic layer, in-context study and the Level-1 / windows / rounds tracks
   live on the archive branch, §7)
```

## 4. What it looks like to use

### Python

```python
from crypsis import Experiment

exp = Experiment.from_config("v4d", model="qwen35-2b",
                             overrides={"objectives.kl.weight": 3.0, "train.steps": 1800})
exp.train()                                   # resumes automatically from exp.run_dir
res = exp.evaluate("final")                   # EvalSuite: held-out race, quality, panel, hide-and-solve, capability
print(res.race["heldout_val"].score, res.hide_solve["sqltest"].by_layer)
```

Lower level, for a new research idea:

```python
from crypsis.models import Policy
from crypsis.probes import fit, GateConfig
from crypsis.episodes import EpisodeSampler, rollout
from crypsis.adversary import Defender, refit_policies

pol = Policy.load("Qwen/Qwen3.5-4B", lora=dict(r=128, alpha=64))
defender = Defender(kinds=["linear@0.375", "attnpool@0.5"], policy=refit_policies.RefitAfterPass(judge="newest"),
                    gate=GateConfig(min_auroc=0.95, fpr=0.01))
traj = rollout(pol, EpisodeSampler(concepts, conditions={"real_score": 0.75, "nobrief": 0.1, "mismatch": 0.1,
               "neg_text": 0.05}), defender, turns=24)
```

### Extending: one registration per idea

```python
from crypsis.objectives import LOSS_TERMS, LossTerm

@LOSS_TERMS.register("entropy_floor")
class EntropyFloor(LossTerm):
    def compute(self, batch, out):        # batch: turns + reads; out: logits / probe probs
        ...
        return loss, {"entropy_floor/value": v}
```

The same pattern covers a probe kind (`PROBE_KINDS`), a refit policy (`REFIT_POLICIES`), an episode condition
(`CONDITIONS`), a feedback channel (`FEEDBACK_CHANNELS`), a checkable task (`TASKS`) and an eval (`EVALS`). External
packs register through the `crypsis.plugins` entry point, as the original design intended.

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
| tica/branching.py | adversary/refit_policies.py (branch mode) |
| tica/anchor.py | objectives/anchor.py |
| tica/probe_bank.py, probes.py, evasion_detector.py | probes/kinds/* (literature bank; the L2 logistic fit merges into probes/fit.py; evdet) |
| tica/panel.py, run_report.py, analysis.py, judge.py | evals/panel.py, analysis/run_report.py, analysis/episodes.py, evals/judge.py |
| trainers/mathrl.py pure pieces (PROMPT, is_correct, work_shown, load_questions, build_base_gens) | data/tasks/gsm8k.py |
| trainers/mathrl.py MathRL, trainers/rl.py RLTrainer | archive branch; reference workloads for training/backends (§13) |
| level1.py prompt strings (opening, VERDICT_FB) | episodes/formats.py |
| level1.py protocol, textcheck.py, env.py, backend.py, evaluation.py, SupervisedTrainer (Level-1, supervised.py:135-606) | archive branch |
| tica/episodes.py, level1.py (prompt strings) | episodes/conditions.py, formats.py, sampler.py |
| trainers/supervised.py `rollout` | episodes/rollout.py + adversary/defender.py + adversary/first_probe.py |
| trainers/supervised.py `step`, `_targets`, KL, stay, cap_* | objectives/* + objectives/capability/* |
| trainers/supervised.py state, DP, logging, workers | training/state.py, distributed.py, callbacks.py, probes/executors.py |
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
| hf/, real/, incontext/, synthetic layer, rounds.py, windows.py + their 32 configs and tests | archive branch |

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

## 7. Decisions (10-31, user)

- **Base:** the tica code. The library is structured from `next_chameleons/tica/` (newer, what we run). The older
  layer contributes only the config search path, `benchmarks/capability*` and the registry / plugin pattern.
- **Paper replication -> archive branch.** `hf/`, `real/`, the synthetic layer, `incontext/`, their configs,
  scripts and tests move to an `archive/paper-replication` branch, not to `legacy/` in the tree. The Level-1 /
  windows / rounds tracks go there too unless a current result depends on them.
- **Saved state:** branch `snapshot/tica-2026-10-31` and tag `tica-v4d-2026-10-31` (both pushed) = the exact
  code of the running v4d jobs. Every result so far stays reproducible from there.
- **Every component is verified** against the current code (§8) before the old path is removed.
- **No implementation yet:** this document is the plan.

## 8. Verification protocol (per component)

Each extracted component lands with an EQUIVALENCE test against the snapshot code on fixed inputs. It is not
"done" until that test passes on CPU, plus GPU where noted.

| Component | Equivalence check |
|---|---|
| models.reading | same pooled / token activations (bit-equal on CPU fp32; GPU within kernel numerics) for Qwen3.5-2B and Gemma-4 on fixed conversations, every read path (full, shared-prefix, incremental, truncated) |
| models.chat | identical rendered token ids for every layout / format x model |
| models.depth | the new layer resolution reproduces every layer list in the current configs (4B / 2B / Gemma) |
| probes.* | same fitted weights, thresholds, gate verdicts (AUROC, TPR) for every kind on fixed activations + seeds; batched == serial; every executor (thread / stream / CPU) == serial |
| data.tasks / pools | same pools (sizes, rows, hold-out split) as `cap_free_requests`; same check verdicts on a fixed answer set (incl. the 298 / 300 fallback self-check) |
| episodes | same sampled specs and rendered prompts / feedback for a fixed seed |
| adversary | same refit decisions (when, which kind, kept / gate) on a recorded rollout |
| objectives | each loss term returns the same value + gradient on a fixed batch |
| training (GPU) | 3-5-step golden trace per model: train_log metrics, refits.jsonl, turns.jsonl verdicts, adapter checksum, resume-from-checkpoint == continuous |
| evals | same race / hide-solve / panel JSON on a fixed adapter (seeded) as the current scripts |
| config.migrate | every one of the ~130 `configs/tica/*.yaml` loads to an identical runtime config |

CPU equivalence tests run in the normal suite. GPU goldens run as one smoke node per model before merging each
phase.

## 9. Training backends

Two kinds of training, two answers:

1. **Activation-gradient training (the core TICA arm): stays native.** The hide / match / panel losses
   differentiate through the probes on the policy's own activations, read with hooks mid-forward. Probes refit
   INSIDE the episode, and the next turn's prompt depends on the verdict, in lockstep across data-parallel ranks.
   RL frameworks optimise a scalar reward from rollouts. They have no differentiable activation-space loss and no
   in-episode adversary, so this arm is our own `training/trainer.py`, built only from library components.
2. **Reward-based training (RL arms: 1-bit verdict as reward, capability correctness, GRPO / PPO / DAPO):
   adapters to existing libraries, not our own RL code.** The library exposes an environment interface:
   - `HidingEnv`: episodes, conditions, feedback channel and defender; `step(turn_text) -> (next prompt,
     verdict)`.
   - `RewardFn`: probe verdict / score, task checks, KL handled by the library.
   - A `ProbeReader` that scores activations of generated text under the CURRENT policy weights.

   Then thin optional adapters:
   - **TRL** (first): HF-native, PEFT / LoRA, GRPO with custom reward functions, single node. Matches our 2B-12B
     scale and our venv, and it is the quickest to verify against our existing `trainers/rl.py` GRPO.
   - **verl** (second, for scale and multi-turn): custom agent loops (`AgentLoop` classes set in config), vLLM /
     SGLang rollouts, FSDP / Megatron, LoRA. The hard part is the probe reward. It needs the policy's
     activations, so it must run inside verl's actor forward pass (log-prob computation) or in a reward worker
     with synced weights. Refits happen between rollout batches (our "accumulate" mode), not mid-episode.

   Adapters live in `training/backends/{trl,verl}.py` behind extras (`pip install crypsis[trl]`, `crypsis[verl]`) and
   never become core dependencies. Each is verified by reproducing the current GRPO arm on a small config before
   use.

## 10. Name

Decided (10-31, user): **`crypsis`**. In biology, crypsis is an organism's ability to avoid detection by others, covering
camouflage, mimicry and the chameleon's colour change. That is exactly "models avoiding detection by activation
probes". It is short and free on PyPI (checked 10-31). Alternatives also free on PyPI: `probewar` (the arms
race), `hidelab`, `latentcloak`, `covertlab`. (`anole`, `cloak`, `veil`, `mimicry`, `obscura`, `masque` are
taken.) Package `crypsis`, CLI `crypsis`, plugin entry-point group `crypsis.plugins`, extras `crypsis[trl]`, `crypsis[verl]`,
`crypsis[vllm]`. `next_chameleons` stays the package name on the archive branch only; `lib` ships no shim for it.

## 11. Open questions

1. ~~Name~~: `crypsis` (decided).
2. ~~Math-RL as a native strategy~~: no (decided, §13).
3. Timing: Phases 0-2 on a `lib` branch in parallel with the running jobs, the trainer split after the current
   runs are evaluated?

## 12. Audit of this plan against the code (10-31, second pass)

Method: every claim in §1-§9 was re-checked on the `tamia` tree at d8e2b4c (module sizes, method sizes, the
intra-package import graph, config field usage, environment variables, scripts, configs, tests, the venv). The
numbers in §1 were corrected in place. What follows is what the first draft got structurally wrong or left
unspecified, and the fix now folded into §3 / §5.

### 12.1 Import cycles the layering must break

The current graph has four cycles or inversions. The library layout only works if each is cut where stated.

| Today | Problem | Cut |
|---|---|---|
| `policy -> answer_format -> {adaptive, data, episodes, level1, monitors, neg_labels}`, and `neg_labels -> answer_format` | the activation reader depends on the whole episode layer for one helper (`split_system`); label cleaning and prompt rendering depend on each other through `text_id` | `models/chat.py` owns the system split; `text_id` moves to `core/types`; `neg_labels` becomes `data/concepts` label cleaning |
| `config -> {trainers.supervised, trainers.rl, rounds, windows, evaluation, objectives, probe_zoo}` | the config module imports the implementations to assemble its nested dataclasses, so loading a config imports the trainers | `config/schema.py` is a leaf; components take their config section as an argument |
| `supervised -> trainers.mathrl` (cap_gen imports `PROMPT`, `is_correct`, `work_shown`, `load_questions`) | the main trainer depends on a side trainer for a task definition | `data/tasks/gsm8k.py` holds the pure GSM8K pieces; both the capability term and any RL backend import it |
| `validity <-> free_pools` (lazy imports both ways) | task checks and pool composition are interleaved | each `data/tasks/*` module owns its check; `data/pools.py` composes tasks and never the reverse |

Two cycles the first draft would have CREATED, now avoided:

- `probes/answer_probes.py` needed prompt rendering (`episodes/formats`) and the model, so `probes` would have
  depended on `episodes`. It is the defender's starting bank, so it is `adversary/base_bank.py`.
- `episodes/rollout.py` taking a `Defender` from `adversary/` while `adversary/` consumes `Trajectory` from
  `episodes/`. The rollout now defines the small Judge protocol it needs; `adversary/` implements it. Likewise the
  rollout exposes `fork(traj, turn, user_msg)` and `objectives/capability/forks.py` calls it, so `episodes` never
  imports `objectives`.
- `RunDir` sat in `training/state.py` but evals, analysis and runtime all need it. It is `core/rundir.py`.

### 12.2 Modules the mapping table had missed

Thirteen tica modules had no row in §5: `textcheck`, `anchor`, `probe_bank`, `probes`, `evasion_detector`,
`judge`, `branching`, `panel`, `run_report`, `evaluation`, `backend`, `env`, `analysis` (plus `level1`, which is
half protocol, half prompt strings). All are now mapped. Five of them (`textcheck`, `env`, `backend`,
`evaluation`, the Level-1 `SupervisedTrainer`) are the Level-1 repeat-the-text track and go to the archive branch
with `rounds` and `windows`; 32 of the 119 configs belong to those tracks. `level1`'s prompt strings are still
imported by the adaptive trainer, `rl` and `mathrl`, so they move to `episodes/formats.py`.

### 12.3 Speed knobs versus semantics

Of the 25 `refit_*` fields, 11 are speed-only and belong in the `perf:` section, each with an equivalence test
against the serial path: `refit_np_cache`, `refit_batched_fits`, `refit_rlobf_prepad`, `refit_batched_rlobf`,
`refit_batched_logistic`, `refit_overlap`, `refit_overlap_stream`, `refit_cpu_fits`, `refit_cpu_threads`,
`refit_gpu_procs`, `refit_stack_episodes`. The same applies to `cap_gen_gen_batch`, `cap_gen_shard`,
`cap_gen_merged`, `merged_reads`, `incremental_rollout`, `truncate_reads`, `episode_fork_batch`, `panel_cache_code`
and the DP / read-batch sizes. `truncate_reads` is exact only because reads stop after the deepest requested layer;
the equivalence test must cover it.

### 12.4 Field usage (input to config v2)

Of the 163 `AdaptiveConfig` fields: 142 are read by the adaptive trainer, 15 only outside it (`rl.py`, the
capability-pool builder in `scripts/tica.py`, the panel and reasoning-budget scripts), and 6 are read nowhere by
attribute name (`pass_reward`, `selectivity_penalty`, `refit_rlobf_prepad`, `refit_batched_rlobf`,
`refit_batched_logistic`, `neg_label_thr`). `config/migrate.py` therefore needs a `deprecated` list that maps a
dead field to "no effect" explicitly, and the migration test asserts that no config in the tree sets a dead field
to a non-default value. The 15 outside-only fields are the first sign of which pieces are not the trainer's
business: the capability-pool keys belong to `data/pools`, the panel keys to `evals/panel`.

### 12.5 Packaging facts

The venv that runs every job (`venvs/q35t29`) has transformers 5.14.1, torch 2.9.1, peft 0.19.1, fasttext 0.9.3
and no vllm (it is loaded through `TICA_VLLM_PY`, a second interpreter). pyproject pins `transformers<5` and
`torch<2.7`, declares `hydra-core`, `omegaconf`, `lm-eval` and `trl` (none imported), and declares neither
`fasttext` nor `vllm`. Phase 0 sets the floors to what runs today and moves vllm / fasttext / trl / verl to extras.

### 12.6 Design rules added

- **Registries only where a config picks by name.** `PROBE_KINDS`, `CONDITIONS`, `FEEDBACK_CHANNELS`,
  `REFIT_POLICIES`, `LOSS_TERMS`, `TASKS`, `EVALS`, `GENERATORS`, `LAUNCHERS`. Everything else is composed in
  Python with plain constructors. One generic `core/registry.py`, no per-module copies.
- **`Experiment` is a builder, not the next god-object.** `Experiment.from_config(cfg) -> Components` (policy,
  concepts, pool, sampler, defender, objectives, capability terms, eval suite) with `train()` and `evaluate()` as
  ten-line conveniences over the components. Every eval and script receives objects, never a config path; the
  CLI does config -> objects.
- **Trajectory is a plain dataclass** that serialises to the current `turns.jsonl` rows, so analysis code and
  the golden traces read old and new runs alike.
- **Equivalence oracles are recorded fixtures, not a dual install.** A script on the snapshot tag records CPU
  goldens on a tiny random model (fixed seed) into `tests/golden/*.json` (aggregates only, small), and GPU goldens
  as JSON aggregates under the artifacts root. The new code is tested against the fixtures; re-recording is a
  documented, reviewed step. This avoids importing two versions of the package into one process.
- **No absolute paths in code.** `trainers/mathrl.py` hard-codes the raw cache root and the configs hard-code
  `/scratch/s/sneheel`; both become `${paths.raw}` / `${paths.artifacts}` through `core/paths.py`.

### 12.7 Phase adjustments

- Phase 0 also: create the archive branch and remove its modules, configs and tests from `lib`; fix the pins;
  record the goldens from the snapshot tag.
- Phase 1 also: the dead-field list (§12.4) and `core/rundir.py` first, since the bash resume logic and the
  eval queues call it.
- New Phase 2b (≈2 days): `adversary/base_bank.py` + `probes/cache.py`, replacing the base-probe block pasted
  into 4 eval scripts and the 11 private trainer attributes read by 5 scripts (`tr._transient` alone 8 times).
  After this, no script constructs `AdaptiveSupervisedTrainer` to evaluate.
- Phase 3 order unchanged; the Judge protocol (§12.1) is introduced at step 4 so step 5 can lift the rollout
  without touching the defender again.

## 13. Decision: the math-RL trainer

**Not a native strategy.** Reasons, all verified on the tree:

1. The recipe we run (v4d) preserves math through `cap_gen` inside the adaptive trainer, which imports only the
   PURE pieces of `mathrl.py` (prompt, answer check, work check, question loading, base gens). Those pieces are
   kept, as `data/tasks/gsm8k.py`, covered by the existing CPU tests (`test_tica_mathrl.py`) and the §8
   equivalence check.
2. `MathRL` (797 lines) and `RLTrainer` (664 lines) are two more GRPO trainers with their own rollout and refit
   code: two of the five rollout engines and two of the four refit implementations the plan exists to remove.
   Carrying either natively keeps that duplication and gives the library two training loops to keep exact.
3. Everything `MathRL` does is expressible in the §9 environment interface: a task turn per step, a pass forks
   the conversation at t+1 and refits on the forks (`HidingEnv.on_pass`), 1-bit verdict feedback, reward =
   correctness × hide bonus, plus the dense probe term through `ProbeReader`. That is exactly the shape the TRL /
   verl adapters need to support, so `MathRL` and `RLTrainer` become the adapters' two REFERENCE WORKLOADS: the
   TRL backend is "supported" only once it reproduces the snapshot `MathRL` and `RLTrainer` on a small config
   (same reward, pass and correctness statistics on fixed seeds, within sampling noise), and verl is verified
   the same way against TRL.
4. Until then the math-RL pilot stays runnable from `snapshot/tica-2026-10-31` / the archive branch. Its 6
   `mathrl_*` configs go there; the `mathrl_*_full/base_gens.json` artifacts the current runs read are data, not
   code, and stay where they are.

What this costs: if we want a solve-while-hiding RL result before the TRL adapter exists, we run it from the
snapshot. What it buys: one native training loop to keep exact, and RL work that rides on a maintained library.
